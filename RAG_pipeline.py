import asyncio
from typing import Any, Dict, List, Optional, Tuple
from Embeder_module import Embedder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from Vector_db import VectorDB


class RAGPipeline:

    def __init__(self, embedder: Embedder, vector_db: VectorDB, llm: ChatOllama):
        self.embedder = embedder
        self.db = vector_db
        self.llm = llm

    def _extract_payload(self, item: Any) -> Dict[str, Any]:
        """Универсально извлекает dict из объекта ответа Qdrant (ScoredPoint) или возвращает сам dict."""
        if isinstance(item, dict):
            if "payload" in item and isinstance(item["payload"], dict):
                return item["payload"]
            return item
        if hasattr(item, "payload") and isinstance(item.payload, dict):
            return item.payload
        return {}

    def _build_product_card(self, payload: Dict[str, Any]) -> str:
        """Динамически формирует карточку товара из всех доступных полей metadata."""
        product_id = payload.get("id", payload.get("row_id", "Без ID"))
        excluded_keys = {"id", "row_id", "_id", "page_content", "full_text"}
        
        card_lines = [f"=== ТОВАР ID: {product_id} ==="]
        for key, value in payload.items():
            if key not in excluded_keys and value is not None:
                card_lines.append(f"{key}: {value}")

        card_lines.append("=========================")
        return "\n".join(card_lines)

    def _generate_context(self, raw_results: List[Any]) -> str:
        """Приводит результаты к словарям, удаляет дубликаты по ID и собирает единый контекст."""
        seen_ids = set()
        cards = []
        
        for item in raw_results:
            payload = self._extract_payload(item)
            if not payload:
                continue

            p_id = payload.get("id", payload.get("row_id"))
            if p_id and p_id in seen_ids:
                continue
            if p_id:
                seen_ids.add(p_id)
            
            cards.append(self._build_product_card(payload))

        if not cards:
            return "В каталоге не найдено подходящих товаров по вашему запросу."

        return "\n\n".join(cards)

    async def get_relevant_docs(self, user_query: str, limit: int = 4) -> str:
        """Извлекает контекст из Qdrant в виде форматированного текста карточек товаров."""
        search_text = user_query.strip()
        try:
            # Неблокирующий асинхронный получение эмбеддингов и векторный поиск
            if hasattr(self.embedder, "aget_embedding"):
                query_vector = await self.embedder.aget_embedding(search_text)
            else:
                query_vector = await asyncio.to_thread(self.embedder.get_embedding, search_text)

            if hasattr(self.db, "asearch"):
                raw_results = await self.db.asearch(query_vector, limit=limit)
            else:
                raw_results = await asyncio.to_thread(self.db.search, query_vector, limit=limit)

            return self._generate_context(raw_results)
        except Exception as e:
            print(f"[RAG ERROR] Ошибка при поиске в БД: {e}")
            return "Ошибка получения данных из базы знаний."

    async def async_answer_question_with_docs(
        self,
        user_query: str,
        chat_history: Optional[List[Any]] = None,
    ) -> Tuple[str, str]:
        """Возвращает кортеж: (ответ_модели, текстовый_контекст)."""
        context = await self.get_relevant_docs(user_query)
        answer = await self.async_answer_question(user_query, chat_history, pre_fetched_context=context)
        return answer, context

    async def async_answer_question(
        self,
        user_query: str,
        chat_history: Optional[List[Any]] = None,
        pre_fetched_context: Optional[str] = None,
    ) -> str:
        """Асинхронный поиск и генерация ответа."""
        search_text = user_query.strip()
        print(f"\n[RAG] Ищем в Qdrant по тексту: '{search_text}'")

        context = pre_fetched_context if pre_fetched_context is not None else await self.get_relevant_docs(search_text)

        # Вынос фиксированной системной инструкции наверх для эффективного Prompt Caching
        system_instructions = (
            "Ты — профессиональный AI-консультант магазина климатической техники SmartKlimat74.\n"
            "ОБЯЗАТЕЛЬНОЕ ПРАВИЛО: ОТВЕЧАЙ СТРОГО НА РУССКОМ ЯЗЫКЕ.\n\n"
            "ТВОЯ ЗАДАЧА:\n"
            "Помочь клиенту подобрать подходящий товар из предоставленного ниже каталога.\n\n"
            "ПРАВИЛА ОТВЕТА:\n"
            "1. Предлагай ТОЛЬКО модели из блока 'КАТАЛОГ ТОВАРОВ ИЗ БАЗЫ ДАННЫХ'. Не придумывай несуществующие товары или характеристики.\n"
            "2. Опирайся только на ту информацию и параметры (название, цена, описание, URL), которые переданы в каталоге.\n"
            "3. Указывай точные названия товаров, их ключевые особенности и цены (если они присутствуют в карточке).\n"
            "4. Если в каталоге нет подходящей модели, вежливо сообщи об этом и предложи уточнить требования."
        )

        user_content = f"КАТАЛОГ ТОВАРОВ ИЗ БАЗЫ ДАННЫХ:\n{context}\n\nВОПРОС: {search_text}"

        formatted_messages: List[Any] = [SystemMessage(content=system_instructions)]

        if chat_history:
            for msg in chat_history[-4:]:
                if isinstance(msg, (HumanMessage, AIMessage)):
                    formatted_messages.append(msg)
                elif isinstance(msg, dict):
                    role = msg.get("role")
                    content = msg.get("content", "")
                    if role == "user":
                        formatted_messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        formatted_messages.append(AIMessage(content=content))

        if not formatted_messages or formatted_messages[-1].content != user_content:
            formatted_messages.append(HumanMessage(content=user_content))

        print("[RAG] Генерация ответа через ChatOllama...")
        response = await self.llm.ainvoke(formatted_messages)

        return response.content