import asyncio
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from Embeder_module import Embedder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from Vector_db import VectorDB


class RAGPipeline:

    def __init__(self, embedder: Embedder, vector_db: VectorDB, llm: ChatOllama):
        self.embedder = embedder
        self.db = vector_db
        self.llm = llm

    def _build_product_card(self, payload: Dict[str, Any]) -> str:
        """Форматирует метаданные товара из Qdrant в структурированный вид."""
        brand = payload.get("Бренд", payload.get("brand", "Неизвестный бренд"))
        series = (
            payload.get("Серия Lessar")
            or payload.get("Серия Ballu")
            or payload.get("Серия Royal Clima")
            or payload.get("Серия Electrolux")
            or payload.get("series", "Базовая")
        )
        area = payload.get(
            "Эффективен для помещ. площадью до (м2)",
            payload.get("area", "Не указана"),
        )
        inverter = payload.get(
            "Инверторная технология", payload.get("inverter", "Нет")
        )
        wifi = payload.get(
            "Управление c мобильного приложения по Wi-Fi",
            payload.get("wifi", "Нет"),
        )
        noise = payload.get(
            "Уровень шума внутр. блока", payload.get("noise", "Не указан")
        )
        product_id = payload.get("id", "Без ID")
        description = payload.get("text", payload.get("description", ""))

        return f"""
=== ТОВАР ID: {product_id} ===
Модель: Кондиционер {brand} (Серия: {series})
Рекомендуемая площадь: до {area} м²
Инвертор: {inverter}
Поддержка Wi-Fi: {wifi}
Уровень шума: {noise}
Описание: {description}
=========================""".strip()

    def _generate_context(self, payloads: List[Dict[str, Any]]) -> str:
        """Удаляет дубликаты товаров по ID и собирает контекст."""
        seen_ids = set()
        cards = []
        for payload in payloads:
            p_id = payload.get("id")
            if p_id and p_id in seen_ids:
                continue
            if p_id:
                seen_ids.add(p_id)
            cards.append(self._build_product_card(payload))

        if not cards:
            return "В каталоге не найдено подходящих товаров по вашему запросу."

        return "\n\n".join(cards)

    async def async_answer_question(
        self,
        user_query: str,
        chat_history: Optional[List[Any]] = None,
    ) -> str:
        """Асинхронный поиск и генерация ответа с защитой event loop."""
        
        # 1. Поисковый запрос берем напрямую из user_query (без мусора из истории)
        search_text = user_query.strip()
        
        print(f"\n[RAG] Ищем в Qdrant по тексту: '{search_text}'")

        # 2. Оборачиваем синхронный генератор эмбеддингов и векторный поиск в asyncio.to_thread
        # Это освободит FastAPI event loop и предотвратит зависания
        try:
            query_vector = await asyncio.to_thread(self.embedder.get_embedding, search_text)
            payloads = await asyncio.to_thread(self.db.search, query_vector, limit=5)
        except Exception as e:
            print(f"[RAG ERROR] Ошибка при поиске в БД: {e}")
            payloads = []

        print(f"[RAG] Найдено объектов в БД: {len(payloads)}")

        context = self._generate_context(payloads)

        # 3. Универсальный и четкий системный промпт
        system_instructions = (
            "Ты — профессиональный AI-консультант магазина климатической техники SmartKlimat74.\n"
            "ОБЯЗАТЕЛЬНОЕ ПРАВИЛО: ОТВЕЧАЙ СТРОГО НА РУССКОМ ЯЗЫКЕ.\n\n"
            "ТВОЯ ЗАДАЧА:\n"
            "Помочь клиенту подобрать кондиционер из предоставленного ниже каталога.\n\n"
            "ПРАВИЛА ОТВЕТА:\n"
            "1. Предлагай ТОЛЬКО модели из блока 'КАТАЛОГ ТОВАРОВ ИЗ БАЗЫ ДАННЫХ'. Не придумывай несуществующие товары.\n"
            "2. Выбери 2-3 наиболее подходящие модели по площади и запрошенным функциям (инвертор, Wi-Fi и т.д.).\n"
            "3. Указывай название бренда, серию, рекомендованную площадь и ключевые особенности.\n"
            "4. Если в каталоге нет подходящей модели, вежливо сообщи об этом и предложи уточнить требования.\n\n"
            f"КАТАЛОГ ТОВАРОВ ИЗ БАЗЫ ДАННЫХ:\n{context}"
        )

        formatted_messages = [SystemMessage(content=system_instructions)]

        # 4. Безопасное добавление истории сообщений
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

        # Убеждаемся, что последнее сообщение — текущий запрос пользователя
        if not formatted_messages or formatted_messages[-1].content != user_query:
            formatted_messages.append(HumanMessage(content=user_query))

        print("[RAG] Генерация ответа через ChatOllama...")
        response = await self.llm.ainvoke(formatted_messages)

        return response.content