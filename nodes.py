import asyncio
import re
import time
from functools import wraps
from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import db, embedder, route_llm, llm, llm_structured, rag_pipeline
from state import (
    AgentState,
    ClientProfile,
    ExtractName,
    ExtractPhone,
    ExtractAddress,
)
from sql_module import Customer

# ============================================================================
# КОНФИГ / ПЕРЕКЛЮЧАТЕЛИ
# ============================================================================

STRICT_GUARDRAIL = False

SECURITY_TRIGGERS = ["игнорируй", "предыдущие инструкции", "напиши код", "системный промпт", "jailbreak"]
CANCEL_WORDS = ["отмена", "стоп", "назад", "не хочу", "сброс"]


# ============================================================================
# ЗАМЕР ВРЕМЕНИ — ЕДИНАЯ ТОЧКА ДЛЯ ВСЕХ УЗЛОВ
# ============================================================================

def timed_node(node_name: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            print(f"[TIMING] node={node_name} status=start")
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - t0
                print(f"[TIMING] node={node_name} status=done elapsed={elapsed:.2f}s")
        return wrapper
    return decorator


class StageTimer:
    """Помощник для точечных замеров времени внутри узлов."""
    def __init__(self, node_name: str):
        self.node_name = node_name
        self._t_start = time.perf_counter()
        self._t_last = self._t_start

    def mark(self, stage_name: str):
        now = time.perf_counter()
        print(f"[TIMING]   node={self.node_name} stage={stage_name} "
              f"stage_elapsed={now - self._t_last:.2f}s total_elapsed={now - self._t_start:.2f}s")
        self._t_last = now


# ============================================================================
# ОЧИСТКА ОТВЕТОВ МОДЕЛИ
# ============================================================================

def clean_llm_response(text: str) -> str:
    """Полностью очищает ответ LLM от JSON-блоков, служебной информации и фигурных скобок."""
    if not text:
        return ""

    text = re.sub(r'\{[\s\S]*?\}', '', text)
    text = re.sub(r'["\']?(is_faithful|reasoning|query)["\']?\s*:\s*.*?,?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^(Подскажи|Какие|Поисковый запрос|Вопрос|Перефразированный вопрос).*?:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[([^\]]+)\]\((https?://[^\s)]*)?$', r'\1', text)
    text = re.sub(r'\b(Об|Ответ|Итог)\s*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


# ============================================================================
# GUARDRAIL ДОСТОВЕРНОСТИ ОТВЕТА
# ============================================================================

class FaithfulnessGuardrail(BaseModel):
    is_faithful: bool = Field(
        description="True, если ответ не содержит явных противоречий контексту или вымышленных брендов. False в случае критических галлюцинаций."
    )
    reasoning: str = Field(description="Причина решения.")


async def validate_rag_answer(user_query: str, retrieved_context: str, bot_answer: str) -> bool:
    """LLM-проверка ответа на соответствие контексту (антигаллюцинация)."""
    if not retrieved_context.strip():
        return True

    evaluator = llm_structured.with_structured_output(FaithfulnessGuardrail)

    system_prompt = (
        "Ты — модуль проверки достоверности ответов климатического магазина.\n"
        "Проверь, не выдумал ли бот несуществующие бренды или характеристики, которых явно нет в контексте.\n"
        "Если ответ логичен и опирается на предоставленный контекст — ставь is_faithful = True."
    )

    user_prompt = f"Контекст:\n{retrieved_context}\n\nВопрос: {user_query}\n\nОтвет бота: {bot_answer}"

    t0 = time.perf_counter()
    try:
        result = await evaluator.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ],
            config={"tags": ["internal"]},
        )
        print(f"[TIMING]   guardrail_llm_call elapsed={time.perf_counter() - t0:.2f}s")
        return result.is_faithful
    except Exception as e:
        print(f"[Guardrail Error]: {e} (elapsed={time.perf_counter() - t0:.2f}s)")
        return True


def _check_facts_in_context(bot_answer: str, retrieved_context: str) -> bool:
    """Быстрая программная проверка ссылок в ответе против контекста (без LLM)."""
    urls_in_answer = set(re.findall(r'https?://\S+', bot_answer))
    if not urls_in_answer:
        return True
    return all(url.rstrip(').,') in retrieved_context for url in urls_in_answer)


async def _log_guardrail_async(user_query: str, context: str, answer: str) -> None:
    """Фоновая (не блокирующая ответ пользователю) проверка — только для логов/алертов."""
    try:
        is_faithful = await validate_rag_answer(user_query, context, answer)
        facts_ok = _check_facts_in_context(answer, context)
        if not is_faithful or not facts_ok:
            print(f"[GUARDRAIL ALERT] Подозрение на галлюцинацию. faithful={is_faithful}, facts_ok={facts_ok}\n"
                  f"Запрос: {user_query}\nОтвет: {answer[:300]}...")
    except Exception as e:
        print(f"[Guardrail Log Error]: {e}")


# ============================================================================
# РОУТИНГ + ПЕРЕФРАЗИРОВАНИЕ ЗАПРОСА
# ============================================================================

class RouteDecision(BaseModel):
    intent: Literal["rag", "about_company", "register", "off_topic"] = Field(
        description="Намерение пользователя с учётом контекста беседы."
    )
    search_query: Optional[str] = Field(
        default=None,
        description=(
            "ТОЛЬКО если intent == 'rag': автономный поисковый запрос по каталогу "
            "кондиционеров, переформулированный с учётом истории диалога "
            "(площадь, бюджет, бренд, тип помещения). Для остальных intent — null."
        ),
    )


_route_query_cache: Dict[str, str] = {}


def _cache_key(state: AgentState, message_text: str) -> str:
    return f"{state.get('user_id')}::{message_text}"


@timed_node("route_question")
async def route_question(state: AgentState) -> Literal["rag", "about_company", "register", "off_topic"]:
    messages = state.get("messages", [])
    if not messages:
        return "rag"

    last_message = messages[-1].content.strip()
    last_lower = last_message.lower()

    if any(word in last_lower for word in CANCEL_WORDS):
        print("[TIMING]   route_question: cancel-word shortcut, LLM не вызывался")
        return "rag"

    if any(trigger in last_lower for trigger in SECURITY_TRIGGERS):
        print("[TIMING]   route_question: security-trigger shortcut, LLM не вызывался")
        return "off_topic"

    history_context = []
    for msg in messages[-4:]:
        role = "Пользователь" if isinstance(msg, HumanMessage) else "Бот"
        history_context.append(f"{role}: {msg.content}")
    context_str = "\n".join(history_context)

    t0 = time.perf_counter()
    try:
        system_prompt = (
            "Ты — классификатор намерений клиента магазина климатической техники SmartKlimat74.\n"
            "Определи intent последнего сообщения:\n"
            "- 'register': желание купить, оформить заказ, оставить заявку или передача контактных данных (имя, телефон, адрес).\n"
            "- 'rag': вопросы по подбору кондиционеров, сплит-систем, площади помещения, ценам, характеристикам, наличию и монтажу.\n"
            "- 'about_company': вопросы о магазине, контактах, режиме работы, адресе или возможностях бота.\n"
            "- 'off_topic': любые сторонние темы, не связанные с климатической техникой, микроклиматом и услугами магазина.\n\n"
            "Если intent == 'rag' — ДОПОЛНИТЕЛЬНО сформулируй search_query: автономный "
            "поисковый запрос по каталогу, учитывающий контекст диалога (площадь, бюджет, "
            "бренд). Если intent != 'rag' — search_query оставь null."
        )
        user_prompt = f"Контекст:\n{context_str}\n\nПоследнее сообщение: {last_message}"

        classifier = route_llm.with_structured_output(RouteDecision)
        res: RouteDecision = await classifier.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ],
            config={"tags": ["internal"]},
        )
        print(f"[TIMING]   route_llm_classify elapsed={time.perf_counter() - t0:.2f}s -> intent={res.intent}")

        if res.intent == "rag" and res.search_query:
            _route_query_cache[_cache_key(state, last_message)] = res.search_query.strip('"')

        return res.intent
    except Exception as e:
        print(f"[Router Error]: {e} (elapsed={time.perf_counter() - t0:.2f}s)")
        return "rag"


async def contextualize_question(user_query: str, chat_history: list) -> str:
    """Резервный путь перефразирования — используется в call_rag только при промахе кэша."""
    if not chat_history:
        return user_query

    recent_history = chat_history[-4:]
    history_str = "\n".join([
        f"{'Пользователь' if isinstance(m, HumanMessage) else 'Бот'}: {m.content}"
        for m in recent_history
    ])

    system_prompt = (
        "Ты — внутренний поисковый модуль. Перефразируй последний вопрос пользователя в один автономный поисковый запрос по каталогу кондиционеров.\n"
        "Учти контекст диалога (площадь, бренд, бюджет, тип помещения).\n"
        "Выдавай ИСКЛЮЧИТЕЛЬНО итоговую строку запроса. Не пиши никакой вводный текст, пояснения или приветствия."
    )

    prompt = f"История:\n{history_str}\n\nВопрос: {user_query}\n\nПоисковый запрос:"
    t0 = time.perf_counter()
    try:
        res = await route_llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=prompt)],
            config={"tags": ["internal"]},
        )
        print(f"[TIMING]   contextualize_question (кэш-промах, резервный вызов) elapsed={time.perf_counter() - t0:.2f}s")
        cleaned = clean_llm_response(res.content).strip('"')
        return cleaned if cleaned else user_query
    except Exception as e:
        print(f"[Contextualize Error]: {e} (elapsed={time.perf_counter() - t0:.2f}s)")
        return user_query


# ============================================================================
# АВТОРИЗАЦИЯ
# ============================================================================

@timed_node("auth_user")
async def auth_user(state: AgentState) -> Dict[str, Any]:
    user_id = state.get("user_id")
    client_info = state.get("client_info")

    if not client_info and user_id:
        try:
            t0 = time.perf_counter()
            existing_user = await asyncio.to_thread(Customer.get_by_id, str(user_id))
            print(f"[TIMING]   auth_db_lookup elapsed={time.perf_counter() - t0:.2f}s")
            if existing_user:
                profile = ClientProfile(
                    status="already_registered",
                    name=existing_user.get("name"),
                    phone=existing_user.get("phone"),
                    address=existing_user.get("address")
                )
                return {"client_info": profile.model_dump()}
        except Exception as e:
            print(f"[Auth DB Error]: {e}")

    if not client_info:
        profile = ClientProfile()
        return {"client_info": profile.model_dump()}

    return {}


# ============================================================================
# УЗЕЛ: О КОМПАНИИ
# ============================================================================

@timed_node("handle_about_company")
async def handle_about_company(state: AgentState) -> Dict[str, Any]:
    system_prompt = (
        "Ты — официальный консультант компании SmartKlimat74.\n"
        "Отвечай строго естественным текстом от лица компании.\n"
        "КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО использовать JSON, фигурные скобки {}, или шаблоны вида [Имя].\n\n"
        "Информация о компании:\n"
        "• Специализация: Продажа, подбор и профессиональный монтаж кондиционеров и сплит-систем.\n"
        "• Адрес: г. Челябинск, ул. Примерная, д. 10\n"
        "• Режим работы: Пн-Пт с 09:00 до 18:00\n"
        "• Телефон: +7 (351) 000-00-00\n\n"
        "Ответь клиенту вежливо, конкретно и без лишних вводных слов."
    )

    messages = [SystemMessage(content=system_prompt)] + state["messages"][-3:]

    t0 = time.perf_counter()
    first_token_at: Optional[float] = None
    response = ""
    async for chunk in llm.astream(messages):
        if first_token_at is None and chunk.content:
            first_token_at = time.perf_counter()
            print(f"[TIMING]   handle_about_company: time_to_first_token={first_token_at - t0:.2f}s")
        response += chunk.content
    print(f"[TIMING]   handle_about_company: full_generation elapsed={time.perf_counter() - t0:.2f}s")

    return {"messages": [AIMessage(content=clean_llm_response(response))]}


# ============================================================================
# УЗЕЛ: OFF-TOPIC
# ============================================================================

@timed_node("handle_off_topic")
async def handle_off_topic(state: AgentState) -> Dict[str, Any]:
    system_prompt = (
        "Ты — консультант климатического магазина SmartKlimat74.\n"
        "Запрос пользователя не относится к тематике магазина.\n"
        "Вежливо и коротко объясни, что ты специализируешься только на подборе и монтаже климатического оборудования (кондиционеры, сплит-системы) и предложи помощь по этой теме.\n"
        "НЕ ИСПОЛЬЗУЙ фигурные скобки {}, JSON или шаблоны с переменными."
    )

    messages = [SystemMessage(content=system_prompt)] + [state["messages"][-1]]

    t0 = time.perf_counter()
    first_token_at: Optional[float] = None
    response = ""
    async for chunk in llm.astream(messages):
        if first_token_at is None and chunk.content:
            first_token_at = time.perf_counter()
            print(f"[TIMING]   handle_off_topic: time_to_first_token={first_token_at - t0:.2f}s")
        response += chunk.content
    print(f"[TIMING]   handle_off_topic: full_generation elapsed={time.perf_counter() - t0:.2f}s")

    return {"messages": [AIMessage(content=clean_llm_response(response))]}


# ============================================================================
# УЗЕЛ: RAG (подбор товара)
# ============================================================================

@timed_node("call_rag")
async def call_rag(state: AgentState) -> Dict[str, Any]:
    st = StageTimer("call_rag")

    messages = state.get("messages", [])
    if not messages:
        return {"messages": [AIMessage(content="Чем я могу помочь вам в выборе климатической техники?")]}

    last_user_message = messages[-1].content
    history = messages[:-1]

    # 1. Переформулированный запрос из кэша либо резервный вызов
    cache_key = _cache_key(state, last_user_message)
    standalone_query = _route_query_cache.pop(cache_key, None)
    cache_hit = standalone_query is not None
    if not standalone_query:
        standalone_query = await contextualize_question(last_user_message, history)
    st.mark(f"rewrite (cache_hit={cache_hit})")

    # 2. Поиск документов в базе данных
    retrieved_docs = ""
    try:
        if hasattr(rag_pipeline, "get_relevant_docs"):
            retrieved_docs = await rag_pipeline.get_relevant_docs(standalone_query)
        elif hasattr(rag_pipeline, "retriever"):
            docs = await rag_pipeline.retriever.ainvoke(standalone_query)
            retrieved_docs = "\n\n".join([d.page_content for d in docs])
        else:
            if hasattr(embedder, "aget_embedding"):
                query_vector = await embedder.aget_embedding(standalone_query)
            else:
                query_vector = await asyncio.to_thread(embedder.get_embedding, standalone_query)
            
            if hasattr(db, "asearch"):
                search_results = await db.asearch(query_vector=query_vector, limit=4)
            else:
                search_results = await asyncio.to_thread(db.search, query_vector=query_vector, limit=4)

            retrieved_docs = "\n\n".join([
                (res["payload"].get("text", "") if isinstance(res, dict) else res.payload.get("text", ""))
                for res in search_results if res
            ])
    except Exception as e:
        print(f"[RAG Retrieval Error]: {e}")
    st.mark("retrieval (embedding + vector_db)")

    if not retrieved_docs:
        return {"messages": [AIMessage(content="К сожалению, по вашему запросу ничего не найдено.")]}

    # 3. Промпт для генерации ответа
    rag_system_prompt = (
        "Ты — консультант магазина климатической техники SmartKlimat74.\n"
        "Ответь на вопрос пользователя, используя только предоставленный контекст.\n\n"
        "СТРОГИЕ ПРАВИЛА ВЫВОДА:\n"
        "1. Начинай ответ сразу с готового решения или подборки товаров.\n"
        "2. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО выводить повтор вопроса, служебные заголовки, JSON или скобки {}.\n"
        "3. Обязательно для каждого предложенного варианта форматируй ссылки строго в формате Markdown: [Посмотреть на сайте](URL).\n"
        "4. Очень коротко поясняй почему представленные варианты подходят."
    )
    full_prompt = f"Контекст из базы знаний:\n{retrieved_docs}\n\nВопрос пользователя: {last_user_message}"
    llm_messages = [
        SystemMessage(content=rag_system_prompt),
        HumanMessage(content=full_prompt),
    ]

    if STRICT_GUARDRAIL:
        response = await llm.ainvoke(llm_messages, config={"tags": ["internal"]})
        st.mark("generation (ainvoke, strict mode)")
        clean_content = clean_llm_response(response.content)

        is_faithful = await validate_rag_answer(last_user_message, retrieved_docs, clean_content)
        facts_ok = _check_facts_in_context(clean_content, retrieved_docs)
        st.mark("guardrail_check")

        if not is_faithful or not facts_ok:
            print(f"[GUARDRAIL BLOCKED] faithful={is_faithful}, facts_ok={facts_ok}. Ответ заменён на безопасный fallback.")
            clean_content = (
                "К сожалению, я не могу уверенно подобрать вариант по вашему запросу — "
                "уточните, пожалуйста, требования (площадь помещения, бюджет, бренд), "
                "и я подберу подходящие модели из каталога."
            )

        return {"messages": [AIMessage(content=clean_content)]}

    else:
        t_gen = time.perf_counter()
        first_token_at: Optional[float] = None
        full_response = ""
        async for chunk in llm.astream(llm_messages):
            if first_token_at is None and chunk.content:
                first_token_at = time.perf_counter()
                print(f"[TIMING]   call_rag: TIME TO FIRST TOKEN={first_token_at - t_gen:.2f}s "
                      f"(с начала узла: {first_token_at - st._t_start:.2f}s)")
            full_response += chunk.content
        st.mark(f"generation (llm.astream total={time.perf_counter() - t_gen:.2f}s)")

        clean_content = clean_llm_response(full_response)

        asyncio.create_task(_log_guardrail_async(last_user_message, retrieved_docs, clean_content))

        return {"messages": [AIMessage(content=clean_content)]}


# ============================================================================
# УЗЕЛ: РЕГИСТРАЦИЯ КЛИЕНТА
# ============================================================================

@timed_node("register_client")
async def register_client(state: AgentState) -> Dict[str, Any]:
    st = StageTimer("register_client")

    current_info_raw = state.get("client_info") or {}
    profile = (
        ClientProfile(**current_info_raw)
        if isinstance(current_info_raw, dict)
        else current_info_raw
    )

    messages = state.get("messages", [])
    last_message = messages[-1].content.strip() if messages else ""
    current_step = state.get("next_step")

    profile.status = "registering"

    if not current_step:
        system_prompt = "Ты — менеджер магазина SmartKlimat74. Вежливо спроси у клиента его имя для оформления заявки. Без JSON и скобок. Не привествуй пользователя повторно"
        res = await llm.ainvoke([SystemMessage(content=system_prompt)])
        st.mark("ask_name_prompt (ainvoke)")
        return {
            "messages": [AIMessage(content=clean_llm_response(res.content))],
            "client_info": profile.model_dump(),
            "next_step": "ask_name",
        }

    if current_step == "ask_name":
        try:
            ext = await llm_structured.with_structured_output(ExtractName).ainvoke(
                last_message, config={"tags": ["internal"]}
            )
            name = ext.name if ext and ext.name else last_message
        except Exception:
            name = last_message
        st.mark("extract_name (structured output)")

        profile.name = name
        system_prompt = f"Клиента зовут {profile.name}. Поблагодари его по имени и попроси указать контактный номер телефона. Пиши только текст без JSON."
        res = await llm.ainvoke([SystemMessage(content=system_prompt)])
        st.mark("ask_phone_prompt (ainvoke)")
        return {
            "messages": [AIMessage(content=clean_llm_response(res.content))],
            "client_info": profile.model_dump(),
            "next_step": "ask_phone",
        }

    if current_step == "ask_phone":
        try:
            ext = await llm_structured.with_structured_output(ExtractPhone).ainvoke(
                last_message, config={"tags": ["internal"]}
            )
            phone_raw = ext.phone if ext and ext.phone else last_message
        except Exception:
            phone_raw = last_message
        st.mark("extract_phone (structured output)")

        digits_only = re.sub(r"\D", "", phone_raw)
        if len(digits_only) < 7:
            system_prompt = "Пользователь указал некорректный номер. Вежливо попроси ввести правильный номер телефона. Без JSON и скобок."
            res = await llm.ainvoke([SystemMessage(content=system_prompt)])
            st.mark("ask_phone_retry_prompt (ainvoke)")
            return {
                "messages": [AIMessage(content=clean_llm_response(res.content))],
                "client_info": profile.model_dump(),
                "next_step": "ask_phone",
            }

        profile.phone = phone_raw
        system_prompt = "Номер принят. Вежливо попроси пользователя указать адрес доставки или установки оборудования. Без JSON и скобок."
        res = await llm.ainvoke([SystemMessage(content=system_prompt)])
        st.mark("ask_address_prompt (ainvoke)")
        return {
            "messages": [AIMessage(content=clean_llm_response(res.content))],
            "client_info": profile.model_dump(),
            "next_step": "ask_address",
        }

    if current_step == "ask_address":
        try:
            ext = await llm_structured.with_structured_output(ExtractAddress).ainvoke(
                last_message, config={"tags": ["internal"]}
            )
            address = ext.address if ext and ext.address else last_message
        except Exception:
            address = last_message
        st.mark("extract_address (structured output)")

        profile.address = address
        profile.status = "already_registered"

        return {
            "client_info": profile.model_dump(),
            "next_step": "checkout",
        }

    return {}


# ============================================================================
# УЗЕЛ: ОФОРМЛЕНИЕ ЗАКАЗА
# ============================================================================

@timed_node("handle_checkout")
async def handle_checkout(state: AgentState) -> Dict[str, Any]:
    st = StageTimer("handle_checkout")

    current_info_raw = state.get("client_info") or {}
    profile = (
        ClientProfile(**current_info_raw)
        if isinstance(current_info_raw, dict)
        else current_info_raw
    )
    user_id = state.get("user_id")

    if profile.name and profile.phone and profile.address and user_id:
        try:
            customer = Customer(
                name=profile.name,
                phone=profile.phone,
                address=profile.address
            )
            await asyncio.to_thread(customer.save, str(user_id))
            st.mark("db_save")
        except Exception as e:
            print(f"[DB Error]: {e}")

    profile.status = "already_registered"

    system_prompt = (
        "Сформируй итоговое подтверждение заказа обычным понятным текстом.\n"
        "НЕ ИСПОЛЬЗУЙ JSON И ФИГУРНЫЕ СКОБКИ!\n"
        f"Имя: {profile.name}\nТелефон: {profile.phone}\nАдрес: {profile.address}\n"
        "Сообщи, что заявка успешно принята и менеджер SmartKlimat74 свяжется с ним в ближайшее время."
    )

    messages = [SystemMessage(content=system_prompt)]
    t0 = time.perf_counter()
    first_token_at: Optional[float] = None
    response = ""
    async for chunk in llm.astream(messages):
        if first_token_at is None and chunk.content:
            first_token_at = time.perf_counter()
            print(f"[TIMING]   handle_checkout: time_to_first_token={first_token_at - t0:.2f}s")
        response += chunk.content
    st.mark("generation")

    return {
        "messages": [AIMessage(content=clean_llm_response(response))],
        "client_info": profile.model_dump(),
        "next_step": None
    }