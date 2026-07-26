import asyncio
from typing import Any, Dict, Literal
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import db, embedder, llm, llm_structured, rag_pipeline
from state import (
    AgentState,
    ClientProfile,
    ExtractName,
    ExtractPhone,
    ExtractAddress,
    IntentDecision,
)
from sql_module import Customer  # Подключаем модуль с классом Customer


# --- Перефразирование вопросов для RAG (Контекстная память) ---
async def contextualize_question(user_query: str, chat_history: list) -> str:
    """Делает короткий запрос автономным на основе истории общения."""
    if not chat_history:
        return user_query

    recent_history = chat_history[-6:]
    history_str = "\n".join([
        f"{'Пользователь' if isinstance(m, HumanMessage) else 'Бот'}: {m.content}"
        for m in recent_history
    ])

    system_prompt = (
        "Сформулируй автономный вопрос по выбору/характеристикам кондиционеров на основе истории диалога и последнего сообщения.\n"
        "Если последнее сообщение содержит местоимения ('из них', 'этот', 'первый'), замени их реальными названиями или параметрами из истории.\n"
        "НЕ ОТВЕЧАЙ на вопрос, только перефразируй его. Если перефразирование не требуется, верни исходный текст."
    )

    prompt = f"История:\n{history_str}\n\nПоследнее сообщение: {user_query}\n\nАвтономный вопрос:"
    try:
        res = await llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=prompt)])
        return res.content.strip()
    except Exception as e:
        print(f"[Contextualize Error]: {e}")
        return user_query


# --- Маршрутизатор (Router) ---
async def route_question(
    state: AgentState,
) -> Literal["rag", "about_company", "register", "off_topic"]:
    messages = state.get("messages", [])
    if not messages:
        return "rag"

    last_message = messages[-1].content.strip()
    last_lower = last_message.lower()

    # 1. Отмена процесса регистрации
    if any(word in last_lower for word in ["отмена", "стоп", "назад", "не хочу", "сброс"]):
        return "rag"

    # 2. Формируем контекст сообщений для роутера
    history_context = []
    for msg in messages[-4:]:
        role = "Пользователь" if isinstance(msg, HumanMessage) else "Бот"
        history_context.append(f"{role}: {msg.content}")

    context_str = "\n".join(history_context)

    # 3. Контекстная классификация через LLM
    try:
        classifier = llm_structured.with_structured_output(IntentDecision)
        system_prompt = (
            "Ты — классификатор намерений в диалоговом боте компании SmartKlimat74 (климатическая техника).\n"
            "Определи intent ПОСЛЕДНЕГО сообщения с учетом контекста.\n\n"
            "Категории:\n"
            "- 'register': явная просьба оформить заявку/заказ ИЛИ ответ пользователя с данными (имя, телефон, адрес), когда бот просит их указать.\n"
            "- 'rag': подбор оборудования, расчёт мощности по площади, характеристики, цены, сравнение моделей, монтаж.\n"
            "- 'about_company': контакты, адрес, время работы, а также общие вопросы 'Чем вы занимаетесь?', 'Что умеет бот?'.\n"
            "- 'off_topic': любые сторонние темы, не относящиеся к климату и услугам магазина."
        )

        user_prompt = f"История диалога:\n{context_str}\n\nОпредели intent для последнего сообщения."
        res = await classifier.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        return res.intent
    except Exception as e:
        print(f"[Router Error]: {e}")
        return "rag"


# --- 0. Авторизация и профиль ---
async def auth_user(state: AgentState) -> Dict[str, Any]:
    user_id = state.get("user_id")
    client_info = state.get("client_info")

    # Если профиль еще не загружен в стейт и у нас есть user_id
    if not client_info and user_id:
        try:
            # Вызываем синхронный staticmethod get_by_id в отдельном потоке
            existing_user = await asyncio.to_thread(Customer.get_by_id, str(user_id))
            
            if existing_user:
                profile = ClientProfile(
                    status="already_registered",
                    name=existing_user.get("name"),
                    phone=existing_user.get("phone"),
                    address=existing_user.get("address")
                )
                print(f"[Auth Success]: Найдена запись в БД для user_id={user_id}")
                return {"client_info": profile.model_dump()}
        except Exception as e:
            print(f"[Auth DB Error]: Ошибка при поиске пользователя: {e}")

    # Если в БД пользователя нет, создаем пустой гостевой профиль
    if not client_info:
        profile = ClientProfile()
        return {"client_info": profile.model_dump()}

    return {}


# --- 1. Информация о компании ---
async def handle_about_company(state: AgentState) -> Dict[str, Any]:
    reply = (
        "🏢 **Компания SmartKlimat74**\n\n"
        "Мы занимаемся подбором, продажей, доставкой и монтажом климатического оборудования "
        "(кондиционеры, сплит-системы, мульти-сплит системы).\n\n"
        "🤖 **Чем я могу помочь:**\n"
        "• Рассчитать мощность кондиционера по площади помещения;\n"
        "• Подобрать модели под ваш бюджет и предпочтения;\n"
        "• Проконсультировать по установке и техническим характеристикам;\n"
        "• Оформить заявку на покупку или выезд замерщика.\n\n"
        "📍 **Контакты:**\n"
        "• **Адрес:** г. Челябинск, ул. Примерная, д. 10\n"
        "• **Режим работы:** Пн-Пт с 09:00 до 18:00\n"
        "• **Телефон:** +7 (351) 000-00-00\n"
        "• **Сайт:** smartklimat74.ru"
    )
    return {"messages": [AIMessage(content=reply)]}


# --- 2. Оффтопик ---
async def handle_off_topic(state: AgentState) -> Dict[str, Any]:
    reply = (
        "Я — специализированный AI-консультант магазина **SmartKlimat74**.\n\n"
        "Я могу отвечать только на вопросы, связанные с климатической техникой, "
        "подбором кондиционеров, их монтажом и работой нашего магазина.\n\n"
        "Подскажите, какая площадь у вашего помещения или какой бюджет вы рассматриваете?"
    )
    return {"messages": [AIMessage(content=reply)]}


# --- 3. Вызов RAG ---
async def call_rag(state: AgentState) -> Dict[str, Any]:
    messages = state.get("messages", [])
    if not messages:
        return {"messages": [AIMessage(content="Напишите ваш запрос, и я подберу кондиционер.")]}

    last_user_message = messages[-1].content
    history = messages[:-1]

    # Перефразируем запрос с учетом памяти перед поиском в RAG
    standalone_query = await contextualize_question(last_user_message, history)

    answer = await rag_pipeline.async_answer_question(
        user_query=standalone_query, chat_history=history
    )
    return {"messages": [AIMessage(content=answer)]}


# --- 4. Регистрация клиента (FSM) ---
async def register_client(state: AgentState) -> Dict[str, Any]:
    current_info_raw = state.get("client_info") or {}
    profile = (
        ClientProfile(**current_info_raw)
        if isinstance(current_info_raw, dict)
        else current_info_raw
    )

    messages = state.get("messages", [])
    last_message = messages[-1].content.strip()
    current_step = state.get("next_step")

    profile.status = "registering"

    # Шаг 0: Запуск процесса
    if not current_step:
        reply = (
            "С удовольствием помогу оформить заявку на покупку или монтаж!\n\n"
            "Подскажите, пожалуйста, **как к вам обращаться (Ваше имя)?**"
        )
        return {
            "messages": [AIMessage(content=reply)],
            "client_info": profile.model_dump(),
            "next_step": "ask_name",
        }

    # Шаг 1: Извлекаем имя
    if current_step == "ask_name":
        try:
            ext = await llm_structured.with_structured_output(ExtractName).ainvoke(last_message)
            name = ext.name if ext and ext.name else last_message
        except Exception:
            name = last_message

        profile.name = name
        reply = f"Отлично, {profile.name}! Укажите ваш **номер телефона** для связи:"
        return {
            "messages": [AIMessage(content=reply)],
            "client_info": profile.model_dump(),
            "next_step": "ask_phone",
        }

    # Шаг 2: Извлекаем телефон
    if current_step == "ask_phone":
        try:
            ext = await llm_structured.with_structured_output(ExtractPhone).ainvoke(last_message)
            phone = ext.phone if ext and ext.phone else last_message
        except Exception:
            phone = last_message

        profile.phone = phone
        reply = "Принято! Напишите **адрес доставки или установки** оборудования:"
        return {
            "messages": [AIMessage(content=reply)],
            "client_info": profile.model_dump(),
            "next_step": "ask_address",
        }

    # Шаг 3: Извлекаем адрес
    if current_step == "ask_address":
        try:
            ext = await llm_structured.with_structured_output(ExtractAddress).ainvoke(last_message)
            address = ext.address if ext and ext.address else last_message
        except Exception:
            address = last_message

        profile.address = address
        profile.status = "already_registered"

        return {
            "client_info": profile.model_dump(),
            "next_step": "checkout",  # Включаем автоматический переход в checkout
        }

    return {}


# --- 5. Оформление заявки (Checkout) ---
async def handle_checkout(state: AgentState) -> Dict[str, Any]:
    current_info_raw = state.get("client_info") or {}
    profile = (
        ClientProfile(**current_info_raw)
        if isinstance(current_info_raw, dict)
        else current_info_raw
    )
    user_id = state.get("user_id")

    db_saved = False

    # Сохраняем в Postgres, если есть данные и user_id
    if profile.name and profile.phone and profile.address and user_id:
        try:
            customer = Customer(
                name=profile.name,
                phone=profile.phone,
                address=profile.address
            )
            # Вызываем метод save(user_id) в потоке
            await asyncio.to_thread(customer.save, str(user_id))
            db_saved = True
            print(f"[DB Success]: Пользователь {profile.name} сохранен в Postgres!")
        except Exception as e:
            print(f"[DB Error]: Не удалось сохранить в БД: {e}")

    # Переводим статус в зарегистрированный
    profile.status = "already_registered"

    if db_saved:
        bot_reply = (
            f"✅ **Заявка успешно оформлена!**\n\n"
            f"• **Имя:** {profile.name}\n"
            f"• **Телефон:** {profile.phone}\n"
            f"• **Адрес:** {profile.address}\n\n"
            "Менеджер компании **SmartKlimat74** свяжется с вами в ближайшее время для уточнения деталей!"
        )
    else:
        bot_reply = (
            f"⚠️ **Данные приняты!**\n\n"
            f"• **Имя:** {profile.name}\n"
            f"• **Телефон:** {profile.phone}\n"
            f"• **Адрес:** {profile.address}\n\n"
            "Заявка зафиксирована. Менеджер свяжется с вами в ближайшее время."
        )

    return {
        "messages": [AIMessage(content=bot_reply)],
        "client_info": profile.model_dump(),
        "next_step": None
    }