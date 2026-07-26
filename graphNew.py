from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import AgentState
from nodes import (
    auth_user,
    route_question,
    call_rag,
    register_client,
    handle_checkout,
    handle_about_company,
    handle_off_topic,
)

# 1. Инициализация графа
workflow = StateGraph(AgentState)

# 2. Регистрация узлов (Nodes)
workflow.add_node("auth_node", auth_user)
workflow.add_node("rag", call_rag)
workflow.add_node("register", register_client)
workflow.add_node("checkout", handle_checkout)
workflow.add_node("about_company", handle_about_company)
workflow.add_node("off_topic", handle_off_topic)

# 3. Точка входа
workflow.add_edge(START, "auth_node")

# 4. Главная маршрутизация после проверки авторизации
workflow.add_conditional_edges(
    "auth_node",
    route_question,
    {
        "rag": "rag",
        "about_company": "about_company",
        "register": "register",
        "off_topic": "off_topic",
    },
)

# 5. Переход из ветки регистрации в checkout при завершении сбора данных
def route_register_step(state: AgentState) -> str:
    if state.get("next_step") == "checkout":
        return "checkout"
    return "end"

workflow.add_conditional_edges(
    "register",
    route_register_step,
    {
        "checkout": "checkout",
        "end": END,
    },
)

# 6. Завершение прохода
workflow.add_edge("rag", END)
workflow.add_edge("about_company", END)
workflow.add_edge("checkout", END)
workflow.add_edge("off_topic", END)

# 7. Компиляция с чекпоинтером памяти
memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)