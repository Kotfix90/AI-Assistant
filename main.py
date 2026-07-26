import asyncio
import json
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import uvicorn

# Импортируем скомпилированный граф из graphNew.py
from graphNew import app_graph

app = FastAPI(title="SmartKlimat74 AI Backend")

# Настройка CORS для работы с веб-фронтендом
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    user_id: str
    message: str
    thread_id: str


async def generate_chat_stream(
    user_id: str, message: str, thread_id: str
) -> AsyncGenerator[str, None]:
    """Генератор SSE-событий с фильтрацией технических токенов LLM."""
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [HumanMessage(content=message)], "user_id": user_id}

    # Множество для отслеживания узлов, которые уже отстримили токены в реальном времени
    streamed_nodes = set()

    try:
        async for event in app_graph.astream_events(inputs, config, version="v2"):
            kind = event["event"]

            # --- 1. Потоковый стриминг по токенам ---
            if kind == "on_chat_model_stream":
                node_name = event.get("metadata", {}).get("langgraph_node", "")

                # ВНИМАНИЕ: Стримятся токены ТОЛЬКО для узлов генерации обычного текста!
                # Узел 'register' исключен, так как внутри него LLM генерирует служебный JSON (Pydantic schema).
                if node_name in ["rag", "off_topic"]:
                    chunk = event["data"].get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        content = chunk.content
                        # Проверяем, что это обычная текстовая строка, а не структура/tool_calls
                        if isinstance(content, str) and content:
                            streamed_nodes.add(node_name)
                            payload = json.dumps({"content": content}, ensure_ascii=False)
                            yield f"data: {payload}\n\n"
                            await asyncio.sleep(0.001)

            # --- 2. Фолбэк/Цельный вывод по завершении работы узла ---
            elif kind == "on_chain_end":
                node_name = event.get("name") or event.get("metadata", {}).get("langgraph_node")

                # Список всех узлов, отправляющих финальные текстовые сообщения в state
                if node_name in ["rag", "register", "off_topic", "checkout", "about_company"]:
                    # Если узел НЕ стримил токены поштучно (например, register или about_company)
                    if node_name not in streamed_nodes:
                        output_data = event["data"].get("output", {})
                        if isinstance(output_data, dict) and "messages" in output_data:
                            messages = output_data["messages"]
                            if messages:
                                last_msg = messages[-1]
                                content = getattr(last_msg, "content", "")
                                if content and isinstance(content, str):
                                    payload = json.dumps({"content": content}, ensure_ascii=False)
                                    yield f"data: {payload}\n\n"

    except Exception as e:
        print(f"[STREAM ERROR]: {str(e)}")
        err_payload = json.dumps({"error": f"Ошибка сервера: {str(e)}"}, ensure_ascii=False)
        yield f"data: {err_payload}\n\n"

    # Сигнал клиенту о завершении потока
    yield "data: [DONE]\n\n"


@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    if app_graph is None:
        raise HTTPException(status_code=500, detail="LangGraph не инициализирован")

    return StreamingResponse(
        generate_chat_stream(
            user_id=request.user_id,
            message=request.message,
            thread_id=request.thread_id,
        ),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)