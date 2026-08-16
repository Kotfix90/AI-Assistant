# main.py

import json
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import uvicorn

from graphNew import app_graph
from nodes import clean_llm_response

app = FastAPI(title="SmartKlimat74 AI Backend")

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


STREAMABLE_NODES = {"rag", "off_topic", "about_company", "checkout", "register"}


async def generate_chat_stream(user_id: str, message: str, thread_id: str) -> AsyncGenerator[str, None]:
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [HumanMessage(content=message)], "user_id": user_id}

    streamed_nodes = set()
    logged_starts = set()

    try:
        async for event in app_graph.astream_events(inputs, config, version="v2"):
            kind = event["event"]
            node_name = event.get("metadata", {}).get("langgraph_node")

            # Логирование старта узлов
            if kind == "on_chain_start" and node_name:
                if node_name not in logged_starts:
                    logged_starts.add(node_name)
                    print(f"\n🚀 [LANGGRAPH NODE START]: ===> {node_name} <===")

            elif kind == "on_chain_end":
                name = event.get("name")
                if name in ["route_question", "route_register_step"]:
                    output = event["data"].get("output")
                    print(f"🔀 [ROUTER DECISION]: {name} -> выбрал путь: '{output}'")

            # -------------------------------------------------------------
            # ПОТОКОВЫЙ ВЫВОД ПО ТОКЕНАМ ДЛЯ ВСЕХ ТЕКСТОВЫХ УЗЛОВ (ВКЛЮЧАЯ RAG)
            # -------------------------------------------------------------
            if kind == "on_chat_model_stream":
                tags = event.get("tags", []) or []

                # Служебные вызовы (перефразирование запроса, классификация intent,
                # извлечение имени/телефона/адреса, guardrail) помечены тегом "internal"
                # в nodes.py и не должны попадать в поток к пользователю.
                if "internal" in tags:
                    continue

                if node_name in STREAMABLE_NODES:
                    chunk = event["data"].get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        token = chunk.content
                        if isinstance(token, str) and token:
                            streamed_nodes.add(node_name)
                            payload = json.dumps({"content": token}, ensure_ascii=False)
                            yield f"data: {payload}\n\n"

            # -------------------------------------------------------------
            # ФОЛЛБЕК: Если узел завершился, но стриминг токенов не сработал
            # -------------------------------------------------------------
            elif kind == "on_chain_end":
                if node_name in STREAMABLE_NODES:
                    if node_name not in streamed_nodes:
                        output_data = event["data"].get("output", {})
                        if isinstance(output_data, dict) and "messages" in output_data:
                            messages = output_data["messages"]
                            if messages:
                                last_msg = messages[-1]
                                content = getattr(last_msg, "content", "")
                                if content and isinstance(content, str):
                                    cleaned_content = clean_llm_response(content)
                                    if cleaned_content:
                                        streamed_nodes.add(node_name)
                                        payload = json.dumps({"content": cleaned_content}, ensure_ascii=False)
                                        yield f"data: {payload}\n\n"

    except Exception as e:
        print(f"❌ [STREAM ERROR]: {str(e)}")
        err_payload = json.dumps({"error": f"Ошибка сервера: {str(e)}"}, ensure_ascii=False)
        yield f"data: {err_payload}\n\n"

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