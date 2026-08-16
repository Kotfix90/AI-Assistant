# config.py
import os
from langchain_ollama import ChatOllama
from Embeder_module import Embedder
from Vector_db import VectorDB
from RAG_pipeline import RAGPipeline
from dotenv import load_dotenv
load_dotenv()

# Конфигурация
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "documents")
LLM_NAME = os.getenv("LLM_MODEL", "akdengi/saiga-llama3-8b:latest")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "qwen2.5:3b")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# Инициализируем компоненты
embedder = Embedder()
db = VectorDB(
    url=QDRANT_URL, 
    collection_name=COLLECTION_NAME, 
    vector_size=embedder.vector_size
)

route_llm = ChatOllama(
    model=ROUTER_MODEL,
    temperature=0.0,
    num_predict=100,
    num_ctx=2048,
    request_timeout=5.0,
    keep_alive="30m",   # добавить
)

llm = ChatOllama(
    model=LLM_NAME,
    temperature=0.2,
    num_predict=400,
    num_ctx=4096,
    request_timeout=20.0,
    keep_alive="30m",   # добавить
)

llm_structured = ChatOllama(
    model=LLM_NAME,
    temperature=0.0,
    num_predict=200,
    request_timeout=20.0,
    keep_alive="30m",   # добавить
)

# 3. Создаем пайплайн RAG, который импортируют nodes.py
rag_pipeline = RAGPipeline(
    embedder=embedder, 
    vector_db=db, 
    llm=llm
)