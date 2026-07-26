# config.py
import os
from langchain_ollama import ChatOllama
from Embeder_module import Embedder
from Vector_db import VectorDB
from RAG_pipeline import RAGPipeline  # <-- Добавили импорт класса

# Конфигурация
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "documents")
LLM_NAME = os.getenv("LLM_MODEL", "qwen2.5:7b")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# Инициализируем компоненты
embedder = Embedder()
db = VectorDB(
    url=QDRANT_URL, 
    collection_name=COLLECTION_NAME, 
    vector_size=embedder.vector_size
)

# 1. Основная модель для разговорных задач
llm = ChatOllama(
    model=LLM_NAME, 
    temperature=0.3,
    request_timeout=120.0
)

# 2. Модель для строгого вывода (JSON / Router)
llm_structured = ChatOllama(
    model=LLM_NAME,
    temperature=0.0,
    request_timeout=60.0
)

# 3. Создаем пайплайн RAG, который импортируют nodes.py
rag_pipeline = RAGPipeline(
    embedder=embedder, 
    vector_db=db, 
    llm=llm
)