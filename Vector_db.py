from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List
from Embeder_module import Embedder
from typing import List, Dict, Any

class VectorDB:
    def __init__(self, url: str, collection_name: str, vector_size: int):
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._init_collection()

    def _init_collection(self):
        """Создает коллекцию, если её нет."""
        if not self.client.collection_exists(self.collection_name):
            print(f"Создание новой коллекции Qdrant: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
        else:
            print(f"Коллекция '{self.collection_name}' уже существует.")

    def upsert_documents(self, splitted_documents: List[Document], embedder: Embedder):
        """Кодирует тексты чанков и загружает в Qdrant."""
        points = []
        print("Кодирование чанков и подготовка к загрузке...")
        
        for idx, doc in enumerate(splitted_documents):
            vector = embedder.get_embedding(doc.page_content)
            
            payload = doc.metadata.copy()
            payload["text"] = payload.get("full_text", doc.page_content)
            if "full_text" in payload:
                del payload["full_text"]

            points.append(PointStruct(id=idx, vector=vector, payload=payload))

        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"Успешно загружено {len(points)} точек в Qdrant.")

    def search(self, query_vector: List[float], limit: int = 3) -> List[Dict[str, Any]]:
        """Ищет похожие векторы в базе."""
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True
        )
        return [hit.payload for hit in results.points]