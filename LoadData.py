# ingest.py
import os
from Embeder_module import Embedder
from Vector_db import VectorDB
from MetaData import DataLoader

def main():
    print("[Ingest] Инициализация моделей для загрузки данных...")
    embedder = Embedder()
    
    print("[Ingest] Подключение к Qdrant...")
    db = VectorDB(url="http://localhost:6333", collection_name="documents", vector_size=embedder.vector_size)
    
    excel_path = "data/cleaned_data.xlsx"
    if not os.path.exists(excel_path):
        print(f"[Ingest] Ошибка: Файл {excel_path} не найден!")
        return

    print(f"[Ingest] Чтение документов из {excel_path}...")
    loader = DataLoader(file_path=excel_path)
    splitted_docs = loader.load_and_split()
    
    print(f"[Ingest] Получено чанков после сплиттера: {len(splitted_docs)}")
    print("[Ingest] Кодирование чанков и загрузка в Qdrant...")
    
    # Загружаем данные в базу
    db.upsert_documents(splitted_docs, embedder)
    
    print(" Все данные успешно загружены в Qdrant.")

if __name__ == "__main__":
    main()