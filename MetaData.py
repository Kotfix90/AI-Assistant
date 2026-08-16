import pandas as pd
import re
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DataLoader:
    def __init__(self, file_path: str, chunk_size: int = 700, chunk_overlap: int = 100):
        self.file_path = file_path
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )

    def _clean_html(self, raw_text: str) -> str:
        """Очищает текст от HTML-тегов и спецсимволов (&nbsp;, &#43; и т.д.)."""
        if not isinstance(raw_text, str) or not raw_text:
            return ""
        
        text = raw_text.replace('&nbsp;', ' ').replace('&#43;', '+').replace('&laquo;', '«').replace('&raquo;', '»')
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def load_and_split(self) -> List[Document]:
        """
        Загружает файл (Excel/CSV), сопоставляет метаданные динамически 
        по колонкам датафрейма и разбивает основной текст на чанки.
        """
        if self.file_path.endswith('.csv'):
            df = pd.read_csv(self.file_path)
        else:
            df = pd.read_excel(self.file_path)

        expected_columns = ["row_id", "id", "name", "description", "price", "url"]
        
        if list(df.columns) == list(range(len(df.columns))):
            df.columns = expected_columns[:len(df.columns)]
        
        documents = []
        
        for _, row in df.iterrows():
            raw_description = str(row.get("description", row.iloc[3] if len(row) > 3 else ""))
            cleaned_text = self._clean_html(raw_description)
            
            if not cleaned_text:
                continue

            metadata: Dict[str, Any] = {}
            
            for col in df.columns:
                val = row[col]
                if pd.isna(val):
                    continue
                
                if col in ["name", "description", 2, 3]:
                    val = self._clean_html(str(val))
                    
                metadata[str(col)] = val

            documents.append(Document(page_content=cleaned_text, metadata=metadata))

        print(f"Загружено исходных документов: {len(documents)}")
        splitted_docs = self.splitter.split_documents(documents)
        print(f"Получено чанков после сплиттера: {len(splitted_docs)}")
        
        return splitted_docs