import re
import os
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Очистка текста
def df_clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    text = re.sub(r'&#\d+;', '', text)
    return text.strip()

# Класс для добавления префиксов к запросам и документам
class PrefixedEmbeddings(Embeddings):
    def __init__(self, base, query_prefix="", doc_prefix=""):
        self.base = base
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix

    def embed_documents(self, texts):
        texts_prefixed = [self.doc_prefix + t for t in texts]
        return self.base.embed_documents(texts_prefixed)

    def embed_query(self, text):
        return self.base.embed_query(self.query_prefix + text)

# 1. Загрузка данных
df = pd.read_csv(
    "data/archive/Air_condition_dataset.csv",
    encoding="cp1251",
    delimiter=";"
)

# 2. Фильтрация данных
selected_columns = ["Наименование", "Подробное описание", "Характеристики"]
filtered_df = df[selected_columns].dropna()

# 3. Преобразование в документы LangChain
docs = []
for idx, row in filtered_df.iterrows():
    content = "\n".join(
        f"{col}: {df_clean_text(str(row[col]))}"
        for col in selected_columns
    )
    metadata = {"source": "Air_condition_dataset.csv", "row": idx}
    docs.append(Document(page_content=content, metadata=metadata))

# Разбиение на чанки
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

# Создание эмбеддингов
base_embeddings = HuggingFaceEmbeddings(model_name="ai-forever/ru-en-RoSBERTa")
embeddings = PrefixedEmbeddings(base_embeddings, query_prefix="search_query: ", doc_prefix="search_document: ")

# Директория для векторной базы
persist_directory = "./mix_chroma_db"

if os.path.isdir(persist_directory):
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
else:
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

# Retriever с MMR
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 8,
        "fetch_k": 32,
    },
)

# Промпт для LLM
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Ты специалист в области систем кондиционирования. "
        "Используй следующую информацию для ответа на вопрос. "
        "Отвечай только на русском. "
        "Отвечай только на те вопросы, которые связаны с областью твоей специальности. "
        "Если вопрос не относится к твоей специальности, вежливо уведоми об этом пользователя. "
        "Если не можешь найти ответ на вопрос, вежливо уведоми пользователя о том, что не обладаешь знаниями об запрашиваемой информации. "
        "Если пользователя интересует какой-то товар, порекомендуй несколько вариантов, указывай бренд, модель и кратко о характеристиках, но постарайся выделить лучший, исходя из вопроса пользователя."
    ),
    (
        "human",
        "Контекст: {context}\n\nВопрос: {question}"
    ),
])

# Модель Ollama
llm = ChatOllama(
    model="gemma2:9b",
    temperature=0.4,
    max_tokens=1024,
    top_p=0.9,
)

max_chars = 8000

# Форматирование документов
def format_docs(docs):
    formatted = []
    total_len = 0
    for doc in docs:
        source = doc.metadata.get("source", "unknown_source")
        text = doc.page_content.strip()
        block = f"Source: {source}\n{text}"
        if total_len + len(block) > max_chars:
            break
        formatted.append(block)
        total_len += len(block)
    return "\n\n---\n\n".join(formatted)

# Цепочка RAG
def get_rag_chain():
    return (
        {
            "context": lambda d: format_docs(d["context"]),
            "question": lambda d: d["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

# Функция для ответа на вопрос
def answer_question(question: str) -> str:
    context = mmr_retriever.invoke(question)
    rag_chain = get_rag_chain()
    answer = rag_chain.invoke({"context": context, "question": question})
    return answer

# Переформулировка вопросов
question_rewrite_llm = ChatOpenAI(
    api_key="None",
    base_url="http://127.0.0.1:11434/v1",
    model="llama3.1:latest",
    temperature=0.2,
    max_tokens=1024,
    top_p=0.9,
)

question_rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Ты AI-агент сайта от компании smartclimat74, который подготавливает пользовательские вопросы для RAG-системы, отвечающей на вопросы по сервисам, продуктам и услугам компании SmartKlimat74 на основе её публичных веб-ресурсов (сайт, FAQ, справочные материалы, каталоги оборудования, инструкции)."
        "Твоя задача:"
        "1) Точно определить суть вопроса пользователя."
        "2) Оценить, насколько формулировка подходит для семантического поиска по документации и материалам SmartKlimat74."
        "3) Если вопрос слишком общий, разговорный, содержит неясные местоимения (например, 'это', 'там', 'оно') или требует уточнения, переписать его в чёткий, формальный и самодостаточный вид."
        "Правила переписывания:"
        "- Всегда явно упоминай SmartKlimat74, если это релевантно контексту вопроса."
        "- Сохраняй исходный смысл, не добавляй новых фактов или предположений."
        "- Формулируй вопрос так, чтобы по нему можно было найти точный ответ в документации, FAQ, каталогах или инструкциях компании."
        "- Используй нейтральный, технический стиль без разговорных выражений и эмоциональной окраски."
        "Если исходный вопрос уже оптимально сформулирован для поиска — верни его без изменений."
        "Отвечай ТОЛЬКО итоговой формулировкой вопроса, без кавычек, пояснений и комментариев."
    ),
    ("human", "{question}")
])

question_rewrite_chain = (
    question_rewrite_prompt
    | question_rewrite_llm
    | StrOutputParser()
)

def rewrite_question_if_needed(question: str) -> str:
    rewritten = question_rewrite_chain.invoke({"question": question}).strip()
    return rewritten