import json
import torch
from pathlib import Path
from opensearchpy import OpenSearch
from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer

# Настройки OpenSearch
OPENSEARCH_HOST = "http://localhost:9200"
INDEX_NAME = "documents"

# Подключение к OpenSearch
client = OpenSearch(
    hosts=[OPENSEARCH_HOST],
    http_compress=True,
    use_ssl=False,
    verify_certs=False
)

# Создание индекса (если не существует)
if not client.indices.exists(index=INDEX_NAME):
    client.indices.create(index=INDEX_NAME, body={"settings": {"index": {"number_of_shards": 1}}})

# Загрузка локальной модели Qwen2.5-1.5-Instruct
MODEL_PATH = "models/Qwen2.5-1.5-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

# Функция обработки PDF и DOCX
def load_document(file_path):
    file_type = Path(file_path).suffix.lower()
    if file_type == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_type == ".docx":
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError("Неподдерживаемый формат файла")
    
    return loader.load()

# Функция чанкинга текста
def chunk_text(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    return splitter.split_documents(documents)

# Функция получения эмбеддингов
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

# Функция извлечения ключевых слов с помощью Qwen2.5-1.5-Instruct
def extract_keywords(text):
    prompt = f"Выдели ключевые слова из следующего текста:\n{text}\nКлючевые слова:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100)
    
    keywords = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return keywords.split(", ")

# Функция загрузки документа в OpenSearch
def index_document(file_path, department):
    documents = load_document(file_path)
    chunks = chunk_text(documents)

    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk.page_content)
        keywords = extract_keywords(chunk.page_content)

        doc = {
            "document_name": Path(file_path).name,
            "document_link": str(file_path),
            "department": department,
            "chunk_id": i,
            "text": chunk.page_content,
            "embedding": embedding,
            "keywords": keywords
        }

        client.index(index=INDEX_NAME, body=json.dumps(doc))

# **Пример загрузки документов**
index_document("example.pdf", "Финансовый отдел")
index_document("example.docx", "Юридический отдел")

print("Документы успешно загружены в OpenSearch!")



import json
import numpy as np
from pathlib import Path
from opensearchpy import OpenSearch
from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
import onnxruntime as ort

# Настройки OpenSearch
OPENSEARCH_HOST = "http://localhost:9200"
INDEX_NAME = "documents"

# Подключение к OpenSearch
client = OpenSearch(
    hosts=[OPENSEARCH_HOST],
    http_compress=True,
    use_ssl=False,
    verify_certs=False
)

# Создание индекса (если не существует)
if not client.indices.exists(index=INDEX_NAME):
    client.indices.create(index=INDEX_NAME, body={"settings": {"index": {"number_of_shards": 1}}})

# Загрузка локальной модели Qwen2.5-1.5-Instruct (ONNX)
MODEL_PATH = "models/Qwen2.5-1.5-Instruct"
TOKENIZER_PATH = MODEL_PATH
ONNX_MODEL_PATH = f"{MODEL_PATH}/model.onnx"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
session = ort.InferenceSession(ONNX_MODEL_PATH)

# Функция обработки PDF и DOCX
def load_document(file_path):
    file_type = Path(file_path).suffix.lower()
    if file_type == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_type == ".docx":
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError("Неподдерживаемый формат файла")
    
    return loader.load()

# Функция чанкинга текста
def chunk_text(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    return splitter.split_documents(documents)

# Функция получения эмбеддингов без torch
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
    outputs = session.run(None, {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]})
    return np.mean(outputs[0], axis=1).squeeze().tolist()

# Функция извлечения ключевых слов с помощью Qwen2.5-1.5-Instruct
def extract_keywords(text):
    prompt = f"Выдели ключевые слова из следующего текста:\n{text}\nКлючевые слова:"
    inputs = tokenizer(prompt, return_tensors="np", padding=True, truncation=True)
    
    outputs = session.run(None, {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]})
    keywords = tokenizer.decode(outputs[0][0], skip_special_tokens=True)
    return keywords.split(", ")

# Функция загрузки документа в OpenSearch
def index_document(file_path, department):
    documents = load_document(file_path)
    chunks = chunk_text(documents)

    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk.page_content)
        keywords = extract_keywords(chunk.page_content)

        doc = {
            "document_name": Path(file_path).name,
            "document_link": str(file_path),
            "department": department,
            "chunk_id": i,
            "text": chunk.page_content,
            "embedding": embedding,
            "keywords": keywords
        }

        client.index(index=INDEX_NAME, body=json.dumps(doc))

# **Пример загрузки документов**
index_document("example.pdf", "Финансовый отдел")
index_document("example.docx", "Юридический отдел")

print("Документы успешно загружены в OpenSearch!")
