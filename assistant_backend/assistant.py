from pathlib import Path
from typing import Optional, Tuple

from loguru import logger
from langchain_community.llms import VLLMOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.core.output_parsers import StrOutputParser

PROJECT_DIR = Path(__file__).resolve().parent
LOGS_DIR = PROJECT_DIR / "logs"
SERVED_MODEL_NAME = "Qwen3-8B"
IP_ADDRESS = "localhost"
PORT = "5600"
openai_api_key = "EMPTY"
openai_api_base = f"http://{IP_ADDRESS}:{PORT}/v1"

VECTOR_SEARCH_TOP_K = 3

logger.add(LOGS_DIR / "assistant.txt", rotation="5 MB")


class R1OutputParser(BaseOutputParser[Tuple[Optional[str], str]]):
    """Parser for DeepSeek R1 model output that includes thinking and response sections."""

    def parse(self, text: str) -> Tuple[Optional[str], str]:
        """Parse the model output into thinking and response sections.

        Args:
            text: Raw text output from the model

        Returns:
            Tuple containing (thinking_text, response_text)
            - thinking_text will be None if no thinking section is found
        """
        if "</think>" in text:
            parts = text.split("</think>")
            thinking_text = parts[0].replace("<think>", "").strip()
            response_text = parts[1].strip()
            return thinking_text, response_text
        return None, text.strip()

    @property
    def _type(self) -> str:
        """Return type key for serialization."""
        return "r1_output_parser"
    

def query_rag(chain, question):
    """Queries the RAG chain and prints the response."""
    logger.info("Querying RAG chain...")
    logger.info(f"Question: {question}")
    thinking, response = chain.invoke(question)
    logger.info("\nTHINKING:\n")
    logger.info(thinking)
    logger.info("\nRESPONSE:\n")
    logger.info(response)


def create_rag_chain(vector_store=None, llm_model_name: str=SERVED_MODEL_NAME):
    """Creates the RAG chain."""
    model = VLLMOpenAI(
        base_url=openai_api_base,
        api_key=openai_api_key,
        model_name=SERVED_MODEL_NAME,
        temperature=0.1,
        streaming=True,
        max_tokens=100,
        top_p=0.9,
    )
    logger.info(f"Initialized ChatVLLM with model: {llm_model_name}")

    # Create parser
    parser = R1OutputParser()

    # Define the prompt template
    template = """
        Известная информация: {context}
        Позиционируй себя как компетентного специалиста в данной области, но при этом не забывай излагать информацию в доступной и понятной форме.
        Основываясь только на вышеуказанной известной информации, ответь на вопрос пользователя кратко и профессионально.
        Если же невозможно получить ответ, скажи: "На вопрос невозможно ответить с учетом предоставленной информации"
        или "Было предоставлено недостаточно соответствующей информации" и не пытайся придумать ответ.
        Пожалуйста, ответь на русском. И всегда спрашивай: "У вас есть еще вопросы?" в конце ответа.
        Вопрос: {question}
        Полезный ответ:
    """
    # Create prompt template
    prompt = ChatPromptTemplate.from_template(template)
    logger.info("Prompt template created.")

    # Create chain
    retriever = lambda x: """
    Постановлением Совета Министров Республики Беларусь от 16 ноября 2024 года № 848 «Об установлении размера базовой величины»
    с 1 января 2025 года установлен размер базовой величины в сумме 42 рубля. Таким образом, с 1 января 2025 года изменятся платежи,
    размер которых зависит от базовой величины (например: государственная пошлина).
    """
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | parser
    )
    logger.info("RAG chain created.")
    return rag_chain

if __name__ == "__main__":
    try:
        # 5. Create RAG Chain
        rag_chain = create_rag_chain() # vector store

        # 6. Query
        query_question = "Какая базовая величина в Республике Беларусь?"
        query_rag(rag_chain, query_question)

    except Exception as err:
        logger.exception(f"Error occurred: {err}")



import re
from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader


def detect_heading_level(text: str) -> int:
    text = text.strip()
    if re.match(r"^Глава\s+\d+", text):
        return 1
    elif re.match(r"^\d+(\.\d+)*\s+", text):  # 1.1, 2.3.4
        return 2
    elif re.match(r"^§\s*\d+(\.\d+)*", text):  # § 1.5
        return 3
    return 4  # обычный абзац


def load_document(file_path: str) -> List[Document]:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(file_path)
    elif suffix == ".docx":
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError("Поддерживаются только .pdf и .docx")
    return loader.load()


def split_document_with_headings(
    docs: List[Document], chunk_size=1000, chunk_overlap=200
) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    output_docs = []

    for doc in docs:
        lines = doc.page_content.splitlines()
        heading_stack = []
        buffer = []
        buffer_len = 0

        def flush_buffer():
            nonlocal buffer, buffer_len
            if not buffer:
                return
            full_section = "\n".join(buffer).strip()
            sub_docs = text_splitter.create_documents([full_section])
            for d in sub_docs:
                heading_prefix = " → ".join(heading_stack)
                d.page_content = f"{heading_prefix}\n\n{d.page_content}" if heading_prefix else d.page_content
                d.metadata = {**doc.metadata, "heading_path": heading_stack.copy()}
                output_docs.append(d)
            buffer.clear()
            buffer_len = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue
            level = detect_heading_level(line)
            if level < 4:
                flush_buffer()
                heading_stack = heading_stack[:level - 1]
                heading_stack.append(line)
            else:
                buffer.append(line)
                buffer_len += len(line)
                if buffer_len >= chunk_size:
                    flush_buffer()

        flush_buffer()

    return output_docs
