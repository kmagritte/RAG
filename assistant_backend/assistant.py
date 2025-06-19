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



from docling.pipeline.base_pipeline import BasePipeline
from docling.datamodel.base_models import DocItem, InputFormat

class ParagraphHeaderPipeline(BasePipeline):
    def parse_text(self, lines, filepath, fileformat=InputFormat.DOCX):
        items = []
        section_keywords = ("ГЛАВА", "СТАТЬЯ", "РАЗДЕЛ", "ЧАСТЬ", "SECTION", "PART")

        for line in lines:
            text = line.strip()
            if not text:
                continue

            line_upper = text.upper()
            if any(line_upper.startswith(kw) for kw in section_keywords):
                item_type = "heading"
            else:
                item_type = "paragraph"

            items.append(DocItem(
                type=item_type,
                text=text
            ))

        return items
    

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.document_converter import WordFormatOption

converter = DocumentConverter(
    format_options={
        InputFormat.DOCX: WordFormatOption(
            pipeline_cls=ParagraphHeaderPipeline
        )
    }
)
