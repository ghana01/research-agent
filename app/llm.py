
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import settings

load_dotenv()

def get_embeddings():
    return OpenAIEmbeddings(
        model=settings.embedding_model
    )


def get_llm():
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=0.0,
        timeout=settings.openai_timeout,
        max_retries=settings.openai_max_retries,
    )
