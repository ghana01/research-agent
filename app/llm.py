from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")

def get_llm():
    return ChatOpenAI(model="gpt-4o", temperature=0.0)
