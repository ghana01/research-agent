from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")