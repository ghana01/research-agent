from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
# this function load a makdown file and return a list of documents
def load_markdown(path:str):
    loader =TextLoader(path, encoding="utf-8")
    documents =loader.load()
    
    return documents

def load_pdf(path:str):
    loader =PyPDFLoader(path)
    documents =loader.load()
    
    return documents