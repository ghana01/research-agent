from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
# this function load a makdown file and return a list of documents
def load_markdown(path:str):
    loader =TextLoader(path, encoding="utf-8")
    documents =loader.load()
    
    return documents

def load_pdf(path:str):
    loader =PyPDFLoader(path)
    documents =loader.load()
    
    return documents


def split_documents(documents, source_path: str, file_type: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=250,
    )

    chunks = splitter.split_documents(documents)

    document_id = Path(source_path).stem

    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "document_id": document_id,
            "chunk_id": f"{document_id}_chunk_{i + 1:03d}",
            "source": source_path,
            "file_type": file_type,
        })

    return chunks