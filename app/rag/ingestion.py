from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_documents(data_dir: str | Path) -> list[Document]:
    """
    Scans a directory for .txt and .pdf files, loads them, and chunks the text.
    """
    data_path = Path(data_dir)
    documents: list[Document] = []
    
    # Iterate through all files in the directory
    for file_path in data_path.rglob("*"):
        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
            documents.extend(loader.load())
        elif file_path.suffix.lower() == ".txt":
            loader = TextLoader(str(file_path))
            documents.extend(loader.load())
            
    if not documents:
        print(f"No .pdf or .txt documents found in {data_dir}")
        return []
        
    # Split text into chunks to fit within LLM context windows
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True
    )
    
    chunked_docs = text_splitter.split_documents(documents)
    return chunked_docs