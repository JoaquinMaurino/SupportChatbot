import os
import shutil
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CARPETA_DATA = "./data"
CSV_FILE = os.path.join(CARPETA_DATA, "base_conocimiento.csv")
CHROMA_PATH = os.path.join(CARPETA_DATA, "chroma_db")

def ingestar():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    loader = CSVLoader(file_path=CSV_FILE, encoding="utf-8")
    docs = loader.load()
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        persist_directory=CHROMA_PATH
    )
    print("Ingesta v1.0 Completa")

if __name__ == "__main__":
    ingestar()