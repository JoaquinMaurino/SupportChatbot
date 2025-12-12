import os
import shutil
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- CONFIGURACIÓN ---
DATA_DIR = "./data"
CSV_FILE = os.path.join(DATA_DIR, "base_conocimiento.csv")
PATH_LOCAL = os.path.join(DATA_DIR, "chroma_local") # Base para Ollama
PATH_CLOUD = os.path.join(DATA_DIR, "chroma_cloud") # Base para Gemini

os.environ["GOOGLE_API_KEY"] = "AIzaSyDqqKm07HnPw-A5jyyjEYUVJIHAQkTtrBQ"

def load_docs():
    if not os.path.exists(CSV_FILE):
        print(f"❌ Error: No existe {CSV_FILE}")
        return []
    loader = CSVLoader(file_path=CSV_FILE, encoding="utf-8")
    return loader.load()

def ingestar_local(docs):
    print("\n🔵 [LOCAL] Generando vectores con HuggingFace (CPU)...")
    if os.path.exists(PATH_LOCAL):
        shutil.rmtree(PATH_LOCAL)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    Chroma.from_documents(docs, embeddings, persist_directory=PATH_LOCAL)
    print(f"✅ Base Local lista en: {PATH_LOCAL}")

def ingestar_cloud(docs):
    print("\n🟠 [CLOUD] Generando vectores con Google (Cloud)...")
    if "GOOGLE_API_KEY" not in os.environ:
        print("⚠️ Saltando Cloud: No se encontró GOOGLE_API_KEY.")
        return

    if os.path.exists(PATH_CLOUD):
        shutil.rmtree(PATH_CLOUD)
    
    # Usamos el modelo de embeddings de Google 
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    Chroma.from_documents(docs, embeddings, persist_directory=PATH_CLOUD)
    print(f"✅ Base Cloud lista en: {PATH_CLOUD}")

if __name__ == "__main__":
    documentos = load_docs()
    if documentos:
        ingestar_local(documentos) # Siempre se ejecuta
        ingestar_cloud(documentos) # Solo si hay API Key