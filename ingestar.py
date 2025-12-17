import os
import shutil
from dotenv import load_dotenv

from langchain_community.document_loaders import CSVLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- CONFIGURACIÓN ---
load_dotenv()

DATA_DIR = "./data"
CSV_FILE = os.path.join(DATA_DIR, "base_conocimiento.csv")
CHROMA_PATH = os.path.join(DATA_DIR, "chroma_gemini")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- VALIDACIONES ---
if not GOOGLE_API_KEY:
    raise RuntimeError("❌ GOOGLE_API_KEY no encontrada en el entorno (.env)")

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"❌ No existe el archivo: {CSV_FILE}")

# --- CARGA DE DOCUMENTOS ---
def load_docs():
    loader = CSVLoader(
        file_path=CSV_FILE,
        encoding="utf-8"
    )
    return loader.load()

# --- INGESTA ---
def ingest_gemini(docs):
    print("🟠 Generando base vectorial con Gemini...")

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )

    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print(f"✅ Base Gemini creada en: {CHROMA_PATH}")

# --- MAIN ---
if __name__ == "__main__":
    docs = load_docs()
    ingest_gemini(docs)
