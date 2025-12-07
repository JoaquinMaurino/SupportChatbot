import os
import shutil
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configuración
CARPETA_DATA = "./data"
CSV_FILE = os.path.join(CARPETA_DATA, "base_conocimiento.csv")
CHROMA_PATH = os.path.join(CARPETA_DATA, "chroma_db")

def ingestar():
    # 1. Limpieza preventiva (Borrar DB vieja si existe para evitar conflictos)
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("🧹 Base de datos anterior eliminada.")

    # 2. Cargar CSV
    if not os.path.exists(CSV_FILE):
        print(f"❌ Error: No existe {CSV_FILE}. Ejecuta generar_datos.py")
        return

    loader = CSVLoader(file_path=CSV_FILE, encoding="utf-8")
    docs = loader.load()
    print(f"📄 Cargados {len(docs)} registros.")

    # 3. Embeddings
    print("🧠 Calculando vectores (esto puede tardar un poco)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Guardar en ChromaDB
    print("💾 Guardando en disco...")
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print("✅ ¡Ingesta completada! Ahora puedes ejecutar app.py")

if __name__ == "__main__":
    ingestar()