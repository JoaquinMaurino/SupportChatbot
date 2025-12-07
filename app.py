import streamlit as st
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
# IMPORTACIONES MODERNAS (Compatibles con LangChain 0.3)
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- CONFIGURACIÓN ---
CARPETA_DATA = "./data"
CHROMA_PATH = os.path.join(CARPETA_DATA, "chroma_db")
MODELO_OLLAMA = "llama3.1" # Asegúrate de tener este modelo en Ollama

# --- UI SETUP ---
st.set_page_config(page_title="Soporte AI - Demo", page_icon="🛡️", layout="wide")
st.title("🛡️ Asistente Inteligente SRE/DevOps")

# --- CARGA DE RECURSOS ---
@st.cache_resource
def cargar_motor_ia():
    # 1. Embeddings (Usamos el mismo modelo que en ingestar.py)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Verificar DB
    if not os.path.exists(CHROMA_PATH):
        st.error(f"No se encontró la base de datos en {CHROMA_PATH}. Ejecuta 'python ingestar.py' primero.")
        return None, None
        
    # 3. Conectar a ChromaDB
    vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # 4. Conectar a Ollama
    try:
        llm = OllamaLLM(model=MODELO_OLLAMA)
    except Exception as e:
        st.error(f"Error conectando a Ollama: {e}")
        return None, None

    return vector_store, llm

vector_store, llm = cargar_motor_ia()

if vector_store and llm:
    # --- LÓGICA RAG (MODERNA) ---
    
    # 1. Prompt para el Chatbot
    prompt = ChatPromptTemplate.from_template("""
    Eres un experto en Soporte Técnico Bancario (SRE).
    Usa el siguiente contexto de incidentes pasados para responder.
    
    <contexto>
    {context}
    </contexto>
    
    <error_actual>
    {input}
    </error_actual>
    
    Tu respuesta debe tener:
    1. **Severidad estimada**
    2. **Diagnóstico técnico**
    3. **Pasos de solución sugeridos**
    """)
    
    # 2. Cadena para procesar documentos (Stuff Chain)
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # 3. Configurar el buscador (Retriever)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # 4. Cadena final de recuperación
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # --- INTERFAZ VISUAL ---
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📥 Reporte de Incidencia")
        log_input = st.text_area("Pega aquí el log del error:", height=200, placeholder="Ej: ORA-12541: TNS:no listener...")
        analizar_btn = st.button("🔍 Analizar Error", type="primary", use_container_width=True)

    with col2:
        st.subheader("🤖 Diagnóstico IA")
        if analizar_btn and log_input:
            with st.spinner(f"Consultando a {MODELO_OLLAMA}..."):
                try:
                    # Ejecutar la cadena
                    respuesta = retrieval_chain.invoke({"input": log_input})
                    
                    st.success("Análisis completado")
                    st.markdown(respuesta["answer"])
                    
                    with st.expander("📚 Ver casos históricos similares usados"):
                        for doc in respuesta["context"]:
                            st.info(doc.page_content)
                            
                except Exception as e:
                    st.error(f"Ocurrió un error: {e}")