import streamlit as st
import os

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# DATA LOCATION
CARPETA_DATA = "./data"
CHROMA_PATH = os.path.join(CARPETA_DATA, "chroma_db")

# MODEL AND PROVIDER
MODEL_PROVIDER = "ollama"
MODEL_NAME = "llama3.1"

# UI CONFIG
st.set_page_config(page_title="Soporte AI - Pro", page_icon="🛡️", layout="wide")
st.title("🛡️ Support ChatBot")

# CARGA DE RECURSOS (Cached)
@st.cache_resource
def cargar_stack():
    # Cargar Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if not os.path.exists(CHROMA_PATH):
        return None, None
    
    # Conexión a DB de vectores
    vector_store = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embeddings
    )

    # Inicializar Modelo 
    try:
        # Mantenemos temperature=0.1 para precisión técnica
        llm = init_chat_model(MODEL_NAME, model_provider=MODEL_PROVIDER, temperature=0.1)
    except Exception as e:
        st.error(f"Error iniciando modelo: {e}")
        return None, None

    return vector_store, llm

vector_store, llm = cargar_stack()

if not vector_store:
    st.error("⚠️ Ejecuta 'python ingestar.py'")
    st.stop()

# RAG CONFIG
retriever = vector_store.as_retriever(search_kwargs={"k": 4}) 

def format_docs(docs):
    return "\n\n".join(f"- {d.page_content}" for d in docs)

# PROMPT ROBUSTO (Mantenido intacto)
template = """
Eres un Ingeniero SRE Senior experto en análisis de incidentes.
Tu tarea es ayudar al usuario usando la base de conocimientos proporcionada.

<base_conocimientos>
{context}
</base_conocimientos>

<consulta_usuario>
{question}
</consulta_usuario>

INSTRUCCIONES DE RESPUESTA:

1. SI EL USUARIO DA UN LOG EXACTO:
   - Analiza la base de conocimientos.
   - Da la solución precisa y directa.

2. SI EL USUARIO DESCRIBE EL PROBLEMA EN LENGUAJE NATURAL (ej: "tengo timeout en oracle"):
   - PRIMERO: Busca en la <base_conocimientos> errores que coincidan conceptualmente (ej: busca temas de Oracle o Timeouts).
   - SEGUNDO: Dile al usuario algo como: "Para un diagnóstico preciso necesito el log completo, pero basándome en tu descripción, aquí tienes sugerencias de nuestra base histórica que podrían servir:".
   - TERCERO: Lista las posibles causas y soluciones encontradas en el contexto que se parezcan al problema.

3. SI NO HAY NADA RELACIONADO EN EL CONTEXTO:
   - Pide amablemente el log del error.

Tu tono debe ser técnico pero colaborativo.
"""

prompt = ChatPromptTemplate.from_template(template)

# OPTIMIZACIÓN 1: Cadena de Generación Pura
# Quitamos el retriever de aquí dentro. Le pasaremos el contexto manualmente.
generation_chain = (
    prompt
    | llm
    | StrOutputParser()
)

# UI LAYOUT
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🗣️ Cuéntame el problema")
    user_input = st.text_area("Describe el error o pega el log:", height=200, placeholder="Ej: Tengo problemas de conexión con Oracle...")
    btn = st.button("Consultar Base de Conocimientos", type="primary")

with col2:
    st.subheader("💡 Respuesta Sugerida")
    
    if btn and user_input:
        # OPTIMIZACIÓN 2: Recuperación explícita antes de generar
        with st.status("🔍 Buscando en registros históricos...", expanded=True) as status:
            try:
                # 1. Recuperamos documentos (Solo una vez)
                docs = retriever.invoke(user_input)
                
                # 2. Formateamos el contexto
                context_text = format_docs(docs)
                
                # 3. Mostramos las fuentes inmediatamente (Mejora UX)
                status.update(label="✅ Contexto recuperado", state="complete", expanded=False)
                
                with st.expander("📚 Registros históricos consultados (Evidencia)"):
                    if not docs:
                        st.warning("No se encontraron registros similares.")
                    for d in docs:
                        st.info(d.page_content)

                # 4. Generación con Streaming (Para que se vea escribir rápido)
                st.markdown("### Diagnóstico:")
                
                # Preparamos el input exacto que pide el prompt
                chain_input = {
                    "context": context_text, 
                    "question": user_input
                }
                
                # Escribimos en tiempo real
                st.write_stream(generation_chain.stream(chain_input))
                
            except Exception as e:
                st.error(f"Error en el proceso: {e}")