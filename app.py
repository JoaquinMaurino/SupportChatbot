import os
import streamlit as st
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- ENV ---
load_dotenv()

# --- CONSTANTES ---
DATA_DIR = "./data"
CHROMA_PATH = os.path.join(DATA_DIR, "chroma_gemini")
MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/text-embedding-004"
TOP_K = 4

# --- UI CONFIG ---
st.set_page_config(
    page_title="Supoort Chatbot RAG",
    page_icon="🧠",
    layout="wide"
)
st.title("🧠 Support Chatbot (Gemini + RAG)")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuración")

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        api_key_input = st.text_input("Google API Key:", type="password")
        if api_key_input:
            api_key = api_key_input
            st.success("🟢 Conectado a Google Cloud")
        else:
            st.warning("⚠️ API Key requerida")
            st.stop()
    else:
        st.success("🟢 Conectado a Google Cloud (.env)")

# --- CARGA DE RECURSOS ---
@st.cache_resource
def load_vector_store(path: str, api_key: str):
    if not os.path.exists(path):
        raise RuntimeError(
            "❌ Base vectorial no encontrada. Ejecuta `python ingestar.py`."
        )

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key
    )

    return Chroma(
        persist_directory=path,
        embedding_function=embeddings
    )

@st.cache_resource
def load_llm(api_key: str):
    return init_chat_model(
        MODEL_NAME,
        model_provider="google_genai",
        temperature=0.1,
        api_key=api_key
    )

# --- INIT MODELOS ---
try:
    vector_store = load_vector_store(CHROMA_PATH, api_key)
    llm = load_llm(api_key)
except Exception as e:
    st.error(str(e))
    st.stop()

# --- RAG ---
retriever = vector_store.as_retriever(
    search_kwargs={"k": TOP_K}
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# --- PROMPT ---
template = """
Eres un Asistente Técnico SRE (Site Reliability Engineer) amigable y profesional.
Tu objetivo es ayudar a desarrolladores a resolver incidencias basándote en la base de conocimientos.

<base_conocimientos>
{context}
</base_conocimientos>

<consulta_usuario>
{question}
</consulta_usuario>

INSTRUCCIONES DE COMPORTAMIENTO:

1. **CASO: SALUDO O DUDA GENERAL** (ej: "Hola", "¿Cómo funciona esto?", "Ayuda"):
   - Saluda cordialmente.
   - Explica brevemente: "Puedo ayudarte a diagnosticar errores. Por favor, describe el problema (ej: 'fallo en Oracle') o, para mayor precisión, pega el log completo del error aquí mismo."
   - NO inventes soluciones si no hay un error técnico en la consulta.

2. **CASO: DESCRIPCIÓN VAGA** (ej: "tengo un error de node", "falla la base de datos"):
   - Analiza la <base_conocimientos> buscando palabras clave relacionadas.
   - Si encuentras coincidencias, responde: "Basándome en tu descripción, he encontrado casos similares en nuestra base histórica que podrían ser la causa:".
   - Lista brevemente las causas y soluciones encontradas.
   - **IMPORTANTE:** Termina diciendo: "⚠️ Esta es una sugerencia basada en similitud. Para un diagnóstico exacto, por favor copia y pega el mensaje de error completo."

3. **CASO: LOG DE ERROR EXACTO** (ej: "ORA-12541...", "Error: connection refused"):
   - Usa el contexto para identificar la causa raíz y la solución.
   - Responde de forma directa y concisa: "He identificado el error. Se trata de [Causa]. La solución recomendada es: [Solución]."
   - Mantén un tono profesional y seguro.

Recuerda: Si la información no está en la base de conocimientos, dilo honestamente y pide el log completo para intentar analizarlo mejor.
"""
prompt = ChatPromptTemplate.from_template(template)

chain = prompt | llm | StrOutputParser()

# --- UI ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🗣️ Reporte")
    user_input = st.text_area("Consulta:", height=200)
    run = st.button("Analizar")

with col2:
    st.subheader("💡 Diagnóstico (Gemini)")
    if run and user_input:
        with st.status("Procesando...", expanded=True):
            docs = retriever.invoke(user_input)
            context = format_docs(docs)

            st.write("✅ Contexto recuperado")

            with st.expander("Fuentes"):
                for d in docs:
                    st.info(d.page_content)

            st.markdown("### Solución:")
            st.write_stream(
                chain.stream(
                    {"context": context, "question": user_input}
                )
            )
