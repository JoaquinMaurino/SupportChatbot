import streamlit as st
import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# --- PATHS ---
DATA_DIR = "./data"
PATH_LOCAL = os.path.join(DATA_DIR, "chroma_local")
PATH_CLOUD = os.path.join(DATA_DIR, "chroma_cloud")

# --- UI CONFIG ---
st.set_page_config(page_title="Soporte AI - Híbrido Pro", page_icon="🔀", layout="wide")
st.title("🔀 Asistente SRE (Arquitectura Multi-Modelo)")

# --- SIDEBAR: CONFIGURACIÓN ---
with st.sidebar:
    st.header("⚙️ Configuración del Motor")

    mode = st.radio("Modo de Ejecución:", ["Local (Privado)", "Nube (Google)"])

    api_key_val = None

    if mode == "Local (Privado)":
        provider = "ollama"
        model_name = "llama3.1"
        db_path = PATH_LOCAL
        embedding_type = "local"
        st.success("🟢 Modo: Offline / Privado")

    else:
        provider = "google_genai"
        model_name = "gemini-2.5-flash"
        db_path = PATH_CLOUD
        embedding_type = "cloud"

        # 1) Intentar desde entorno (.env)
        api_key_val = os.getenv("GOOGLE_API_KEY")

        # 2) Fallback por UI
        if not api_key_val:
            api_key_input = st.text_input("Google API Key:", type="password")

            if api_key_input:
                api_key_val = api_key_input
                st.success("🟢 Conectado a Google Cloud (input)")
            else:
                st.warning("⚠️ API Key requerida")
                st.stop()
        else:
            st.success("🟢 Conectado a Google Cloud (env)")

# --- CARGA DE RECURSOS (CORREGIDO) ---

@st.cache_resource
def get_vector_store(path, type_embed, api_key_trigger):
    if not os.path.exists(path):
        return None

    try:
        if type_embed == "local":
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        else:
            if not api_key_trigger:
                return None
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=api_key_trigger
            )

        return Chroma(
            persist_directory=path,
            embedding_function=embeddings
        )
    except Exception:
        return None

@st.cache_resource
def get_llm(prov, mod, api_key_trigger):
    if prov == "google_genai" and not api_key_trigger:
        return None

    try:
        return init_chat_model(
            mod,
            model_provider=prov,
            temperature=0.1,
            api_key=api_key_trigger
        )
    except Exception:
        return None


# Inicialización (Pasamos la key para invalidar caché viejo)
vector_store = get_vector_store(db_path, embedding_type, api_key_val)
llm = get_llm(provider, model_name, api_key_val)

# --- VALIDACIONES ---
if not vector_store:
    # Mensaje de error más detallado
    if mode == "Nube (Google)":
        st.error(f"❌ Error accediendo a la base de datos Nube en: {db_path}")
        st.info("Posibles causas:\n1. No ejecutaste 'python ingestar.py' CON la API Key puesta.\n2. La API Key actual es incorrecta.\n3. Intenta borrar caché (arriba a la derecha 'C' -> 'Clear Cache').")
    else:
        st.error(f"❌ No se encontró la base de datos Local en: {db_path}. Ejecuta 'python ingestar.py'.")
    st.stop()

if not llm:
    st.error("⚠️ Error cargando el modelo. Verifica tus credenciales.")
    st.stop()

# --- RAG PIPELINE ---
K_ARG = 4
retriever = vector_store.as_retriever(search_kwargs={"k": K_ARG})

def format_docs(docs):
    return "\n\n".join(f"- {d.page_content}" for d in docs)

# PROMPT INTELIGENTE (Sintaxis Memorizada)
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

# --- INTERFAZ ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🗣️ Reporte")
    user_input = st.text_area("Consulta:", height=200)
    btn = st.button("Analizar")

with col2:
    st.subheader(f"💡 Diagnóstico ({mode})")
    if btn and user_input:
        with st.status(f"Procesando con {model_name}...", expanded=True):
            # 1. Recuperación
            docs = retriever.invoke(user_input)
            ctx = format_docs(docs)
            st.write("✅ Contexto recuperado")
            
            with st.expander("Fuentes"):
                for d in docs: st.info(d.page_content)
                
            # 2. Generación
            st.markdown("### Solución:")
            st.write_stream(chain.stream({"context": ctx, "question": user_input}))