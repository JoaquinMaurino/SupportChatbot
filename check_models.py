import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv() 

# --- SETUP ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

print("🔍 Consultando modelos disponibles para tu API Key...")

try:
    # Listamos todos los modelos disponibles
    for m in genai.list_models():
        # Filtramos solo los que sirven para generar contenido (chat)
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ Disponible: {m.name}")
except Exception as e:
    print(f"❌ Error: {e}")