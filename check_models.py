import os
import google.generativeai as genai

# --- SETUP ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyDaJG3JxKdaxM3kpcbfvEtGoAvDgqCMKnM"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

print("🔍 Consultando modelos disponibles para tu API Key...")

try:
    # Listamos todos los modelos disponibles
    for m in genai.list_models():
        # Filtramos solo los que sirven para generar contenido (chat)
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ Disponible: {m.name}")
except Exception as e:
    print(f"❌ Error: {e}")