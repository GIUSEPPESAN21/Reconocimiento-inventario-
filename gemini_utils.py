import google.generativeai as genai
from PIL import Image
import streamlit as st

def identify_item(image: Image.Image, inventory_list: list):
    """
    Usa Gemini para identificar un artículo, con el nombre del modelo actualizado.
    """
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        raise ValueError("La clave GEMINI_API_KEY no está configurada en los secretos de Streamlit.")

    genai.configure(api_key=api_key)

    prompt = f"""
    Eres un experto en clasificación de inventarios. Tu única tarea es identificar el objeto en la imagen.
    Compara el objeto con esta lista de mi inventario: {', '.join(inventory_list)}
    Responde únicamente con el nombre exacto del artículo de la lista.
    Si no coincide con ninguno, responde 'Artículo no encontrado'.
    No incluyas texto adicional.
    """

    # Se actualiza el nombre del modelo al más reciente.
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    try:
        response = model.generate_content([prompt, image])
        return response.text.strip()
    except Exception as e:
        return f"Error al contactar la API de Gemini: {e}"

