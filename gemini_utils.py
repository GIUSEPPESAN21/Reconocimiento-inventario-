import google.generativeai as genai
from PIL import Image
import streamlit as st

def identify_item(image: Image.Image, inventory_list: list):
    """
    Usa Gemini Pro Vision para identificar un artículo.
    Lee la API Key desde los secretos de Streamlit.
    """
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        raise ValueError("La variable GEMINI_API_KEY no está configurada en los secretos de Streamlit.")

    genai.configure(api_key=api_key)

    prompt = f"""
    Eres un experto en clasificación de inventarios. Tu única tarea es identificar el objeto principal en la imagen.
    Compara el objeto con la siguiente lista de artículos de mi inventario:
    {', '.join(inventory_list)}
    Responde únicamente con el nombre exacto del artículo de la lista que mejor corresponda.
    Si el objeto no coincide con ninguno, responde con 'Artículo no encontrado'.
    No incluyas explicaciones ni texto adicional.
    """

    model = genai.GenerativeModel('gemini-pro-vision')
    
    try:
        response = model.generate_content([prompt, image])
        return response.text.strip()
    except Exception as e:
        return f"Error al contactar la API de Gemini: {e}"

