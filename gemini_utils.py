import google.generativeai as genai
import os
from dotenv import load_dotenv
from PIL import Image

# Cargar variables de entorno del archivo .env
load_dotenv()

def identify_item(image: Image.Image, inventory_list: list):
    """
    Usa Gemini Pro Vision para identificar un artículo en una imagen
    basándose en una lista de artículos de inventario.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("La variable de entorno GEMINI_API_KEY no está definida.")

    genai.configure(api_key=api_key)

    prompt = f"""
    Eres un experto en clasificación de inventarios. Tu única tarea es identificar el objeto principal en la imagen.
    
    Compara el objeto con la siguiente lista de artículos de mi inventario:
    {', '.join(inventory_list)}

    Responde únicamente con el nombre exacto del artículo de la lista que mejor corresponda.
    Si el objeto no coincide con ninguno de la lista, responde con 'Artículo no encontrado'.
    No incluyas explicaciones, saludos ni texto adicional. Solo el nombre del artículo o el mensaje de error.
    """

    model = genai.GenerativeModel('gemini-pro-vision')
    
    try:
        response = model.generate_content([prompt, image])
        return response.text.strip()
    except Exception as e:
        return f"Error al contactar la API de Gemini: {e}"
