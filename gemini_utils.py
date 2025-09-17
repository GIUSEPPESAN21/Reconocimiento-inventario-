import google.generativeai as genai
from PIL import Image
import streamlit as st

def get_image_attributes(image: Image.Image):
    """
    Usa Gemini para extraer una lista detallada de atributos de la imagen de un objeto,
    solicitando la respuesta en formato JSON para un análisis más preciso.
    """
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        raise ValueError("La clave GEMINI_API_KEY no está configurada en los secretos de Streamlit.")

    genai.configure(api_key=api_key)

    # Prompt avanzado que fuerza a Gemini a analizar atributos antes de decidir.
    prompt = """
    Analiza la imagen de este objeto y proporciona sus atributos. Responde únicamente con un objeto JSON válido.
    El objeto JSON debe tener las siguientes claves:
    - "main_object": (string) El nombre genérico del objeto principal (ej: "taza", "teclado", "destornillador").
    - "main_color": (string) El color dominante del objeto.
    - "secondary_colors": (array of strings) Una lista de otros colores significativos presentes.
    - "shape": (string) Una breve descripción de la forma principal (ej: "cilíndrica", "rectangular").
    - "material": (string) Tu mejor suposición sobre el material principal (ej: "plástico", "metal", "cerámica").
    - "features": (array of strings) Una lista de características visuales notables (ej: "tiene un asa", "con logo", "teclas negras").

    Ejemplo de respuesta para una taza de café blanca:
    ```json
    {
      "main_object": "taza",
      "main_color": "blanco",
      "secondary_colors": ["negro"],
      "shape": "cilíndrica",
      "material": "cerámica",
      "features": ["tiene un asa", "interior oscuro"]
    }
    ```
    Ahora, analiza la imagen que te proporciono y devuelve solo el JSON.
    """

    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    try:
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        # Devuelve un JSON de error si la API falla, para mantener la consistencia.
        return f'{{"error": "Error al contactar la API de Gemini: {e}"}}'

