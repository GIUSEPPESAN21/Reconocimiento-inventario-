import google.generativeai as genai
from PIL import Image
import streamlit as st
import json

def get_image_attributes(image: Image.Image):
    """
    Paso 1: Actúa como el "Ojo". Extrae atributos visuales de la imagen y los devuelve en formato JSON.
    """
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        raise ValueError("La clave GEMINI_API_KEY no está configurada en los secretos de Streamlit.")

    genai.configure(api_key=api_key)

    prompt = """
    Analiza la imagen de este objeto y proporciona sus atributos. Responde únicamente con un objeto JSON válido.
    El objeto JSON debe tener las siguientes claves:
    - "main_object": (string) El nombre genérico del objeto principal (ej: "taza", "teclado").
    - "main_color": (string) El color dominante del objeto.
    - "secondary_colors": (array of strings) Otros colores presentes.
    - "shape": (string) La forma principal (ej: "cilíndrica", "rectangular").
    - "material": (string) Tu mejor suposición sobre el material (ej: "plástico", "metal", "cerámica").
    - "features": (array of strings) Características visuales notables (ej: "tiene un asa", "con logo", "teclas negras").
    Asegúrate de que el JSON sea sintácticamente correcto.
    """

    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    try:
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return f'{{"error": "Error al contactar la API de Gemini: {e}"}}'

def get_best_match_from_attributes(attributes: dict, inventory_names: list):
    """
    Paso 2: Actúa como el "Cerebro Logístico". Recibe los atributos y la lista del inventario,
    y razona cuál es la mejor coincidencia.
    """
    if not inventory_names:
        return '{"best_match": "Artículo no encontrado", "reasoning": "El inventario está vacío."}'
        
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)

    # Convertir el diccionario de atributos a una cadena JSON para el prompt
    attributes_json_string = json.dumps(attributes, indent=2)

    prompt = f"""
    Eres un experto en logística de inventarios. Tu tarea es encontrar la mejor coincidencia para un objeto basado en sus atributos observados.
    
    Atributos observados del objeto:
    ```json
    {attributes_json_string}
    ```

    Mi lista de inventario actual:
    - {"\n- ".join(inventory_names)}

    Basado en los atributos observados, ¿cuál de los artículos de mi lista de inventario es la coincidencia más probable?
    
    Responde únicamente con un objeto JSON válido que contenga dos claves:
    1. "best_match": (string) El nombre exacto del artículo de la lista que mejor coincide. Si ninguna coincidencia es buena, usa la cadena "Artículo no encontrado".
    2. "reasoning": (string) Una frase corta explicando tu elección.
    
    Ejemplo de respuesta:
    ```json
    {{
      "best_match": "Taza de cerámica blanca con asa",
      "reasoning": "El objeto es una taza blanca de cerámica con un asa, lo que coincide perfectamente con este artículo del inventario."
    }}
    ```
    Ahora, proporciona tu análisis.
    """

    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f'{{"best_match": "Error", "reasoning": "Error al contactar la API de Gemini: {e}"}}'

