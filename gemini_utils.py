import google.generativeai as genai
from PIL import Image
import streamlit as st
import json
import logging

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _call_gemini_with_fallback(prompt, image=None):
    """
    Función interna para llamar a la API de Gemini con una lista de modelos y fallback automático.
    Esta función centraliza la lógica de conexión para evitar duplicar código.
    """
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
    except KeyError:
        error_msg = "La clave GEMINI_API_KEY no está configurada en los secretos de Streamlit."
        logging.error(error_msg)
        return f'{{"error": "{error_msg}"}}'

    # --- CONFIGURACIÓN OPTIMIZADA (BASADA EN LA SOLUCIÓN PROPORCIONADA) ---
    generation_config = {
        "temperature": 0.4,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 4096,
    }
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_UP"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_UP"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_UP"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_UP"},
    ]
    
    # --- LISTA DE MODELOS ACTUALIZADA CON FALLBACK (CORRECCIÓN DEL ERROR 404) ---
    # Se intentará usar los modelos en este orden. Si uno falla, prueba con el siguiente.
    model_list = [
        "gemini-1.5-flash-001", # Modelo más reciente y rápido
        "gemini-1.5-pro-latest",   # Modelo pro más estable
        "gemini-1.0-pro",       # Modelo base como último recurso
    ]

    # Contenido a enviar al modelo
    content = [prompt, image] if image else [prompt]

    for model_name in model_list:
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            response = model.generate_content(content)
            
            if response.parts:
                logging.info(f"Respuesta exitosa usando el modelo: {model_name}")
                return response.text
            elif hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                block_reason = response.prompt_feedback.block_reason
                logging.warning(f"Modelo {model_name} bloqueó la respuesta: {block_reason}. Intentando con el siguiente modelo.")
                continue # Intenta con el siguiente modelo de la lista
        
        except Exception as e:
            logging.warning(f"Error con el modelo {model_name}: {e}. Intentando con el siguiente modelo.")
            continue # Intenta con el siguiente modelo

    # Si todos los modelos fallan
    final_error_msg = "Todos los modelos de Gemini fallaron o no están disponibles."
    logging.error(final_error_msg)
    return f'{{"error": "{final_error_msg}"}}'


def get_image_attributes(image: Image.Image):
    """
    Paso 1: Actúa como el "Ojo". Extrae atributos visuales de la imagen y los devuelve en formato JSON.
    Utiliza la nueva función con fallback.
    """
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
    return _call_gemini_with_fallback(prompt, image=image)


def get_best_match_from_attributes(attributes: dict, inventory_names: list):
    """
    Paso 2: Actúa como el "Cerebro Logístico". Recibe los atributos y la lista del inventario,
    y razona cuál es la mejor coincidencia. Utiliza la nueva función con fallback.
    """
    if not inventory_names:
        return '{"best_match": "Artículo no encontrado", "reasoning": "El inventario está vacío."}'
        
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
    return _call_gemini_with_fallback(prompt)
