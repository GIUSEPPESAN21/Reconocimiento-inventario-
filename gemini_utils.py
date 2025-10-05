import google.generativeai as genai
from PIL import Image
import streamlit as st
import json
import logging

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _call_gemini_with_fallback(prompt, image=None):
    """
    Función interna para llamar a la API de Gemini con una lista de modelos, 
    fallback automático y registro de errores detallado.
    """
    last_error = None
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
    except KeyError:
        error_msg = "La clave GEMINI_API_KEY no está configurada en los secretos de Streamlit."
        logging.error(error_msg)
        return json.dumps({"error": error_msg})

    generation_config = {
        "temperature": 0.4,
        "top_p": 1.0,
        "top_k": 32,
        "max_output_tokens": 4096,
    }
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_UP"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_UP"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_UP"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_UP"},
    ]
    
    # Lista de modelos refinada para estabilidad. Flash es rápido, Pro es un buen respaldo.
    model_list = [
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro-latest",
    ]

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
            
            # Guardar la razón del bloqueo si la respuesta fue bloqueada
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                last_error = f"Modelo {model_name} bloqueó la respuesta: {response.prompt_feedback.block_reason}"
                logging.warning(f"{last_error}. Intentando con el siguiente modelo.")
            else:
                last_error = f"El modelo {model_name} no devolvió contenido."
                logging.warning(f"{last_error}. Intentando con el siguiente modelo.")

        except Exception as e:
            # --- MEJORA CLAVE: Registrar el error específico ---
            last_error = f"Error de API con el modelo {model_name}: {e}"
            logging.error(f"{last_error}. Intentando con el siguiente modelo.")
            continue

    # Si todos los modelos fallan, devolver el último error conocido
    final_error_msg = f"No se pudo conectar con ningún modelo de IA disponible. Último error: {last_error}"
    logging.error(final_error_msg)
    return json.dumps({"error": final_error_msg})


def get_image_attributes(image: Image.Image):
    """
    Paso 1: Actúa como el "Ojo". Extrae atributos visuales de la imagen.
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
    Paso 2: Actúa como el "Cerebro Logístico". Encuentra la mejor coincidencia en el inventario.
    """
    if not inventory_names:
        return json.dumps({"best_match": "Artículo no encontrado", "reasoning": "El inventario está vacío."})
        
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
    1. "best_match": (string) El nombre exacto del artículo de la lista. Si no hay coincidencia, usa "Artículo no encontrado".
    2. "reasoning": (string) Una frase corta explicando tu elección.
    """
    return _call_gemini_with_fallback(prompt)

