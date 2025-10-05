import google.generativeai as genai
import logging
import streamlit as st
from typing import Optional, Dict, Any
import json
import base64
import io

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiUtils:
    def __init__(self):
        """Inicializar la clase GeminiUtils con configuración optimizada usando Streamlit secrets."""
        # Obtener API key desde Streamlit secrets
        try:
            self.api_key = st.secrets["GEMINI_API_KEY"]
        except KeyError:
            raise ValueError("GEMINI_API_KEY no encontrada en Streamlit secrets. Configúrala en .streamlit/secrets.toml")
        
        # Configurar la API
        genai.configure(api_key=self.api_key)
        
        # Modelos disponibles ordenados por prioridad (más estables primero)
        self.models = [
            "gemini-1.5-flash-002",
            "gemini-1.5-flash-8b", 
            "gemini-1.5-flash",
            "gemini-1.5-pro-002",
            "gemini-1.5-pro",
            "gemini-1.0-pro"
        ]
        
        self.current_model = None
        self.model_config = {
            "temperature": 0.1,
            "top_p": 0.8,
            "top_k": 20,
            "max_output_tokens": 2048,
            "response_mime_type": "text/plain"
        }
        
        # Configuración de seguridad más permisiva
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH", 
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]

    def get_available_model(self) -> Optional[str]:
        """Obtener un modelo disponible de la lista."""
        for model_name in self.models:
            try:
                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=self.model_config,
                    safety_settings=self.safety_settings
                )
                # Probar el modelo con una consulta simple
                response = model.generate_content("Hola")
                if response.text:
                    logger.info(f"Modelo {model_name} disponible y funcionando")
                    self.current_model = model_name
                    return model_name
            except Exception as e:
                logger.warning(f"Modelo {model_name} no disponible: {str(e)}")
                continue
        
        return None

    def generate_content(self, prompt: str, image_data: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Generar contenido usando Gemini AI.
        
        Args:
            prompt (str): Texto del prompt
            image_data (Optional[bytes]): Datos de imagen si es necesario
            
        Returns:
            Dict[str, Any]: Respuesta del modelo o error
        """
        try:
            # Asegurar que tenemos un modelo disponible
            if not self.current_model:
                model_name = self.get_available_model()
                if not model_name:
                    return {
                        "success": False,
                        "error": "No hay modelos de IA disponibles",
                        "response": None
                    }

            # Crear el modelo
            model = genai.GenerativeModel(
                model_name=self.current_model,
                generation_config=self.model_config,
                safety_settings=self.safety_settings
            )

            # Preparar el contenido
            content_parts = [prompt]
            if image_data:
                content_parts.append({
                    "mime_type": "image/jpeg",
                    "data": image_data
                })

            # Generar respuesta
            response = model.generate_content(content_parts)
            
            # Verificar si la respuesta fue bloqueada
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                logger.warning(f"Respuesta bloqueada: {response.prompt_feedback.block_reason}")
                return {
                    "success": False,
                    "error": f"Contenido bloqueado: {response.prompt_feedback.block_reason}",
                    "response": None
                }

            if response.candidates and response.candidates[0].finish_reason == "SAFETY":
                logger.warning("Respuesta bloqueada por filtros de seguridad")
                return {
                    "success": False,
                    "error": "Respuesta bloqueada por filtros de seguridad",
                    "response": None
                }

            if response.text:
                return {
                    "success": True,
                    "error": None,
                    "response": response.text.strip(),
                    "model_used": self.current_model
                }
            else:
                return {
                    "success": False,
                    "error": "No se generó respuesta válida",
                    "response": None
                }

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error de API con el modelo {self.current_model}: {error_msg}")
            
            # Intentar con el siguiente modelo
            if self.current_model in self.models:
                current_index = self.models.index(self.current_model)
                if current_index < len(self.models) - 1:
                    self.current_model = self.models[current_index + 1]
                    logger.info(f"Intentando con el siguiente modelo: {self.current_model}")
                    return self.generate_content(prompt, image_data)
            
            return {
                "success": False,
                "error": f"No se pudo conectar con ningún modelo de IA disponible. Último error: {error_msg}",
                "response": None
            }

    def analyze_inventory_item(self, item_description: str, image_data: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Analizar un elemento del inventario.
        
        Args:
            item_description (str): Descripción del elemento
            image_data (Optional[bytes]): Imagen del elemento
            
        Returns:
            Dict[str, Any]: Análisis del elemento
        """
        prompt = f"""
        Analiza este elemento del inventario y proporciona información detallada:
        
        Descripción: {item_description}
        
        Por favor, proporciona:
        1. Nombre del producto
        2. Categoría
        3. Estado (nuevo, usado, dañado, etc.)
        4. Material principal
        5. Estimación de valor
        6. Recomendaciones de almacenamiento
        7. Código de barras o número de serie si es visible
        
        Responde en formato JSON estructurado.
        """
        
        return self.generate_content(prompt, image_data)

    def get_model_info(self) -> Dict[str, Any]:
        """Obtener información sobre el modelo actual."""
        return {
            "current_model": self.current_model,
            "available_models": self.models,
            "config": self.model_config,
            "safety_settings": self.safety_settings
        }

# Función de utilidad para uso directo
def get_gemini_response(prompt: str, image_data: Optional[bytes] = None) -> Dict[str, Any]:
    """
    Función de conveniencia para obtener respuesta de Gemini.
    
    Args:
        prompt (str): Texto del prompt
        image_data (Optional[bytes]): Datos de imagen
        
    Returns:
        Dict[str, Any]: Respuesta del modelo
    """
    try:
        gemini = GeminiUtils()
        return gemini.generate_content(prompt, image_data)
    except Exception as e:
        logger.error(f"Error inicializando Gemini: {str(e)}")
        return {
            "success": False,
            "error": f"Error inicializando Gemini: {str(e)}",
            "response": None
        }
