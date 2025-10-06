import google.generativeai as genai
import logging
from PIL import Image
import streamlit as st
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiUtils:
    def __init__(self):
        self.api_key = st.secrets.get('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY no encontrada en secrets")
        
        genai.configure(api_key=self.api_key)
        self.model = self._get_available_model()
    
    def _get_available_model(self):
        """Obtiene el primer modelo disponible de la lista"""
        # Modelos actualizados y disponibles en 2025
        modelos_disponibles = [
            "gemini-2.0-flash-exp",  # Modelo experimental más reciente
            "gemini-1.5-flash-latest",  # Versión más reciente de 1.5
            "gemini-1.5-pro-latest",   # Versión más reciente de 1.5 pro
            "gemini-1.5-flash",        # Modelo básico
            "gemini-1.5-pro",          # Modelo pro básico
        ]
        
        for modelo in modelos_disponibles:
            try:
                m = genai.GenerativeModel(modelo)
                _ = m.generate_content(["test", Image.new('RGB', (1, 1))]) 
                logger.info(f"Modelo de visión {modelo} inicializado correctamente.")
                return m
            except Exception as e:
                logger.warning(f"Modelo de visión {modelo} no disponible o no compatible: {e}")
                continue
        
        raise Exception("No se pudo inicializar ningún modelo de visión de Gemini compatible.")
    
    def analyze_image(self, image_pil: Image, description: str = ""):
        """Analiza una imagen PIL y devuelve una respuesta JSON."""
        try:
            prompt = f"""
            Analiza esta imagen de un objeto de inventario.
            Descripción adicional: "{description}"
            
            Tu tarea es identificar y describir el objeto principal. Responde únicamente con un objeto JSON válido con estas claves:
            - "elemento_identificado": (string) El nombre específico del objeto.
            - "cantidad_aproximada": (integer) El número de unidades que ves.
            - "estado_condicion": (string) La condición aparente (ej: "Nuevo", "Usado").
            - "caracteristicas_distintivas": (string) Una lista de características visuales en una sola cadena de texto.
            - "posible_categoria_de_inventario": (string) Una categoría de inventario (ej: "Suministros de Oficina").

            Ejemplo:
            {{
              "elemento_identificado": "Taza de cerámica blanca",
              "cantidad_aproximada": 1,
              "estado_condicion": "Usado",
              "caracteristicas_distintivas": "Color blanco, material de cerámica, tiene un asa",
              "posible_categoria_de_inventario": "Menaje de Cocina"
            }}
            
            IMPORTANTE: Tu respuesta debe ser solo el objeto JSON, sin incluir ```json al principio o al final.
            """
            
            response = self.model.generate_content([prompt, image_pil])
            
            if response and response.text:
                return response.text.strip()
            else:
                return json.dumps({"error": "No se pudo analizar la imagen"})
                
        except Exception as e:
            logger.error(f"Error al analizar imagen con Gemini: {e}")
            return json.dumps({"error": f"Error en el análisis de Gemini: {str(e)}"})
