import google.generativeai as genai
import logging
import os

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiUtils:
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY no encontrada en las variables de entorno")
        
        genai.configure(api_key=self.api_key)
        self.model = self._get_available_model()
    
    def _get_available_model(self):
        """Obtiene el primer modelo disponible de la lista"""
        # Modelos actualizados y disponibles en 2025
        modelos_disponibles = [
            "gemini-2.0-flash-exp",  # Modelo experimental más reciente
            "gemini-1.5-flash-latest",  # Versión más reciente de 1.5
            "gemini-1.5-pro-latest",   # Versión más reciente de 1.5 pro
        ]
        
        for modelo in modelos_disponibles:
            try:
                model = genai.GenerativeModel(modelo)
                logger.info(f"Modelo {modelo} inicializado correctamente")
                return model
            except Exception as e:
                logger.warning(f"Modelo {modelo} no disponible: {e}")
                continue
        
        # Fallback a modelo básico si ninguno funciona
        try:
            model = genai.GenerativeModel("models/gemini-1.5-flash")
            logger.info("Usando modelo fallback: models/gemini-1.5-flash")
            return model
        except Exception as e:
            logger.error(f"Error al inicializar modelo fallback: {e}")
            raise Exception("No se pudo inicializar ningún modelo de Gemini")
    
    def analyze_image(self, image_bytes, description=""):
        """Analiza una imagen usando Gemini AI"""
        try:
            # Crear el prompt
            prompt = f"""
            Analiza esta imagen y proporciona información detallada sobre los elementos de inventario que observes.
            Descripción proporcionada: {description}
            
            Por favor, identifica:
            1. Tipo de elemento(s)
            2. Cantidad aproximada
            3. Estado/condición
            4. Características distintivas
            5. Posible categoría de inventario
            
            Responde en formato JSON estructurado.
            """
            
            # Generar contenido
            response = self.model.generate_content([prompt, image_bytes])
            
            if response and response.text:
                return response.text
            else:
                return "No se pudo analizar la imagen"
                
        except Exception as e:
            logger.error(f"Error al analizar imagen: {e}")
            return f"Error en el análisis: {str(e)}"
    
    def generate_description(self, text_input):
        """Genera una descripción usando texto"""
        try:
            prompt = f"""
            Analiza la siguiente descripción de inventario y proporciona información estructurada:
            
            Descripción: {text_input}
            
            Extrae:
            1. Tipo de elemento
            2. Cantidad
            3. Estado
            4. Características
            5. Categoría sugerida
            
            Responde en formato JSON.
            """
            
            response = self.model.generate_content(prompt)
            
            if response and response.text:
                return response.text
            else:
                return "No se pudo procesar la descripción"
                
        except Exception as e:
            logger.error(f"Error al procesar descripción: {str(e)}")
            return f"Error en el procesamiento: {str(e)}"
