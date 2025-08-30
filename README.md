📦 Inventario Inteligente con IA
Este proyecto es una aplicación web construida con Streamlit que utiliza la API de Gemini para identificar artículos de un inventario a partir de imágenes capturadas por la cámara. La lista de artículos se gestiona en una base de datos de Firebase Firestore en tiempo real.

✨ Características
Interfaz Web Interactiva: Carga imágenes fácilmente a través de la cámara del dispositivo.

Gestión de Inventario en Tiempo Real: Añade nuevos artículos a tu base de datos de Firebase directamente desde la aplicación.

Reconocimiento por IA: Utiliza el potente modelo gemini-pro-vision para una clasificación precisa de objetos.

Seguro y Escalable: Construido sobre la infraestructura de Google Cloud y Firebase.

🚀 Cómo Desplegar
Clonar el Repositorio: git clone <url-del-repositorio>

Instalar Dependencias: pip install -r requirements.txt

Configurar Credenciales:

Crear un archivo .env con las claves GEMINI_API_KEY y GOOGLE_APPLICATION_CREDENTIALS.

Añadir el archivo serviceAccountKey.json de Firebase.

Ejecutar Localmente: streamlit run streamlit_app.py

Este proyecto está listo para ser desplegado en Streamlit Community Cloud.
