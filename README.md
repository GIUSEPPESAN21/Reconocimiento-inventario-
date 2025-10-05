# Sistema de Reconocimiento de Inventario con IA

Este proyecto utiliza Google Gemini AI para analizar elementos de inventario mediante imágenes y descripciones, con almacenamiento seguro en Firebase usando Streamlit Secrets.

## 🚀 Características

- Reconocimiento de elementos usando Gemini AI
- Almacenamiento seguro en Firebase Firestore
- Interfaz web con Streamlit
- Análisis de imágenes con IA
- Gestión completa de inventario
- Configuración segura con Streamlit Secrets

## 📋 Requisitos Previos

1. **Python 3.8+**
2. **Cuenta de Google Cloud** con acceso a Gemini AI
3. **Proyecto de Firebase** configurado
4. **Cuenta de Streamlit** para deployment

## ⚙️ Configuración

### 1. Clonar el repositorio
```bash
git clone <tu-repositorio>
cd Inventario-Code
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Configurar Streamlit Secrets

#### Para desarrollo local:
Crea el archivo `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "tu_api_key_de_gemini"
FIREBASE_SERVICE_ACCOUNT_BASE64 = "tu_credencial_base64"
```

#### Para deployment en Streamlit Cloud:
1. Ve a tu aplicación en [Streamlit Cloud](https://share.streamlit.io/)
2. En "Settings" → "Secrets"
3. Agrega los siguientes secrets:
   - `GEMINI_API_KEY`: Tu API key de Gemini
   - `FIREBASE_SERVICE_ACCOUNT_BASE64`: Tu credencial de Firebase en base64

### 4. Configurar Gemini AI
1. Ve a [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Crea una nueva API key
3. Agrega la key a tus Streamlit secrets

### 5. Configurar Firebase
1. Ve a [Firebase Console](https://console.firebase.google.com/)
2. Crea un nuevo proyecto o usa uno existente
3. Habilita Firestore Database
4. Descarga el archivo de credenciales de servicio
5. Convierte el archivo JSON a base64:
   ```bash
   base64 -i firebase-credentials.json
   ```
6. Agrega el resultado a tus Streamlit secrets como `FIREBASE_SERVICE_ACCOUNT_BASE64`

## 🏃‍♂️ Ejecutar la aplicación

### Desarrollo local:
```bash
streamlit run streamlit_app.py
```

### Deployment en Streamlit Cloud:
1. Sube tu código a GitHub
2. Conecta tu repositorio a Streamlit Cloud
3. Configura los secrets en la interfaz de Streamlit
4. Deploy!

## 🔧 Solución de Problemas

### Error "block_medium_and_up"
Este error se ha solucionado con:
- Uso de modelos más recientes y estables
- Configuración de seguridad optimizada
- Manejo de errores mejorado
- Fallback entre múltiples modelos

### Error de secrets en Streamlit
- Verifica que los secrets estén configurados correctamente
- Asegúrate de que el formato base64 sea válido
- Revisa que los nombres de los secrets coincidan exactamente

### Error de conexión a Firebase
- Verifica que el base64 de las credenciales sea correcto
- Confirma que el proyecto ID sea correcto
- Asegúrate de que Firestore esté habilitado

### Error de API de Gemini
- Verifica que tu API key sea válida
- Confirma que tienes cuota disponible
- Revisa que el proyecto tenga acceso a Gemini API

## 📁 Estructura del Proyecto
