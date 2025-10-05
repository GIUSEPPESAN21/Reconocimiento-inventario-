# Sistema de Reconocimiento de Inventario con IA

Este proyecto utiliza Google Gemini AI para analizar elementos de inventario mediante imágenes y descripciones.

## 🚀 Características

- Reconocimiento de elementos usando Gemini AI
- Almacenamiento en Firebase Firestore
- Interfaz web con Streamlit
- Análisis de imágenes con IA
- Gestión completa de inventario

## 📋 Requisitos Previos

1. **Python 3.8+**
2. **Cuenta de Google Cloud** con acceso a Gemini AI
3. **Proyecto de Firebase** configurado

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

### 3. Configurar variables de entorno
Copia `.env.example` a `.env` y configura las variables:

```bash
cp .env.example .env
```

Edita `.env` con tus credenciales:
```env
GEMINI_API_KEY=tu_api_key_de_gemini
FIREBASE_PROJECT_ID=tu_proyecto_firebase
FIREBASE_CREDENTIALS_PATH=firebase-credentials.json
```

### 4. Configurar Gemini AI
1. Ve a [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Crea una nueva API key
3. Agrega la key a tu archivo `.env`

### 5. Configurar Firebase
1. Ve a [Firebase Console](https://console.firebase.google.com/)
2. Crea un nuevo proyecto o usa uno existente
3. Habilita Firestore Database
4. Descarga el archivo de credenciales de servicio
5. Renombra el archivo a `firebase-credentials.json`
6. Colócalo en la raíz del proyecto

## 🏃‍♂️ Ejecutar la aplicación

```bash
streamlit run streamlit_app.py
```

La aplicación estará disponible en `http://localhost:8501`

## 🔧 Solución de Problemas

### Error "block_medium_and_up"
Este error se ha solucionado con:
- Uso de modelos más recientes y estables
- Configuración de seguridad optimizada
- Manejo de errores mejorado
- Fallback entre múltiples modelos

### Error de conexión a Firebase
- Verifica que el archivo `firebase-credentials.json` esté en la raíz
- Confirma que el proyecto ID sea correcto
- Asegúrate de que Firestore esté habilitado

### Error de API de Gemini
- Verifica que tu API key sea válida
- Confirma que tienes cuota disponible
- Revisa que el proyecto tenga acceso a Gemini API

## 📁 Estructura del Proyecto
