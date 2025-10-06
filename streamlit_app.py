import streamlit as st
import os
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import plotly.express as px
from datetime import datetime
import logging
import platform
import sys
import json

from ultralytics import YOLO
from firebase_utils import FirebaseUtils
from gemini_utils import GeminiUtils

# --- Configuración Inicial ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Sistema de Reconocimiento de Inventario con IA",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado (del repositorio original)
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .stats-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .author-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Inicialización de Estado y Servicios ---

# Usamos st.session_state para mantener los datos entre interacciones
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'data_to_save' not in st.session_state:
    st.session_state.data_to_save = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "🏠 Inicio"

@st.cache_resource
def load_yolo_model():
    try:
        logger.info("Cargando modelo YOLOv8...")
        model = YOLO('yolov8m.pt') # Modelo mediano como en el original
        logger.info("Modelo YOLOv8 cargado.")
        return model
    except Exception as e:
        st.error(f"Error al cargar YOLO: {e}")
        return None

@st.cache_resource
def initialize_services():
    try:
        firebase = FirebaseUtils()
        gemini = GeminiUtils()
        return firebase, gemini
    except Exception as e:
        st.error(f"Error al inicializar servicios (Firebase/Gemini). Verifica tus secrets. Detalle: {e}")
        return None, None

yolo_model = load_yolo_model()
firebase, gemini = initialize_services()

if not all([yolo_model, firebase, gemini]):
    st.error("❌ Faltan servicios clave. La aplicación no puede continuar.")
    st.stop()

# --- Interfaz de Usuario ---

st.markdown('<h1 class="main-header">🤖 Sistema de Reconocimiento de Inventario con IA</h1>', unsafe_allow_html=True)

st.sidebar.title("📋 Navegación")
page_options = [
    "🏠 Inicio", "📸 Cámara en Vivo", "📁 Subir Imagen", "📝 Análisis de Texto", 
    "📊 Dashboard", "🗃️ Base de Datos", "👥 Información del Proyecto", "⚙️ Configuración"
]
page = st.sidebar.selectbox("Selecciona una opción:", page_options)

# Limpiar estado si cambiamos de página
if page != st.session_state.current_page:
    st.session_state.analysis_result = None
    st.session_state.data_to_save = None
    st.session_state.current_page = page

# --- Lógica Funcional Centralizada ---

def handle_save_to_inventory():
    """Función reutilizable para guardar en Firebase desde session_state."""
    if st.session_state.data_to_save:
        try:
            doc_id = firebase.save_inventory_item(st.session_state.data_to_save)
            st.success(f"✅ ¡Guardado en inventario con éxito! ID: {doc_id}")
            st.toast("¡Elemento guardado!", icon="💾")
            st.session_state.analysis_result = None
            st.session_state.data_to_save = None
        except Exception as e:
            st.error(f"Error al guardar en Firebase: {e}")

# --- Lógica de las Páginas ---

if page == "🏠 Inicio":
    st.header("🏠 Bienvenido al Sistema de Reconocimiento de Inventario")
    try:
        items = firebase.get_all_inventory_items()
        col1, col2, col3 = st.columns(3)
        with col1: st.markdown(f'<div class="stats-box"><h3>{len(items)}</h3><p>Total Elementos</p></div>', unsafe_allow_html=True)
        with col2: st.markdown(f'<div class="stats-box"><h3>{len([i for i in items if i.get("tipo") in ["imagen", "camera"]])}</h3><p>Imágenes Analizadas</p></div>', unsafe_allow_html=True)
        with col3: st.markdown(f'<div class="stats-box"><h3>{len([i for i in items if i.get("tipo") == "texto"])}</h3><p>Descripciones Procesadas</p></div>', unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Error al cargar estadísticas: {e}")
    st.subheader("🚀 Características Principales")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="feature-box"><h4>📸 Análisis por Cámara en Vivo</h4><p>Captura imágenes para análisis inmediato con YOLO e IA.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-box"><h4>📁 Subida de Archivos</h4><p>Sube imágenes para un análisis detallado con Gemini AI.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="feature-box"><h4>📝 Análisis de Texto</h4><p>Describe elementos y obtén un análisis estructurado con IA.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-box"><h4>📊 Dashboard Completo</h4><p>Visualiza estadísticas de tu inventario con gráficos interactivos.</p></div>', unsafe_allow_html=True)

elif page == "📸 Cámara en Vivo":
    st.header("📸 Análisis con Cámara en Vivo")
    picture = st.camera_input("📷 Toma una foto")

    if picture:
        st.image(picture, caption="Imagen capturada")
        image_pil = Image.open(picture)
        
        st.subheader("🧠 Análisis con Gemini AI")
        description = st.text_input("Descripción adicional (opcional):", key="cam_desc")

        if st.button("✨ Analizar con IA", key="cam_analyze"):
            with st.spinner("Analizando con Gemini AI..."):
                analysis = gemini.analyze_image(image_pil, description)
                st.session_state.analysis_result = analysis
                st.session_state.data_to_save = {
                    "tipo": "camera", "archivo": f"camera_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                    "descripcion": description, "analisis": analysis, "timestamp": firebase.get_timestamp()
                }
        
        if st.session_state.analysis_result:
            st.subheader("📝 Resultado del Análisis IA:")
            st.text_area("Análisis:", st.session_state.analysis_result, height=200, key="cam_result")
            if st.button("💾 Guardar en Inventario", key="cam_save"):
                handle_save_to_inventory()

elif page == "📁 Subir Imagen":
    st.header("📁 Análisis de Imagen Subida")
    uploaded_file = st.file_uploader("Sube una imagen", type=['png', 'jpg', 'jpeg'])

    if uploaded_file:
        st.image(uploaded_file, caption="Imagen subida")
        image_pil = Image.open(uploaded_file)

        st.subheader("🧠 Análisis con Gemini AI")
        description = st.text_input("Descripción adicional (opcional):", key="upload_desc")

        if st.button("✨ Analizar con IA", key="upload_analyze"):
            with st.spinner("Analizando con Gemini AI..."):
                analysis = gemini.analyze_image(image_pil, description)
                st.session_state.analysis_result = analysis
                st.session_state.data_to_save = {
                    "tipo": "imagen", "archivo": uploaded_file.name,
                    "descripcion": description, "analisis": analysis, "timestamp": firebase.get_timestamp()
                }

        if st.session_state.analysis_result:
            st.subheader("📝 Resultado del Análisis IA:")
            st.text_area("Análisis:", st.session_state.analysis_result, height=200, key="upload_result")
            if st.button("💾 Guardar en Inventario", key="upload_save"):
                handle_save_to_inventory()

elif page == "📝 Análisis de Texto":
    st.header("📝 Análisis de Texto con IA")
    text_input = st.text_area("Describe los elementos de inventario:", height=150, placeholder="Ej: 15 laptops Dell...")

    if st.button("🧠 Analizar Descripción"):
        if text_input.strip():
            with st.spinner("Analizando descripción..."):
                analysis = gemini.generate_description(text_input)
                st.session_state.analysis_result = analysis
                st.session_state.data_to_save = {
                    "tipo": "texto", "descripcion": text_input, "analisis": analysis,
                    "timestamp": firebase.get_timestamp()
                }
        else:
            st.warning("Por favor, ingresa una descripción")
            
    if st.session_state.analysis_result:
        st.subheader("📝 Análisis de la Descripción:")
        st.text_area("Resultado:", st.session_state.analysis_result, height=200, key="text_result")
        if st.button("💾 Guardar Análisis", key="text_save"):
            handle_save_to_inventory()

elif page == "📊 Dashboard":
    st.header("📊 Dashboard de Inventario")
    try:
        items = firebase.get_all_inventory_items()
        if items:
            df = pd.DataFrame(items)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            
            st.subheader("📈 Estadísticas Generales")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Elementos", len(df))
            col2.metric("Tipos de Entradas", df['tipo'].nunique())
            col3.metric("Día más Reciente", df['date'].max().strftime('%Y-%m-%d'))
            
            st.subheader("📊 Gráficos Interactivos")
            type_counts = df['tipo'].value_counts()
            fig_pie = px.pie(type_counts, values=type_counts.values, names=type_counts.index, title="Distribución por Tipo")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No hay elementos en el inventario para mostrar.")
    except Exception as e:
        st.error(f"Error al cargar el dashboard: {e}")

elif page == "🗃️ Base de Datos":
    st.header("🗃️ Gestión de Base de Datos")
    if st.button("🔄 Actualizar Lista"):
        st.rerun()

    try:
        items = firebase.get_all_inventory_items()
        st.subheader(f"📋 Elementos Encontrados ({len(items)})")
        if items:
            for i, item in enumerate(items):
                with st.expander(f"📦 {item.get('tipo', 'N/A').capitalize()} - {item.get('timestamp', 'Sin fecha')[:19]}"):
                    st.json(item) # Muestra todos los datos de forma ordenada
                    if st.button(f"🗑️ Eliminar", key=f"delete_{item['id']}_{i}"):
                        firebase.delete_inventory_item(item['id'])
                        st.success(f"Elemento {item['id']} eliminado.")
                        st.rerun() # Recarga la página para mostrar la lista actualizada
        else:
            st.info("No se encontraron elementos en la base de datos.")
    except Exception as e:
        st.error(f"Error al cargar inventario: {e}")

elif page == "👥 Información del Proyecto":
    st.header("👥 Información del Proyecto")
    st.markdown('<div class="author-info"><h4>🎓 Estudiante Principal</h4><p><strong>Nombre:</strong> Giuseppe Sánchez</p><p><strong>Universidad:</strong> Uniminuto</p><p><strong>Programa:</strong> Ingeniería de Sistemas</p><p><strong>Año:</strong> 2025</p></div>', unsafe_allow_html=True)
    st.subheader("🔧 Tecnologías Utilizadas")
    st.markdown("- **Streamlit:** Framework para la interfaz de usuario\n- **Google Gemini AI:** Modelo de IA para análisis\n- **YOLOv8:** Modelo de detección de objetos\n- **Firebase Firestore:** Base de datos NoSQL")

elif page == "⚙️ Configuración":
    st.header("⚙️ Configuración y Estado del Sistema")
    st.subheader("✔️ Estado de los Servicios")
    col1, col2, col3 = st.columns(3)
    col1.success("✅ Firebase: Conectado") if firebase else col1.error("❌ Firebase: Desconectado")
    col2.success("✅ Gemini AI: Conectado") if gemini else col2.error("❌ Gemini AI: Desconectado")
    col3.success("✅ YOLOv8: Cargado") if yolo_model else col3.error("❌ YOLOv8: No cargado")
    st.subheader("💻 Información del Sistema")
    st.info(f"**Python:** {sys.version.split()[0]} | **Streamlit:** {st.__version__} | **OS:** {platform.system()}")

# Footer
st.markdown("---")
st.markdown('<div style="text-align: center; color: #666;"><p>Desarrollado por <strong>Giuseppe Sánchez</strong> - Uniminuto 2025</p></div>', unsafe_allow_html=True)

