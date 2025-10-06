import streamlit as st
import os
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import plotly.express as px
from datetime import datetime
import logging

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

# CSS personalizado (sin cambios)
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
    .feature-box { background-color: #f0f2f6; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #1f77b4; margin-bottom: 1rem; }
    .stats-box { background: #ffffff; color: #333; padding: 1rem; border-radius: 10px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# --- Inicialización de Servicios y Estado ---

# Usamos st.session_state para mantener los datos entre re-ejecuciones
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'data_to_save' not in st.session_state:
    st.session_state.data_to_save = None

@st.cache_resource
def load_yolo_model():
    try:
        logger.info("Cargando modelo YOLOv8...")
        model = YOLO('yolov8n.pt') # Usamos un modelo más ligero para mayor velocidad
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
page = st.sidebar.radio(
    "Selecciona una opción:",
    ["🏠 Inicio", "📸 Cámara en Vivo", "📁 Subir Imagen", "📝 Análisis de Texto", "📊 Dashboard", "🗃️ Base de Datos"],
    captions=["Visión general", "Análisis en tiempo real", "Analiza tus archivos", "Describe y analiza", "Visualiza tus datos", "Gestiona tu inventario"]
)

# --- Lógica de las Páginas ---

def handle_save_to_inventory():
    """Función reutilizable para guardar en Firebase."""
    if st.session_state.data_to_save:
        try:
            doc_id = firebase.save_inventory_item(st.session_state.data_to_save)
            st.success(f"✅ ¡Guardado en inventario con éxito! ID: {doc_id}")
            st.toast("¡Elemento guardado!", icon="💾")
            # Limpiar estado para evitar re-guardado
            st.session_state.analysis_result = None
            st.session_state.data_to_save = None
        except Exception as e:
            st.error(f"Error al guardar en Firebase: {e}")

# PÁGINA: INICIO
if page == "🏠 Inicio":
    st.header("🏠 Bienvenido al Sistema de Reconocimiento de Inventario")
    # ... (El resto del código de la página de inicio no necesita cambios) ...
    try:
        items = firebase.get_all_inventory_items()
        total_items = len(items)
        image_items = len([item for item in items if item.get('tipo') in ['imagen', 'camera']])
        text_items = len([item for item in items if item.get('tipo') == 'texto'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de Elementos", total_items)
        with col2:
            st.metric("Imágenes Analizadas", image_items)
        with col3:
            st.metric("Descripciones Procesadas", text_items)
            
    except Exception as e:
        st.warning(f"No se pudieron cargar las estadísticas: {e}")

    st.subheader("🚀 Características Principales")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="feature-box"><h4>📸 Análisis por Cámara en Vivo</h4><p>Captura imágenes en tiempo real para análisis inmediato.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-box"><h4>📁 Subida de Archivos</h4><p>Sube imágenes para un análisis detallado con IA.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="feature-box"><h4>📝 Análisis de Texto</h4><p>Describe elementos y obtén un análisis estructurado.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-box"><h4>📊 Dashboard Completo</h4><p>Visualiza estadísticas y tendencias de tu inventario.</p></div>', unsafe_allow_html=True)

# PÁGINA: CÁMARA EN VIVO
elif page == "📸 Cámara en Vivo":
    st.header("📸 Análisis con Cámara en Vivo")
    st.info("Activa tu cámara, toma una foto y analízala con la IA de Gemini.")
    
    picture = st.camera_input("📷 Haz clic aquí para tomar una foto", key="camera_input")

    if picture:
        st.image(picture, caption="Imagen capturada. ¡Lista para analizar!", use_column_width=True)
        image_pil = Image.open(picture)

        description = st.text_input("Añade una descripción (opcional)", key="camera_desc")

        if st.button("🧠 Analizar con Gemini AI", key="camera_analyze"):
            with st.spinner("Analizando con la magia de Gemini..."):
                try:
                    analysis = gemini.analyze_image(image_pil, description)
                    st.session_state.analysis_result = analysis
                    st.session_state.data_to_save = {
                        "tipo": "camera",
                        "archivo": f"camera_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                        "descripcion": description,
                        "analisis": analysis,
                        "timestamp": firebase.get_timestamp()
                    }
                except Exception as e:
                    st.error(f"Error en el análisis de Gemini: {e}")
                    st.session_state.analysis_result = None

        if st.session_state.analysis_result:
            st.subheader("📝 Resultado del Análisis")
            st.text_area("Análisis de IA:", st.session_state.analysis_result, height=200, key="camera_result_area")
            if st.button("💾 Guardar en Inventario", key="camera_save"):
                handle_save_to_inventory()

# PÁGINA: SUBIR IMAGEN
elif page == "📁 Subir Imagen":
    st.header("📁 Análisis de Imagen Subida")
    uploaded_file = st.file_uploader("Sube una imagen", type=['png', 'jpg', 'jpeg'])

    if uploaded_file:
        st.image(uploaded_file, caption="Imagen subida.", use_column_width=True)
        image_pil = Image.open(uploaded_file)

        description = st.text_input("Añade una descripción (opcional)", key="upload_desc")

        if st.button("🧠 Analizar con Gemini AI", key="upload_analyze"):
            with st.spinner("Procesando imagen con Gemini..."):
                try:
                    analysis = gemini.analyze_image(image_pil, description)
                    st.session_state.analysis_result = analysis
                    st.session_state.data_to_save = {
                        "tipo": "imagen",
                        "archivo": uploaded_file.name,
                        "descripcion": description,
                        "analisis": analysis,
                        "timestamp": firebase.get_timestamp()
                    }
                except Exception as e:
                    st.error(f"Error en el análisis de Gemini: {e}")
                    st.session_state.analysis_result = None

        if st.session_state.analysis_result:
            st.subheader("📝 Resultado del Análisis")
            st.text_area("Análisis de IA:", st.session_state.analysis_result, height=200, key="upload_result_area")
            if st.button("💾 Guardar en Inventario", key="upload_save"):
                handle_save_to_inventory()

# PÁGINA: ANÁLISIS DE TEXTO
elif page == "📝 Análisis de Texto":
    st.header("📝 Análisis de Texto con IA")
    text_input = st.text_area("Describe los elementos del inventario:", height=150, placeholder="Ej: 15 laptops Dell Inspiron, estado bueno, modelo 2023...")

    if st.button("🧠 Analizar Descripción", key="text_analyze"):
        if text_input.strip():
            with st.spinner("Generando análisis estructurado..."):
                try:
                    analysis = gemini.generate_description(text_input)
                    st.session_state.analysis_result = analysis
                    st.session_state.data_to_save = {
                        "tipo": "texto",
                        "descripcion": text_input,
                        "analisis": analysis,
                        "timestamp": firebase.get_timestamp()
                    }
                except Exception as e:
                    st.error(f"Error en el análisis: {e}")
                    st.session_state.analysis_result = None
        else:
            st.warning("Por favor, ingresa una descripción.")

    if st.session_state.analysis_result:
        st.subheader("📝 Resultado del Análisis")
        st.text_area("Análisis de IA:", st.session_state.analysis_result, height=200, key="text_result_area")
        if st.button("💾 Guardar en Inventario", key="text_save"):
            handle_save_to_inventory()

# PÁGINA: DASHBOARD
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
            
            st.subheader("📊 Distribución por Tipo")
            type_counts = df['tipo'].value_counts()
            fig_pie = px.pie(type_counts, values=type_counts.values, names=type_counts.index, title="Distribución de Entradas")
            st.plotly_chart(fig_pie, use_container_width=True)

            st.subheader("🕒 Actividad por Día")
            daily_counts = df.groupby('date').size().reset_index(name='count')
            fig_bar = px.bar(daily_counts, x='date', y='count', title="Elementos Agregados por Día")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Aún no hay elementos en el inventario para mostrar.")
    except Exception as e:
        st.error(f"Error al cargar el dashboard: {e}")

# PÁGINA: BASE DE DATOS
elif page == "🗃️ Base de Datos":
    st.header("🗃️ Gestión de la Base de Datos")

    if st.button("🔄 Refrescar Datos"):
        st.cache_data.clear()
        st.toast("Datos actualizados", icon="🔄")

    try:
        items = firebase.get_all_inventory_items()
        if items:
            st.info(f"Se encontraron {len(items)} elementos en el inventario.")
            for item in items:
                with st.expander(f"📦 **{item.get('tipo', 'N/A').capitalize()}** - {item.get('timestamp', 'N/A')}"):
                    st.json(item)
                    if st.button("🗑️ Eliminar", key=f"delete_{item['id']}"):
                        try:
                            firebase.delete_inventory_item(item['id'])
                            st.success(f"Elemento {item['id']} eliminado.")
                            st.rerun() # Vuelve a ejecutar para refrescar la lista
                        except Exception as e:
                            st.error(f"Error al eliminar: {e}")
        else:
            st.warning("La base de datos está vacía.")
    except Exception as e:
        st.error(f"No se pudo conectar con la base de datos: {e}")
