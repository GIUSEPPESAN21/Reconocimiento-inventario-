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

# CSS personalizado
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
    .feature-box { background-color: #f0f2f6; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #1f77b4; margin-bottom: 1rem; }
    .stats-box { background: #ffffff; color: #333; padding: 1rem; border-radius: 10px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .author-info { background-color: #e8f4fd; padding: 1rem; border-radius: 10px; border: 1px solid #1f77b4; margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# --- Inicialización de Servicios y Estado de Sesión ---
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'data_to_save' not in st.session_state:
    st.session_state.data_to_save = None

@st.cache_resource
def load_yolo_model():
    try {
        logger.info("Cargando modelo YOLOv8...")
        model = YOLO('yolov8n.pt') # Modelo ligero para rendimiento
        logger.info("Modelo YOLOv8 cargado.")
        return model
    } except Exception as e:
        st.error(f"Error al cargar YOLO: {e}")
        return None

@st.cache_resource
def initialize_services():
    try {
        firebase = FirebaseUtils()
        gemini = GeminiUtils()
        return firebase, gemini
    } except Exception as e:
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
    ["🏠 Inicio", "📸 Cámara en Vivo", "📁 Subir Imagen", "📝 Análisis de Texto", "📊 Dashboard", "🗃️ Base de Datos", "👥 Información", "⚙️ Configuración"]
)

# --- Lógica de las Páginas ---

def handle_save_to_inventory():
    if st.session_state.data_to_save:
        try:
            doc_id = firebase.save_inventory_item(st.session_state.data_to_save)
            st.success(f"✅ ¡Guardado en inventario con éxito! ID: {doc_id}")
            st.toast("¡Elemento guardado!", icon="💾")
            st.session_state.analysis_result = None
            st.session_state.data_to_save = None
        except Exception as e:
            st.error(f"Error al guardar en Firebase: {e}")

# PÁGINA: INICIO
if page == "🏠 Inicio":
    # ... (código sin cambios) ...
    st.header("🏠 Bienvenido al Sistema de Reconocimiento de Inventario")
    try:
        items = firebase.get_all_inventory_items()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total de Elementos", len(items))
        col2.metric("Imágenes Analizadas", len([i for i in items if i.get('tipo') in ['imagen', 'camera']]))
        col3.metric("Descripciones Procesadas", len([i for i in items if i.get('tipo') == 'texto']))
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

# PÁGINA: CÁMARA EN VIVO Y SUBIR IMAGEN (Lógica YOLO reutilizada)
def yolo_analysis_interface(image_pil):
    st.subheader("🎯 Análisis con YOLOv8")
    if st.button("🔍 Detectar Objetos con YOLO", key=f"yolo_{image_pil.__hash__}"):
        with st.spinner("Detectando objetos..."):
            try:
                results = yolo_model(image_pil)
                annotated_image = results[0].plot() # .plot() dibuja las cajas en la imagen
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                
                st.image(annotated_image_rgb, caption="Objetos detectados por YOLOv8", use_column_width=True)

                detected_objects = []
                for box in results[0].boxes:
                    label = yolo_model.names[int(box.cls)]
                    confidence = float(box.conf)
                    detected_objects.append(f"- {label} (Confianza: {confidence:.2f})")
                
                if detected_objects:
                    st.markdown("**Objetos encontrados:**\n" + "\n".join(detected_objects))
                else:
                    st.info("No se detectaron objetos con el modelo YOLO.")
            except Exception as e:
                st.error(f"Error durante el análisis con YOLO: {e}")

if page == "📸 Cámara en Vivo":
    st.header("📸 Análisis con Cámara en Vivo")
    picture = st.camera_input("📷 Haz clic para tomar una foto", key="camera_input")

    if picture:
        image_pil = Image.open(picture)
        st.image(image_pil, caption="Imagen capturada", use_column_width=True)
        
        yolo_analysis_interface(image_pil) # Interfaz de YOLO
        
        st.subheader("🧠 Análisis con Gemini AI")
        description = st.text_input("Añade una descripción (opcional)", key="camera_desc")

        if st.button("✨ Analizar con IA", key="camera_analyze"):
            with st.spinner("Analizando con Gemini..."):
                analysis = gemini.analyze_image(image_pil, description)
                st.session_state.analysis_result = analysis
                st.session_state.data_to_save = {
                    "tipo": "camera", "archivo": f"camera_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                    "descripcion": description, "analisis": analysis, "timestamp": firebase.get_timestamp()
                }

        if st.session_state.analysis_result:
            st.text_area("Resultado del Análisis:", st.session_state.analysis_result, height=200)
            if st.button("💾 Guardar en Inventario", key="camera_save"):
                handle_save_to_inventory()

elif page == "📁 Subir Imagen":
    st.header("📁 Análisis de Imagen Subida")
    uploaded_file = st.file_uploader("Sube una imagen", type=['png', 'jpg', 'jpeg'])

    if uploaded_file:
        image_pil = Image.open(uploaded_file)
        st.image(image_pil, caption="Imagen subida", use_column_width=True)
        
        yolo_analysis_interface(image_pil) # Interfaz de YOLO

        st.subheader("🧠 Análisis con Gemini AI")
        description = st.text_input("Añade una descripción (opcional)", key="upload_desc")

        if st.button("✨ Analizar con IA", key="upload_analyze"):
            with st.spinner("Procesando con Gemini..."):
                analysis = gemini.analyze_image(image_pil, description)
                st.session_state.analysis_result = analysis
                st.session_state.data_to_save = {
                    "tipo": "imagen", "archivo": uploaded_file.name, "descripcion": description,
                    "analisis": analysis, "timestamp": firebase.get_timestamp()
                }
        
        if st.session_state.analysis_result:
            st.text_area("Resultado del Análisis:", st.session_state.analysis_result, height=200)
            if st.button("💾 Guardar en Inventario", key="upload_save"):
                handle_save_to_inventory()

# PÁGINA: ANÁLISIS DE TEXTO
elif page == "📝 Análisis de Texto":
    # ... (código sin cambios) ...
    st.header("📝 Análisis de Texto con IA")
    text_input = st.text_area("Describe los elementos:", height=150, placeholder="Ej: 15 laptops Dell...")

    if st.button("🧠 Analizar Descripción"):
        if text_input.strip():
            with st.spinner("Generando análisis..."):
                analysis = gemini.generate_description(text_input)
                st.session_state.analysis_result = analysis
                st.session_state.data_to_save = {
                    "tipo": "texto", "descripcion": text_input, "analisis": analysis,
                    "timestamp": firebase.get_timestamp()
                }
        else:
            st.warning("Por favor, ingresa una descripción.")

    if st.session_state.analysis_result:
        st.text_area("Resultado del Análisis:", st.session_state.analysis_result, height=200)
        if st.button("💾 Guardar en Inventario"):
            handle_save_to_inventory()

# PÁGINA: DASHBOARD
elif page == "📊 Dashboard":
    # ... (código sin cambios) ...
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
            col1, col2 = st.columns(2)
            with col1:
                type_counts = df['tipo'].value_counts()
                fig_pie = px.pie(type_counts, values=type_counts.values, names=type_counts.index, title="Distribución por Tipo")
                st.plotly_chart(fig_pie, use_container_width=True)
            with col2:
                daily_counts = df.groupby('date').size().reset_index(name='count')
                fig_bar = px.bar(daily_counts, x='date', y='count', title="Actividad por Día")
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Aún no hay elementos en el inventario.")
    except Exception as e:
        st.error(f"Error al cargar el dashboard: {e}")

# PÁGINA: BASE DE DATOS
elif page == "🗃️ Base de Datos":
    st.header("🗃️ Gestión de la Base de Datos")
    if st.button("🔄 Refrescar Datos"):
        st.rerun()

    try:
        items = firebase.get_all_inventory_items()
        if items:
            df = pd.DataFrame(items)
            st.dataframe(df) # Muestra la tabla completa para una vista rápida

            st.subheader(f"Detalle de {len(items)} Elementos")
            for item in items:
                with st.expander(f"📦 **{item.get('tipo', 'N/A').capitalize()}** - {item.get('timestamp', 'N/A')}"):
                    # Intenta parsear el análisis como JSON para una mejor visualización
                    try:
                        analysis_json = json.loads(item.get('analisis', '{}'))
                        st.json(analysis_json)
                    except json.JSONDecodeError:
                        st.text(item.get('analisis', '')) # Muestra como texto si no es JSON
                    
                    st.write(f"**ID:** `{item.get('id', 'N/A')}`")
                    
                    if st.button("🗑️ Eliminar", key=f"delete_{item['id']}"):
                        firebase.delete_inventory_item(item['id'])
                        st.success(f"Elemento {item['id']} eliminado.")
                        st.rerun()
        else:
            st.warning("La base de datos está vacía.")
    except Exception as e:
        st.error(f"No se pudo conectar con la base de datos: {e}")

# PÁGINA: INFORMACIÓN
elif page == "👥 Información":
    st.header("👥 Información del Proyecto")
    st.markdown("""
    <div class="author-info">
        <h4><strong>Desarrollador Principal:</strong> Giuseppe Sánchez</h4>
        <p><strong>Institución:</strong> Corporación Universitaria Minuto de Dios (Uniminuto)</p>
        <p><strong>Programa:</strong> Ingeniería de Sistemas</p>
        <p><strong>Año:</strong> 2025</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🔧 Tecnologías Utilizadas")
    st.markdown("""
    - **Streamlit:** Framework para la interfaz de usuario.
    - **Google Gemini AI:** Modelo de IA para análisis de imágenes y texto.
    - **YOLOv8:** Modelo de detección de objetos en tiempo real.
    - **Firebase Firestore:** Base de datos NoSQL en la nube.
    """)

# PÁGINA: CONFIGURACIÓN
elif page == "⚙️ Configuración":
    st.header("⚙️ Configuración y Estado del Sistema")
    st.subheader("✔️ Estado de los Servicios")
    
    col1, col2, col3 = st.columns(3)
    col1.success("✅ Firebase: Conectado") if firebase else col1.error("❌ Firebase: Desconectado")
    col2.success("✅ Gemini AI: Conectado") if gemini else col2.error("❌ Gemini AI: Desconectado")
    col3.success("✅ YOLOv8: Cargado") if yolo_model else col3.error("❌ YOLOv8: No cargado")

    st.subheader("💻 Información del Sistema")
    st.info(f"**Python:** {sys.version.split()[0]} | **Streamlit:** {st.__version__} | **OS:** {platform.system()}")

