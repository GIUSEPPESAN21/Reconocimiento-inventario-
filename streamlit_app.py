import streamlit as st
from PIL import Image
import firebase_utils
import gemini_utils
import cv2  # Importamos OpenCV para dibujar sobre la imagen
import numpy as np
from ultralytics import YOLO

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Inventario Inteligente Híbrido",
    page_icon="🤖",
    layout="wide"
)

# --- CARGA DE MODELOS Y CONEXIONES (Una sola vez) ---

@st.cache_resource
def load_yolo_model():
    """Carga el modelo YOLO pre-entrenado una sola vez."""
    # 'yolov8n.pt' es un modelo pequeño y rápido, ideal para empezar.
    model = YOLO('yolov8n.pt')
    return model

try:
    # Cargamos el modelo YOLO al iniciar la app
    yolo_model = load_yolo_model()
    # Inicializamos Firebase
    firebase_utils.initialize_firebase()
except Exception as e:
    st.error(f"**Error Crítico de Inicialización.** No se pudo cargar un modelo o conectar a la base de datos.")
    st.code(f"Detalle del error: {e}", language="bash")
    st.stop()

# --- TÍTULO Y DESCRIPCIÓN ---
st.title("📦 Inventario Inteligente Híbrido (IA Local + Cloud)")
st.markdown("Usa **YOLO** para detectar objetos localmente y **Gemini** para una clasificación detallada.")

# --- ESTRUCTURA ---
col1, col2 = st.columns([2, 1])

# --- PANEL DE CONTROL (COLUMNA 2) ---
with col2:
    st.header("📊 Panel de Control")
    with st.spinner("Cargando inventario..."):
        inventory_list = firebase_utils.get_inventory()
    
    inventory_names = [item.get('name') for item in inventory_list]

    with st.expander("➕ Añadir Nuevo Artículo", expanded=True):
        new_item_name = st.text_input("Nombre del artículo", key="new_item")
        if st.button("Guardar en Firebase", use_container_width=True):
            if new_item_name and new_item_name.strip() and new_item_name not in inventory_names:
                firebase_utils.add_item(new_item_name.strip())
                st.success(f"'{new_item_name}' añadido.")
                st.rerun()
            else:
                st.warning("El nombre no puede estar vacío o ya existe.")

    st.subheader("📋 Inventario Actual")
    if inventory_names:
        st.dataframe(inventory_names, use_container_width=True, column_config={"value": "Artículo"})
    else:
        st.info("Inventario vacío.")

# --- CAPTURA Y ANÁLISIS (COLUMNA 1) ---
with col1:
    st.header("📷 Captura y Detección Local")
    img_buffer = st.camera_input("Apunta la cámara a los objetos", key="camera")

    if img_buffer:
        # Convertir el buffer de la imagen a un formato que OpenCV pueda usar
        bytes_data = img_buffer.getvalue()
        cv_image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Realizar la detección de objetos con YOLO
        with st.spinner("🧠 Detectando objetos con IA local (YOLO)..."):
            results = yolo_model(cv_image)

        st.subheader("🔍 Objetos Detectados")
        
        # Dibujar las cajas delimitadoras y etiquetas sobre la imagen
        annotated_image = results[0].plot() # El método .plot() de ultralytics hace esto automáticamente!

        # Convertir la imagen de vuelta a un formato que Streamlit pueda mostrar (BGR a RGB)
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        st.image(annotated_image_rgb, caption="Imagen con objetos detectados por YOLO.", use_column_width=True)
        
        # (Próximo paso será hacer estas detecciones interactivas)

