import streamlit as st
from PIL import Image
import firebase_utils
import gemini_utils
import cv2
import numpy as np
from ultralytics import YOLO

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Inventario Inteligente Híbrido",
    page_icon="🤖",
    layout="wide"
)

# --- CARGA DE MODELOS Y CONEXIONES ---

@st.cache_resource
def load_yolo_model():
    """Carga el modelo YOLO pre-entrenado una sola vez."""
    model = YOLO('yolov8n.pt')
    return model

try:
    yolo_model = load_yolo_model()
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
        # --- CORRECCIÓN ---
        # Se cambió la clave de "new_item" a "new_item_input" para garantizar que sea única.
        new_item_name = st.text_input("Nombre del artículo", key="new_item_input")
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
    
    st.subheader("✔️ Resultado Final de Gemini")
    if 'gemini_result' in st.session_state:
        st.success(f"**Artículo clasificado:** {st.session_state.gemini_result}")
    else:
        st.info("Selecciona un objeto detectado para clasificarlo.")


# --- CAPTURA Y ANÁLISIS (COLUMNA 1) ---
with col1:
    st.header("📷 Captura y Detección Local")
    img_buffer = st.camera_input("Apunta la cámara a los objetos", key="camera")

    if img_buffer:
        bytes_data = img_buffer.getvalue()
        cv_image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        with st.spinner("🧠 Detectando objetos con IA local (YOLO)..."):
            results = yolo_model(cv_image)

        st.subheader("🔍 Objetos Detectados")
        annotated_image = results[0].plot()
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        st.image(annotated_image_rgb, caption="Imagen con objetos detectados por YOLO.", use_column_width=True)
        
        st.subheader("▶️ Clasificar un objeto con Gemini")
        
        st.session_state.detections = results[0]

        if not st.session_state.detections.boxes:
            st.info("No se detectó ningún objeto conocido en la imagen.")
        else:
            for i, box in enumerate(st.session_state.detections.boxes):
                class_name = st.session_state.detections.names[box.cls[0].item()]
                
                if st.button(f"Clasificar '{class_name}' #{i+1}", key=f"classify_{i}", use_container_width=True):
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = coords
                    
                    cropped_image_cv = cv_image[y1:y2, x1:x2]
                    
                    cropped_image_rgb = cv2.cvtColor(cropped_image_cv, cv2.COLOR_BGR2RGB)
                    cropped_pil_image = Image.fromarray(cropped_image_rgb)
                    
                    with st.spinner(f"🤖 Gemini está analizando '{class_name}'..."):
                        gemini_result = gemini_utils.identify_item(cropped_pil_image, inventory_names)
                        st.session_state.gemini_result = gemini_result
                        st.rerun()

