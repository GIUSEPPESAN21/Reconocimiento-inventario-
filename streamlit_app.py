import streamlit as st
from PIL import Image
import firebase_utils
import gemini_utils
import cv2
import numpy as np
from ultralytics import YOLO

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Inventario Inteligente Pro",
    page_icon="✨",
    layout="wide"
)

# --- CARGA DE MODELOS Y CONEXIONES ---
@st.cache_resource
def load_yolo_model():
    """
    Carga un modelo YOLO más potente. 'yolov8m.pt' (medium) es un gran
    equilibrio entre velocidad y una precisión significativamente mayor.
    """
    model = YOLO('yolov8m.pt')
    return model

try:
    yolo_model = load_yolo_model()
    firebase_utils.initialize_firebase()
except Exception as e:
    st.error(f"**Error Crítico de Inicialización.** No se pudo cargar un modelo o conectar a la base de datos.")
    st.code(f"Detalle del error: {e}", language="bash")
    st.stop()

# --- TÍTULO Y DESCRIPCIÓN ---
st.title("📦 Inventario Inteligente Pro")
st.markdown("Versión mejorada con un modelo de detección local **más preciso (YOLOv8m)** y un razonamiento de IA **más avanzado (Gemini 1.5)**.")

# --- ESTRUCTURA DE LA INTERFAZ ---
col1, col2 = st.columns([2, 1])

# --- PANEL DE CONTROL (COLUMNA 2) ---
with col2:
    st.header("📊 Panel de Control")
    with st.spinner("Cargando inventario desde Firebase..."):
        inventory_list = firebase_utils.get_inventory()
    
    inventory_names = [item.get('name') for item in inventory_list]

    with st.expander("➕ Añadir Nuevo Artículo", expanded=True):
        new_item_name = st.text_input("Nombre del artículo", key="new_item", placeholder="Ej: Taza de cerámica blanca")
        if st.button("Guardar en Firebase", use_container_width=True):
            if new_item_name and new_item_name.strip() and new_item_name not in inventory_names:
                firebase_utils.add_item(new_item_name.strip())
                st.success(f"'{new_item_name}' añadido.")
                st.rerun()
            else:
                st.warning("El nombre no puede estar vacío o ya existe.")

    st.subheader("📋 Inventario Actual en Firebase")
    if inventory_names:
        st.dataframe(inventory_names, use_container_width=True, column_config={"value": "Artículo"})
    else:
        st.info("Inventario vacío.")
    
    st.subheader("✔️ Resultado Final de la Clasificación")
    # Este placeholder mostrará el resultado final de Gemini
    gemini_result_placeholder = st.empty()
    if 'gemini_result' in st.session_state:
        gemini_result_placeholder.success(f"**Artículo clasificado:** {st.session_state.gemini_result}")
    else:
        gemini_result_placeholder.info("Selecciona un objeto detectado para clasificarlo.")

# --- CAPTURA Y ANÁLISIS (COLUMNA 1) ---
with col1:
    st.header("📷 Fase 1: Detección Local con YOLO")
    img_buffer = st.camera_input("Apunta la cámara a los objetos y captura una imagen", key="camera")

    if img_buffer:
        bytes_data = img_buffer.getvalue()
        cv_image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        with st.spinner("🧠 Analizando escena con IA local (YOLO)..."):
            results = yolo_model(cv_image)

        # Guardar detecciones e imagen original en el estado de la sesión
        st.session_state.detections = results[0]
        st.session_state.original_image = cv_image

        annotated_image = st.session_state.detections.plot()
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        st.image(annotated_image_rgb, caption="Imagen con objetos detectados. Haz clic en un botón para clasificar.", use_column_width=True)
        
        st.header("▶️ Fase 2: Clasificación Específica con Gemini")
        
        if not st.session_state.detections.boxes:
            st.info("YOLO no detectó objetos conocidos en esta imagen.")
        else:
            # Crear columnas para los botones, para un look más limpio
            num_detections = len(st.session_state.detections.boxes)
            button_cols = st.columns(min(num_detections, 4)) # Máximo 4 botones por fila

            for i, box in enumerate(st.session_state.detections.boxes):
                class_name = st.session_state.detections.names[box.cls[0].item()]
                col = button_cols[i % 4]
                
                if col.button(f"Clasificar '{class_name}' #{i+1}", key=f"classify_{i}"):
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = coords
                    
                    # Recortar el objeto de la imagen original
                    cropped_image_cv = st.session_state.original_image[y1:y2, x1:x2]
                    cropped_image_rgb = cv2.cvtColor(cropped_image_cv, cv2.COLOR_BGR2RGB)
                    cropped_pil_image = Image.fromarray(cropped_image_rgb)
                    
                    # Mostrar al usuario exactamente qué se está enviando a Gemini
                    st.image(cropped_pil_image, caption=f"Enviando este recorte de '{class_name}' a Gemini...")
                    
                    with st.spinner(f"🤖 Gemini está razonando sobre '{class_name}'..."):
                        gemini_result = gemini_utils.identify_item(cropped_pil_image, inventory_names)
                        st.session_state.gemini_result = gemini_result
                        st.rerun()

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
        new_item_name = st.text_input("Nombre del artículo", key="new_item")
        # CORRECCIÓN: Se mantiene use_container_width que ya estaba bien.
        if st.button("Guardar en Firebase", use_container_width=True):
            if new_item_name and new_item_name.strip() and new_item_name not in inventory_names:
                firebase_utils.add_item(new_item_name.strip())
                st.success(f"'{new_item_name}' añadido.")
                st.rerun()
            else:
                st.warning("El nombre no puede estar vacío o ya existe.")

    st.subheader("📋 Inventario Actual")
    if inventory_names:
        # CORRECIÓN: Cambiado a use_container_width=True
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
        
        # CORRECIÓN: Cambiado a use_container_width=True
        st.image(annotated_image_rgb, caption="Imagen con objetos detectados por YOLO.", use_column_width=True)
        
        st.subheader("▶️ Clasificar un objeto con Gemini")
        
        st.session_state.detections = results[0]

        if not st.session_state.detections.boxes:
            st.info("No se detectó ningún objeto conocido en la imagen.")
        else:
            for i, box in enumerate(st.session_state.detections.boxes):
                class_name = st.session_state.detections.names[box.cls[0].item()]
                
                # CORRECCIÓN: Se mantiene use_container_width que ya estaba bien.
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

