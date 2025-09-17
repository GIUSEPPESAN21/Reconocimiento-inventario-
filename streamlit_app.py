import streamlit as st
from PIL import Image
import firebase_utils
import gemini_utils
import cv2
import numpy as np
from ultralytics import YOLO
import json
from collections import Counter

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Inventario Inteligente Experto",
    page_icon="🧠",
    layout="wide"
)

# --- CARGA DE MODELOS Y CONEXIONES ---
@st.cache_resource
def load_yolo_model():
    """Carga un modelo YOLO potente para una mejor detección."""
    model = YOLO('yolov8m.pt') # Modelo mediano para un buen balance de velocidad/precisión
    return model

try:
    yolo_model = load_yolo_model()
    firebase_utils.initialize_firebase()
except Exception as e:
    st.error(f"**Error Crítico de Inicialización.** No se pudo cargar un modelo o conectar a la base de datos.")
    st.code(f"Detalle del error: {e}", language="bash")
    st.stop()

# --- TÍTULO Y DESCRIPCIÓN ---
st.title("🧠 Inventario Inteligente Experto")
st.markdown("Sistema de doble análisis: **YOLO** detecta, **Gemini-Visión** extrae atributos y **Gemini-Lógico** razona la mejor coincidencia.")

# --- ESTRUCTURA DE LA INTERFAZ ---
col1, col2 = st.columns([2, 1])

# --- PANEL DE CONTROL (COLUMNA 2) ---
with col2:
    st.header("📊 Panel de Control")
    with st.spinner("Cargando inventario..."):
        inventory_list = firebase_utils.get_inventory()
    
    inventory_names = [item.get('name') for item in inventory_list]

    with st.expander("➕ Añadir Artículo Manualmente", expanded=True):
        new_item_name = st.text_input("Nombre del artículo", key="add_item_input")
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
    
    st.subheader("✔️ Resultado del Análisis")
    if 'analysis_result' in st.session_state:
        result = st.session_state.analysis_result
        
        if result.get('best_match') == "Artículo no encontrado":
            st.warning("Este objeto no parece coincidir con nada en tu inventario.")
            with st.expander("Registrar como Nuevo Artículo", expanded=True):
                attributes = result.get('attributes', {})
                st.write("Atributos detectados por la IA:")
                st.json(attributes)
                
                suggested_name = f"{attributes.get('main_color', '')} {attributes.get('main_object', '')} {attributes.get('features', [''])[0]}".strip().capitalize()
                
                new_item_suggestion = st.text_input("Nombre para el nuevo artículo:", value=suggested_name)
                if st.button("✅ Registrar este artículo", use_container_width=True):
                    if new_item_suggestion and new_item_suggestion.strip():
                        firebase_utils.add_item(new_item_suggestion.strip())
                        st.success(f"'{new_item_suggestion}' ha sido registrado.")
                        del st.session_state['analysis_result']
                        st.rerun()
        else:
            st.success(f"**Mejor Coincidencia:** {result.get('best_match')}")
            st.info(f"**Razón de la IA:** {result.get('reasoning')}")
            with st.expander("Ver atributos detallados"):
                st.json(result.get('attributes', {}))
    else:
        st.info("Selecciona un objeto para analizarlo.")

# --- CAPTURA Y ANÁLISIS (COLUMNA 1) ---
with col1:
    st.header("📷 Captura y Detección")
    img_buffer = st.camera_input("Apunta la cámara a los objetos", key="camera")

    if img_buffer:
        bytes_data = img_buffer.getvalue()
        cv_image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        with st.spinner("🧠 Detectando y contando objetos..."):
            results = yolo_model(cv_image)

        st.subheader("🔍 Objetos Detectados")
        annotated_image = results[0].plot()
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        st.image(annotated_image_rgb, caption="Imagen con objetos detectados por YOLO.", use_container_width=True)
        
        detections = results[0]
        
        if detections.boxes:
            detected_classes = [detections.names[c] for c in detections.boxes.cls.tolist()]
            counts = Counter(detected_classes)
            st.write("**Conteo en la escena:**")
            st.table(counts)

        st.subheader("▶️ Analizar un objeto en detalle")
        
        if not detections.boxes:
            st.info("No se detectó ningún objeto para analizar.")
        else:
            for i, box in enumerate(detections.boxes):
                class_name = detections.names[box.cls[0].item()]
                
                if st.button(f"Analizar '{class_name}' #{i+1}", key=f"classify_{i}", use_container_width=True):
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = coords
                    
                    cropped_image_cv = cv_image[y1:y2, x1:x2]
                    cropped_image_rgb = cv2.cvtColor(cropped_image_cv, cv2.COLOR_BGR2RGB)
                    cropped_pil_image = Image.fromarray(cropped_image_rgb)
                    
                    st.image(cropped_pil_image, caption=f"Recorte enviado para análisis...")

                    # --- FLUJO DE DOBLE ANÁLISIS ---
                    with st.spinner(f"🤖 Paso 1: Extrayendo atributos de '{class_name}'..."):
                        attributes_str = gemini_utils.get_image_attributes(cropped_pil_image)
                    
                    try:
                        clean_json_str = attributes_str.strip().replace("```json", "").replace("```", "")
                        attributes = json.loads(clean_json_str)
                        
                        with st.spinner("🤖 Paso 2: Razonando la mejor coincidencia..."):
                            match_str = gemini_utils.get_best_match_from_attributes(attributes, inventory_names)
                        
                        clean_match_str = match_str.strip().replace("```json", "").replace("```", "")
                        match_data = json.loads(clean_match_str)
                        
                        st.session_state.analysis_result = {
                            "attributes": attributes,
                            "best_match": match_data.get("best_match", "Error"),
                            "reasoning": match_data.get("reasoning", "No se proporcionó una razón.")
                        }

                    except (json.JSONDecodeError, KeyError) as e:
                        st.session_state.analysis_result = {
                            "attributes": {"error": "No se pudo interpretar la respuesta de la IA.", "raw_response": attributes_str},
                            "best_match": "Artículo no encontrado",
                            "reasoning": f"Error de formato: {e}"
                        }
                    st.rerun()

