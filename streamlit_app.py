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
    page_title="Inventario Inteligente Avanzado",
    page_icon="🧠",
    layout="wide"
)

# --- CARGA DE MODELOS Y CONEXIONES ---
@st.cache_resource
def load_yolo_model():
    """Carga un modelo YOLO más potente para una mejor detección."""
    # Usamos el modelo 'medium' (m) para mayor precisión.
    model = YOLO('yolov8m.pt')
    return model

try:
    yolo_model = load_yolo_model()
    firebase_utils.initialize_firebase()
except Exception as e:
    st.error(f"**Error Crítico de Inicialización.** No se pudo cargar un modelo o conectar a la base de datos.")
    st.code(f"Detalle del error: {e}", language="bash")
    st.stop()

# --- FUNCIONES DE LÓGICA ---
def find_best_match(attributes, inventory_names):
    """
    Compara los atributos extraídos por Gemini con la lista de inventario.
    Devuelve el artículo del inventario con la mayor cantidad de coincidencias de palabras clave.
    """
    if not inventory_names:
        return "Inventario vacío."

    best_match = "Artículo no encontrado"
    max_score = 0

    # Construir una cadena de texto unificada con todos los atributos detectados
    attribute_text = (
        f"{attributes.get('main_object', '')} "
        f"{attributes.get('main_color', '')} "
        f"{' '.join(attributes.get('secondary_colors', []))} "
        f"{attributes.get('shape', '')} "
        f"{attributes.get('material', '')} "
        f"{' '.join(attributes.get('features', []))}"
    ).lower()

    # Iterar sobre cada artículo en nuestro inventario de Firebase
    for name in inventory_names:
        score = 0
        # Calcular la puntuación de coincidencia
        for word in name.lower().split():
            if word in attribute_text:
                score += 1
        
        # Actualizar si encontramos una mejor coincidencia
        if score > max_score:
            max_score = score
            best_match = name

    return best_match

# --- TÍTULO Y DESCRIPCIÓN ---
st.title("🧠 Inventario Inteligente Avanzado")
st.markdown("Sistema híbrido con **YOLO** para detección/conteo y **Gemini** para análisis detallado de atributos.")

# --- ESTRUCTURA DE LA INTERFAZ ---
col1, col2 = st.columns([2, 1])

# --- PANEL DE CONTROL (COLUMNA 2) ---
with col2:
    st.header("📊 Panel de Control")
    with st.spinner("Cargando inventario..."):
        inventory_list = firebase_utils.get_inventory()
    
    inventory_names = [item.get('name') for item in inventory_list]

    with st.expander("➕ Añadir Nuevo Artículo", expanded=True):
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
        st.success(f"**Mejor Coincidencia:** {result['best_match']}")
        with st.expander("Ver atributos detallados extraídos por Gemini"):
            st.json(result['attributes'])
    else:
        st.info("Selecciona un objeto detectado para analizarlo.")

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
        st.session_state.detections = detections
        
        # Conteo de objetos detectados
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
                    
                    st.image(cropped_pil_image, caption=f"Enviando este recorte de '{class_name}' a Gemini...")

                    with st.spinner(f"🤖 Gemini está extrayendo atributos..."):
                        attributes_str = gemini_utils.get_image_attributes(cropped_pil_image)
                        try:
                            clean_json_str = attributes_str.strip().replace("```json", "").replace("```", "")
                            attributes = json.loads(clean_json_str)
                            best_match = find_best_match(attributes, inventory_names)
                            
                            st.session_state.analysis_result = {
                                "attributes": attributes,
                                "best_match": best_match
                            }
                        except (json.JSONDecodeError, KeyError) as e:
                            st.session_state.analysis_result = {
                                "attributes": {"error": "No se pudo interpretar la respuesta de Gemini.", "raw_response": attributes_str},
                                "best_match": "Error de análisis"
                            }
                        st.rerun()

