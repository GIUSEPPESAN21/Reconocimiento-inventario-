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
    """Carga el modelo YOLO pre-entrenado una sola vez."""
    model = YOLO('yolov8m.pt')
    return model

try:
    yolo_model = load_yolo_model()
    firebase_utils.initialize_firebase()
except Exception as e:
    st.error(f"**Error Crítico de Inicialización.** {e}")
    st.stop()

# --- TÍTULO PRINCIPAL DE LA APLICACIÓN ---
st.title("🧠 Inventario Inteligente Experto")

# --- NAVEGACIÓN POR PESTAÑAS ---
tab_inventario, tab_acerca_de = st.tabs(["📷 Reconocimiento de Inventario", "👥 Acerca de Nosotros"])

# --- CONTENIDO DE LA PESTAÑA DE INVENTARIO ---
with tab_inventario:
    st.markdown("Sistema híbrido con **YOLO** para detección y **Gemini** para análisis de atributos.")
    
    col1, col2 = st.columns([2, 1])

    # --- PANEL DE CONTROL (COLUMNA 2) ---
    with col2:
        st.header("📊 Panel de Control")
        with st.spinner("Cargando inventario..."):
            inventory_list = firebase_utils.get_inventory()
        inventory_names = [item.get('name') for item in inventory_list]

        with st.expander("➕ Añadir Artículo", expanded=True):
            new_item_name = st.text_input("Nombre del artículo", key="add_item_input")
            if st.button("Guardar", use_container_width=True):
                if new_item_name and new_item_name.strip() and new_item_name not in inventory_names:
                    firebase_utils.add_item(new_item_name.strip())
                    st.success(f"'{new_item_name}' añadido.")
                    st.rerun()
                else:
                    st.warning("El nombre no puede estar vacío o ya existe.")

        st.subheader("📋 Inventario Actual")
        st.dataframe(inventory_names, use_container_width=True, column_config={"value": "Artículo"})
        
        st.subheader("✔️ Resultado del Análisis")
        if 'analysis_result' in st.session_state:
            result = st.session_state.analysis_result
            if result.get('best_match') == "Artículo no encontrado":
                st.warning("Este objeto no coincide con tu inventario.")
                with st.expander("Registrar Nuevo Artículo", expanded=True):
                    attributes = result.get('attributes', {})
                    st.write("Atributos detectados por la IA:")
                    st.json(attributes)
                    suggested_name = f"{attributes.get('main_color', '')} {attributes.get('main_object', '')}".strip().capitalize()
                    new_item_suggestion = st.text_input("Nombre para el nuevo artículo:", value=suggested_name, key="suggestion_input")
                    if st.button("✅ Registrar este artículo", use_container_width=True, key="register_new"):
                        if new_item_suggestion and new_item_suggestion.strip():
                            firebase_utils.add_item(new_item_suggestion.strip())
                            st.success(f"'{new_item_suggestion}' registrado con éxito.")
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
            
            with st.spinner("🧠 Detectando objetos..."):
                results = yolo_model(cv_image)

            st.subheader("🔍 Objetos Detectados")
            annotated_image = results[0].plot()
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            st.image(annotated_image_rgb, caption="Imagen con objetos detectados por YOLO.", use_container_width=True)
            
            detections = results[0]
            if detections.boxes:
                detected_classes = [detections.names[c] for c in detections.boxes.cls.tolist()]
                counts = Counter(detected_classes)
                st.write("**Conteo en escena:**"); st.table(counts)

            st.subheader("▶️ Analizar un objeto en detalle")
            if not detections.boxes:
                st.info("No se detectó ningún objeto para analizar.")
            else:
                for i, box in enumerate(detections.boxes):
                    class_name = detections.names[box.cls[0].item()]
                    if st.button(f"Analizar '{class_name}' #{i+1}", key=f"classify_{i}", use_container_width=True):
                        coords = box.xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = coords
                        cropped_pil_image = Image.fromarray(cv2.cvtColor(cv_image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
                        st.image(cropped_pil_image, caption=f"Recorte enviado para análisis...")

                        with st.spinner(f"🤖 Paso 1: Extrayendo atributos..."):
                            attributes_str = gemini_utils.get_image_attributes(cropped_pil_image)
                        
                        try:
                            clean_json_str = attributes_str.strip().lstrip("```json").rstrip("```")
                            attributes = json.loads(clean_json_str)
                            
                            with st.spinner("🤖 Paso 2: Razonando la mejor coincidencia..."):
                                match_str = gemini_utils.get_best_match_from_attributes(attributes, inventory_names)
                            
                            clean_match_str = match_str.strip().lstrip("```json").rstrip("```")
                            match_data = json.loads(clean_match_str)
                            
                            st.session_state.analysis_result = {
                                "attributes": attributes, "best_match": match_data.get("best_match"), "reasoning": match_data.get("reasoning")
                            }
                        except (json.JSONDecodeError, KeyError) as e:
                            st.session_state.analysis_result = {"attributes": {"error": "Respuesta no válida de la IA."}, "best_match": "Artículo no encontrado", "reasoning": f"Error: {e}"}
                        st.rerun()

# --- CONTENIDO DE LA PESTAÑA "ACERCA DE" ---
with tab_acerca_de:
    st.header("Sobre el Proyecto y sus Creadores")

    # Información del Estudiante
    with st.container(border=True):
        col_img_est, col_info_est = st.columns([1, 3])
        with col_img_est:
            st.image("https://placehold.co/250x250/000000/FFFFFF?text=J.S.", caption="Joseph Javier Sánchez Acuña")
        
        with col_info_est:
            st.title("Joseph Javier Sánchez Acuña")
            st.subheader("_Estudiante de Ingeniería Industrial_")
            st.subheader("_Experto en Inteligencia Artificial y Desarrollo de Software._")
            st.markdown(
                """
                - 🔗 **LinkedIn:** [joseph-javier-sánchez-acuña](https://www.linkedin.com/in/joseph-javier-sánchez-acuña-150410275)
                - 📂 **GitHub:** [GIUSEPPESAN21](https://github.com/GIUSEPPESAN21)
                - 📧 **Email:** [joseph.sanchez@uniminuto.edu.co](mailto:joseph.sanchez@uniminuto.edu.co)
                """
            )
    
    st.markdown("---")

    # Información del Profesor
    with st.container(border=True):
        col_img_prof, col_info_prof = st.columns([1, 3])
        with col_img_prof:
            # Puedes cambiar el texto del placeholder
            st.image("https://placehold.co/250x250/2B3137/FFFFFF?text=J.M.", caption="Jhon Alejandro Mojica")
        
        with col_info_prof:
            st.title("Jhon Alejandro Mojica")
            st.subheader("_Profesor y Tutor del Proyecto_")
            st.markdown(
                """
                - 📧 **Email:** [jhon.mojica@uniminuto.edu.co](mailto:jhon.mojica@uniminuto.edu.co)
                """
            )

    st.markdown("---")

    # Descripción de la Herramienta
    with st.container(border=True):
        st.subheader("Acerca de esta Herramienta")
        st.markdown("""
        Esta aplicación representa una solución avanzada para la **gestión inteligente de inventarios**. 
        Utiliza un modelo híbrido de inteligencia artificial que combina la **detección de objetos en tiempo real (YOLO)** con el **análisis profundo de atributos de imagen (Google Gemini)**.

        El objetivo es proporcionar una herramienta que no solo identifique objetos, sino que también los cuente, 
        analice sus características y permita un registro dinámico en una base de datos en la nube (Firebase). 
        Cada componente está diseñado para maximizar la precisión, la velocidad y la facilidad de uso, 
        facilitando la toma de decisiones basada en datos y mejorando la eficiencia operativa.
        """)
