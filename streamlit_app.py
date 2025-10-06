import streamlit as st
from PIL import Image
import numpy as np
import cv2
import json
from collections import Counter

# Importa las nuevas clases que creaste
from firebase_utils import FirebaseUtils
from gemini_utils import GeminiUtils
from ultralytics import YOLO

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Inventario Inteligente Híbrido",
    page_icon="🤖",
    layout="wide"
)

# --- CARGA DE MODELOS Y CONEXIONES (Método robusto) ---
@st.cache_resource
def initialize_services():
    """Carga YOLO e inicializa Firebase y Gemini una sola vez."""
    try:
        yolo_model = YOLO('yolov8m.pt')
        firebase_handler = FirebaseUtils()
        gemini_handler = GeminiUtils()
        return yolo_model, firebase_handler, gemini_handler
    except Exception as e:
        # Mostramos el error de forma clara si algo falla al inicio
        st.error(f"**Error Crítico de Inicialización.** No se pudo cargar un modelo o conectar a un servicio.")
        st.code(f"Detalle del error: {e}", language="bash")
        return None, None, None

# Inicializa todos los servicios
yolo_model, firebase, gemini = initialize_services()

# Si la inicialización falla, detenemos la app
if not all([yolo_model, firebase, gemini]):
    st.stop()

# --- TÍTULO PRINCIPAL Y PESTAÑAS ---
st.title("🧠 Inventario Inteligente Híbrido")
tab_inventario, tab_acerca_de = st.tabs(["📷 Reconocimiento de Inventario", "👥 Acerca de Nosotros"])

# --- PESTAÑA PRINCIPAL DE LA APLICACIÓN ---
with tab_inventario:
    col1, col2 = st.columns([2, 1])

    # --- PANEL DE CONTROL (COLUMNA 2) ---
    with col2:
        st.header("📊 Panel de Control")
        try:
            with st.spinner("Cargando inventario..."):
                # Usamos el objeto firebase inicializado para obtener los datos
                inventory_list = firebase.get_all_inventory_items()
            # Extraemos el nombre de cada item, que ahora está en el campo 'descripcion' o 'name'
            inventory_names = [item.get('descripcion') or item.get('name', 'Nombre no encontrado') for item in inventory_list]
        except Exception as e:
            st.error(f"Error al cargar inventario: {e}")
            inventory_names = []

        with st.expander("➕ Añadir Nuevo Artículo", expanded=True):
            new_item_name = st.text_input("Nombre del artículo", key="add_item_input")
            if st.button("Guardar en Firebase", use_container_width=True):
                if new_item_name and new_item_name.strip() and new_item_name.strip() not in inventory_names:
                    # Creamos el diccionario de datos que tu nueva función espera
                    data_to_save = {
                        "tipo": "manual",
                        "descripcion": new_item_name.strip(),
                        "analisis": "N/A",
                        "timestamp": firebase.get_timestamp()
                    }
                    firebase.save_inventory_item(data_to_save)
                    st.success(f"'{new_item_name.strip()}' añadido.")
                    st.rerun()
                else:
                    st.warning("El nombre no puede estar vacío o ya existe.")

        st.subheader("📋 Inventario Actual")
        if inventory_names:
            st.dataframe([name for name in inventory_names if name != 'Nombre no encontrado'], use_container_width=True, column_config={"value": "Artículo"})
        else:
            st.info("Inventario vacío o no se pudo cargar.")
        
        st.subheader("✔️ Resultado del Análisis")
        if 'analysis_result' in st.session_state:
            st.success("Análisis completado:")
            st.json(st.session_state.analysis_result)
        else:
            st.info("Selecciona un objeto para analizarlo.")

    # --- CAPTURA Y ANÁLISIS (COLUMNA 1) ---
    with col1:
        st.header("📷 Captura y Detección")
        img_buffer = st.camera_input("Apunta la cámara a los objetos", key="camera")

        if img_buffer:
            bytes_data = img_buffer.getvalue()
            cv_image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            
            with st.spinner("🧠 Detectando objetos con IA local (YOLO)..."):
                results = yolo_model(pil_image)

            st.subheader("🔍 Objetos Detectados")
            annotated_image = results[0].plot()
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            st.image(annotated_image_rgb, caption="Imagen con objetos detectados por YOLO.", use_container_width=True)

            detections = results[0]
            st.subheader("▶️ Analizar un objeto en detalle")

            if not detections.boxes:
                st.info("No se detectó ningún objeto para analizar.")
            else:
                for i, box in enumerate(detections.boxes):
                    class_name = detections.names[box.cls[0].item()]
                    if st.button(f"Analizar '{class_name}' #{i+1}", key=f"classify_{i}", use_container_width=True):
                        # Recortar la imagen original de PIL para el análisis de Gemini
                        coords = box.xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = coords
                        cropped_pil_image = pil_image.crop((x1, y1, x2, y2))
                        
                        st.image(cropped_pil_image, caption=f"Recorte de '{class_name}' enviado para análisis...")

                        with st.spinner(f"🤖 Gemini está analizando el recorte..."):
                            # Usamos el objeto gemini inicializado
                            analysis_text = gemini.analyze_image(cropped_pil_image, f"Objeto detectado como {class_name}")
                            try:
                                # Guardamos el JSON resultante en el estado de la sesión
                                st.session_state.analysis_result = json.loads(analysis_text)
                            except json.JSONDecodeError:
                                st.session_state.analysis_result = {"error": "La respuesta de la IA no fue un JSON válido.", "raw_response": analysis_text}
                            st.rerun()

# --- PESTAÑA "ACERCA DE" ---
with tab_acerca_de:
    st.header("👥 Sobre el Proyecto y sus Creadores")
    
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

    with st.container(border=True):
        col_img_prof, col_info_prof = st.columns([1, 3])
        with col_img_prof:
            st.image("https://placehold.co/250x250/2B3137/FFFFFF?text=J.M.", caption="Jhon Alejandro Mojica")
        with col_info_prof:
            st.title("Jhon Alejandro Mojica")
            st.subheader("_Profesor y Tutor del Proyecto_")
            st.markdown("- 📧 **Email:** [jhon.mojica@uniminuto.edu.co](mailto:jhon.mojica@uniminuto.edu.co)")

    st.markdown("---")

    with st.container(border=True):
        st.subheader("💡 Acerca de esta Herramienta")
        st.markdown("""
        Esta aplicación es una solución avanzada para la gestión inteligente de inventarios. 
        Utiliza un modelo híbrido de IA que combina la detección de objetos en tiempo real (**YOLO**) 
        con el análisis profundo de imágenes (**Google Gemini**), todo gestionado a través de una base de datos en la nube (**Firebase**).
        """)
