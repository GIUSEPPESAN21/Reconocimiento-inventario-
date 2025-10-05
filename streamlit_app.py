import streamlit as st
import logging
from gemini_utils import GeminiUtils
from firebase_utils import FirebaseUtils
from PIL import Image
import io
import pandas as pd
import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter

# --- CONFIGURACIÓN DE LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- CARGA DE MODELOS Y CONEXIONES (CACHEADO) ---
@st.cache_resource
def load_yolo_model():
    """Carga el modelo YOLO pre-entrenado una sola vez."""
    logger.info("Cargando modelo YOLOv8...")
    model = YOLO('yolov8m.pt')
    logger.info("Modelo YOLOv8 cargado.")
    return model

@st.cache_resource
def initialize_connections():
    """Inicializa Firebase y Gemini una sola vez."""
    try:
        firebase = FirebaseUtils()
        gemini = GeminiUtils()
        # Probar conexión inicial para asegurar que el modelo está disponible
        gemini.get_available_model()
        return firebase, gemini
    except Exception as e:
        st.error(f"❌ Error crítico en la inicialización: {str(e)}")
        st.stop()

def main():
    # --- CONFIGURACIÓN DE PÁGINA ---
    st.set_page_config(
        page_title="Inventario Inteligente Experto",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Inventario Inteligente Experto")
    st.markdown("---")
    
    # --- VERIFICACIÓN DE SECRETS ---
    try:
        if not st.secrets["GEMINI_API_KEY"] or not st.secrets["FIREBASE_SERVICE_ACCOUNT_BASE64"]:
            st.error("❌ GEMINI_API_KEY o FIREBASE_SERVICE_ACCOUNT_BASE64 no configuradas en Streamlit secrets.")
            st.stop()
    except KeyError as e:
        st.error(f"❌ Secret no encontrado: {str(e)}. Por favor, configúralo en los 'Secrets' de tu app.")
        st.stop()

    # --- INICIALIZACIÓN DE SERVICIOS ---
    firebase, gemini = initialize_connections()
    yolo_model = load_yolo_model()
    
    model_info = gemini.get_model_info()
    st.success(f"✅ Conexiones establecidas. Modelo IA activo: **{model_info.get('current_model', 'N/A')}**")

    # --- NAVEGACIÓN POR PESTAÑAS ---
    tab_live, tab_file, tab_inventory, tab_stats, tab_about = st.tabs([
        "📷 Detección en Vivo",
        "📂 Analizar Archivo",
        "📊 Inventario",
        "📈 Estadísticas",
        "👥 Acerca de"
    ])

    # --- PESTAÑA 1: DETECCIÓN EN VIVO (FUNCIONALIDAD RECUPERADA) ---
    with tab_live:
        st.header("📹 Detección de Objetos en Tiempo Real con YOLO")
        st.markdown("Usa tu cámara para detectar y analizar objetos en la escena al instante.")

        col1, col2 = st.columns([2, 1])

        with col1:
            img_buffer = st.camera_input("Apunta la cámara a los objetos", key="camera")

            if img_buffer:
                bytes_data = img_buffer.getvalue()
                cv_image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

                with st.spinner("🧠 Detectando objetos con YOLO..."):
                    results = yolo_model(cv_image)

                st.subheader("🔍 Objetos Detectados")
                annotated_image = results[0].plot()
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                st.image(annotated_image_rgb, caption="Imagen con objetos detectados.", use_container_width=True)

                detections = results[0]
                if detections.boxes:
                    detected_classes = [detections.names[c] for c in detections.boxes.cls.tolist()]
                    counts = Counter(detected_classes)
                    st.write("**Conteo en escena:**")
                    st.table(counts)
                else:
                    st.info("No se detectaron objetos en la imagen.")

        with col2:
            st.subheader("▶️ Analizar un objeto en detalle")
            if 'detections' in locals() and detections.boxes:
                for i, box in enumerate(detections.boxes):
                    class_name = detections.names[box.cls[0].item()]
                    if st.button(f"Analizar '{class_name}' #{i+1}", key=f"classify_{i}", use_container_width=True):
                        coords = box.xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = coords
                        cropped_pil_image = Image.fromarray(cv2.cvtColor(cv_image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
                        
                        st.image(cropped_pil_image, caption=f"Objeto a analizar...")

                        # Convertir a bytes para Gemini
                        img_byte_arr = io.BytesIO()
                        cropped_pil_image.save(img_byte_arr, format='JPEG')
                        
                        with st.spinner("🤖 Analizando con Gemini AI..."):
                            analysis_result = gemini.analyze_inventory_item(f"Objeto: {class_name}", img_byte_arr.getvalue())
                            st.session_state.last_analysis = analysis_result
                            st.rerun() # Refrescar para mostrar el resultado

            else:
                st.info("Apunta la cámara a una escena para habilitar el análisis.")
            
            if 'last_analysis' in st.session_state:
                st.subheader("✔️ Resultado del Análisis")
                result = st.session_state.last_analysis
                if result and result.get("success"):
                    st.success("Análisis completado con éxito.")
                    analysis_data = json.loads(result["response"])
                    st.json(analysis_data)
                    if st.button("💾 Guardar en Inventario", key="save_live"):
                         inventory_data = {
                            "description": analysis_data.get("Nombre del producto", "Objeto detectado en vivo"),
                            "analysis": result["response"],
                            "timestamp": firebase.get_timestamp(),
                            "model_used": result.get("model_used", "unknown")
                        }
                         firebase.add_inventory_item(inventory_data)
                         st.success("¡Guardado en el inventario!")
                elif result:
                    st.error(f"❌ Error en el análisis: {result.get('error')}")


    # --- PESTAÑA 2: ANALIZAR ARCHIVO (FUNCIONALIDAD NUEVA) ---
    with tab_file:
        st.header("📂 Analizar un Elemento desde Archivo")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader("Selecciona una imagen", type=['png', 'jpg', 'jpeg'])
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Imagen subida", use_column_width=True)
        
        with col2:
            description = st.text_area("Describe brevemente el elemento (opcional)", height=100)
            if st.button("🔍 Analizar Elemento", type="primary"):
                if uploaded_file:
                    with st.spinner("Analizando con IA..."):
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format='JPEG')
                        result = gemini.analyze_inventory_item(description or "Elemento del inventario", img_byte_arr.getvalue())
                        
                        if result["success"]:
                            st.success("✅ Análisis completado")
                            analysis_data = json.loads(result["response"])
                            st.json(analysis_data)
                            if st.button("💾 Guardar en Inventario", key="save_file"):
                                inventory_data = {
                                    "description": description or analysis_data.get("Nombre del producto", "Item de archivo"),
                                    "analysis": result["response"],
                                    "timestamp": firebase.get_timestamp(),
                                    "model_used": result.get("model_used", "unknown")
                                }
                                firebase.add_inventory_item(inventory_data)
                                st.success(f"✅ Elemento guardado.")
                        else:
                            st.error(f"❌ Error en el análisis: {result['error']}")
                else:
                    st.warning("⚠️ Por favor, sube una imagen primero")

    # --- PESTAÑA 3: GESTIÓN DE INVENTARIO ---
    with tab_inventory:
        st.header("📦 Gestión de Inventario")
        try:
            inventory_items = firebase.get_inventory_items()
            if inventory_items:
                st.write(f"Total de elementos: {len(inventory_items)}")
                search_term = st.text_input("🔍 Buscar elemento")
                filtered_items = [item for item in inventory_items if search_term.lower() in item.to_dict().get("description", "").lower()] if search_term else inventory_items
                
                for i, item_doc in enumerate(filtered_items):
                    item = item_doc.to_dict()
                    with st.expander(f"📦 {item.get('description', 'Sin descripción')}"):
                        st.write(f"**ID:** {item_doc.id}")
                        st.write(f"**Fecha:** {item.get('timestamp', 'N/A')}")
                        st.json(item.get('analysis', '{}'))
                        if st.button("🗑️ Eliminar", key=f"delete_{i}", type="secondary"):
                            firebase.delete_inventory_item(item_doc.id)
                            st.success("✅ Elemento eliminado")
                            st.rerun()
            else:
                st.info("📭 No hay elementos en el inventario")
        except Exception as e:
            st.error(f"❌ Error cargando inventario: {str(e)}")

    # --- PESTAÑA 4: ESTADÍSTICAS ---
    with tab_stats:
        st.header("📈 Estadísticas del Sistema")
        try:
            stats = firebase.get_inventory_stats()
            if stats['total_items'] > 0:
                col1, col2 = st.columns(2)
                col1.metric("Total Elementos", stats['total_items'])
                col2.metric("Último Análisis", pd.to_datetime(stats['last_update']).strftime('%Y-%m-%d %H:%M') if stats['last_update'] else "N/A")
                
                st.subheader("📊 Distribución de Modelos Usados")
                if stats['models_used']:
                    st.bar_chart(stats['models_used'])
                else:
                    st.info("No hay datos sobre los modelos usados.")
            else:
                st.info("📊 No hay datos suficientes para mostrar estadísticas")
        except Exception as e:
            st.error(f"❌ Error cargando estadísticas: {str(e)}")
            
    # --- PESTAÑA 5: ACERCA DE (CONTENIDO RECUPERADO) ---
    with tab_about:
        st.header("Sobre el Proyecto y sus Creadores")
        st.info(f"**Modelo IA actual:** {model_info.get('current_model', 'N/A')} | **Proyecto Firebase:** {firebase.project_id}")
        
        with st.container(border=True):
            st.title("Joseph Javier Sánchez Acuña")
            st.subheader("_Estudiante de Ingeniería Industrial_")
            st.subheader("_Experto en Inteligencia Artificial y Desarrollo de Software._")
            st.markdown("- 🔗 **LinkedIn:** [joseph-javier-sánchez-acuña](https://www.linkedin.com/in/joseph-javier-sánchez-acuña-150410275)\n- 📂 **GitHub:** [GIUSEPPESAN21](https://github.com/GIUSEPPESAN21)\n- 📧 **Email:** [joseph.sanchez@uniminuto.edu.co](mailto:joseph.sanchez@uniminuto.edu.co)")

        st.markdown("---")
        with st.container(border=True):
            st.title("Xammy Alexander Victoria Gonzalez")
            st.subheader("Profesor Tiempo Completo")
            st.markdown("- 📧 **Email:** [xammy.victoria@uniminuto.edu.co](mailto:xammy.victoria@uniminuto.edu.co)")
            
        st.markdown("---")
        with st.container(border=True):
            st.subheader("Acerca de esta Herramienta")
            st.markdown("""
            Esta aplicación representa una solución avanzada para la **gestión inteligente de inventarios**. 
            Utiliza un modelo híbrido de IA que combina la **detección de objetos en tiempo real (YOLO)** con el **análisis profundo de atributos de imagen (Google Gemini)**.
            El objetivo es proporcionar una herramienta que identifique, cuente, analice y registre dinámicamente los objetos en una base de datos en la nube (Firebase), optimizando la eficiencia operativa.
            """)

if __name__ == "__main__":
    main()
