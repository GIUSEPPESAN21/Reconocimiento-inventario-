import streamlit as st
import logging
from gemini_utils import _call_gemini_with_fallback # Usamos la función corregida directamente
from firebase_utils import FirebaseUtils
from PIL import Image
import io
import pandas as pd
import json
from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter

# --- CONFIGURACIÓN DE LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# --- CARGA DE MODELOS Y CONEXIONES ---
@st.cache_resource
def load_yolo_model():
    """Carga el modelo YOLO pre-entrenado una sola vez."""
    try:
        model = YOLO('yolov8m.pt')
        return model
    except Exception as e:
        logger.error(f"Error cargando el modelo YOLO: {e}")
        st.error(f"No se pudo cargar el modelo de detección de objetos. La aplicación no puede continuar. Error: {e}")
        st.stop()

@st.cache_resource
def initialize_firebase_connection():
    """Inicializa la conexión con Firebase una sola vez."""
    try:
        firebase = FirebaseUtils()
        logger.info("Conexión con Firebase establecida correctamente.")
        return firebase
    except Exception as e:
        logger.error(f"Error crítico al conectar con Firebase: {e}")
        st.error(f"No se pudo conectar a la base de datos. Verifique los secretos de Streamlit. Error: {e}")
        st.stop()

# --- FUNCIÓN PRINCIPAL DE LA APP ---
def main():
    st.set_page_config(
        page_title="Inventario Inteligente Experto",
        page_icon="📦",
        layout="wide"
    )
    
    st.title("🤖 Sistema de Reconocimiento de Inventario con IA")
    st.markdown("---")

    # --- VERIFICACIÓN DE SECRETS Y CONEXIONES ---
    try:
        if not st.secrets.get("GEMINI_API_KEY"):
            st.error("❌ GEMINI_API_KEY no configurada en Streamlit secrets.")
            st.stop()
        if not st.secrets.get("FIREBASE_SERVICE_ACCOUNT_BASE64"):
            st.error("❌ FIREBASE_SERVICE_ACCOUNT_BASE64 no configurada en Streamlit secrets.")
            st.stop()
    except Exception:
        st.error("❌ No se pudieron cargar los secretos. Asegúrate de que tu archivo `secrets.toml` está configurado.")
        st.stop()

    firebase = initialize_firebase_connection()
    yolo_model = load_yolo_model()
    
    # --- NAVEGACIÓN POR PESTAÑAS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📷 Reconocimiento", 
        "📊 Inventario", 
        "📈 Estadísticas", 
        "⚙️ Configuración", 
        "👥 Acerca de"
    ])
    
    # --- PESTAÑA DE RECONOCIMIENTO ---
    with tab1:
        st.header("Reconocimiento de Elementos con YOLO y Gemini")
        
        img_buffer = st.camera_input("Apunta la cámara a los objetos", key="camera_input")

        if img_buffer:
            bytes_data = img_buffer.getvalue()
            image = Image.open(io.BytesIO(bytes_data))
            cv_image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            with st.spinner("Detectando objetos con YOLO..."):
                results = yolo_model(cv_image)
            
            st.subheader("🔍 Objetos Detectados")
            annotated_image = results[0].plot()
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            st.image(annotated_image_rgb, caption="Imagen con objetos detectados.", use_container_width=True)

            detections = results[0]
            if detections.boxes:
                detected_classes = [detections.names[c] for c in detections.boxes.cls.tolist()]
                counts = Counter(detected_classes)
                st.write("**Conteo en escena:**"); st.table(counts)

                st.subheader("▶️ Analizar un objeto en detalle con Gemini")
                for i, box in enumerate(detections.boxes):
                    class_name = detections.names[box.cls[0].item()]
                    if st.button(f"Analizar '{class_name}' #{i+1}", key=f"classify_{i}"):
                        coords = box.xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = coords
                        cropped_pil_image = Image.fromarray(cv2.cvtColor(cv_image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
                        
                        st.image(cropped_pil_image, caption=f"Recorte enviado para análisis...")

                        with st.spinner("Analizando con Gemini AI..."):
                            prompt = """
                            Analiza la imagen de este objeto y proporciona un análisis detallado.
                            Responde únicamente con un objeto JSON válido con las siguientes claves:
                            - "nombre_producto": (string)
                            - "categoria": (string)
                            - "estado": (string) "nuevo", "usado", etc.
                            - "material_principal": (string)
                            - "caracteristicas": (array of strings)
                            """
                            img_byte_arr = io.BytesIO()
                            cropped_pil_image.save(img_byte_arr, format='JPEG')
                            img_byte_arr = img_byte_arr.getvalue()

                            result_text = _call_gemini_with_fallback(prompt, image=cropped_pil_image)
                            
                            try:
                                clean_json_str = result_text.strip().lstrip("```json").rstrip("```")
                                analysis_data = json.loads(clean_json_str)
                                st.session_state.last_analysis = analysis_data
                                st.success("Análisis completado.")
                            except (json.JSONDecodeError, KeyError) as e:
                                st.error(f"Error al procesar la respuesta de la IA: {e}")
                                st.text_area("Respuesta recibida:", result_text, height=150)

            else:
                st.info("No se detectaron objetos en la imagen.")

        if 'last_analysis' in st.session_state:
            st.subheader("📋 Resultados del Último Análisis")
            st.json(st.session_state.last_analysis)
            if st.button("💾 Guardar en Inventario"):
                try:
                    inventory_data = {
                        "description": st.session_state.last_analysis.get("nombre_producto", "Elemento sin nombre"),
                        "analysis": json.dumps(st.session_state.last_analysis, indent=2),
                        "timestamp": firebase.get_timestamp(),
                    }
                    firebase.add_inventory_item(inventory_data)
                    st.success(f"✅ Elemento '{inventory_data['description']}' guardado con éxito.")
                    del st.session_state.last_analysis
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error guardando en Firebase: {e}")

    # --- PESTAÑA DE INVENTARIO ---
    with tab2:
        st.header("Gestión de Inventario")
        try:
            inventory_items_docs = firebase.get_inventory_items()
            inventory_items = [doc.to_dict() for doc in inventory_items_docs]
            for item, doc in zip(inventory_items, inventory_items_docs):
                item['id'] = doc.id
            
            if inventory_items:
                st.write(f"📦 Total de elementos: {len(inventory_items)}")
                search_term = st.text_input("🔍 Buscar elemento por descripción")
                
                filtered_items = [
                    item for item in inventory_items 
                    if search_term.lower() in item.get("description", "").lower()
                ] if search_term else inventory_items
                
                for i, item in enumerate(filtered_items):
                    with st.expander(f"📦 {item.get('description', 'Sin descripción')}"):
                        st.write(f"**ID:** {item.get('id', 'N/A')}")
                        st.write(f"**Fecha:** {item.get('timestamp', 'N/A')}")
                        analysis = item.get('analysis', '{}')
                        try:
                            st.json(json.loads(analysis))
                        except json.JSONDecodeError:
                            st.text(analysis)
                        
                        if st.button("🗑️ Eliminar", key=f"delete_{i}"):
                            firebase.delete_inventory_item(item['id'])
                            st.success("✅ Elemento eliminado.")
                            st.rerun()
            else:
                st.info("📭 No hay elementos en el inventario.")
        except Exception as e:
            st.error(f"❌ Error cargando inventario: {e}")

    # --- PESTAÑA DE ESTADÍSTICAS ---
    with tab3:
        st.header("Estadísticas del Sistema")
        try:
            stats = firebase.get_inventory_stats()
            if stats['total_items'] > 0:
                col1, col2 = st.columns(2)
                col1.metric("Total de Elementos", stats['total_items'])
                col2.metric("Última Actualización", pd.to_datetime(stats['last_update']).strftime('%Y-%m-%d %H:%M') if stats['last_update'] else "N/A")

                if stats['categories']:
                    st.subheader("📊 Distribución por Categorías")
                    st.bar_chart(stats['categories'])
            else:
                st.info("📊 No hay datos suficientes para mostrar estadísticas.")
        except Exception as e:
            st.error(f"❌ Error cargando estadísticas: {e}")

    # --- PESTAÑA DE CONFIGURACIÓN ---
    with tab4:
        st.header("Configuración y Estado del Sistema")
        st.subheader("🔧 Streamlit Secrets")
        st.success("✅ GEMINI_API_KEY Configurada")
        st.success("✅ FIREBASE_SERVICE_ACCOUNT_BASE64 Configurada")
        
        st.subheader("🔥 Información de Firebase")
        st.write(f"**Proyecto ID:** {firebase.project_id}")
        st.write("**Estado de conexión:** ✅ Conectado")

        if st.button("🧪 Probar Conexión con Gemini"):
            with st.spinner("Probando..."):
                response_text = _call_gemini_with_fallback("Hola, esto es una prueba de conexión.")
                try:
                    response_json = json.loads(response_text)
                    if "error" in response_json:
                        st.error(f"❌ Prueba fallida: {response_json['error']}")
                    else:
                        st.success("✅ Conexión con Gemini AI exitosa.")
                except json.JSONDecodeError:
                    st.success("✅ Conexión con Gemini AI exitosa.")
                    st.text(response_text)


    # --- PESTAÑA "ACERCA DE" ---
    with tab5:
        st.header("Sobre el Proyecto y sus Creadores")
        st.markdown("""
        Esta aplicación representa una solución avanzada para la **gestión inteligente de inventarios**. 
        Utiliza un modelo híbrido de inteligencia artificial que combina la **detección de objetos en tiempo real (YOLO)** con el **análisis profundo de atributos de imagen (Google Gemini)**.
        """)
        st.info("Creado por Joseph Javier Sánchez Acuña y supervisado por Xammy Alexander Victoria Gonzalez.")

if __name__ == "__main__":
    main()
