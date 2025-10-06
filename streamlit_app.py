import streamlit as st
import os
from PIL import Image
import json
import base64
import logging
from ultralytics import YOLO
from firebase_utils import FirebaseUtils
from gemini_utils import GeminiUtils

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar directorio de YOLO para evitar advertencias
os.environ['YOLO_CONFIG_DIR'] = '/tmp/Ultralytics'

# Configurar página
st.set_page_config(
    page_title="Sistema de Reconocimiento de Inventario",
    page_icon="📦",
    layout="wide"
)

# Título principal
st.title("🤖 Sistema de Reconocimiento de Inventario con IA")

# Inicializar servicios
@st.cache_resource
def load_yolo_model():
    """Carga el modelo YOLO con cache"""
    try:
        logger.info("Cargando modelo YOLOv8...")
        model = YOLO('yolov8m.pt')
        logger.info("Modelo YOLOv8 cargado.")
        return model
    except Exception as e:
        logger.error(f"Error al cargar YOLO: {e}")
        return None

@st.cache_resource
def initialize_services():
    """Inicializa Firebase y Gemini con cache"""
    try:
        firebase = FirebaseUtils()
        gemini = GeminiUtils()
        return firebase, gemini
    except Exception as e:
        logger.error(f"Error al inicializar servicios: {e}")
        return None, None

# Cargar modelos y servicios
yolo_model = load_yolo_model()
firebase, gemini = initialize_services()

if yolo_model is None or firebase is None or gemini is None:
    st.error("Error al inicializar los servicios. Verifica la configuración.")
    st.stop()

# Sidebar para navegación
st.sidebar.title("📋 Navegación")
page = st.sidebar.selectbox(
    "Selecciona una opción:",
    ["🏠 Inicio", "📸 Análisis de Imagen", "📝 Análisis de Texto", "📊 Ver Inventario"]
)

if page == "🏠 Inicio":
    st.header("Bienvenido al Sistema de Reconocimiento de Inventario")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Análisis por Imagen")
        st.write("Sube una imagen para que la IA analice los elementos de inventario")
        
    with col2:
        st.subheader("📝 Análisis por Texto")
        st.write("Describe elementos de inventario para obtener análisis estructurado")

elif page == "📸 Análisis de Imagen":
    st.header("Análisis de Imagen con IA")
    
    uploaded_file = st.file_uploader(
        "Sube una imagen para analizar",
        type=['png', 'jpg', 'jpeg'],
        help="Formatos soportados: PNG, JPG, JPEG"
    )
    
    if uploaded_file is not None:
        # Mostrar imagen
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen subida", width='stretch')
        
        # Análisis con YOLO
        if st.button("🔍 Analizar con YOLO"):
            with st.spinner("Analizando con YOLO..."):
                try:
                    results = yolo_model(image)
                    st.subheader("Resultados YOLO:")
                    
                    # Mostrar resultados
                    for r in results:
                        im_array = r.plot()
                        st.image(im_array, caption="Detección YOLO", width='stretch')
                        
                        # Mostrar información de detección
                        if len(r.boxes) > 0:
                            st.write("Elementos detectados:")
                            for box in r.boxes:
                                conf = box.conf.item()
                                cls = int(box.cls.item())
                                label = yolo_model.names[cls]
                                st.write(f"- {label}: {conf:.2f} confianza")
                        else:
                            st.write("No se detectaron objetos")
                            
                except Exception as e:
                    st.error(f"Error en análisis YOLO: {e}")
        
        # Análisis con Gemini
        description = st.text_input("Descripción adicional (opcional):")
        
        if st.button("🧠 Analizar con Gemini AI"):
            with st.spinner("Analizando con Gemini AI..."):
                try:
                    # Convertir imagen a bytes
                    image_bytes = uploaded_file.getvalue()
                    
                    # Analizar con Gemini
                    analysis = gemini.analyze_image(image_bytes, description)
                    
                    st.subheader("Análisis Gemini AI:")
                    st.text_area("Resultado:", analysis, height=200)
                    
                    # Opción para guardar en Firebase
                    if st.button("💾 Guardar en Inventario"):
                        try:
                            # Preparar datos
                            data = {
                                "tipo": "imagen",
                                "archivo": uploaded_file.name,
                                "descripcion": description,
                                "analisis": analysis,
                                "timestamp": firebase.get_timestamp()
                            }
                            
                            # Guardar en Firebase
                            doc_id = firebase.save_inventory_item(data)
                            st.success(f"✅ Guardado en inventario con ID: {doc_id}")
                            
                        except Exception as e:
                            st.error(f"Error al guardar: {e}")
                            
                except Exception as e:
                    st.error(f"Error en análisis Gemini: {e}")

elif page == "📝 Análisis de Texto":
    st.header("Análisis de Texto con IA")
    
    text_input = st.text_area(
        "Describe los elementos de inventario:",
        placeholder="Ej: 10 laptops Dell, estado bueno, modelo Inspiron 15...",
        height=100
    )
    
    if st.button("🧠 Analizar Descripción"):
        if text_input.strip():
            with st.spinner("Analizando descripción..."):
                try:
                    analysis = gemini.generate_description(text_input)
                    
                    st.subheader("Análisis de la Descripción:")
                    st.text_area("Resultado:", analysis, height=200)
                    
                    # Opción para guardar
                    if st.button("💾 Guardar Análisis"):
                        try:
                            data = {
                                "tipo": "texto",
                                "descripcion": text_input,
                                "analisis": analysis,
                                "timestamp": firebase.get_timestamp()
                            }
                            
                            doc_id = firebase.save_inventory_item(data)
                            st.success(f"✅ Guardado en inventario con ID: {doc_id}")
                            
                        except Exception as e:
                            st.error(f"Error al guardar: {e}")
                except Exception as e:
                    st.error(f"Error en análisis: {e}")
        else:
            st.warning("Por favor, ingresa una descripción")

elif page == "📊 Ver Inventario":
    st.header("Inventario Guardado")
    
    try:
        # Obtener todos los elementos del inventario
        items = firebase.get_all_inventory_items()
        
        if items:
            st.write(f"Total de elementos: {len(items)}")
            
            # Mostrar elementos
            for item in items:
                with st.expander(f"📦 {item.get('tipo', 'Sin tipo')} - {item.get('timestamp', 'Sin fecha')}"):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.write("**Tipo:**", item.get('tipo', 'N/A'))
                        st.write("**Fecha:**", item.get('timestamp', 'N/A'))
                    
                    with col2:
                        if item.get('tipo') == 'imagen':
                            st.write("**Archivo:**", item.get('archivo', 'N/A'))
                        st.write("**Descripción:**", item.get('descripcion', 'N/A'))
                        st.write("**Análisis:**", item.get('analisis', 'N/A'))
                    
                    # Botón para eliminar
                    if st.button(f"🗑️ Eliminar", key=f"delete_{item['id']}"):
                        try:
                            firebase.delete_inventory_item(item['id'])
                            st.success("✅ Elemento eliminado")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error al eliminar: {e}")
        else:
            st.info("No hay elementos en el inventario")
            
    except Exception as e:
        st.error(f"Error al cargar inventario: {e}")

# Footer
st.markdown("---")
st.markdown("🤖 Sistema de Reconocimiento de Inventario con IA - Powered by Gemini AI & YOLO")
