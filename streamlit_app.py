import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import json
import base64
import logging
from datetime import datetime
from ultralytics import YOLO
from firebase_utils import FirebaseUtils
from gemini_utils import GeminiUtils
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar directorio de YOLO
os.environ['YOLO_CONFIG_DIR'] = '/tmp/Ultralytics'

# Configurar página
st.set_page_config(
    page_title="Sistema de Reconocimiento de Inventario con IA",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .stats-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .author-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">🤖 Sistema de Reconocimiento de Inventario con IA</h1>', unsafe_allow_html=True)

# Inicializar servicios con cache
@st.cache_resource
def load_yolo_model():
    """Carga el modelo YOLO"""
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
    """Inicializa Firebase y Gemini"""
    try:
        firebase = FirebaseUtils()
        gemini = GeminiUtils()
        return firebase, gemini
    except Exception as e:
        logger.error(f"Error al inicializar servicios: {e}")
        return None, None

# Cargar servicios
yolo_model = load_yolo_model()
firebase, gemini = initialize_services()

if yolo_model is None or firebase is None or gemini is None:
    st.error("❌ Error al inicializar los servicios. Verifica la configuración.")
    st.stop()

# Sidebar con navegación
st.sidebar.title("📋 Navegación")
page = st.sidebar.selectbox(
    "Selecciona una opción:",
    [
        "🏠 Inicio", 
        "📸 Cámara en Vivo", 
        "📁 Subir Imagen", 
        "📝 Análisis de Texto", 
        "📊 Dashboard", 
        "🗃️ Base de Datos", 
        "👥 Información del Proyecto",
        "⚙️ Configuración"
    ]
)

# Página de Inicio
if page == "🏠 Inicio":
    st.header("🏠 Bienvenido al Sistema de Reconocimiento de Inventario")
    
    # Estadísticas rápidas
    try:
        items = firebase.get_all_inventory_items()
        total_items = len(items)
        image_items = len([item for item in items if item.get('tipo') == 'imagen'])
        text_items = len([item for item in items if item.get('tipo') == 'texto'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="stats-box">
                <h3>{total_items}</h3>
                <p>Total Elementos</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stats-box">
                <h3>{image_items}</h3>
                <p>Imágenes Analizadas</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="stats-box">
                <h3>{text_items}</h3>
                <p>Descripciones Procesadas</p>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.warning(f"Error al cargar estadísticas: {e}")
    
    # Características principales
    st.subheader("🚀 Características Principales")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h4>📸 Análisis por Cámara en Vivo</h4>
            <p>Captura imágenes en tiempo real usando tu cámara web para análisis inmediato con YOLO e IA.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-box">
            <h4>📁 Subida de Archivos</h4>
            <p>Sube imágenes desde tu dispositivo para análisis detallado con Gemini AI.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <h4>📝 Análisis de Texto</h4>
            <p>Describe elementos de inventario y obtén análisis estructurado con IA.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-box">
            <h4>📊 Dashboard Completo</h4>
            <p>Visualiza estadísticas y tendencias de tu inventario con gráficos interactivos.</p>
        </div>
        """, unsafe_allow_html=True)

# Página de Cámara en Vivo
elif page == "📸 Cámara en Vivo":
    st.header("📸 Análisis con Cámara en Vivo")
    
    # Opciones de cámara
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("🎥 Activa tu cámara para capturar y analizar elementos de inventario en tiempo real")
    
    with col2:
        auto_analyze = st.checkbox("🔄 Análisis Automático", value=True)
        confidence_threshold = st.slider("🎯 Umbral de Confianza", 0.1, 1.0, 0.5, 0.1)
    
    # Componente de cámara
    picture = st.camera_input("📷 Toma una foto")
    
    if picture is not None:
        # Convertir a imagen PIL
        image = Image.open(picture)
        
        # Mostrar imagen original
        st.subheader("📸 Imagen Capturada")
        st.image(image, caption="Imagen capturada", width='stretch')
        
        # Convertir a formato OpenCV
        img_array = np.array(image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Análisis con YOLO
        if st.button("🔍 Analizar con YOLO") or auto_analyze:
            with st.spinner("Analizando con YOLO..."):
                try:
                    results = yolo_model(img_cv)
                    
                    # Procesar resultados
                    detections = []
                    annotated_image = img_cv.copy()
                    
                    for r in results:
                        if len(r.boxes) > 0:
                            for box in r.boxes:
                                conf = box.conf.item()
                                if conf >= confidence_threshold:
                                    cls = int(box.cls.item())
                                    label = yolo_model.names[cls]
                                    
                                    # Coordenadas del bounding box
                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                    
                                    # Dibujar bounding box
                                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.putText(annotated_image, f"{label}: {conf:.2f}", 
                                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                    
                                    detections.append({
                                        'label': label,
                                        'confidence': conf,
                                        'bbox': [x1, y1, x2, y2]
                                    })
                    
                    # Mostrar imagen anotada
                    if detections:
                        st.subheader("🎯 Detecciones YOLO")
                        annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                        st.image(annotated_rgb, caption="Detecciones YOLO", width='stretch')
                        
                        # Mostrar información de detecciones
                        st.subheader("📋 Información de Detecciones")
                        for i, detection in enumerate(detections, 1):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"**Elemento {i}:** {detection['label']}")
                            with col2:
                                st.write(f"**Confianza:** {detection['confidence']:.2f}")
                            with col3:
                                st.write(f"**Posición:** {detection['bbox']}")
                    else:
                        st.warning("No se detectaron objetos con el umbral de confianza seleccionado")
                        
                except Exception as e:
                    st.error(f"Error en análisis YOLO: {e}")
        
        # Análisis con Gemini AI
        st.subheader("🧠 Análisis con Gemini AI")
        description = st.text_input("Descripción adicional (opcional):")
        
        if st.button("🧠 Analizar con IA"):
            with st.spinner("Analizando con Gemini AI..."):
                try:
                    # Convertir imagen a bytes
                    image_bytes = picture.getvalue()
                    
                    # Analizar con Gemini
                    analysis = gemini.analyze_image(image_bytes, description)
                    
                    st.subheader("📝 Resultado del Análisis IA:")
                    st.text_area("Análisis:", analysis, height=200)
                    
                    # Opción para guardar
                    if st.button("💾 Guardar en Inventario"):
                        try:
                            data = {
                                "tipo": "camera",
                                "archivo": f"camera_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                                "descripcion": description,
                                "analisis": analysis,
                                "detections": detections if 'detections' in locals() else [],
                                "timestamp": firebase.get_timestamp()
                            }
                            
                            doc_id = firebase.save_inventory_item(data)
                            st.success(f"✅ Guardado en inventario con ID: {doc_id}")
                            
                        except Exception as e:
                            st.error(f"Error al guardar: {e}")
                            
                except Exception as e:
                    st.error(f"Error en análisis Gemini: {e}")

# Página de Subir Imagen
elif page == "📁 Subir Imagen":
    st.header("📁 Análisis de Imagen Subida")
    
    uploaded_file = st.file_uploader(
        "Sube una imagen para analizar",
        type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
        help="Formatos soportados: PNG, JPG, JPEG, GIF, BMP"
    )
    
    if uploaded_file is not None:
        # Mostrar imagen
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen subida", width='stretch')
        
        # Información del archivo
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"📁 **Archivo:** {uploaded_file.name}")
        with col2:
            st.info(f"📏 **Tamaño:** {image.size}")
        with col3:
            st.info(f"🎨 **Formato:** {image.format}")
        
        # Análisis con YOLO
        if st.button("🔍 Analizar con YOLO"):
            with st.spinner("Analizando con YOLO..."):
                try:
                    results = yolo_model(image)
                    st.subheader("🎯 Resultados YOLO:")
                    
                    for r in results:
                        im_array = r.plot()
                        st.image(im_array, caption="Detección YOLO", width='stretch')
                        
                        if len(r.boxes) > 0:
                            st.write("**Elementos detectados:**")
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
                    image_bytes = uploaded_file.getvalue()
                    analysis = gemini.analyze_image(image_bytes, description)
                    
                    st.subheader("📝 Análisis Gemini AI:")
                    st.text_area("Resultado:", analysis, height=200)
                    
                    if st.button("💾 Guardar en Inventario"):
                        try:
                            data = {
                                "tipo": "imagen",
                                "archivo": uploaded_file.name,
                                "descripcion": description,
                                "analisis": analysis,
                                "timestamp": firebase.get_timestamp()
                            }
                            
                            doc_id = firebase.save_inventory_item(data)
                            st.success(f"✅ Guardado en inventario con ID: {doc_id}")
                            
                        except Exception as e:
                            st.error(f"Error al guardar: {e}")
                            
                except Exception as e:
                    st.error(f"Error en análisis Gemini: {e}")

# Página de Análisis de Texto
elif page == "📝 Análisis de Texto":
    st.header("📝 Análisis de Texto con IA")
    
    # Plantillas de texto
    st.subheader("📋 Plantillas de Ejemplo")
    template = st.selectbox(
        "Selecciona una plantilla:",
        ["Personalizado", "Laptops", "Muebles", "Equipos Electrónicos", "Herramientas"]
    )
    
    templates = {
        "Laptops": "15 laptops Dell Inspiron, estado bueno, modelo 2023, color negro",
        "Muebles": "5 escritorios de oficina, madera, estado regular, necesita mantenimiento",
        "Equipos Electrónicos": "10 monitores Samsung 24 pulgadas, estado excelente, LED",
        "Herramientas": "20 destornilladores Phillips, metal, estado bueno, varios tamaños"
    }
    
    if template != "Personalizado":
        text_input = st.text_area(
            "Describe los elementos de inventario:",
            value=templates[template],
            height=100
        )
    else:
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
                    
                    st.subheader("📝 Análisis de la Descripción:")
                    st.text_area("Resultado:", analysis, height=200)
                    
                    # Botón para guardar
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

# Página de Dashboard
elif page == "📊 Dashboard":
    st.header("📊 Dashboard de Inventario")
    
    try:
        items = firebase.get_all_inventory_items()
        
        if items:
            # Estadísticas generales
            st.subheader("📈 Estadísticas Generales")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Elementos", len(items))
            
            with col2:
                image_count = len([item for item in items if item.get('tipo') == 'imagen'])
                st.metric("Imágenes", image_count)
            
            with col3:
                text_count = len([item for item in items if item.get('tipo') == 'texto'])
                st.metric("Textos", text_count)
            
            with col4:
                camera_count = len([item for item in items if item.get('tipo') == 'camera'])
                st.metric("Cámara", camera_count)
            
            # Gráfico de tipos
            st.subheader("📊 Distribución por Tipo")
            type_counts = {}
            for item in items:
                tipo = item.get('tipo', 'desconocido')
                type_counts[tipo] = type_counts.get(tipo, 0) + 1
            
            if type_counts:
                fig = px.pie(
                    values=list(type_counts.values()),
                    names=list(type_counts.keys()),
                    title="Distribución de Elementos por Tipo"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Gráfico temporal
            st.subheader("📅 Actividad Temporal")
            dates = []
            for item in items:
                timestamp = item.get('timestamp', '')
                if timestamp:
                    try:
                        date = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).date()
                        dates.append(date)
                    except:
                        pass
            
            if dates:
                date_counts = pd.Series(dates).value_counts().sort_index()
                fig = px.line(
                    x=date_counts.index,
                    y=date_counts.values,
                    title="Elementos Agregados por Fecha"
                )
                fig.update_xaxes(title="Fecha")
                fig.update_yaxes(title="Cantidad")
                st.plotly_chart(fig, use_container_width=True)
            
            # Tabla de elementos recientes
            st.subheader("🕒 Elementos Recientes")
            recent_items = sorted(items, key=lambda x: x.get('timestamp', ''), reverse=True)[:10]
            
            df_data = []
            for item in recent_items:
                df_data.append({
                    'Tipo': item.get('tipo', 'N/A'),
                    'Archivo': item.get('archivo', 'N/A'),
                    'Fecha': item.get('timestamp', 'N/A')[:10] if item.get('timestamp') else 'N/A',
                    'ID': item.get('id', 'N/A')[:8] + '...'
                })
            
            if df_data:
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)
            
        else:
            st.info("No hay elementos en el inventario para mostrar estadísticas")
            
    except Exception as e:
        st.error(f"Error al cargar dashboard: {e}")

# Página de Base de Datos
elif page == "🗃️ Base de Datos":
    st.header("🗃️ Gestión de Base de Datos")
    
    # Filtros
    st.subheader("🔍 Filtros de Búsqueda")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_type = st.selectbox("Filtrar por tipo:", ["Todos", "imagen", "texto", "camera"])
    
    with col2:
        date_filter = st.date_input("Filtrar por fecha:")
    
    with col3:
        search_term = st.text_input("Buscar en descripciones:")
    
    # Botones de acción
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Actualizar Lista"):
            st.rerun()
    
    with col2:
        if st.button("📊 Exportar CSV"):
            try:
                items = firebase.get_all_inventory_items()
                if items:
                    df_data = []
                    for item in items:
                        df_data.append({
                            'ID': item.get('id', ''),
                            'Tipo': item.get('tipo', ''),
                            'Archivo': item.get('archivo', ''),
                            'Descripción': item.get('descripcion', ''),
                            'Análisis': item.get('analisis', ''),
                            'Timestamp': item.get('timestamp', '')
                        })
                    
                    df = pd.DataFrame(df_data)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Descargar CSV",
                        data=csv,
                        file_name=f"inventario_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No hay datos para exportar")
            except Exception as e:
                st.error(f"Error al exportar: {e}")
    
    with col3:
        if st.button("🗑️ Limpiar Base de Datos"):
            if st.checkbox("⚠️ Confirmar eliminación de todos los elementos"):
                try:
                    items = firebase.get_all_inventory_items()
                    for item in items:
                        firebase.delete_inventory_item(item['id'])
                    st.success("✅ Base de datos limpiada")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error al limpiar: {e}")
    
    # Mostrar elementos
    try:
        items = firebase.get_all_inventory_items()
        
        # Aplicar filtros
        filtered_items = items
        
        if filter_type != "Todos":
            filtered_items = [item for item in filtered_items if item.get('tipo') == filter_type]
        
        if search_term:
            filtered_items = [item for item in filtered_items 
                            if search_term.lower() in item.get('descripcion', '').lower()]
        
        if date_filter:
            filtered_items = [item for item in filtered_items 
                            if item.get('timestamp', '').startswith(str(date_filter))]
        
        st.subheader(f"📋 Elementos Encontrados ({len(filtered_items)})")
        
        if filtered_items:
            for i, item in enumerate(filtered_items):
                with st.expander(f"📦 {item.get('tipo', 'Sin tipo')} - {item.get('timestamp', 'Sin fecha')[:19]}"):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.write("**Tipo:**", item.get('tipo', 'N/A'))
                        st.write("**Fecha:**", item.get('timestamp', 'N/A')[:19])
                        st.write("**ID:**", item.get('id', 'N/A'))
                        
                        if item.get('tipo') == 'imagen' or item.get('tipo') == 'camera':
                            st.write("**Archivo:**", item.get('archivo', 'N/A'))
                    
                    with col2:
                        st.write("**Descripción:**", item.get('descripcion', 'N/A'))
                        st.write("**Análisis:**", item.get('analisis', 'N/A')[:200] + "..." if len(item.get('analisis', '')) > 200 else item.get('analisis', 'N/A'))
                    
                    # Botones de acción
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button(f"👁️ Ver Completo", key=f"view_{item['id']}"):
                            st.json(item)
                    
                    with col2:
                        if st.button(f"✏️ Editar", key=f"edit_{item['id']}"):
                            st.session_state.editing_item = item
                            st.rerun()
                    
                    with col3:
                        if st.button(f"🗑️ Eliminar", key=f"delete_{item['id']}"):
                            try:
                                firebase.delete_inventory_item(item['id'])
                                st.success("✅ Elemento eliminado")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error al eliminar: {e}")
        else:
            st.info("No se encontraron elementos con los filtros aplicados")
            
    except Exception as e:
        st.error(f"Error al cargar inventario: {e}")

# Página de Información del Proyecto
elif page == "👥 Información del Proyecto":
    st.header("👥 Información del Proyecto")
    
    # Información de los autores
    st.subheader("👨‍💻 Desarrolladores")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="author-info">
            <h4>🎓 Estudiante Principal</h4>
            <p><strong>Nombre:</strong> Giuseppe Sánchez</p>
            <p><strong>Universidad:</strong> Uniminuto</p>
            <p><strong>Programa:</strong> Ingeniería de Sistemas</p>
            <p><strong>Rol:</strong> Desarrollador Principal</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="author-info">
            <h4>🏫 Institución</h4>
            <p><strong>Universidad:</strong> Corporación Universitaria Minuto de Dios</p>
            <p><strong>Facultad:</strong> Ingeniería</p>
            <p><strong>Programa:</strong> Ingeniería de Sistemas</p>
            <p><strong>Año:</strong> 2025</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Información técnica del proyecto
    st.subheader("🔧 Información Técnica")
    
    st.markdown("""
    <div class="feature-box">
        <h4>🤖 Tecnologías Utilizadas</h4>
        <ul>
            <li><strong>Streamlit:</strong> Framework web para la interfaz de usuario</li>
            <li><strong>Google Gemini AI:</strong> Modelo de inteligencia artificial para análisis de imágenes y texto</li>
            <li><strong>YOLOv8:</strong> Modelo de detección de objetos en tiempo real</li>
            <li><strong>Firebase Firestore:</strong> Base de datos NoSQL en la nube</li>
            <li><strong>OpenCV:</strong> Procesamiento de imágenes y visión por computadora</li>
            <li><strong>Python:</strong> Lenguaje de programación principal</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-box">
        <h4>🎯 Objetivos del Proyecto</h4>
        <ul>
            <li>Automatizar el reconocimiento de elementos de inventario</li>
            <li>Implementar análisis inteligente con IA</li>
            <li>Crear una interfaz web intuitiva y fácil de usar</li>
            <li>Integrar múltiples tecnologías de vanguardia</li>
            <li>Facilitar la gestión de inventarios empresariales</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Características implementadas
    st.subheader("✨ Características Implementadas")
    
    features = [
        "📸 Captura de imágenes con cámara web",
        "📁 Subida de archivos de imagen",
        "🧠 Análisis inteligente con Gemini AI",
        "🎯 Detección de objetos con YOLOv8",
        "🗃️ Almacenamiento en Firebase",
        "📊 Dashboard con estadísticas",
        "📈 Gráficos interactivos",
        "🔍 Sistema de búsqueda y filtros",
        "📤 Exportación de datos a CSV",
        "⚙️ Configuración personalizable"
    ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        for feature in features[:5]:
            st.write(f"✅ {feature}")
    
    with col2:
        for feature in features[5:]:
            st.write(f"✅ {feature}")
    
    # Contacto y soporte
    st.subheader("📞 Contacto y Soporte")
    
    st.markdown("""
    <div class="author-info">
        <h4>📧 Información de Contacto</h4>
        <p><strong>Desarrollador:</strong> Giuseppe Sánchez</p>
        <p><strong>Institución:</strong> Uniminuto</p>
        <p><strong>Proyecto:</strong> Sistema de Reconocimiento de Inventario con IA</p>
        <p><strong>Versión:</strong> 2.0</p>
        <p><strong>Última actualización:</strong> Enero 2025</p>
    </div>
    """, unsafe_allow_html=True)

# Página de Configuración
elif page == "⚙️ Configuración":
    st.header("⚙️ Configuración del Sistema")
    
    # Estado de los servicios
    st.subheader("🔧 Estado de los Servicios")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if yolo_model:
            st.success("✅ YOLOv8: Conectado")
        else:
            st.error("❌ YOLOv8: Desconectado")
    
    with col2:
        if firebase:
            st.success("✅ Firebase: Conectado")
        else:
            st.error("❌ Firebase: Desconectado")
    
    with col3:
        if gemini:
            st.success("✅ Gemini AI: Conectado")
        else:
            st.error("❌ Gemini AI: Desconectado")
    
    # Configuración de YOLO
    st.subheader("🎯 Configuración de YOLO")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Modelo actual:** YOLOv8 Medium")
        st.info("**Confianza por defecto:** 0.5")
    
    with col2:
        st.info("**Clases detectables:** 80 objetos")
        st.info("**Velocidad:** ~200ms por imagen")
    
    # Configuración de Gemini
    st.subheader("🧠 Configuración de Gemini AI")
    
    try:
        if hasattr(gemini, 'model') and gemini.model:
            model_name = str(gemini.model.model_name) if hasattr(gemini.model, 'model_name') else "Modelo activo"
            st.info(f"**Modelo actual:** {model_name}")
        else:
            st.info("**Modelo actual:** Modelo por defecto")
    except:
        st.info("**Modelo actual:** Modelo por defecto")
    
    # Información del sistema
    st.subheader("💻 Información del Sistema")
    
    import platform
    import sys
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Python:** {sys.version.split()[0]}")
        st.write(f"**Streamlit:** {st.__version__}")
        st.write(f"**Sistema:** {platform.system()}")
    
    with col2:
        st.write(f"**Arquitectura:** {platform.architecture()[0]}")
        st.write(f"**Procesador:** {platform.processor()}")
        st.write(f"**Máquina:** {platform.machine()}")
    
    # Logs del sistema
    st.subheader("📋 Logs del Sistema")
    
    if st.button("🔄 Actualizar Logs"):
        st.rerun()
    
    # Mostrar información de la sesión
    st.subheader("🔍 Información de la Sesión")
    
    session_info = {
        "Timestamp de inicio": datetime.now().isoformat(),
        "Servicios cargados": len([x for x in [yolo_model, firebase, gemini] if x is not None]),
        "Página actual": page,
        "Estado de la aplicación": "Activa"
    }
    
    for key, value in session_info.items():
        st.write(f"**{key}:** {value}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🤖 <strong>Sistema de Reconocimiento de Inventario con IA</strong></p>
    <p>Desarrollado por <strong>Giuseppe Sánchez</strong> - Uniminuto 2025</p>
    <p>Powered by <strong>Gemini AI</strong>, <strong>YOLOv8</strong> y <strong>Firebase</strong></p>
</div>
""", unsafe_allow_html=True)
