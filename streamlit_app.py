import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import plotly.express as px
import json
from collections import Counter

# Importa las clases que creaste
from firebase_utils import FirebaseUtils
from gemini_utils import GeminiUtils
from ultralytics import YOLO

# --- CONFIGURACIÓN DE PÁGINA Y ESTILOS ---
st.set_page_config(
    page_title="Sistema de Reconocimiento de Inventario",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 1.5rem; }
    .st-emotion-cache-16txtl3 { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# --- INICIALIZACIÓN DE SERVICIOS (Método robusto con cache) ---
@st.cache_resource
def initialize_services():
    """Carga YOLO e inicializa Firebase y Gemini una sola vez para toda la sesión."""
    try:
        yolo_model = YOLO('yolov8m.pt')
        firebase_handler = FirebaseUtils()
        gemini_handler = GeminiUtils()
        return yolo_model, firebase_handler, gemini_handler
    except Exception as e:
        st.error(f"**Error Crítico de Inicialización.** No se pudo cargar un modelo o conectar a un servicio. Revisa los logs y tus secretos.")
        st.code(f"Detalle: {e}", language="bash")
        return None, None, None

yolo_model, firebase, gemini = initialize_services()

if not all([yolo_model, firebase, gemini]):
    st.stop()

# --- BARRA LATERAL DE NAVEGACIÓN ---
st.sidebar.title("Navegación Principal")
page = st.sidebar.radio(
    "Selecciona una sección:",
    ["🏠 Inicio", "📸 Análisis de Imagen", "🗃️ Base de Datos", "📊 Dashboard", "👥 Acerca de"]
)

# --- LÓGICA DE LAS PÁGINAS ---

if page == "🏠 Inicio":
    st.markdown('<h1 class="main-header">Bienvenido al Sistema de Inventario con IA</h1>', unsafe_allow_html=True)
    st.info("Utiliza el menú de la izquierda para navegar entre las diferentes funcionalidades de la aplicación.")
    
    try:
        items = firebase.get_all_inventory_items()
        item_count = len(items)
        image_items = sum(1 for item in items if item.get("tipo") in ["camera", "imagen"])
        text_items = sum(1 for item in items if item.get("tipo") == "texto")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total de Elementos en Inventario", f"{item_count} 📦")
        col2.metric("Análisis desde Imágenes", f"{image_items} 🖼️")
        col3.metric("Análisis desde Texto", f"{text_items} 📝")

    except Exception as e:
        st.warning(f"No se pudieron cargar las estadísticas del inventario: {e}")

    st.subheader("Funcionalidades Clave")
    st.markdown("""
    - **Análisis de Imagen:** Captura una foto con tu cámara o sube un archivo para que la IA detecte y analice los objetos.
    - **Base de Datos:** Visualiza y gestiona todos los elementos que has guardado en tu inventario en la nube.
    - **Dashboard:** Obtén una vista gráfica y resumida de la composición de tu inventario.
    - **Acerca de:** Conoce a los creadores de este proyecto.
    """)

elif page == "📸 Análisis de Imagen":
    st.header("📸 Detección y Análisis de Objetos por Imagen")

    img_source = st.radio("Elige la fuente de la imagen:", ["Cámara en vivo", "Subir un archivo"], horizontal=True)

    img_buffer = None
    if img_source == "Cámara en vivo":
        img_buffer = st.camera_input("Apunta la cámara a los objetos", key="camera_input")
    else:
        img_buffer = st.file_uploader("Sube un archivo de imagen", type=['png', 'jpg', 'jpeg'], key="file_uploader")

    if img_buffer:
        pil_image = Image.open(img_buffer)
        
        with st.spinner("🧠 Detectando objetos con IA local (YOLO)..."):
            results = yolo_model(pil_image)

        st.subheader("🔍 Objetos Detectados")
        # Convertir la imagen a un formato que OpenCV pueda manejar para dibujar
        cv_image = np.array(pil_image.convert('RGB'))[:, :, ::-1].copy() # RGB a BGR
        annotated_image = results[0].plot() # El método .plot() de ultralytics dibuja sobre la imagen
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        st.image(annotated_image_rgb, caption="Imagen con objetos detectados por YOLO.", use_container_width=True)

        detections = results[0]
        
        # Conteo de objetos
        if detections.boxes:
            detected_classes = [detections.names[c] for c in detections.boxes.cls.tolist()]
            counts = Counter(detected_classes)
            st.write("**Conteo en la escena:**")
            st.table(counts)
        else:
            st.info("No se detectaron objetos conocidos en la imagen.")

        st.subheader("▶️ Analizar un objeto en detalle con Gemini")
        if detections.boxes:
            for i, box in enumerate(detections.boxes):
                class_name = detections.names[box.cls[0].item()]
                if st.button(f"Analizar '{class_name}' #{i+1}", key=f"classify_{i}", use_container_width=True):
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = coords
                    cropped_pil_image = pil_image.crop((x1, y1, x2, y2))
                    
                    st.image(cropped_pil_image, caption=f"Recorte de '{class_name}' enviado para análisis...")

                    with st.spinner("🤖 Gemini está analizando el recorte..."):
                        analysis_text = gemini.analyze_image(cropped_pil_image, f"Objeto detectado como {class_name}")
                        st.session_state.last_analysis = analysis_text
                        st.session_state.last_image_name = img_buffer.name if hasattr(img_buffer, 'name') else f"camera_{firebase.get_timestamp()}.jpg"
                        st.rerun()

    # --- Lógica de visualización de resultados (mejorada) ---
    if 'last_analysis' in st.session_state:
        st.subheader("✔️ Resultado del Análisis de Gemini")
        analysis_text = st.session_state.last_analysis
        
        try:
            # Intenta interpretar la respuesta como JSON
            analysis_data = json.loads(analysis_text)
            
            # Si es una lista (múltiples objetos), itera y muéstralos
            if isinstance(analysis_data, list):
                st.info(f"Se encontraron {len(analysis_data)} elementos en el análisis:")
                for item in analysis_data:
                    with st.container(border=True):
                        st.write(f"**Elemento:** {item.get('tipo_de_elemento', 'N/A')}")
                        st.write(f"**Cantidad:** {item.get('cantidad_aproximada', 'N/A')}")
                        st.write(f"**Características:** {item.get('caracteristicas_distintivas', 'N/A')}")

            # Si es un diccionario (un solo objeto), muéstralo
            elif isinstance(analysis_data, dict):
                 st.json(analysis_data)
            
            # Botón para guardar en la base de datos
            if st.button("💾 Guardar Análisis en Inventario", key="save_analysis"):
                data_to_save = {
                    "tipo": "imagen" if hasattr(st.session_state, 'last_image_name') else "camera",
                    "archivo": st.session_state.get('last_image_name', 'desconocido'),
                    "analisis": analysis_data,
                    "timestamp": firebase.get_timestamp()
                }
                firebase.save_inventory_item(data_to_save)
                st.success("¡Análisis guardado en Firebase!")
                # Limpiar para el próximo análisis
                del st.session_state['last_analysis']
                st.rerun()

        except json.JSONDecodeError:
            # Si falla la interpretación de JSON, muestra un error amigable
            st.error("La IA devolvió una respuesta con formato inesperado.")
            with st.expander("Ver detalles técnicos (respuesta sin procesar)"):
                st.code(analysis_text, language='text')

elif page == "🗃️ Base de Datos":
    st.header("🗃️ Gestión de la Base de Datos")
    if st.button("🔄 Refrescar Datos"):
        st.rerun()

    try:
        with st.spinner("Cargando datos desde Firebase..."):
            items = firebase.get_all_inventory_items()
        
        if items:
            st.info(f"Se encontraron **{len(items)}** registros en el inventario.")
            for item in items:
                # Usamos el timestamp o el ID para crear un expander único
                header = item.get('descripcion', item.get('archivo', item['id']))
                with st.expander(f"📦 Registro: **{header}**"):
                    st.json(item)
                    if st.button("🗑️ Eliminar este registro", key=f"delete_{item['id']}", type="primary"):
                        firebase.delete_inventory_item(item['id'])
                        st.success(f"Registro '{item['id']}' eliminado.")
                        st.rerun()
        else:
            st.warning("El inventario está vacío.")
            
    except Exception as e:
        st.error(f"No se pudo conectar con la base de datos: {e}")

elif page == "📊 Dashboard":
    st.header("📊 Dashboard del Inventario")
    try:
        with st.spinner("Generando estadísticas..."):
            items = firebase.get_all_inventory_items()
        
        if items:
            df = pd.DataFrame(items)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            st.subheader("Distribución de Análisis por Tipo")
            type_counts = df['tipo'].value_counts()
            fig_pie = px.pie(
                type_counts, 
                values=type_counts.values, 
                names=type_counts.index, 
                title="Tipos de Registros en el Inventario"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            st.subheader("Actividad Reciente en el Inventario")
            df_recent = df.sort_values('timestamp', ascending=False).head(10)
            st.dataframe(df_recent[['timestamp', 'tipo', 'descripcion', 'archivo']], use_container_width=True)

        else:
            st.warning("No hay datos en el inventario para generar un dashboard.")

    except Exception as e:
        st.error(f"Error al crear el dashboard: {e}")

elif page == "👥 Acerca de":
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

