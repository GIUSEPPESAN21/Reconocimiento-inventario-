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
    page_title="Sistema de Inventario IA Profesional",
    page_icon="🚀",
    layout="wide"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 1.5rem; }
    .st-emotion-cache-16txtl3 { padding-top: 2rem; }
    .report-box { background-color: #f0f2f6; padding: 1.5rem; border-radius: 10px; border-left: 6px solid #1f77b4; margin-bottom: 1rem;}
    .report-header { font-size: 1.2rem; font-weight: bold; color: #333; }
    .report-data { font-size: 1.1rem; color: #555; }
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
    st.markdown('<h1 class="main-header">🚀 Bienvenido al Sistema de Inventario Profesional</h1>', unsafe_allow_html=True)
    st.subheader("Una solución inteligente para la gestión y reconocimiento de tus activos.")

    st.markdown("---")

    try:
        items = firebase.get_all_inventory_items()
        item_count = len(items)
        image_items = sum(1 for item in items if item.get("tipo") in ["camera", "imagen"])
        text_items = sum(1 for item in items if item.get("tipo") == "texto")

        col1, col2, col3 = st.columns(3)
        col1.metric("📦 Total de Artículos Registrados", item_count)
        col2.metric("🖼️ Análisis desde Imágenes", image_items)
        col3.metric("📝 Registros Manuales/Texto", text_items)

    except Exception as e:
        st.warning(f"No se pudieron cargar las estadísticas del inventario: {e}")
    
    st.markdown("---")

    st.subheader("Pasos para empezar:")
    st.markdown("""
    1.  **Ve a 'Análisis de Imagen'**: Usa tu cámara o sube una foto para que la IA detecte los objetos.
    2.  **Analiza y Registra**: Selecciona un objeto detectado para obtener un análisis detallado de sus características. Luego, asígnale un ID personalizado, define las unidades y guárdalo en tu inventario.
    3.  **Gestiona tu Inventario**: En la sección de 'Base de Datos', puedes ver, buscar y eliminar cualquier artículo registrado.
    4.  **Visualiza tus Datos**: El 'Dashboard' te ofrece gráficos interactivos para entender la composición de tu inventario de un solo vistazo.
    """)

elif page == "📸 Análisis de Imagen":
    st.header("📸 Detección y Análisis de Objetos por Imagen")

    if 'last_analysis' in st.session_state:
        st.subheader("✔️ Resultado del Análisis de Gemini")
        analysis_text = st.session_state.last_analysis
        
        try:
            analysis_data = json.loads(analysis_text)
            
            # --- NUEVO: Visualización en texto normal ---
            if "error" not in analysis_data:
                st.markdown('<div class="report-box">', unsafe_allow_html=True)
                st.write(f"<span class='report-header'>Elemento Identificado:</span> <span class='report-data'>{analysis_data.get('elemento_identificado', 'No especificado')}</span>", unsafe_allow_html=True)
                st.write(f"<span class='report-header'>Cantidad Detectada:</span> <span class='report-data'>{analysis_data.get('cantidad', 'No especificada')}</span>", unsafe_allow_html=True)
                st.write(f"<span class='report-header'>Estado Aparente:</span> <span class='report-data'>{analysis_data.get('estado', 'No especificado')}</span>", unsafe_allow_html=True)
                st.write(f"<span class='report-header'>Categoría Sugerida:</span> <span class='report-data'>{analysis_data.get('categoria_sugerida', 'No especificada')}</span>", unsafe_allow_html=True)
                
                features = analysis_data.get('caracteristicas', [])
                if features:
                    st.markdown("<span class='report-header'>Características Notables:</span>", unsafe_allow_html=True)
                    for feature in features:
                        st.markdown(f"- {feature}")
                st.markdown('</div>', unsafe_allow_html=True)

                # --- NUEVO: Formulario de guardado avanzado ---
                with st.form("save_to_db_form"):
                    st.subheader("💾 Registrar en la Base de Datos")
                    custom_id = st.text_input("ID Personalizado (SKU, Código de Producto, etc.):", key="custom_id")
                    description = st.text_input("Descripción del Producto:", value=analysis_data.get('elemento_identificado', ''))
                    quantity = st.number_input("Unidades Existentes:", min_value=1, value=analysis_data.get('cantidad', 1), step=1)
                    
                    submitted = st.form_submit_button("Añadir a la Base de Datos")

                    if submitted:
                        if not custom_id or not description:
                            st.warning("El ID Personalizado y la Descripción son obligatorios.")
                        else:
                            with st.spinner("Guardando..."):
                                data_to_save = {
                                    "custom_id": custom_id,
                                    "name": description, # 'name' para compatibilidad con el listado
                                    "quantity": quantity,
                                    "tipo": "imagen" if hasattr(st.session_state, 'last_image_name') else "camera",
                                    "analisis_ia": analysis_data,
                                    "timestamp": firebase.get_timestamp()
                                }
                                firebase.save_inventory_item(data_to_save, custom_id)
                                st.success(f"¡Artículo '{description}' con ID '{custom_id}' guardado con éxito!")
                                del st.session_state['last_analysis']
                                st.rerun()

            else:
                 st.error(f"Error en el análisis de Gemini: {analysis_data['error']}")

        except json.JSONDecodeError:
            st.error("La IA devolvió una respuesta con formato inesperado.")
            with st.expander("Ver detalles técnicos (respuesta sin procesar)"):
                st.code(analysis_text, language='text')

    else:
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
            annotated_image = results[0].plot()
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            st.image(annotated_image_rgb, caption="Imagen con objetos detectados por YOLO.", use_container_width=True)

            detections = results[0]
            
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
                        cropped_pil_image = pil_image.crop(tuple(coords))
                        
                        st.image(cropped_pil_image, caption=f"Recorte de '{class_name}' enviado para análisis...")

                        with st.spinner("🤖 Gemini está analizando el recorte..."):
                            analysis_text = gemini.analyze_image(cropped_pil_image, f"Objeto detectado como {class_name}")
                            st.session_state.last_analysis = analysis_text
                            st.session_state.last_image_name = img_buffer.name if hasattr(img_buffer, 'name') else f"camera_{firebase.get_timestamp()}.jpg"
                            st.rerun()

elif page == "🗃️ Base de Datos":
    st.header("🗃️ Gestión de la Base de Datos")

    with st.expander("➕ Añadir Artículo Manualmente con ID Personalizado"):
        with st.form("manual_add_form"):
            manual_custom_id = st.text_input("ID Personalizado (SKU, Código, etc.)")
            manual_name = st.text_input("Nombre o Descripción del Artículo")
            manual_quantity = st.number_input("Cantidad", min_value=0, step=1)
            
            manual_submit = st.form_submit_button("Guardar Artículo")

            if manual_submit:
                if not manual_custom_id or not manual_name:
                    st.warning("El ID Personalizado y el Nombre son obligatorios.")
                else:
                    data_to_save = {
                        "custom_id": manual_custom_id,
                        "name": manual_name,
                        "quantity": manual_quantity,
                        "tipo": "manual",
                        "timestamp": firebase.get_timestamp()
                    }
                    try:
                        firebase.save_inventory_item(data_to_save, manual_custom_id)
                        st.success(f"Artículo '{manual_name}' guardado con éxito.")
                    except ValueError as e:
                        st.error(str(e)) # Muestra el error si el ID ya existe
                    except Exception as e:
                        st.error(f"Ocurrió un error inesperado: {e}")

    st.markdown("---")
    st.subheader("Inventario Actual")

    if st.button("🔄 Refrescar Datos"):
        st.rerun()

    try:
        with st.spinner("Cargando datos desde Firebase..."):
            items = firebase.get_all_inventory_items()
        
        if items:
            st.info(f"Se encontraron **{len(items)}** registros en el inventario.")
            for item in items:
                header = item.get('custom_id') or item.get('name', item['id'])
                with st.expander(f"📦 **{header}** (Cantidad: {item.get('quantity', 'N/A')})"):
                    st.json(item)
                    if st.button("🗑️ Eliminar", key=f"delete_{item['id']}", type="primary"):
                        firebase.delete_inventory_item(item['id'])
                        st.success(f"Registro '{header}' eliminado.")
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
            # --- CORRECCIÓN DEL ERROR 'timestamp' ---
            # Filtramos los items para asegurarnos de que tengan los campos necesarios
            valid_items = [item for item in items if 'timestamp' in item and 'tipo' in item]
            if not valid_items:
                 st.warning("No hay registros con datos suficientes para generar un dashboard.")
            else:
                df = pd.DataFrame(valid_items)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                st.subheader("Distribución de Registros por Tipo")
                type_counts = df['tipo'].value_counts()
                fig_pie = px.pie(
                    type_counts, 
                    values=type_counts.values, 
                    names=type_counts.index, 
                    title="Tipos de Registros en el Inventario",
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                st.plotly_chart(fig_pie, use_container_width=True)

                st.subheader("Actividad Reciente en el Inventario")
                df_recent = df.sort_values('timestamp', ascending=False).head(10)
                # Seleccionamos columnas que sabemos que existen
                display_cols = ['timestamp', 'tipo']
                if 'name' in df_recent.columns: display_cols.append('name')
                if 'custom_id' in df_recent.columns: display_cols.append('custom_id')

                st.dataframe(df_recent[display_cols], use_container_width=True)
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
