import streamlit as st
from PIL import Image
import firebase_utils

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Inventario Inteligente",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- INICIALIZACIÓN DE FIREBASE ---
try:
    db = firebase_utils.get_db()
except Exception as e:
    st.error(f"**Error Crítico de Conexión.** No se pudo inicializar Firebase. Detalle: {e}")
    st.stop()

# --- TÍTULO Y DESCRIPCIÓN ---
st.title("📦 Inventario Inteligente con IA")
st.markdown("Identifica artículos de tu inventario en tiempo real con la cámara, usando **Gemini AI** y **Firebase**.")

# --- ESTRUCTURA DE LA INTERFAZ ---
col1, col2 = st.columns([2, 1])

# --- Columna 2: Panel de Control ---
with col2:
    st.header("📊 Panel de Control")

    # --- CARGA DEL INVENTARIO CON MANEJO DE ERRORES ---
    if 'inventory_list' not in st.session_state:
        try:
            with st.spinner("Cargando inventario desde Firebase..."):
                st.session_state.inventory_list = firebase_utils.get_inventory()
        except Exception as e:
            st.error("⚠️ Error al Cargar el Inventario desde Firebase.")
            st.warning("""
                La aplicación no pudo obtener los datos. Esto suele ocurrir por:
                1.  **Secretos Incorrectos:** Un error al copiar la variable `FIREBASE_SERVICE_ACCOUNT_BASE64` en Streamlit Cloud.
                2.  **Reglas de Firestore:** Las reglas de seguridad podrían estar bloqueando el acceso.
                3.  **Problema de Red:** Un bloqueo de red temporal entre Streamlit y Google.
            """)
            st.code(f"Detalles del error: {e}", language="python")
            st.session_state.inventory_list = [] # Se asigna lista vacía para que la app no se rompa.

    inventory_list = st.session_state.inventory_list
    inventory_names = [item['name'] for item in inventory_list if 'name' in item]

    with st.expander("➕ Añadir Nuevo Artículo", expanded=True):
        new_item_name = st.text_input("Nombre del artículo", key="new_item", placeholder="Ej: Laptop, Taza, Silla")
        if st.button("Guardar en Firebase", use_container_width=True):
            if new_item_name and new_item_name.strip() and new_item_name not in inventory_names:
                firebase_utils.add_item(new_item_name.strip())
                st.success(f"'{new_item_name}' añadido.")
                del st.session_state.inventory_list
                st.rerun()
            else:
                st.warning("El nombre no puede estar vacío o ya existe.")

    st.subheader("📋 Inventario Actual")
    if inventory_names:
        st.dataframe(inventory_names, use_container_width=True, column_config={"value": "Artículo"})
    else:
        st.info("No se pudo cargar el inventario o está vacío.")

    st.subheader("🤖 Resultado del Análisis")
    result_placeholder = st.empty()
    if 'last_result' in st.session_state:
        result_placeholder.success(f"**Artículo Identificado:** {st.session_state.last_result}")
    else:
        result_placeholder.info("Esperando análisis de imagen...")

# --- Columna 1: Cámara y Visualización ---
with col1:
    st.header("📷 Captura y Análisis")
    img_buffer = st.camera_input("Apunta al artículo y toma una foto", key="camera")

    if img_buffer:
        img_pil = Image.open(img_buffer)
        
        max_width = 512
        if img_pil.width > max_width:
            aspect_ratio = img_pil.height / img_pil.width
            new_height = int(max_width * aspect_ratio)
            img_pil = img_pil.resize((max_width, new_height))

        st.image(img_pil, caption="Imagen lista para analizar", use_column_width=True)

        if st.button("✨ Analizar con Gemini", type="primary", use_container_width=True):
            if not inventory_names:
                st.warning("El inventario está vacío. Añade artículos antes de analizar.")
            else:
                with st.spinner("🧠 Gemini está identificando el artículo..."):
                    import gemini_utils
                    result = gemini_utils.identify_item(img_pil, inventory_names)
                    st.session_state.last_result = result
                    st.rerun()


