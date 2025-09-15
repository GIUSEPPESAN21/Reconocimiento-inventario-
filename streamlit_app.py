import streamlit as st
from PIL import Image
import firebase_utils  # Asegúrate de que este import esté presente

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Inventario Inteligente",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- INICIALIZACIÓN DE FIREBASE (Ocurre una sola vez) ---
# Esta función ahora solo asegura que la app esté conectada, no trae datos.
try:
    firebase_utils.initialize_firebase()
    # Muestra un mensaje de éxito discreto en la barra lateral o en un log.
    # st.sidebar.success("Conexión con Firebase OK.") 
except Exception as e:
    st.error(f"**Error Crítico de Conexión.** No se pudo inicializar Firebase. Revisa tus secretos. Detalle: {e}")
    st.stop()

# --- TÍTULO Y DESCRIPCIÓN (Carga inmediata) ---
st.title("📦 Inventario Inteligente con IA")
st.markdown("Identifica artículos de tu inventario en tiempo real con la cámara, usando **Gemini AI** y **Firebase**.")

# --- ESTRUCTURA DE LA INTERFAZ (Carga inmediata) ---
col1, col2 = st.columns([2, 1])

# --- Columna 2: Panel de Control ---
with col2:
    st.header("📊 Panel de Control")

    # --- CARGA DEL INVENTARIO ---
    # Usamos un placeholder para mostrar un spinner solo en esta sección.
    inventory_placeholder = st.empty()
    with inventory_placeholder.container():
        with st.spinner("Cargando inventario..."):
            try:
                # La función ahora se llama desde aquí y tiene manejo de errores
                inventory_list = firebase_utils.get_inventory()
                st.session_state.inventory_list = inventory_list
            except Exception as e:
                st.error("⚠️ Error al Cargar Inventario.")
                st.warning("Revisa tus secretos y las reglas de seguridad de Firestore.")
                st.code(f"Detalles: {e}", language="bash")
                st.session_state.inventory_list = [] # Evita que la app se rompa

    inventory_list = st.session_state.get('inventory_list', [])
    inventory_names = [item['name'] for item in inventory_list if 'name' in item]

    with st.expander("➕ Añadir Nuevo Artículo", expanded=True):
        new_item_name = st.text_input("Nombre del artículo", key="new_item", placeholder="Ej: Laptop, Taza, Silla")
        if st.button("Guardar en Firebase", use_container_width=True):
            if new_item_name and new_item_name.strip() and new_item_name not in inventory_names:
                firebase_utils.add_item(new_item_name.strip())
                st.success(f"'{new_item_name}' añadido.")
                # Forzar la recarga de la lista de inventario en la siguiente ejecución
                if 'inventory_list' in st.session_state:
                    del st.session_state['inventory_list']
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
        
        # Optimización de imagen
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

