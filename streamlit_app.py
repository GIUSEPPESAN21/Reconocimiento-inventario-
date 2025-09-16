import streamlit as st
from PIL import Image
import firebase_utils

# --- Page Configuration ---
st.set_page_config(
    page_title="Inventario Inteligente",
    page_icon="📦",
    layout="wide"
)

# --- Firebase Initialization ---
# This will run once and show a clear error if secrets are wrong.
try:
    firebase_utils.initialize_firebase()
except Exception as e:
    st.error(f"**Error Crítico de Conexión:** No se pudo inicializar Firebase.")
    st.error("Por favor, revisa tus secretos en Streamlit Cloud, especialmente `FIREBASE_SERVICE_ACCOUNT_BASE64`.")
    st.code(f"Detalle del error: {e}", language="bash")
    st.stop()

# --- Title and Description ---
st.title("📦 Inventario Inteligente con IA")
st.markdown("Identifica artículos de tu inventario en tiempo real con la cámara.")

# --- Layout ---
col1, col2 = st.columns([2, 1])

# --- Control Panel (Right Column) ---
with col2:
    st.header("📊 Panel de Control")
    # Spinner appears only in this section while loading data
    inventory_placeholder = st.empty()
    with inventory_placeholder, st.spinner("Cargando inventario..."):
        try:
            # Load inventory and save it to the session state
            inventory_list = firebase_utils.get_inventory()
            st.session_state.inventory_list = inventory_list
        except Exception as e:
            st.error(f"⚠️ Error al Cargar Inventario: {e}")
            st.session_state.inventory_list = []

    inventory_names = [item.get('name') for item in st.session_state.get('inventory_list', [])]

    with st.expander("➕ Añadir Nuevo Artículo", expanded=True):
        new_item_name = st.text_input("Nombre del artículo", key="new_item")
        if st.button("Guardar", use_container_width=True):
            if new_item_name and new_item_name.strip() and new_item_name not in inventory_names:
                firebase_utils.add_item(new_item_name.strip())
                st.success(f"'{new_item_name}' añadido.")
                # Clear inventory from state to force a reload
                if 'inventory_list' in st.session_state:
                    del st.session_state['inventory_list']
                st.rerun()
            else:
                st.warning("El nombre no puede estar vacío o ya existe.")

    st.subheader("📋 Inventario Actual")
    if inventory_names:
        st.dataframe(inventory_names, use_container_width=True, column_config={"value": "Artículo"})
    else:
        st.info("Inventario vacío o no se pudo cargar.")

    st.subheader("🤖 Resultado")
    if 'last_result' in st.session_state:
        st.success(f"**Identificado:** {st.session_state.last_result}")
    else:
        st.info("Esperando análisis...")

# --- Camera and Analysis (Left Column) ---
with col1:
    st.header("📷 Captura y Análisis")
    img_buffer = st.camera_input("Apunta al artículo y toma una foto", key="camera")

    if img_buffer:
        img_pil = Image.open(img_buffer)
        # Image optimization before sending
        max_width = 512
        if img_pil.width > max_width:
            h = int((max_width / img_pil.width) * img_pil.height)
            img_pil = img_pil.resize((max_width, h))
        st.image(img_pil, caption="Imagen a analizar")

        if st.button("✨ Analizar", type="primary", use_container_width=True):
            if not inventory_names:
                st.warning("Añade artículos al inventario primero.")
            else:
                with st.spinner("🧠 Identificando..."):
                    import gemini_utils
                    result = gemini_utils.identify_item(img_pil, inventory_names)
                    st.session_state.last_result = result
                    st.rerun()

