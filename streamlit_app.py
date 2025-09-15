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

# --- INICIALIZACIÓN DE FIREBASE (rápida y una sola vez) ---
# Esto asegura que la conexión esté lista.
try:
    db = firebase_utils.get_db()
    st.success("✅ Conexión con Firebase establecida.")
except Exception as e:
    st.error(f"**Error de Conexión.** No se pudo conectar con Firebase. Revisa los secretos. Detalle: {e}")
    st.stop()

# --- TÍTULO Y DESCRIPCIÓN (Se muestra al instante) ---
st.title("📦 Inventario Inteligente con IA")
st.markdown("Identifica artículos de tu inventario en tiempo real con la cámara, usando **Gemini AI** y **Firebase**.")

# --- ESTRUCTURA DE LA INTERFAZ ---
col1, col2 = st.columns([2, 1])

# --- Columna 2: Panel de Control ---
with col2:
    st.header("📊 Panel de Control")

    # --- CARGA DEL INVENTARIO EN SEGUNDO PLANO ---
    # Usamos session_state para cargar el inventario solo una vez por sesión.
    if 'inventory_list' not in st.session_state:
        with st.spinner("Cargando inventario desde Firebase..."):
            st.session_state.inventory_list = firebase_utils.get_inventory()

    inventory_list = st.session_state.inventory_list
    inventory_names = [item['name'] for item in inventory_list]

    with st.expander("➕ Añadir Nuevo Artículo", expanded=True):
        new_item_name = st.text_input("Nombre del artículo", key="new_item", placeholder="Ej: Laptop, Taza, Silla")
        if st.button("Guardar en Firebase", use_container_width=True):
            if new_item_name and new_item_name.strip() and new_item_name not in inventory_names:
                firebase_utils.add_item(new_item_name.strip())
                st.success(f"'{new_item_name}' añadido.")
                # Limpiamos el inventario del estado para forzar la recarga
                del st.session_state.inventory_list
                st.rerun()
            else:
                st.warning("El nombre no puede estar vacío o ya existe.")

    st.subheader("📋 Inventario Actual")
    if inventory_names:
        st.dataframe(inventory_names, use_container_width=True, column_config={"value": "Artículo"})
    else:
        st.info("Tu inventario está vacío. Añade un artículo.")

    st.subheader("🤖 Resultado del Análisis")
    result_placeholder = st.empty()
    if 'last_result' in st.session_state:
        result_placeholder.success(f"**Artículo Identificado:** {st.session_state.last_result}")
    else:
        result_placeholder.info("Esperando análisis de imagen...")

# --- Columna 1: Cámara y Visualización (Se muestra al instante) ---
with col1:
    st.header("📷 Captura y Análisis")
    img_buffer = st.camera_input("Apunta al artículo y toma una foto", key="camera")

    if img_buffer:
        img_pil = Image.open(img_buffer)
        
        # Optimización: Reducir tamaño de imagen antes de enviar
        max_width = 512
        if img_pil.width > max_width:
            aspect_ratio = img_pil.height / img_pil.width
            new_height = int(max_width * aspect_ratio)
            img_pil = img_pil.resize((max_width, new_height))

        st.image(img_pil, caption="Imagen lista para analizar", use_column_width=True)

        if st.button("✨ Analizar con Gemini", type="primary", use_container_width=True):
            if not inventory_names:
                st.warning("Añade al menos un artículo a tu inventario antes de analizar.")
            else:
                with st.spinner("🧠 Gemini está identificando el artículo..."):
                    # NOTA: Importamos gemini_utils aquí para una carga más rápida
                    import gemini_utils
                    result = gemini_utils.identify_item(img_pil, inventory_names)
                    st.session_state.last_result = result
                    st.rerun()

