import streamlit as st
from PIL import Image
import firebase_utils
import gemini_utils
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Inventario Inteligente",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- INICIALIZACIÓN ROBUSTA DE FIREBASE ---
# Se llama a la función de inicialización al principio para asegurar la conexión.
# Gracias a @st.cache_resource en firebase_utils, esto solo se ejecuta una vez.
try:
    firebase_utils.initialize_firebase()
except Exception as e:
    st.error(f"Error crítico al iniciar la conexión con la base de datos: {e}")
    st.stop()


# --- TITLE AND DESCRIPTION ---
st.title("📦 Inventario Inteligente con IA")
st.markdown("Identifica artículos de tu inventario en tiempo real con la cámara, usando **Gemini AI** y **Firebase**.")

# --- INTERFACE STRUCTURE ---
col1, col2 = st.columns([2, 1])

# --- Column 2: Control Panel (Inventory and Results) ---
with col2:
    st.header("📊 Panel de Control")

    try:
        # Usamos cache_data para la función que lee los datos.
        @st.cache_data(ttl=60)
        def get_inventory_cached():
            return firebase_utils.get_inventory()

        inventory_list = get_inventory_cached()
        inventory_names = [item['name'] for item in inventory_list]

    except Exception as e:
        st.error(f"Error al cargar inventario desde Firebase: {e}")
        inventory_names = []

    with st.expander("➕ Añadir Nuevo Artículo", expanded=True):
        new_item_name = st.text_input("Nombre del artículo", key="new_item", placeholder="Ej: Laptop, Taza, Silla")
        if st.button("Guardar en Firebase", use_container_width=True):
            if new_item_name and new_item_name.strip() and new_item_name not in inventory_names:
                firebase_utils.add_item(new_item_name.strip())
                st.success(f"'{new_item_name}' añadido.")
                st.cache_data.clear()
                st.rerun()
            else:
                st.warning("El nombre no puede estar vacío o ya existe.")

    st.subheader("📋 Inventario Actual en Firebase")
    if inventory_names:
        st.dataframe(inventory_names, use_container_width=True, column_config={"value": "Artículo"})
    else:
        st.info("Tu inventario está vacío. Añade un artículo para empezar.")
    
    st.subheader("🤖 Resultado del Análisis")
    result_placeholder = st.empty()
    if 'last_result' in st.session_state:
        result_placeholder.success(f"**Artículo Identificado:** {st.session_state.last_result}")
    else:
        result_placeholder.info("Esperando análisis de imagen...")

# --- Column 1: Camera and Visualization ---
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
            img_pil_resized = img_pil.resize((max_width, new_height))
        else:
            img_pil_resized = img_pil

        st.image(img_pil, caption="Imagen lista para analizar", use_column_width=True)

        if st.button("✨ Analizar con Gemini", type="primary", use_container_width=True):
            if not inventory_names:
                st.warning("Añade al menos un artículo a tu inventario antes de analizar.")
            else:
                with st.spinner("🧠 Gemini está identificando el artículo..."):
                    result = gemini_utils.identify_item(img_pil_resized, inventory_names)
                    st.session_state.last_result = result
                    st.rerun()

