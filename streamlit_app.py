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

# --- INITIALIZATION ---
# Usamos un placeholder mientras se conecta
status_placeholder = st.empty()
with status_placeholder.container():
    with st.spinner("Conectando con la base de datos de Firebase..."):
        try:
            firebase_utils.initialize_firebase()
            status_placeholder.success("✅ Conexión con Firebase exitosa.")
        except Exception as e:
            st.error(f"""
                **Error de Conexión con Firebase.** Asegúrate de que los secretos en Streamlit Cloud estén configurados correctamente.  
                **Detalles del error:** {e}
            """)
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
        # Usamos cache para no llamar a la base de datos en cada re-render
        @st.cache_data(ttl=60)
        def get_inventory_cached():
            return firebase_utils.get_inventory()

        inventory_list = get_inventory_cached()
        inventory_names = [item['name'] for item in inventory_list]
    except Exception as e:
        st.error(f"Error al cargar inventario: {e}")
        inventory_names = []

    with st.expander("➕ Añadir Nuevo Artículo", expanded=True):
        new_item_name = st.text_input("Nombre del artículo", key="new_item", placeholder="Ej: Laptop, Taza, Silla")
        if st.button("Guardar en Firebase", use_container_width=True):
            if new_item_name and new_item_name.strip() and new_item_name not in inventory_names:
                firebase_utils.add_item(new_item_name.strip())
                st.success(f"'{new_item_name}' añadido.")
                st.cache_data.clear() # Limpiamos el cache para que se actualice la lista
                st.rerun()
            else:
                st.warning("El nombre no puede estar vacío o ya existe.")

    st.subheader("📋 Inventario Actual en Firebase")
    if inventory_names:
        st.dataframe(inventory_names, use_container_width=True, column_config={"value": "Artículo"})
    else:
        st.info("Tu inventario está vacío. Añade un artículo o crea la colección 'inventory' en Firebase.")
    
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

        # --- INICIO DE LA MEJORA DE VELOCIDAD ---
        # Reducimos el tamaño de la imagen antes de enviarla a Gemini.
        # Imágenes más pequeñas se procesan más rápido.
        max_width = 512
        if img_pil.width > max_width:
            aspect_ratio = img_pil.height / img_pil.width
            new_height = int(max_width * aspect_ratio)
            img_pil_resized = img_pil.resize((max_width, new_height))
        else:
            img_pil_resized = img_pil
        # --- FIN DE LA MEJORA ---

        st.image(img_pil, caption="Imagen lista para analizar", use_column_width=True)

        if st.button("✨ Analizar con Gemini", type="primary", use_container_width=True):
            if not inventory_names:
                st.warning("Añade al menos un artículo a tu inventario antes de analizar.")
            else:
                with st.spinner("🧠 Gemini está identificando el artículo..."):
                    # Usamos la imagen redimensionada para el análisis
                    result = gemini_utils.identify_item(img_pil_resized, inventory_names)
                    st.session_state.last_result = result
                    st.rerun()
