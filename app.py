import streamlit as st
import numpy as np
import cv2
from PIL import Image

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Inventario de Bodega",
    page_icon="📦",
    layout="wide"
)

# --- TÍTULO Y DESCRIPCIÓN ---
st.title("📦 Sistema de Inventario de Bodega")
st.write(
    "Utiliza la cámara para capturar una imagen de la estantería. "
    "El sistema detectará los productos, los contará y mostrará su ubicación."
)

# --- FUNCIÓN DE DETECCIÓN SIMULADA ---
def detectar_productos_simulado(imagen_np):
    """
    Esta es una función SIMULADA de detección de objetos.
    En un caso real, aquí es donde llamarías a tu modelo de IA (ej. YOLO, TensorFlow, PyTorch).

    Retorna una lista de diccionarios, donde cada diccionario representa un producto detectado.
    """
    # Coordenadas y etiquetas de los productos "detectados"
    productos_fijos = [
        {"label": "Caja", "box": (50, 70, 250, 300), "confidence": 0.92},
        {"label": "Botella", "box": (300, 150, 400, 400), "confidence": 0.88},
        {"label": "Caja", "box": (450, 100, 600, 280), "confidence": 0.95},
        {"label": "Lata", "box": (80, 320, 180, 450), "confidence": 0.76},
    ]
    return productos_fijos

# --- FUNCIÓN PARA DIBUJAR EN LA IMAGEN ---
def dibujar_detecciones(imagen, detecciones, producto_buscado=""):
    """
    Dibuja los cuadros delimitadores y las etiquetas sobre la imagen.
    """
    img_con_dibujos = np.array(imagen).copy()
    for det in detecciones:
        label = det["label"]
        box = det["box"]
        x1, y1, x2, y2 = box

        # Color por defecto (verde)
        color = (36, 255, 12)
        
        # Si hay una búsqueda, resalta el producto buscado en otro color (amarillo)
        if producto_buscado.lower() in label.lower():
            color = (255, 255, 0)

        # Dibujar el rectángulo
        cv2.rectangle(img_con_dibujos, (x1, y1), (x2, y2), color, 2)
        
        # Dibujar la etiqueta
        texto = f"{label}: {det['confidence']:.2f}"
        cv2.putText(img_con_dibujos, texto, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    return img_con_dibujos

# --- ESTRUCTURA DE LA INTERFAZ ---
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📷 Captura de la Cámara")
    img_buffer = st.camera_input("Toma una foto", key="camera")

    if img_buffer:
        # Convertir el buffer de imagen a un objeto de imagen PIL
        img_pil = Image.open(img_buffer)
        
        # Convertir la imagen PIL a un array de NumPy para OpenCV
        img_np = np.array(img_pil)

        # --- LÓGICA PRINCIPAL ---
        # 1. Detectar productos (usando la función simulada)
        productos_detectados = detectar_productos_simulado(img_np)
        
        # 2. Obtener el producto buscado en la otra columna
        # (Se usa session_state para acceder al valor desde aquí)
        producto_a_buscar = st.session_state.get("search_query", "")

        # 3. Dibujar las detecciones en la imagen
        imagen_resultado = dibujar_detecciones(img_np, productos_detectados, producto_a_buscar)

        # 4. Mostrar la imagen con las detecciones
        st.image(imagen_resultado, caption="Imagen Procesada", use_column_width=True)

with col2:
    st.header("📊 Análisis y Búsqueda")
    
    # Cuadro de texto para buscar producto
    buscar_producto = st.text_input("Buscar Producto", key="search_query", placeholder="Ej: Caja, Botella...")

    st.subheader("Descripción")
    
    # Lógica de descripción
    if img_buffer:
        # Contar productos
        conteo_productos = {}
        for producto in productos_detectados:
            label = producto["label"]
            conteo_productos[label] = conteo_productos.get(label, 0) + 1
        
        # Si no hay búsqueda, mostrar el resumen general
        if not buscar_producto:
            st.info("Resumen de inventario detectado en la imagen.")
            for nombre, cantidad in conteo_productos.items():
                st.markdown(f"- **{nombre}**: {cantidad} unidad(es)")
        else:
            # Si hay búsqueda, mostrar detalles del producto buscado
            resultados_busqueda = [p for p in productos_detectados if buscar_producto.lower() in p["label"].lower()]
            
            if resultados_busqueda:
                cantidad = len(resultados_busqueda)
                st.success(f"Se encontraron {cantidad} unidad(es) de '{buscar_producto}'.")
                
                st.markdown("#### Ubicaciones:")
                for i, res in enumerate(resultados_busqueda):
                    x1, y1, x2, y2 = res["box"]
                    st.markdown(f"- **Item {i+1}**: Ubicado en el área de coordenadas ({x1}, {y1}) a ({x2}, {y2}).")
            else:
                st.warning(f"No se encontró ningún producto con el nombre '{buscar_producto}'.")
    else:
        st.info("Esperando una imagen para analizar...")

# Botón para refrescar la página (opcional, ya que Streamlit es reactivo)
if st.button("🔄 Actualizar"):
    st.rerun()
