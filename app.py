import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- SIMULACIÓN DEL MODELO DE IA ---
# En un caso real, esta función usaría un modelo de deep learning (como YOLO, SSD, etc.)
# para detectar objetos en la imagen. Por ahora, devolvemos datos de ejemplo.
def detectar_productos(imagen_np):
    """
    Esta función simula la detección de productos en una imagen.
    Devuelve una lista de diccionarios, donde cada uno representa un producto.
    """
    # Coordenadas de los cuadros delimitadores (bounding boxes) en formato:
    # [x_inicial, y_inicial, x_final, y_final]
    # Estas coordenadas serían la salida de tu modelo de IA.
    productos_detectados = [
        {"etiqueta": "Caja de Tornillos", "confianza": 0.95, "caja": [50, 70, 250, 220]},
        {"etiqueta": "Botella de Agua", "confianza": 0.89, "caja": [300, 150, 400, 350]},
        {"etiqueta": "Lata de Pintura", "confianza": 0.92, "caja": [450, 100, 600, 280]},
        {"etiqueta": "Caja de Tornillos", "confianza": 0.91, "caja": [60, 250, 260, 400]},
    ]
    return productos_detectados

# --- FUNCIONES AUXILIARES DE DIBUJO ---
def dibujar_cajas(imagen, detecciones, producto_buscado=""):
    """
    Dibuja los cuadros delimitadores y etiquetas sobre la imagen.
    Resalta el producto que se está buscando.
    """
    imagen_con_cajas = np.array(imagen).copy()
    color_normal = (0, 255, 0) # Verde para cajas normales
    color_resaltado = (255, 0, 0) # Rojo para el producto buscado
    
    for producto in detecciones:
        caja = producto["caja"]
        etiqueta = producto["etiqueta"]
        
        # Determinar el color de la caja
        color = color_resaltado if producto_buscado.lower() in etiqueta.lower() and producto_buscado != "" else color_normal
        
        # Dibujar el rectángulo
        cv2.rectangle(imagen_con_cajas, (caja[0], caja[1]), (caja[2], caja[3]), color, 2)
        
        # Dibujar la etiqueta con fondo
        texto = f"{etiqueta}"
        (ancho_texto, alto_texto), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        cv2.rectangle(imagen_con_cajas, (caja[0], caja[1] - alto_texto - 10), (caja[0] + ancho_texto, caja[1]), color, -1)
        cv2.putText(imagen_con_cajas, texto, (caja[0], caja[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
    return imagen_con_cajas

# --- INTERFAZ DE STREAMLIT ---
st.set_page_config(layout="wide", page_title="Inventario por Imagen")

st.title("Sistema de Inventario por Reconocimiento de Imagen")

# Dividir la pantalla en dos columnas
col1, col2 = st.columns([2, 1])

# --- Columna 1: Visualización de la imagen ---
with col1:
    st.header("Cámara de la Bodega")
    
    # Opción para subir una imagen (o usar la cámara)
    # Nota: st.camera_input es ideal para producción, pero st.file_uploader es mejor para probar.
    imagen_subida = st.file_uploader("Sube una imagen de la bodega", type=["jpg", "jpeg", "png"])
    
    imagen_placeholder = st.empty()

# --- Columna 2: Controles y Descripción ---
with col2:
    st.header("Análisis de Inventario")
    
    producto_a_buscar = st.text_input("Buscar Producto")
    
    if st.button("Actualizar"):
        # En una aplicación real, este botón podría forzar una nueva captura de la cámara
        st.experimental_rerun()

    descripcion_placeholder = st.empty()


# --- LÓGICA PRINCIPAL ---
if imagen_subida is not None:
    # Cargar la imagen usando Pillow y convertirla a un formato que OpenCV pueda usar
    imagen_pil = Image.open(imagen_subida)
    imagen_np = np.array(imagen_pil.convert('RGB'))
    
    # 1. Simular la detección de productos
    detecciones = detectar_productos(imagen_np)
    
    # 2. Dibujar las cajas en la imagen
    imagen_procesada = dibujar_cajas(imagen_np, detecciones, producto_a_buscar)
    
    # 3. Mostrar la imagen procesada en el placeholder
    imagen_placeholder.image(imagen_procesada, caption="Productos detectados en la bodega", use_column_width=True)
    
    # 4. Generar la descripción
    texto_descripcion = ""
    if producto_a_buscar:
        # Filtrar detecciones para el producto buscado
        productos_encontrados = [p for p in detecciones if producto_a_buscar.lower() in p["etiqueta"].lower()]
        cantidad = len(productos_encontrados)
        
        texto_descripcion += f"Buscando: '{producto_a_buscar}'\n"
        texto_descripcion += f"Cantidad encontrada: {cantidad}\n\n"
        if cantidad > 0:
            texto_descripcion += "Ubicaciones (coordenadas x,y):\n"
            for i, p in enumerate(productos_encontrados):
                texto_descripcion += f"- Producto {i+1}: ({p['caja'][0]}, {p['caja'][1]})\n"
    else:
        # Si no hay búsqueda, mostrar un resumen general
        resumen = {}
        for p in detecciones:
            etiqueta = p["etiqueta"]
            resumen[etiqueta] = resumen.get(etiqueta, 0) + 1
            
        texto_descripcion = "Resumen de Inventario:\n"
        for nombre, cant in resumen.items():
            texto_descripcion += f"- {nombre}: {cant} unidad(es)\n"

    descripcion_placeholder.text_area("Descripción", value=texto_descripcion, height=300)

else:
    # Mostrar una imagen de ejemplo si no se ha subido ninguna
    imagen_placeholder.image("https://placehold.co/1200x800/e2e8f0/64748b?text=Esperando+imagen+de+la+bodega...", use_column_width=True)
    descripcion_placeholder.text_area("Descripción", "Sube una imagen para analizar el inventario.", height=300)
