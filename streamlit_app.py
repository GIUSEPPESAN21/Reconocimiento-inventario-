import streamlit as st
import logging
from gemini_utils import GeminiUtils, get_gemini_response
from firebase_utils import FirebaseUtils
from PIL import Image
import io
import pandas as pd

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    st.set_page_config(
        page_title="Sistema de Reconocimiento de Inventario",
        page_icon="📦",
        layout="wide"
    )
    
    st.title("🤖 Sistema de Reconocimiento de Inventario con IA")
    st.markdown("---")
    
    # Verificar que los secrets estén configurados
    try:
        # Verificar GEMINI_API_KEY
        gemini_key = st.secrets["GEMINI_API_KEY"]
        if not gemini_key:
            st.error("❌ GEMINI_API_KEY no configurada en Streamlit secrets")
            return
        
        # Verificar FIREBASE_SERVICE_ACCOUNT_BASE64
        firebase_secret = st.secrets["FIREBASE_SERVICE_ACCOUNT_BASE64"]
        if not firebase_secret:
            st.error("❌ FIREBASE_SERVICE_ACCOUNT_BASE64 no configurada en Streamlit secrets")
            return
            
    except KeyError as e:
        st.error(f"❌ Secret no encontrado: {str(e)}")
        st.info("💡 Configura tus secrets en la sección 'Secrets' de tu aplicación Streamlit")
        return
    
    # Inicializar Firebase
    try:
        firebase = FirebaseUtils()
        st.success("✅ Conexión a Firebase establecida")
    except Exception as e:
        st.error(f"❌ Error conectando a Firebase: {str(e)}")
        return
    
    # Inicializar Gemini
    try:
        gemini = GeminiUtils()
        model_info = gemini.get_model_info()
        st.success(f"✅ Gemini AI inicializado - Modelo: {model_info['current_model']}")
        
        # Mostrar información del modelo en sidebar
        with st.sidebar:
            st.header("🔧 Información del Sistema")
            st.write(f"**Modelo actual:** {model_info['current_model']}")
            st.write(f"**Modelos disponibles:** {len(model_info['available_models'])}")
            st.write(f"**Proyecto Firebase:** {firebase.project_id}")
            
    except Exception as e:
        st.error(f"❌ Error inicializando Gemini AI: {str(e)}")
        return
    
    # Crear pestañas
    tab1, tab2, tab3, tab4 = st.tabs(["📷 Reconocimiento", "📊 Inventario", "📈 Estadísticas", "⚙️ Configuración"])
    
    with tab1:
        st.header("Reconocimiento de Elementos")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Subir Imagen")
            uploaded_file = st.file_uploader(
                "Selecciona una imagen del inventario",
                type=['png', 'jpg', 'jpeg'],
                help="Sube una imagen clara del elemento que deseas analizar"
            )
            
            if uploaded_file is not None:
                # Mostrar imagen
                image = Image.open(uploaded_file)
                st.image(image, caption="Imagen subida", use_column_width=True)
                
                # Convertir a bytes para Gemini
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()
        
        with col2:
            st.subheader("Descripción del Elemento")
            description = st.text_area(
                "Describe brevemente el elemento (opcional)",
                placeholder="Ej: Laptop Dell, Silla de oficina, etc.",
                height=100
            )
            
            if st.button("🔍 Analizar Elemento", type="primary"):
                if uploaded_file is not None:
                    with st.spinner("Analizando elemento con IA..."):
                        try:
                            # Analizar con Gemini
                            result = gemini.analyze_inventory_item(
                                description or "Elemento del inventario",
                                img_byte_arr
                            )
                            
                            if result["success"]:
                                st.success("✅ Análisis completado")
                                
                                # Mostrar resultados
                                st.subheader("📋 Resultados del Análisis")
                                
                                try:
                                    # Intentar parsear como JSON
                                    import json
                                    analysis_data = json.loads(result["response"])
                                    
                                    # Mostrar en formato organizado
                                    for key, value in analysis_data.items():
                                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                                        
                                except json.JSONDecodeError:
                                    # Si no es JSON, mostrar como texto
                                    st.write(result["response"])
                                
                                # Botón para guardar en Firebase
                                if st.button("💾 Guardar en Inventario"):
                                    try:
                                        # Preparar datos para Firebase
                                        inventory_data = {
                                            "description": description or "Elemento analizado",
                                            "analysis": result["response"],
                                            "image_url": "uploaded_image",  # Aquí podrías subir la imagen a Firebase Storage
                                            "timestamp": firebase.get_timestamp(),
                                            "model_used": result.get("model_used", "unknown")
                                        }
                                        
                                        # Guardar en Firebase
                                        doc_ref = firebase.add_inventory_item(inventory_data)
                                        st.success(f"✅ Elemento guardado con ID: {doc_ref.id}")
                                        
                                    except Exception as e:
                                        st.error(f"❌ Error guardando en Firebase: {str(e)}")
                            else:
                                st.error(f"❌ Error en el análisis: {result['error']}")
                                
                        except Exception as e:
                            st.error(f"❌ Error procesando imagen: {str(e)}")
                            logger.error(f"Error procesando imagen: {str(e)}")
                else:
                    st.warning("⚠️ Por favor, sube una imagen primero")
    
    with tab2:
        st.header("Gestión de Inventario")
        
        try:
            # Obtener elementos del inventario
            inventory_items = firebase.get_inventory_items()
            
            if inventory_items:
                st.write(f"📦 Total de elementos: {len(inventory_items)}")
                
                # Buscar elementos
                search_term = st.text_input("🔍 Buscar elemento")
                
                if search_term:
                    filtered_items = [
                        item for item in inventory_items 
                        if search_term.lower() in item.get("description", "").lower()
                    ]
                else:
                    filtered_items = inventory_items
                
                # Mostrar elementos
                for i, item in enumerate(filtered_items):
                    with st.expander(f"📦 {item.get('description', 'Sin descripción')}"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.write(f"**ID:** {item.id}")
                            st.write(f"**Fecha:** {item.get('timestamp', 'N/A')}")
                            st.write(f"**Modelo IA:** {item.get('model_used', 'N/A')}")
                            
                            # Mostrar análisis
                            analysis = item.get('analysis', '')
                            if analysis:
                                st.write("**Análisis:**")
                                st.text_area("", value=analysis, height=100, disabled=True, key=f"analysis_{i}")
                        
                        with col2:
                            if st.button("🗑️ Eliminar", key=f"delete_{i}"):
                                try:
                                    firebase.delete_inventory_item(item.id)
                                    st.success("✅ Elemento eliminado")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Error eliminando: {str(e)}")
            else:
                st.info("📭 No hay elementos en el inventario")
                
        except Exception as e:
            st.error(f"❌ Error cargando inventario: {str(e)}")
    
    with tab3:
        st.header("Estadísticas del Sistema")
        
        try:
            inventory_items = firebase.get_inventory_items()
            
            if inventory_items:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Elementos", len(inventory_items))
                
                with col2:
                    # Contar modelos usados
                    models_used = {}
                    for item in inventory_items:
                        model = item.get('model_used', 'unknown')
                        models_used[model] = models_used.get(model, 0) + 1
                    
                    most_used_model = max(models_used, key=models_used.get) if models_used else "N/A"
                    st.metric("Modelo Más Usado", most_used_model)
                
                with col3:
                    st.metric("Último Análisis", "Hoy" if inventory_items else "N/A")
                
                # Gráfico de modelos usados
                if len(models_used) > 1:
                    st.subheader("📊 Distribución de Modelos")
                    st.bar_chart(models_used)
            else:
                st.info("📊 No hay datos suficientes para mostrar estadísticas")
                
        except Exception as e:
            st.error(f"❌ Error cargando estadísticas: {str(e)}")
    
    with tab4:
        st.header("Configuración del Sistema")
        
        # Información del modelo
        st.subheader("🤖 Configuración de Gemini AI")
        try:
            model_info = gemini.get_model_info()
            
            st.write(f"**Modelo actual:** {model_info['current_model']}")
            st.write("**Modelos disponibles:**")
            for i, model in enumerate(model_info['available_models']):
                status = "✅" if model == model_info['current_model'] else "⏳"
                st.write(f"{status} {model}")
            
            # Configuración actual
            st.write("**Configuración actual:**")
            config_df = pd.DataFrame([model_info['config']])
            st.dataframe(config_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error obteniendo información del modelo: {str(e)}")
        
        # Variables de entorno
        st.subheader("🔧 Streamlit Secrets")
        secrets_status = {
            "GEMINI_API_KEY": "✅ Configurada" if st.secrets.get("GEMINI_API_KEY") else "❌ No configurada",
            "FIREBASE_SERVICE_ACCOUNT_BASE64": "✅ Configurada" if st.secrets.get("FIREBASE_SERVICE_ACCOUNT_BASE64") else "❌ No configurada"
        }
        
        for secret, status in secrets_status.items():
            st.write(f"**{secret}:** {status}")
        
        # Información del proyecto Firebase
        st.subheader("🔥 Información de Firebase")
        try:
            st.write(f"**Proyecto ID:** {firebase.project_id}")
            st.write("**Estado de conexión:** ✅ Conectado")
        except Exception as e:
            st.write(f"**Estado de conexión:** ❌ Error: {str(e)}")
        
        # Botón para probar conexiones
        if st.button("🧪 Probar Conexiones"):
            with st.spinner("Probando conexiones..."):
                # Probar Gemini
                try:
                    test_response = gemini.generate_content("Hola, esto es una prueba")
                    if test_response["success"]:
                        st.success("✅ Gemini AI funcionando correctamente")
                    else:
                        st.error(f"❌ Error en Gemini: {test_response['error']}")
                except Exception as e:
                    st.error(f"❌ Error probando Gemini: {str(e)}")
                
                # Probar Firebase
                try:
                    test_items = firebase.get_inventory_items()
                    st.success(f"✅ Firebase conectado - {len(test_items)} elementos")
                except Exception as e:
                    st.error(f"❌ Error probando Firebase: {str(e)}")

if __name__ == "__main__":
    main()
