import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import os
import json
import base64

# Usamos st.cache_resource para asegurar que la inicialización ocurra una sola vez.
@st.cache_resource
def initialize_firebase():
    """
    Inicializa la conexión con Firebase de forma segura y eficiente.
    Esta función se ejecuta solo una vez gracias al cache de Streamlit.
    """
    try:
        # MÉTODO PARA STREAMLIT CLOUD (lee el secreto Base64)
        if "FIREBASE_SERVICE_ACCOUNT_BASE64" in os.environ:
            base64_creds = os.environ["FIREBASE_SERVICE_ACCOUNT_BASE64"]
            # Aseguramos que el padding sea correcto para base64
            missing_padding = len(base64_creds) % 4
            if missing_padding:
                base64_creds += '=' * (4 - missing_padding)
            
            json_creds_str = base64.b64decode(base64_creds).decode('utf-8')
            creds_dict = json.loads(json_creds_str)
            cred = credentials.Certificate(creds_dict)
        
        # MÉTODO LOCAL (usa el archivo .json)
        else:
            cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "serviceAccountKey.json")
            if not os.path.exists(cred_path):
                raise FileNotFoundError(f"No se encontró el archivo de credenciales: {cred_path}")
            cred = credentials.Certificate(cred_path)
        
        # Inicializar la app solo si no existe ya una
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
            
        print("Firebase App inicializada exitosamente.")

    except Exception as e:
        # Este error se mostrará en la app de Streamlit si algo sale mal
        st.error(f"Error fatal al inicializar Firebase: {e}")
        # Detenemos la app si Firebase no puede conectar.
        st.stop()
        raise e # También lo lanzamos para que se vea en los logs

def get_db():
    """Retorna una instancia del cliente de Firestore."""
    # Asegurarse de que Firebase está inicializado antes de obtener el cliente
    if not firebase_admin._apps:
        initialize_firebase()
    return firestore.client()

def get_inventory():
    """
    Obtiene todos los artículos de la colección 'inventory' en Firestore.
    """
    db = get_db()
    inventory_ref = db.collection('inventory')
    docs = inventory_ref.stream()
    inventory_list = [{"id": doc.id, "name": doc.to_dict().get('name')} for doc in docs]
    return inventory_list

def add_item(item_name):
    """
    Añade un nuevo artículo a la colección 'inventory'.
    """
    db = get_db()
    inventory_ref = db.collection('inventory')
    inventory_ref.add({'name': item_name})

