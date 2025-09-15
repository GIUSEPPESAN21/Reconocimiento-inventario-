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
    Inicializa la conexión con Firebase de forma segura. Se ejecuta solo una vez.
    """
    try:
        # Lee las credenciales desde los secretos de Streamlit
        creds_json_str = st.secrets["FIREBASE_SERVICE_ACCOUNT_BASE64"]
        creds_b64 = base64.b64decode(creds_json_str)
        creds_dict = json.loads(creds_b64)
        
        cred = credentials.Certificate(creds_dict)
        
        # Inicializar la app solo si no existe
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
            
    except Exception as e:
        # Lanza una excepción clara si falla, que será capturada en la app principal
        raise RuntimeError(f"Fallo en la inicialización de Firebase: {e}")

def get_db():
    """Retorna una instancia del cliente de Firestore, asegurando la inicialización."""
    if not firebase_admin._apps:
        initialize_firebase()
    return firestore.client()

def get_inventory():
    """Obtiene todos los artículos de la colección 'inventory'."""
    db = get_db()
    inventory_ref = db.collection('inventory')
    docs = inventory_ref.stream() # Esta es la llamada que puede ser lenta
    return [{"id": doc.id, "name": doc.to_dict().get('name')} for doc in docs]

def add_item(item_name):
    """Añade un nuevo artículo a la colección 'inventory'."""
    db = get_db()
    db.collection('inventory').add({'name': item_name})

