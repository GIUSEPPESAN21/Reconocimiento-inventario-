import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import json
import base64

@st.cache_resource
def initialize_firebase():
    """
    Inicializa Firebase usando los secretos de Streamlit.
    Ahora corrige automáticamente los errores de padding en Base64.
    """
    try:
        creds_b64 = st.secrets["FIREBASE_SERVICE_ACCOUNT_BASE64"]
        
        # --- SOLUCIÓN AL ERROR "INCORRECT PADDING" ---
        # Esta sección revisa si a la clave le faltan caracteres de relleno y los añade.
        missing_padding = len(creds_b64) % 4
        if missing_padding:
            creds_b64 += '=' * (4 - missing_padding)
        # --- FIN DE LA SOLUCIÓN ---

        creds_json = base64.b64decode(creds_b64).decode("utf-8")
        creds_dict = json.loads(creds_json)
        
        cred = credentials.Certificate(creds_dict)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
            
    except Exception as e:
        raise RuntimeError(f"Fallo en la inicialización de Firebase: {e}")

def get_db():
    """Retorna una instancia del cliente de Firestore."""
    if not firebase_admin._apps:
        initialize_firebase()
    return firestore.client()

def get_inventory():
    """Obtiene la lista de artículos del inventario."""
    db = get_db()
    docs = db.collection('inventory').stream()
    return [{"id": doc.id, "name": doc.to_dict().get('name')} for doc in docs]

def add_item(item_name):
    """Añade un nuevo artículo al inventario."""
    db = get_db()
    db.collection('inventory').add({'name': item_name})

