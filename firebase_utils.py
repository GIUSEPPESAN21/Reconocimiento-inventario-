import firebase_admin
from firebase_admin import credentials, firestore
import os
import json # Importante: se necesita para leer el texto del JSON
from dotenv import load_dotenv

load_dotenv()

def initialize_firebase():
    """
    Inicializa Firebase de forma inteligente.
    Busca el secreto en Streamlit Cloud primero, y si no lo encuentra,
    usa el archivo local serviceAccountKey.json.
    """
    try:
        if not firebase_admin._apps:
            # MÉTODO PARA STREAMLIT CLOUD (lee el secreto directamente)
            if "FIREBASE_SERVICE_ACCOUNT_JSON" in os.environ:
                creds_json_str = os.environ["FIREBASE_SERVICE_ACCOUNT_JSON"]
                creds_dict = json.loads(creds_json_str)
                cred = credentials.Certificate(creds_dict)
            
            # MÉTODO LOCAL (usa el archivo .json)
            else:
                cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
                if not cred_path or not os.path.exists(cred_path):
                    raise FileNotFoundError("No se encontró 'serviceAccountKey.json'. Asegúrate de que el archivo existe o que el secreto FIREBASE_SERVICE_ACCOUNT_JSON está configurado en Streamlit Cloud.")
                cred = credentials.Certificate(cred_path)
            
            firebase_admin.initialize_app(cred)
            
        return firestore.client()
    except Exception as e:
        print(f"Error fatal al inicializar Firebase: {e}")
        raise

def get_inventory():
    """
    Obtiene todos los artículos de la colección 'inventory' en Firestore.
    """
    db = firestore.client()
    inventory_ref = db.collection('inventory')
    docs = inventory_ref.stream()
    inventory_list = [{"id": doc.id, "name": doc.to_dict().get('name')} for doc in docs]
    return inventory_list

def add_item(item_name):
    """
    Añade un nuevo artículo a la colección 'inventory'.
    """
    db = firestore.client()
    inventory_ref = db.collection('inventory')
    inventory_ref.add({'name': item_name})
