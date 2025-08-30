import firebase_admin
from firebase_admin import credentials, firestore
import os
import json
import base64 # Importante: se necesita para decodificar
from dotenv import load_dotenv

load_dotenv()

def initialize_firebase():
    """
    Inicializa Firebase de forma robusta usando Base64 para las credenciales
    en el despliegue de Streamlit Cloud.
    """
    try:
        if not firebase_admin._apps:
            # MÉTODO PARA STREAMLIT CLOUD (lee el secreto Base64)
            if "FIREBASE_SERVICE_ACCOUNT_BASE64" in os.environ:
                # Decodifica la cadena Base64 a un string JSON
                base64_creds = os.environ["FIREBASE_SERVICE_ACCOUNT_BASE64"]
                json_creds_str = base64.b64decode(base64_creds).decode('utf-8')
                creds_dict = json.loads(json_creds_str)
                
                cred = credentials.Certificate(creds_dict)
            
            # MÉTODO LOCAL (usa el archivo .json)
            else:
                cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
                if not cred_path or not os.path.exists(cred_path):
                    raise FileNotFoundError("No se encontró 'serviceAccountKey.json'. Asegúrate de que el archivo existe o que el secreto FIREBASE_SERVICE_ACCOUNT_BASE64 está configurado.")
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

