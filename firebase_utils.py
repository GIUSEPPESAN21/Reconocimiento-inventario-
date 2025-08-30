import firebase_admin
from firebase_admin import credentials, firestore
import os
from dotenv import load_dotenv

# Cargar variables de entorno del archivo .env
load_dotenv()

def initialize_firebase():
    """
    Inicializa la app de Firebase usando las credenciales.
    Evita la reinicialización si ya existe una instancia.
    """
    try:
        if not firebase_admin._apps:
            cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if not os.path.exists(cred_path):
                raise FileNotFoundError(f"El archivo de credenciales no se encuentra en la ruta: {cred_path}")
            
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
