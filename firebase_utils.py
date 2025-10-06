import firebase_admin
from firebase_admin import credentials, firestore
import json
import base64
import logging
from datetime import datetime
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FirebaseUtils:
    def __init__(self):
        self.db = None
        self.project_id = "reconocimiento-inventario" # Asegúrate que este sea tu Project ID
        self._initialize_firebase()
    
    def _initialize_firebase(self):
        """Inicializa Firebase usando Streamlit secrets."""
        try:
            if not firebase_admin._apps:
                creds_base64 = st.secrets.get('FIREBASE_SERVICE_ACCOUNT_BASE64')
                if not creds_base64:
                    raise ValueError("El secret 'FIREBASE_SERVICE_ACCOUNT_BASE64' no fue encontrado.")
                
                creds_json_str = base64.b64decode(creds_base64).decode('utf-8')
                creds_dict = json.loads(creds_json_str)
                
                cred = credentials.Certificate(creds_dict)
                firebase_admin.initialize_app(cred, {'projectId': self.project_id})
                logger.info("Firebase inicializado correctamente.")
            
            self.db = firestore.client()
        except Exception as e:
            logger.error(f"Error fatal al inicializar Firebase: {e}")
            raise  # Relanza la excepción para que Streamlit la muestre

    def get_timestamp(self):
        """Retorna el timestamp actual en formato ISO."""
        return datetime.now().isoformat()

    def save_inventory_item(self, data):
        """Guarda un nuevo elemento en la colección 'inventario'."""
        try:
            _, doc_ref = self.db.collection('inventario').add(data)
            logger.info(f"Elemento guardado con ID: {doc_ref.id}")
            return doc_ref.id
        except Exception as e:
            logger.error(f"Error al guardar en Firestore: {e}")
            raise

    def get_all_inventory_items(self):
        """Obtiene todos los elementos del inventario, ordenados por fecha."""
        try:
            docs = self.db.collection('inventario').order_by('timestamp', direction=firestore.Query.DESCENDING).stream()
            items = []
            for doc in docs:
                item = doc.to_dict()
                item['id'] = doc.id
                items.append(item)
            return items
        except Exception as e:
            logger.error(f"Error al obtener datos de Firestore: {e}")
            return []

    def delete_inventory_item(self, doc_id):
        """Elimina un elemento por su ID de documento."""
        try:
            self.db.collection('inventario').document(doc_id).delete()
            logger.info(f"Elemento {doc_id} eliminado.")
        except Exception as e:
            logger.error(f"Error al eliminar de Firestore: {e}")
            raise
