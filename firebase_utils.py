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
        self.project_id = "reconocimiento-inventario"
        self._initialize_firebase()
    
    def _initialize_firebase(self):
        """Inicializa Firebase usando Streamlit secrets."""
        try:
            if not firebase_admin._apps:
                creds_base64 = st.secrets.get('FIREBASE_SERVICE_ACCOUNT_BASE64')
                if not creds_base64:
                    raise ValueError("El secret 'FIREBASE_SERVICE_ACCOUNT_BASE64' no fue encontrado.")
                
                missing_padding = len(creds_base64) % 4
                if missing_padding:
                    creds_base64 += '=' * (4 - missing_padding)

                creds_json_str = base64.b64decode(creds_base64).decode('utf-8')
                creds_dict = json.loads(creds_json_str)
                
                cred = credentials.Certificate(creds_dict)
                firebase_admin.initialize_app(cred, {'projectId': self.project_id})
                logger.info("Firebase inicializado correctamente.")
            
            self.db = firestore.client()
        except Exception as e:
            logger.error(f"Error fatal al inicializar Firebase: {e}")
            raise

    def get_timestamp(self):
        """Retorna el timestamp actual en formato ISO."""
        return datetime.now().isoformat()

    def save_inventory_item(self, data, custom_id=None):
        """
        Guarda un nuevo elemento en la colección 'inventory'.
        Si se proporciona un custom_id, se usa como ID del documento,
        previa verificación de que no exista.
        """
        try:
            collection_ref = self.db.collection('inventory')
            
            if custom_id:
                doc_ref = collection_ref.document(custom_id)
                if doc_ref.get().exists:
                    raise ValueError(f"El ID personalizado '{custom_id}' ya existe en el inventario.")
                doc_ref.set(data)
                logger.info(f"Elemento guardado con ID personalizado: {custom_id}")
                return custom_id
            else:
                _, doc_ref = collection_ref.add(data)
                logger.info(f"Elemento guardado con ID autogenerado: {doc_ref.id}")
                return doc_ref.id
                
        except Exception as e:
            logger.error(f"Error al guardar en Firestore: {e}")
            raise

    def get_all_inventory_items(self):
        """Obtiene todos los elementos de la colección 'inventory'."""
        try:
            docs = self.db.collection('inventory').stream()
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
            self.db.collection('inventory').document(doc_id).delete()
            logger.info(f"Elemento {doc_id} eliminado.")
        except Exception as e:
            logger.error(f"Error al eliminar de Firestore: {e}")
            raise
