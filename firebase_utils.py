import firebase_admin
from firebase_admin import credentials, firestore
import json
import base64
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FirebaseUtils:
    def __init__(self):
        self.db = None
        self.project_id = "reconocimiento-inventario"
        self._initialize_firebase()
    
    def _initialize_firebase(self):
        """Inicializa Firebase con las credenciales"""
        try:
            # Obtener credenciales desde Streamlit secrets
            import streamlit as st
            
            # Usar el nombre exacto de tu secret
            if 'FIREBASE_SERVICE_ACCOUNT_BASE64' in st.secrets:
                # Decodificar credenciales desde base64
                creds_base64 = st.secrets['FIREBASE_SERVICE_ACCOUNT_BASE64']
                creds_json = base64.b64decode(creds_base64).decode('utf-8')
                creds_dict = json.loads(creds_json)
                
                # Crear credenciales
                cred = credentials.Certificate(creds_dict)
                
                # Inicializar Firebase (solo si no está inicializado)
                if not firebase_admin._apps:
                    firebase_admin.initialize_app(cred, {
                        'projectId': self.project_id
                    })
                
                self.db = firestore.client()
                logger.info(f"Proyecto Firebase: {self.project_id}")
                logger.info("Firebase inicializado correctamente")
                
            else:
                raise ValueError("FIREBASE_SERVICE_ACCOUNT_BASE64 no encontrado en secrets")
                
        except Exception as e:
            logger.error(f"Error al inicializar Firebase: {e}")
            raise
    
    def get_timestamp(self):
        """Obtiene timestamp actual"""
        return datetime.now().isoformat()
    
    def save_inventory_item(self, data):
        """Guarda un elemento en el inventario"""
        try:
            doc_ref = self.db.collection('inventario').add(data)
            return doc_ref[1].id
        except Exception as e:
            logger.error(f"Error al guardar elemento: {e}")
            raise
    
    def get_all_inventory_items(self):
        """Obtiene todos los elementos del inventario"""
        try:
            docs = self.db.collection('inventario').stream()
            items = []
            
            for doc in docs:
                item = doc.to_dict()
                item['id'] = doc.id
                items.append(item)
            
            # Ordenar por timestamp (más recientes primero)
            items.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return items
            
        except Exception as e:
            logger.error(f"Error al obtener elementos: {e}")
            return []
    
    def delete_inventory_item(self, doc_id):
        """Elimina un elemento del inventario"""
        try:
            self.db.collection('inventario').document(doc_id).delete()
        except Exception as e:
            logger.error(f"Error al eliminar elemento: {e}")
            raise
