import firebase_admin
from firebase_admin import credentials, firestore, storage
import logging
import streamlit as st
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import base64

# --- CONFIGURACIÓN DE LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FirebaseUtils:
    def __init__(self):
        """Inicializa Firebase Admin SDK usando Streamlit secrets."""
        if not firebase_admin._apps:
            try:
                firebase_service_account_base64 = st.secrets["FIREBASE_SERVICE_ACCOUNT_BASE64"]
                service_account_json = base64.b64decode(firebase_service_account_base64).decode('utf-8')
                service_account_info = json.loads(service_account_json)
                self.project_id = service_account_info.get('project_id')
                
                if not self.project_id:
                    raise ValueError("project_id no encontrado en las credenciales de Firebase")
                
                cred = credentials.Certificate(service_account_info)
                firebase_admin.initialize_app(cred, {
                    'storageBucket': f'{self.project_id}.appspot.com'
                })
                logger.info("Firebase inicializado correctamente.")
            except Exception as e:
                logger.error(f"Error inicializando Firebase: {e}")
                raise
        
        self.db = firestore.client()
        self.bucket = storage.bucket()

    def get_timestamp(self) -> str:
        """Obtiene el timestamp actual en formato ISO."""
        return datetime.now().isoformat()

    def add_inventory_item(self, item_data: Dict[str, Any]) -> Any:
        """Agrega un elemento a la colección 'inventory'."""
        if 'timestamp' not in item_data:
            item_data['timestamp'] = self.get_timestamp()
        _, doc_ref = self.db.collection('inventory').add(item_data)
        logger.info(f"Elemento agregado con ID: {doc_ref.id}")
        return doc_ref

    def get_inventory_items(self, limit: int = 100) -> List[Any]:
        """Obtiene elementos del inventario, ordenados por fecha."""
        return self.db.collection('inventory').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit).stream()

    def delete_inventory_item(self, item_id: str) -> bool:
        """Elimina un elemento del inventario por su ID."""
        self.db.collection('inventory').document(item_id).delete()
        logger.info(f"Elemento {item_id} eliminado.")
        return True

    def get_inventory_stats(self) -> Dict[str, Any]:
        """Calcula y devuelve estadísticas básicas del inventario."""
        items_docs = list(self.get_inventory_items())
        items = [doc.to_dict() for doc in items_docs]
        
        if not items:
            return {'total_items': 0, 'last_update': None, 'categories': {}}
            
        stats = {
            'total_items': len(items),
            'last_update': max([item.get('timestamp', '') for item in items]),
            'categories': {}
        }
        
        for item in items:
            analysis = item.get('analysis', '')
            if analysis:
                try:
                    analysis_data = json.loads(analysis)
                    category = analysis_data.get('categoria', 'Sin categoría').capitalize()
                    stats['categories'][category] = stats['categories'].get(category, 0) + 1
                except (json.JSONDecodeError, AttributeError):
                    pass # Ignora análisis malformados
        
        return stats
