import firebase_admin
from firebase_admin import credentials, firestore, storage
import logging
import streamlit as st
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import base64
import io

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FirebaseUtils:
    def __init__(self):
        """Inicializar Firebase Admin SDK usando Streamlit secrets."""
        try:
            # Obtener credenciales desde Streamlit secrets
            firebase_service_account_base64 = st.secrets["FIREBASE_SERVICE_ACCOUNT_BASE64"]
            
            # Decodificar el JSON base64
            service_account_json = base64.b64decode(firebase_service_account_base64).decode('utf-8')
            
            # Parsear el JSON
            service_account_info = json.loads(service_account_json)
            
            self.project_id = service_account_info.get('project_id')
            
            if not self.project_id:
                raise ValueError("project_id no encontrado en las credenciales de Firebase")
            
            logger.info(f"Proyecto Firebase: {self.project_id}")
            
        except KeyError:
            raise ValueError("FIREBASE_SERVICE_ACCOUNT_BASE64 no encontrada en Streamlit secrets")
        except Exception as e:
            raise ValueError(f"Error decodificando credenciales de Firebase: {str(e)}")
        
        # Inicializar Firebase Admin SDK
        try:
            # Verificar si ya está inicializado
            if not firebase_admin._apps:
                # Crear credenciales desde el diccionario
                cred = credentials.Certificate(service_account_info)
                firebase_admin.initialize_app(cred, {
                    'storageBucket': f'{self.project_id}.appspot.com'
                })
            
            self.db = firestore.client()
            self.bucket = storage.bucket()
            logger.info("Firebase inicializado correctamente")
            
        except Exception as e:
            logger.error(f"Error inicializando Firebase: {str(e)}")
            raise

    def get_timestamp(self) -> str:
        """Obtener timestamp actual."""
        return datetime.now().isoformat()

    def add_inventory_item(self, item_data: Dict[str, Any]) -> Any:
        """
        Agregar un elemento al inventario.
        
        Args:
            item_data (Dict[str, Any]): Datos del elemento
            
        Returns:
            DocumentReference: Referencia del documento creado
        """
        try:
            # Agregar timestamp si no existe
            if 'timestamp' not in item_data:
                item_data['timestamp'] = self.get_timestamp()
            
            # Agregar a la colección 'inventory'
            doc_ref = self.db.collection('inventory').add(item_data)[1]
            logger.info(f"Elemento agregado con ID: {doc_ref.id}")
            return doc_ref
            
        except Exception as e:
            logger.error(f"Error agregando elemento: {str(e)}")
            raise

    def get_inventory_items(self, limit: int = 100) -> List[Any]:
        """
        Obtener elementos del inventario.
        
        Args:
            limit (int): Límite de elementos a obtener
            
        Returns:
            List[Any]: Lista de documentos
        """
        try:
            items = self.db.collection('inventory').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit).stream()
            return list(items)
            
        except Exception as e:
            logger.error(f"Error obteniendo elementos: {str(e)}")
            raise

    def get_inventory_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtener un elemento específico del inventario.
        
        Args:
            item_id (str): ID del elemento
            
        Returns:
            Optional[Dict[str, Any]]: Datos del elemento o None
        """
        try:
            doc_ref = self.db.collection('inventory').document(item_id)
            doc = doc_ref.get()
            
            if doc.exists:
                data = doc.to_dict()
                data['id'] = doc.id
                return data
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error obteniendo elemento {item_id}: {str(e)}")
            raise

    def update_inventory_item(self, item_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Actualizar un elemento del inventario.
        
        Args:
            item_id (str): ID del elemento
            update_data (Dict[str, Any]): Datos a actualizar
            
        Returns:
            bool: True si se actualizó correctamente
        """
        try:
            doc_ref = self.db.collection('inventory').document(item_id)
            doc_ref.update(update_data)
            logger.info(f"Elemento {item_id} actualizado")
            return True
            
        except Exception as e:
            logger.error(f"Error actualizando elemento {item_id}: {str(e)}")
            raise

    def delete_inventory_item(self, item_id: str) -> bool:
        """
        Eliminar un elemento del inventario.
        
        Args:
            item_id (str): ID del elemento
            
        Returns:
            bool: True si se eliminó correctamente
        """
        try:
            self.db.collection('inventory').document(item_id).delete()
            logger.info(f"Elemento {item_id} eliminado")
            return True
            
        except Exception as e:
            logger.error(f"Error eliminando elemento {item_id}: {str(e)}")
            raise

    def upload_image(self, image_data: bytes, filename: str) -> str:
        """
        Subir imagen a Firebase Storage.
        
        Args:
            image_data (bytes): Datos de la imagen
            filename (str): Nombre del archivo
            
        Returns:
            str: URL de la imagen subida
        """
        try:
            blob = self.bucket.blob(f'inventory_images/{filename}')
            blob.upload_from_string(image_data, content_type='image/jpeg')
            blob.make_public()
            
            url = blob.public_url
            logger.info(f"Imagen subida: {url}")
            return url
            
        except Exception as e:
            logger.error(f"Error subiendo imagen: {str(e)}")
            raise

    def search_inventory(self, search_term: str) -> List[Any]:
        """
        Buscar elementos en el inventario.
        
        Args:
            search_term (str): Término de búsqueda
            
        Returns:
            List[Any]: Lista de elementos encontrados
        """
        try:
            # Búsqueda simple por descripción
            items = self.db.collection('inventory').where('description', '>=', search_term).where('description', '<=', search_term + '\uf8ff').stream()
            return list(items)
            
        except Exception as e:
            logger.error(f"Error buscando: {str(e)}")
            raise

    def get_inventory_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del inventario.
        
        Returns:
            Dict[str, Any]: Estadísticas del inventario
        """
        try:
            items = self.get_inventory_items()
            
            stats = {
                'total_items': len(items),
                'last_update': max([item.get('timestamp', '') for item in items]) if items else None,
                'models_used': {},
                'categories': {}
            }
            
            # Contar modelos usados y categorías
            for item in items:
                model = item.get('model_used', 'unknown')
                stats['models_used'][model] = stats['models_used'].get(model, 0) + 1
                
                # Intentar extraer categoría del análisis
                analysis = item.get('analysis', '')
                if analysis:
                    try:
                        analysis_data = json.loads(analysis)
                        category = analysis_data.get('categoria', 'Sin categoría')
                        stats['categories'][category] = stats['categories'].get(category, 0) + 1
                    except:
                        pass
            
            return stats
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {str(e)}")
            raise

# Función de utilidad para uso directo
def get_firebase_connection() -> FirebaseUtils:
    """
    Función de conveniencia para obtener conexión a Firebase.
    
    Returns:
        FirebaseUtils: Instancia de FirebaseUtils
    """
    try:
        return FirebaseUtils()
    except Exception as e:
        logger.error(f"Error conectando a Firebase: {str(e)}")
        raise
