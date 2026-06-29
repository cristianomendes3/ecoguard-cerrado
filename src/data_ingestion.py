import pystac_client
import rasterio
from rasterio.windows import from_bounds
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

# Configuração de logs para monitoramento do pipeline na nuvem
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestionPipeline:
    def __init__(self, stac_url: str = "https://earth-search.aws.element84.com/v1"):
        """Inicializa a conexão com o catálogo STAC."""
        self.catalog = pystac_client.Client.open(stac_url)
        logger.info("[INGESTÃO] Conexão com STAC estabelecida.")

    def query_mapbiomas_alerts(self, date_range: Tuple[str, str]) -> List[Dict]:
        """
        Consulta alertas validados. 
        Implementação pronta para integração com API GraphQL.
        """
        # Em produção, este método processará a resposta real do MapBiomas
        return [{"id": "ALERT-2026-PI-001", "bbox": [-45.21, -8.45, -45.15, -8.39]}]

    def fetch_sentinel_item(self, bbox: List[float], dt: str, collection: str = "sentinel-2-l2a") -> Optional[dict]:
        """
        Busca o item mais recente no catálogo com tratamento de exceções.
        """
        try:
            search = self.catalog.search(
                collections=[collection],
                bbox=bbox,
                datetime=dt,
                query={"eo:cloud_cover": {"lt": 10}}
            )
            items = list(search.item_collection())
            
            if not items:
                logger.warning(f"[STAC] Nenhuma cena encontrada para BBOX {bbox}")
                return None
                
            return items[0]
            
        except Exception as e:
            logger.error(f"[STAC] Erro crítico na requisição: {e}")
            return None

    def extract_spectral_tensor(self, item: dict, bbox: List[float], bands: List[str] = ['blue', 'green', 'red', 'nir']) -> np.ndarray:
        tensor_layers = []
        try:
            for band in bands:
                asset_key = band if band in item.assets else f"{band}-jp2"
                asset_href = item.assets[asset_key].href
                
                with rasterio.open(asset_href) as src:
                    # 1. Reprojeta o bbox para o CRS da imagem (importante!)
                    from rasterio.warp import transform_bounds
                    dst_crs = src.crs
                    # Transforma o bbox para o CRS do arquivo (ex: UTM para WGS84)
                    new_bbox = transform_bounds("epsg:4326", dst_crs, *bbox)
                    
                    # 2. Tenta ler a janela transformada
                    window = from_bounds(*new_bbox, src.transform)
                    data = src.read(1, window=window, boundless=True, fill_value=0)
                    
                    # 3. Se ainda vier vazio, ler a imagem toda como fallback (debug)
                    if data.size == 0:
                        logger.warning(f"[RASTERIO] Recorte vazio na banda {band}. Lendo cena inteira.")
                        data = src.read(1)
                    
                    tensor_layers.append(data.astype(np.float32))
            
            return np.stack(tensor_layers, axis=0)
        except Exception as e:
            logger.error(f"[RASTERIO] Falha: {e}")
            raise RuntimeError("Falha na montagem.")

# Exemplo de uso para validação do módulo
if __name__ == "__main__":
    pipeline = DataIngestionPipeline()
    alerta = pipeline.query_mapbiomas_alerts(("2026-05-01", "2026-05-31"))[0]
    
    item = pipeline.fetch_sentinel_item(alerta['bbox'], "2026-05-01/2026-06-01")
    
    if item:
        tensor = pipeline.extract_spectral_tensor(item, alerta['bbox'])
        print(f"[SUCESSO] Tensor gerado: {tensor.shape}")