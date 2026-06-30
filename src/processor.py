from src.data_ingestion import DataIngestionPipeline
from src.geo_utils import GeoProcessor
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Orquestrador do Pipeline de Dados.
    Unifica a ingestão (Cloud) e o pré-processamento (Geometria/Tiling).
    """
    def __init__(self):
        self.ingestor = DataIngestionPipeline()
        self.processor = GeoProcessor()
        logger.info("[PROCESSOR] Pipeline orquestrado inicializado.")

    def run_full_pipeline(self, bbox: list, datetime_range: str, geojson_features: list):
        """
        Executa o fluxo completo:
        1. Baixa a cena Sentinel-2 via STAC.
        2. Rasteriza o alerta em uma máscara binária.
        3. Recorta (tiling) em blocos de 256x256.
        """
        # 1. Ingestão
        item = self.ingestor.fetch_sentinel_item(bbox, datetime_range)
        if not item:
            logger.warning(f"[PROCESSOR] Falha na ingestão para BBOX {bbox}. Pulando.")
            return None # Retorna None em vez de dar erro fatal
        
        # Extrai tensor e obtém transform para a rasterização
        tensor = self.ingestor.extract_spectral_tensor(item, bbox)
        
        # 2. Rasterização (Ground Truth)
        # Assumindo que a imagem está em UTM (via rasterio), precisamos do transform da imagem
        # Para este protótipo, usamos o transform do item que lemos no rasterio.open
        import rasterio
        with rasterio.open(item.assets['blue'].href) as src:
            # Capturamos o transform real do arquivo
            img_transform = src.transform 
            mask = self.processor.create_mask_from_geojson(
                geojson_features, img_transform, (tensor.shape[1], tensor.shape[2])
            )
        
        # 3. Tiling
        tiles = self.processor.tile_data(tensor, mask, tile_size=256)
        
        logger.info(f"[PROCESSOR] Pipeline concluído: {len(tiles)} tiles gerados.")
        
        return tiles, img_transform

# Exemplo de teste do orquestrador
if __name__ == "__main__":
    processor = DataProcessor()
    # Exemplo de uso com um alerta mockado para garantir que a amarração funciona
    bbox = [-45.21, -8.45, -45.15, -8.39]
    try:
        tiles = processor.run_full_pipeline(bbox, "2026-05-01/2026-06-01", [])
        print(f"Orquestrador finalizado. Tiles prontos: {len(tiles)}")
    except Exception as e:
        print(f"Erro no pipeline: {e}")