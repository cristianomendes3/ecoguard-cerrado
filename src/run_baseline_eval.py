from src.processor import DataProcessor
from src.metrics import MetricsEvaluator
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_evaluation():
    processor = DataProcessor()
    metrics = MetricsEvaluator()
    
    # 1. Pipeline Real (Dados do Sentinel-2 + Polígono MapBiomas)
    # Usando o mesmo bbox de teste que já validamos
    bbox = [-45.21, -8.45, -45.15, -8.39]
    tiles = processor.run_full_pipeline(bbox, "2026-05-01/2026-06-01", [])
    
    if not tiles:
        print("Pipeline não retornou tiles para avaliação.")
        return

    # 2. Avaliação do primeiro tile (Baseline NDVI)
    tile_scene, tile_mask = tiles[0]
    ndvi = metrics.calculate_ndvi(tile_scene)
    
    # Baseline: NDVI < 0.2 é desmatamento
    prediction = (ndvi < 0.2).astype(np.uint8)
    
    # 3. Métricas
    iou = metrics.calculate_iou(prediction, tile_mask)
    acc = metrics.calculate_accuracy(prediction, tile_mask)
    
    print("\n" + "="*50)
    print("BOLETIM DE DESEMPENHO: BASELINE NDVI")
    print(f"IoU (Classe Desmatamento): {iou:.4f}")
    print(f"Acurácia Global: {acc:.4f}")
    print("="*50)

if __name__ == "__main__":
    run_evaluation()