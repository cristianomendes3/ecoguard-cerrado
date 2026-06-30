import numpy as np
import matplotlib.pyplot as plt
from src.processor import DataProcessor
from src.metrics import MetricsEvaluator
from src.geo_utils import GeoProcessor
from shapely.geometry import Polygon

def run_stress_test():
    processor = DataProcessor()
    
    cenarios = [
        {"titulo": "Cenário 1: Fazenda Consolidada", "bbox": [-44.6, -7.5, -44.5, -7.4], "tipo": "fazenda"},
        {"titulo": "Cenário 2: Desmatamento Recente", "bbox": [-45.9, -8.1, -45.8, -8.0], "tipo": "desmatamento"},
        {"titulo": "Cenário 3: Reserva Natural na Seca", "bbox": [-44.10, -8.85, -44.00, -8.75], "tipo": "reserva"}
    ]

    for c in cenarios:
        print(f"\n[ESTRESSE] Processando: {c['titulo']}...")
        
        # 1. Pipeline de Ingestão: Capturando tiles E o transform real da imagem
        result = processor.run_full_pipeline(c['bbox'], "2023-09-01/2023-09-30", [])
        if result is None: 
            continue
        
        tiles, img_transform = result
        scene, _ = tiles[0]
        
        # 2. Baseline NDVI
        ndvi = MetricsEvaluator.calculate_ndvi(scene)
        previsao = (ndvi < 0.25).astype(np.uint8)
        
        # 3. Ground Truth: Rasterização usando o img_transform correto
        if c['tipo'] == "desmatamento":
            poly = Polygon([(-45.88, -8.08), (-45.81, -8.08), (-45.81, -8.02), (-45.88, -8.02)])
            mask = GeoProcessor.create_mask_from_geojson([{'geometry': poly}], img_transform, (scene.shape[1], scene.shape[2]))
        else:
            mask = np.zeros((scene.shape[1], scene.shape[2]), dtype=np.uint8)

        # 4. Métricas
        iou = MetricsEvaluator.calculate_iou(previsao, mask)
        acc = MetricsEvaluator.calculate_accuracy(previsao, mask)
        
        # 5. Visualização (4 painéis)
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f"{c['titulo']} | IoU: {iou*100:.1f}% | Acurácia: {acc*100:.1f}%", fontsize=16, fontweight='bold')
        
        rgb = np.clip(np.transpose(scene[0:3, :, :], (1, 2, 0)) / 3000.0, 0, 1)
        axes[0].imshow(rgb); axes[0].set_title("1. Visão Satélite"); axes[0].axis('off')
        axes[1].imshow(ndvi, cmap='RdYlGn', vmin=-0.2, vmax=0.8); axes[1].set_title("2. Processamento NDVI"); axes[1].axis('off')
        axes[2].imshow(previsao, cmap='Reds'); axes[2].set_title("3. Alerta do Baseline"); axes[2].axis('off')
        axes[3].imshow(mask, cmap='gray'); axes[3].set_title("4. Gabarito MapBiomas"); axes[3].axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run_stress_test()