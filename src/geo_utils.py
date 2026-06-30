import numpy as np
import rasterio
from rasterio.features import rasterize
from typing import List, Tuple, Dict

class GeoProcessor:
    """
    Ferramentas de processamento geométrico para converter vetores em máscaras
    e preparar tiles para o treinamento de modelos de Deep Learning.
    """

    @staticmethod
    def create_mask_from_geojson(geometries: List[Dict], transform: rasterio.Affine, shape: Tuple[int, int]) -> np.ndarray:
        """
        Converte uma lista de geometrias GeoJSON em uma máscara binária (0: preservado, 1: desmatado).
        
        Args:
            geometries: Lista de geometrias (formato GeoJSON).
            transform: Affine transform da imagem de satélite base.
            shape: Tupla (altura, largura) da máscara.
        """
        # Cria uma lista de (geometria, valor_do_pixel)
        shapes = [(geom, 1) for geom in geometries]
        
        # Rasteriza os polígonos na grade da imagem
        mask = rasterize(
            shapes,
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype=np.uint8
        )
        return mask

    @staticmethod
    def tile_data(scene: np.ndarray, mask: np.ndarray, tile_size: int = 256) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Divide cena (tensor) e máscara em tiles de tamanho fixo.
        """
        _, h, w = scene.shape
        tiles = []
        
        for i in range(0, h, tile_size):
            for j in range(0, w, tile_size):
                # Extrai o tile se ele couber completamente na imagem
                if i + tile_size <= h and j + tile_size <= w:
                    s_tile = scene[:, i:i+tile_size, j:j+tile_size]
                    m_tile = mask[i:i+tile_size, j:j+tile_size]
                    tiles.append((s_tile, m_tile))
        
        return tiles

    @staticmethod
    def analyze_class_balance(mask: np.ndarray) -> Dict:
        """
        Calcula a proporção de pixels desmatados. Útil para definir pesos na Loss Function.
        """
        total = mask.size
        deforested = np.sum(mask == 1)
        return {
            "total_pixels": total,
            "deforested_pixels": deforested,
            "ratio": deforested / total if total > 0 else 0
        }