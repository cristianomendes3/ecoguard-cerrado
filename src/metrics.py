import numpy as np

class MetricsEvaluator:
    """
    Biblioteca de métricas para auditoria de modelos de visão computacional.
    """
    
    @staticmethod
    def calculate_ndvi(tensor_scene: np.ndarray) -> np.ndarray:
        # Canais: 0=Blue, 1=Green, 2=Red, 3=NIR
        red = tensor_scene[2].astype(np.float32)
        nir = tensor_scene[3].astype(np.float32)
        # Adiciona eps para evitar divisão por zero
        return (nir - red) / (nir + red + 1e-8)

    @staticmethod
    def calculate_iou(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5, smooth: float = 1e-6) -> float:
        """Calcula IoU binário entre previsão e gabarito real."""
        pred = (pred > threshold).astype(np.uint8)
        target = (target > 0.5).astype(np.uint8)
        
        intersection = np.logical_and(pred, target).sum()
        union = np.logical_or(pred, target).sum()
        
        return float((intersection + smooth) / (union + smooth))

    @staticmethod
    def calculate_accuracy(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
        """Acurácia Global."""
        pred_binary = (pred > threshold).astype(np.uint8)
        target_binary = (target > 0.5).astype(np.uint8)
        return float(np.mean(pred_binary == target_binary))

    @staticmethod
    def calculate_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
        """Acurácia Global."""
        return float(np.mean(pred == target))