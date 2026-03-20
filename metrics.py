import numpy as np 
import torch 
from skimage.metrics import peak_signal_noise_ratio
from sklearn.metrics import precision_recall_curve, auc
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from typing import Optional

def compute_mse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return np.mean((truth - prediction) ** 2) 

def compute_psnr(truth: np.ndarray, prediction: np.ndarray) -> float:
    return peak_signal_noise_ratio(truth, prediction)

def compute_ssim(truth: np.ndarray, prediction: np.ndarray) -> float:
    if isinstance(truth, np.ndarray):
        truth = torch.from_numpy(truth).unsqueeze(0).unsqueeze(0) 
    if isinstance(prediction, np.ndarray):
        prediction = torch.from_numpy(prediction).unsqueeze(0).unsqueeze(0) 
    mssim_object = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
    return mssim_object(prediction, truth).item()

def compute_aupr(truth: np.ndarray, prediction: np.ndarray) -> float:
    t, p = truth.ravel(), prediction.ravel() 
    if not np.any(t) and not np.any(p):
        return 1.0 
    if not np.any(t):
        if np.any(p > 0.5):
            return 0.0
        else:
            return -1.0
    else:
        precision, recall, _ = precision_recall_curve(t, p) 
        return auc(recall, precision)


def compute_dice(truth: np.ndarray, prediction: np.ndarray) -> float:
    t, p = truth.ravel().astype(bool), prediction.ravel().astype(bool)
    
    intersection = np.sum(t & p)
    denominator = np.sum(t) + np.sum(p)
    if denominator == 0:
        return 1.0
    return 2.0 * intersection / denominator

def compute_metrics(
    truth_image: np.ndarray,
    prediction_image: np.ndarray,
    truth_segmentation: Optional[np.ndarray] = None,
    prediction_segmentation: Optional[np.ndarray] = None,
) -> dict:
    metrics = {}
    metrics["mse"] = compute_mse(truth_image, prediction_image)
    metrics["psnr"] = compute_psnr(truth_image, prediction_image)
    metrics["ssim"] = compute_ssim(truth_image, prediction_image)
    if truth_segmentation is None or prediction_segmentation is None:
        return metrics 
    else:
        metrics["rings_dice"] = compute_dice(truth_segmentation[0], prediction_segmentation[0])
        metrics["fibers_dice"] = compute_dice(truth_segmentation[1], prediction_segmentation[1])
        return metrics

