import numpy as np 
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import precision_recall_curve, auc
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from typing import Optional
import pywt
from scipy.stats import pearsonr
from stedfm import get_pretrained_model_v2 

STEDFM, _ = get_pretrained_model_v2(
    name="mae-lightning-small",
    weights="MAE_SMALL_STED",
    blocks="all",
    as_classifier=True,
    global_pool="token",
)

def compute_mse(truth: np.ndarray, prediction: np.ndarray, foreground: Optional[np.ndarray] = None) -> float:
    residual_sq = (truth - prediction) ** 2 
    if foreground is not None:
        fg = foreground.astype(bool)
        if not np.any(fg):
            return np.nan
        residual_sq = residual_sq[fg]
    return np.mean(residual_sq) 

def compute_mae(truth: np.ndarray, prediction: np.ndarray, foreground: Optional[np.ndarray] = None) -> float:
    residual = np.abs(truth - prediction)
    if foreground is not None:
        fg = foreground.astype(bool)
        if not np.any(fg):
            return np.nan
        residual = residual[fg]
    return np.mean(residual)

def compute_psnr(
    truth: np.ndarray,
    prediction: np.ndarray,
    foreground: Optional[np.ndarray] = None,
) -> float:
    if foreground is None:
        return peak_signal_noise_ratio(truth, prediction, data_range=1.0)
    mse_fg = compute_mse(truth, prediction, foreground=foreground)
    if np.isnan(mse_fg):
        return np.nan
    if mse_fg <= 0.0:
        return np.inf
    return float(10.0 * np.log10(1.0 / mse_fg))

def compute_ssim(
    truth: np.ndarray,
    prediction: np.ndarray,
    foreground: Optional[np.ndarray] = None,
) -> float:
    if foreground is None:
        return structural_similarity(truth, prediction, data_range=1.0)
    _, ssim_map = structural_similarity(
        truth,
        prediction,
        data_range=1.0,
        full=True,
    )
    fg = foreground.astype(bool)
    if not np.any(fg):
        return np.nan
    return float(np.mean(ssim_map[fg]))

def compute_ms_ssim(truth: np.ndarray, prediction: np.ndarray) -> float:
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

def compute_fourier_ncc(image1: np.ndarray, image2: np.ndarray) -> float:
    if image1.shape != image2.shape:
        raise ValueError("Both images must have the same shape.")
    
    if image1.ndim == 2:
        image1 = image1[np.newaxis, ...]
    if image2.ndim == 2:
        image2 = image2[np.newaxis, ...]

    channel_similarities = []

    # Iterate through each channel
    for i in range(image1.shape[0]):
        # Step 1: Compute the 2D FFT of each channel
        fft_image1_channel = np.fft.fft2(image1[i])
        fft_image2_channel = np.fft.fft2(image2[i])

        # Step 2: Get the magnitude spectra
        # The phase information is discarded as it's not needed for this similarity metric
        magnitude1 = np.abs(fft_image1_channel)
        magnitude2 = np.abs(fft_image2_channel)

        # Step 3: Compare the Fourier spectra using a correlation coefficient
        # We flatten the 2D magnitude arrays into 1D for np.corrcoef
        flat_magnitude1 = magnitude1.flatten()
        flat_magnitude2 = magnitude2.flatten()

        # np.corrcoef returns a correlation matrix. The desired score is [0, 1]
        correlation_matrix = np.corrcoef(flat_magnitude1, flat_magnitude2)
        ncc_score = correlation_matrix[0, 1]
        channel_similarities.append(ncc_score)

    # Return the average similarity score across both channels
    return np.mean(channel_similarities)

def compute_wavelet_ncc(image1: np.ndarray, image2: np.ndarray, wavelet: str = 'haar', level: int = 2) -> float:
    if image1.shape != image2.shape:
        raise ValueError("Both images must be 2-channel and have the same shape.")
    
    if image1.ndim == 2:
        image1 = image1[np.newaxis, ...]
    if image2.ndim == 2:
        image2 = image2[np.newaxis, ...]

    similarity_scores = []

    for i in range(image1.shape[0]):
        # Perform 2D DWT on each channel for both images
        # The DWT returns an approximation coefficient (cA) and detail coefficients (cH, cV, cD)
        coeffs1 = pywt.wavedec2(image1[i], wavelet=wavelet, level=level)
        coeffs2 = pywt.wavedec2(image2[i], wavelet=wavelet, level=level)
        
        # Unpack the approximation and detail coefficients
        cA1 = coeffs1[0].flatten()
        cA2 = coeffs2[0].flatten()
        
        # Compare the approximation coefficients (low-frequency content)
        # Using Pearson correlation, which is equivalent to NCC on normalized data
        corr_cA, _ = pearsonr(cA1, cA2)
        similarity_scores.append(corr_cA)
        
        # Compare the detail coefficients (high-frequency content)
        for j in range(1, level + 1):
            cH1, cV1, cD1 = [c.flatten() for c in coeffs1[j]]
            cH2, cV2, cD2 = [c.flatten() for c in coeffs2[j]]
            
            # Compare horizontal, vertical, and diagonal details
            corr_cH, _ = pearsonr(cH1, cH2)
            corr_cV, _ = pearsonr(cV1, cV2)
            corr_cD, _ = pearsonr(cD1, cD2)
            
            similarity_scores.extend([corr_cH, corr_cV, corr_cD])
            
    # Calculate the average of all similarity scores
    return np.mean(similarity_scores) 

def compute_phase_correlation(image1: np.ndarray, image2: np.ndarray) -> float:
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same dimensions.")
    
    if image1.ndim == 2:
        image1 = image1[np.newaxis, ...]
    if image2.ndim == 2:
        image2 = image2[np.newaxis, ...]

    correlation_scores = []
    
    for i in range(image1.shape[0]):
        # 1. Compute the 2D Fourier Transform of each channel
        fourier1 = np.fft.fft2(image1[i])
        fourier2 = np.fft.fft2(image2[i])
        
        # 2. Compute the Cross-Power Spectrum
        # np.conjugate() computes the complex conjugate
        cross_power_spectrum = (fourier1 * np.conjugate(fourier2))
        
        # Normalize the cross-power spectrum
        normalized_cross_power_spectrum = cross_power_spectrum / (np.abs(cross_power_spectrum) + 1e-10) # Add epsilon to avoid division by zero
        
        # 3. Compute the Inverse Fourier Transform
        phase_correlation_matrix = np.fft.ifft2(normalized_cross_power_spectrum)
        
        # 4. Find the Peak in the Phase Correlation Matrix
        # The peak value is a good indicator of similarity
        peak_value = np.max(np.abs(phase_correlation_matrix))
        correlation_scores.append(peak_value)
        
    return np.mean(correlation_scores)

def compute_stedfm_similarity(image1: np.ndarray, image2: np.ndarray) -> float:
    image1 = torch.from_numpy(image1).unsqueeze(0).unsqueeze(0)
    image2 = torch.from_numpy(image2).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        out1 = STEDFM.forward_features(image1)
        out2 = STEDFM.forward_features(image2)
    return torch.cosine_similarity(out1, out2, dim=1).item()

def compute_metrics(
    truth_image: np.ndarray,
    prediction_image: np.ndarray,
    foreground: Optional[np.ndarray] = None,
    truth_segmentation: Optional[np.ndarray] = None,
    prediction_segmentation: Optional[np.ndarray] = None,
) -> dict:
    metrics = {}
    metrics["mse"] = compute_mse(truth_image, prediction_image, foreground)
    metrics["mae"] = compute_mae(truth_image, prediction_image, foreground)
    metrics["psnr"] = compute_psnr(truth_image, prediction_image, foreground)
    metrics["ssim"] = compute_ssim(truth_image, prediction_image, foreground)
    metrics["ms_ssim"] = compute_ms_ssim(truth_image, prediction_image)
    metrics["fourier_ncc"] = compute_fourier_ncc(truth_image, prediction_image)
    metrics["wavelet_ncc"] = compute_wavelet_ncc(truth_image, prediction_image)
    metrics["phase_correlation"] = compute_phase_correlation(truth_image, prediction_image)
    metrics["stedfm"] = compute_stedfm_similarity(truth_image, prediction_image)
    if truth_segmentation is None or prediction_segmentation is None:
        return metrics 
    else:
        metrics["rings_dice"] = compute_dice(truth_segmentation[0], prediction_segmentation[0])
        metrics["fibers_dice"] = compute_dice(truth_segmentation[1], prediction_segmentation[1])
        return metrics

