"""
Signal preprocessing module.
Handles DC offset removal, power normalization, and digital filtering with full audit tracking.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.signal import butter, sosfiltfilt


def remove_dc_offset(samples: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Subtracts the mean (DC component) from I and Q channels independently.
    
    Returns:
        Tuple of (clean_samples, dc_i, dc_q)
    """
    if np.iscomplexobj(samples):
        dc_i = float(np.mean(samples.real))
        dc_q = float(np.mean(samples.imag))
        clean_samples = (samples.real - dc_i) + 1j * (samples.imag - dc_q)
    else:
        dc_i = float(np.mean(samples))
        dc_q = 0.0
        clean_samples = samples - dc_i
    return clean_samples.astype(samples.dtype), dc_i, dc_q


def normalize_amplitude(samples: np.ndarray, target_peak: float = 1.0) -> Tuple[np.ndarray, float]:
    """
    Scales sample amplitude so peak absolute magnitude equals target_peak.
    
    Returns:
        Tuple of (normalized_samples, scale_factor)
    """
    peak_val = np.max(np.abs(samples))
    if peak_val == 0 or np.isnan(peak_val):
        return samples.copy(), 1.0
    scale_factor = target_peak / peak_val
    normalized = samples * scale_factor
    return normalized, float(scale_factor)


def apply_bandpass_filter(
    samples: np.ndarray,
    sample_rate: float,
    low_cutoff: float,
    high_cutoff: float,
    order: int = 4
) -> np.ndarray:
    """
    Applies zero-phase Butterworth bandpass filter to real or complex signals.
    """
    nyquist = 0.5 * sample_rate
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist

    if low <= 0 or high >= 1.0 or low >= high:
        raise ValueError(f"Invalid filter cutoffs: [{low_cutoff}, {high_cutoff}] for Fs={sample_rate}")

    sos = butter(order, [low, high], btype="bandpass", output="sos")

    if np.iscomplexobj(samples):
        filtered_i = sosfiltfilt(sos, samples.real)
        filtered_q = sosfiltfilt(sos, samples.imag)
        return (filtered_i + 1j * filtered_q).astype(samples.dtype)
    else:
        return sosfiltfilt(sos, samples).astype(samples.dtype)
