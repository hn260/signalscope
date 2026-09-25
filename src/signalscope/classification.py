"""
Modulation feature extraction and heuristic candidate classifier module.
Computes key instantaneous signal features (amplitude, phase, frequency variances)
and evaluates candidate modulation types (CW, AM, FM, FSK, PSK).
"""

import numpy as np
from typing import Dict, Any, Tuple


def extract_modulation_features(
    samples: np.ndarray,
    sample_rate: float
) -> Dict[str, float]:
    """
    Computes key statistical modulation features:
    - gamma_max: Max spectral density of normalized instantaneous amplitude
    - sigma_aa: Variance of normalized instantaneous amplitude
    - sigma_ap: Variance of non-linear instantaneous phase
    - sigma_af: Variance of normalized instantaneous frequency
    """
    n_samples = len(samples)
    if n_samples < 32:
        return {"gamma_max": 0.0, "sigma_aa": 0.0, "sigma_ap": 0.0, "sigma_af": 0.0}

    is_complex = np.iscomplexobj(samples)
    
    # 1. Instantaneous Amplitude A(n)
    amp = np.abs(samples)
    mean_amp = np.mean(amp)
    if mean_amp < 1e-9:
        return {"gamma_max": 0.0, "sigma_aa": 0.0, "sigma_ap": 0.0, "sigma_af": 0.0}

    norm_amp = amp / mean_amp
    sigma_aa = float(np.var(norm_amp))

    # Gamma max: max peak of spectral density of centered normalized amplitude
    centered_amp = norm_amp - 1.0
    fft_amp = np.fft.fft(centered_amp)
    gamma_max = float(np.max(np.abs(fft_amp) ** 2) / n_samples)

    if not is_complex:
        # Real audio signal modulation features
        return {
            "gamma_max": gamma_max,
            "sigma_aa": sigma_aa,
            "sigma_ap": 0.0,
            "sigma_af": float(np.var(np.diff(samples) * sample_rate / (2 * np.pi)))
        }

    # 2. Instantaneous Phase phi(n)
    phase = np.unwrap(np.angle(samples))
    # Remove linear trend (carrier phase offset)
    time_indices = np.arange(n_samples)
    p_fit = np.polyfit(time_indices, phase, 1)
    phase_nl = phase - np.polyval(p_fit, time_indices)
    sigma_ap = float(np.var(phase_nl))

    # 3. Instantaneous Frequency f(n)
    inst_freq = np.diff(phase) * sample_rate / (2.0 * np.pi)
    mean_freq = np.mean(inst_freq)
    norm_inst_freq = inst_freq - mean_freq
    sigma_af = float(np.var(norm_inst_freq / (sample_rate / 2.0)))

    return {
        "gamma_max": gamma_max,
        "sigma_aa": sigma_aa,
        "sigma_ap": sigma_ap,
        "sigma_af": sigma_af,
    }


def classify_modulation(
    features: Dict[str, float],
    is_complex: bool = True
) -> Tuple[str, float, Dict[str, float]]:
    """
    Classifies candidate modulation format based on instantaneous features.
    
    Returns:
        Tuple of (predicted_class_name, confidence_heuristic, probabilities_dict)
    """
    gamma_max = features.get("gamma_max", 0.0)
    sigma_aa = features.get("sigma_aa", 0.0)
    sigma_ap = features.get("sigma_ap", 0.0)
    sigma_af = features.get("sigma_af", 0.0)

    # Candidate classes
    scores = {
        "CW / Unmodulated": 0.1,
        "AM (Amplitude Modulated)": 0.1,
        "FM (Frequency Modulated)": 0.1,
        "2-FSK (Frequency Shift Keyed)": 0.1,
        "BPSK / QPSK (Phase Shift Keyed)": 0.1,
    }

    # Decision heuristics
    if sigma_aa < 0.02 and sigma_af < 1e-4 and sigma_ap < 0.1:
        # Very clean unmodulated tone or CW
        scores["CW / Unmodulated"] += 0.85
    elif sigma_aa > 0.08:
        # High envelope variation -> AM
        scores["AM (Amplitude Modulated)"] += 0.80
    elif sigma_af > 1e-3 and sigma_aa < 0.05:
        # Continuous frequency variation -> FM or FSK
        if sigma_ap > 0.5:
            scores["2-FSK (Frequency Shift Keyed)"] += 0.75
            scores["FM (Frequency Modulated)"] += 0.40
        else:
            scores["FM (Frequency Modulated)"] += 0.80
    elif sigma_ap > 0.2 and sigma_aa < 0.04:
        # Phase transitions with constant envelope -> PSK
        scores["BPSK / QPSK (Phase Shift Keyed)"] += 0.75
    else:
        scores["CW / Unmodulated"] += 0.4

    # Normalize to probabilities
    total_score = sum(scores.values())
    probs = {k: float(v / total_score) for k, v in scores.items()}

    best_class = max(probs, key=probs.get)
    best_confidence = probs[best_class]

    return best_class, best_confidence, probs
