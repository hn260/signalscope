"""
Signal parameter extraction engine.
Computes time-domain statistics, FFT-based Power Spectral Density (PSD), peak frequencies,
occupied bandwidth (99% power container & X-dB down methods), and estimated SNR noise floor.
"""

import numpy as np
from typing import Tuple, Dict, Any, List
from scipy.signal import welch, get_window

from .metadata import SignalMetadata, ParameterResult, ParameterStatus


def compute_fft_psd(
    samples: np.ndarray,
    sample_rate: float,
    nfft: int = 4096,
    window_name: str = "hann"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Power Spectral Density (PSD) using Hann window and FFT shift.
    
    Returns:
        Tuple of (freq_axis_hz, psd_db)
        freq_axis_hz is centered at 0 Hz for complex IQ, range [-Fs/2, +Fs/2]
        or [0, Fs/2] for real audio.
    """
    is_complex = np.iscomplexobj(samples)
    n_samples = len(samples)
    actual_nfft = min(nfft, n_samples)
    
    if actual_nfft < 128:
        actual_nfft = max(16, n_samples)

    if is_complex:
        freqs, psd = welch(
            samples,
            fs=sample_rate,
            window=window_name,
            nperseg=actual_nfft,
            noverlap=actual_nfft // 2,
            return_onesided=False,
            scaling="density"
        )
        # Shift zero frequency to center
        freqs = np.fft.fftshift(freqs)
        psd = np.fft.fftshift(psd)
    else:
        freqs, psd = welch(
            samples,
            fs=sample_rate,
            window=window_name,
            nperseg=actual_nfft,
            noverlap=actual_nfft // 2,
            return_onesided=True,
            scaling="density"
        )

    # Convert to dB (avoid log of zero)
    psd_db = 10.0 * np.log10(np.maximum(psd, 1e-18))
    return freqs, psd_db


def compute_peak_frequency(
    freqs: np.ndarray,
    psd_db: np.ndarray,
    center_freq_hz: float = 0.0
) -> Tuple[float, float, float]:
    """
    Identifies peak spectral frequency and peak PSD value.
    
    Returns:
        Tuple of (peak_baseband_freq_hz, peak_rf_freq_hz, peak_power_db)
    """
    peak_idx = int(np.argmax(psd_db))
    peak_baseband_freq = float(freqs[peak_idx])
    peak_rf_freq = center_freq_hz + peak_baseband_freq if center_freq_hz is not None else peak_baseband_freq
    peak_power = float(psd_db[peak_idx])
    return peak_baseband_freq, peak_rf_freq, peak_power


def compute_occupied_bandwidth_99(
    freqs: np.ndarray,
    psd_db: np.ndarray
) -> Tuple[float, float, float]:
    """
    Computes 99% Occupied Power Bandwidth by integrating linear power across frequency bins.
    
    Returns:
        Tuple of (obw_hz, f_low_hz, f_high_hz)
    """
    psd_linear = 10.0 ** (psd_db / 10.0)
    total_power = np.sum(psd_linear)

    if total_power <= 0:
        return 0.0, float(freqs[0]), float(freqs[-1])

    cum_power = np.cumsum(psd_linear) / total_power
    
    # 0.5% tail on lower side, 99.5% tail on upper side (99% central power)
    low_idx = int(np.searchsorted(cum_power, 0.005))
    high_idx = int(np.searchsorted(cum_power, 0.995))

    low_idx = max(0, min(low_idx, len(freqs) - 1))
    high_idx = max(0, min(high_idx, len(freqs) - 1))

    f_low = float(freqs[low_idx])
    f_high = float(freqs[high_idx])
    obw_hz = abs(f_high - f_low)

    return obw_hz, f_low, f_high


def compute_xdb_bandwidth(
    freqs: np.ndarray,
    psd_db: np.ndarray,
    x_db: float = 20.0
) -> Tuple[float, float, float]:
    """
    Computes bandwidth at X dB below spectral peak power level.
    
    Returns:
        Tuple of (xdb_bw_hz, f_low_hz, f_high_hz)
    """
    peak_idx = int(np.argmax(psd_db))
    peak_db = psd_db[peak_idx]
    threshold_db = peak_db - abs(x_db)

    above_thresh_indices = np.where(psd_db >= threshold_db)[0]

    if len(above_thresh_indices) == 0:
        return 0.0, float(freqs[peak_idx]), float(freqs[peak_idx])

    low_idx = above_thresh_indices[0]
    high_idx = above_thresh_indices[-1]

    f_low = float(freqs[low_idx])
    f_high = float(freqs[high_idx])
    bw_hz = abs(f_high - f_low)

    return bw_hz, f_low, f_high


def estimate_snr_floor(
    psd_db: np.ndarray,
    signal_percentile: float = 90.0,
    noise_percentile: float = 25.0
) -> Tuple[float, float, float]:
    """
    Estimates Signal-to-Noise Ratio (SNR) using spectral percentile energy estimation.
    Assumes noise floor dominates lower percentiles of PSD bins.
    
    Returns:
        Tuple of (estimated_snr_db, signal_level_db, noise_floor_db)
    """
    noise_floor_db = float(np.percentile(psd_db, noise_percentile))
    signal_level_db = float(np.percentile(psd_db, signal_percentile))
    snr_db = signal_level_db - noise_floor_db
    return snr_db, signal_level_db, noise_floor_db


def extract_all_parameters(
    samples: np.ndarray,
    metadata: SignalMetadata
) -> Tuple[List[ParameterResult], np.ndarray, np.ndarray]:
    """
    Executes full parameter extraction pipeline.
    
    Returns:
        Tuple of (parameters_list, freqs, psd_db)
    """
    mag = np.abs(samples)
    peak_amp = float(np.max(mag)) if len(mag) > 0 else 0.0
    rms_amp = float(np.sqrt(np.mean(mag ** 2))) if len(mag) > 0 else 0.0
    crest_factor = peak_amp / (rms_amp + 1e-12)
    crest_factor_db = 20.0 * np.log10(max(crest_factor, 1e-6))
    rms_power_dbfs = 20.0 * np.log10(max(rms_amp, 1e-6))

    # Spectral Analysis
    freqs, psd_db = compute_fft_psd(samples, metadata.sample_rate_hz)
    
    peak_baseband_hz, peak_rf_hz, peak_power_db = compute_peak_frequency(
        freqs, psd_db, metadata.center_freq_hz if metadata.center_freq_hz else 0.0
    )

    obw_99_hz, f_low_99, f_high_99 = compute_occupied_bandwidth_99(freqs, psd_db)
    bw_20db_hz, _, _ = compute_xdb_bandwidth(freqs, psd_db, x_db=20.0)
    bw_3db_hz, _, _ = compute_xdb_bandwidth(freqs, psd_db, x_db=3.0)

    snr_db, signal_lvl_db, noise_lvl_db = estimate_snr_floor(psd_db)

    # Construct ParameterResult objects
    params = [
        ParameterResult(
            name="sample_rate",
            display_name="Sample Rate",
            value=f"{metadata.sample_rate_hz / 1e6:.6f}",
            unit="MS/s",
            method=f"Header/User Input ({metadata.metadata_origin.value})",
            status=ParameterStatus.CONFIDENT,
            confidence=1.0,
            assumptions="Sample rate as configured by user or file header."
        ),
        ParameterResult(
            name="duration",
            display_name="Duration",
            value=f"{metadata.duration_sec:.4f}",
            unit="seconds",
            method="Exact sample count / Fs",
            status=ParameterStatus.CONFIDENT,
            confidence=1.0,
            assumptions="Continuous recording duration."
        ),
        ParameterResult(
            name="num_samples",
            display_name="Sample Count",
            value=metadata.num_samples,
            unit="samples",
            method="Buffer length count",
            status=ParameterStatus.CONFIDENT,
            confidence=1.0,
            assumptions="Total parsed I/Q or WAV audio samples."
        ),
        ParameterResult(
            name="peak_amplitude",
            display_name="Peak Amplitude",
            value=f"{peak_amp:.4f}",
            unit="FS (Normalized)",
            method="Max absolute magnitude max(|s[n]|)",
            status=ParameterStatus.CONFIDENT,
            confidence=1.0,
            assumptions="Full-scale normalized range [-1.0, +1.0]."
        ),
        ParameterResult(
            name="rms_power",
            display_name="RMS Power",
            value=f"{rms_power_dbfs:.2f}",
            unit="dBFS",
            method="20*log10(sqrt(mean(|s[n]|^2)))",
            status=ParameterStatus.CONFIDENT,
            confidence=1.0,
            assumptions="Root-Mean-Square signal magnitude."
        ),
        ParameterResult(
            name="crest_factor",
            display_name="Crest Factor (PAR)",
            value=f"{crest_factor:.2f} ({crest_factor_db:.2f} dB)",
            unit="ratio / dB",
            method="Peak Magnitude / RMS Magnitude",
            status=ParameterStatus.CONFIDENT,
            confidence=1.0,
            assumptions="Peak-to-Average Power Ratio proxy."
        ),
        ParameterResult(
            name="peak_baseband_freq",
            display_name="Peak Baseband Frequency",
            value=f"{peak_baseband_hz / 1e3:.3f}",
            unit="kHz",
            method="Welch PSD Peak Bin Location (Hann window)",
            status=ParameterStatus.CONFIDENT,
            confidence=0.98,
            assumptions="Relative to zero baseband carrier frequency."
        ),
        ParameterResult(
            name="peak_rf_freq",
            display_name="Peak Absolute RF Frequency",
            value=f"{peak_rf_hz / 1e6:.6f}" if metadata.has_center_frequency else "Relative (Center Freq Unspecified)",
            unit="MHz" if metadata.has_center_frequency else "Baseband Only",
            method="Center Frequency + Baseband Peak Frequency" if metadata.has_center_frequency else "Relative Baseband Only",
            status=ParameterStatus.CONFIDENT if metadata.has_center_frequency else ParameterStatus.ESTIMATED,
            confidence=0.99 if metadata.has_center_frequency else 0.5,
            assumptions="Absolute RF frequency requires known center frequency metadata." if not metadata.has_center_frequency else "Center frequency verified."
        ),
        ParameterResult(
            name="occupied_bw_99",
            display_name="99% Occupied Bandwidth",
            value=f"{obw_99_hz / 1e3:.2f}",
            unit="kHz",
            method="Spectral Power Integration (99% central power container)",
            status=ParameterStatus.CONFIDENT,
            confidence=0.95,
            assumptions="99% of total spectral power contained within interval."
        ),
        ParameterResult(
            name="bw_20db",
            display_name="-20 dB Down Bandwidth",
            value=f"{bw_20db_hz / 1e3:.2f}",
            unit="kHz",
            method="Spectral width at -20 dB relative to peak power",
            status=ParameterStatus.CONFIDENT,
            confidence=0.90,
            assumptions="Measures main lobe bandwidth at -20 dB down."
        ),
        ParameterResult(
            name="bw_3db",
            display_name="-3 dB (Half-Power) Bandwidth",
            value=f"{bw_3db_hz / 1e3:.2f}",
            unit="kHz",
            method="Spectral width at -3 dB relative to peak power",
            status=ParameterStatus.CONFIDENT,
            confidence=0.90,
            assumptions="Half-power bandwidth."
        ),
        ParameterResult(
            name="estimated_snr",
            display_name="Estimated Signal-to-Noise Ratio (SNR)",
            value=f"{snr_db:.2f}",
            unit="dB",
            method="Percentile PSD Floor Estimation (90th pct signal vs 25th pct noise)",
            status=ParameterStatus.EXPERIMENTAL,
            confidence=0.75,
            assumptions="Assumes background noise dominates lowest 25% spectral bins."
        ),
    ]

    return params, freqs, psd_db
