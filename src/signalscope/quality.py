"""
Quality assurance and flag generator module.
Evaluates signals for clipping, missing center frequency, short sample count, low SNR, and DC bias.
"""

import numpy as np
from typing import List, Tuple
from .metadata import SignalMetadata, QualityFlag, QualitySeverity


def evaluate_signal_quality(
    samples: np.ndarray,
    metadata: SignalMetadata,
    snr_db: float
) -> Tuple[List[QualityFlag], float]:
    """
    Evaluates signal parameters against quality criteria.
    
    Returns:
        Tuple of (flags_list, overall_quality_score_percent)
    """
    flags: List[QualityFlag] = []
    score_deductions = 0.0

    mag = np.abs(samples)
    peak_val = float(np.max(mag)) if len(mag) > 0 else 0.0

    # 1. Missing Center Frequency Flag
    if metadata.file_format == "RAW_IQ" and not metadata.has_center_frequency:
        flags.append(
            QualityFlag(
                code="MISSING_CENTER_FREQ",
                severity=QualitySeverity.WARNING,
                message="RF Center Frequency was not provided for this IQ recording.",
                recommendation="Provide the SDR tuned center frequency in metadata settings to display absolute RF values instead of relative baseband frequencies."
            )
        )
        score_deductions += 10.0

    # 2. Clipping / Saturation Flag
    if peak_val >= 0.98:
        flags.append(
            QualityFlag(
                code="ADC_CLIPPING_DETECTED",
                severity=QualitySeverity.WARNING,
                message=f"Signal peak amplitude ({peak_val:.3f}) reaches or exceeds full scale.",
                recommendation="Possible ADC saturation or clipping distortion. Reduce gain during recording if possible."
            )
        )
        score_deductions += 25.0

    # 3. Short Sample Count Flag
    if metadata.num_samples < 2048:
        flags.append(
            QualityFlag(
                code="SHORT_RECORDING_LENGTH",
                severity=QualitySeverity.WARNING,
                message=f"Sample count ({metadata.num_samples}) is low (< 2048 samples).",
                recommendation="FFT frequency resolution may be coarse. Record longer sample durations for precise bandwidth measurement."
            )
        )
        score_deductions += 15.0

    # 4. Low SNR Flag
    if snr_db < 6.0:
        flags.append(
            QualityFlag(
                code="LOW_SNR_WARNING",
                severity=QualitySeverity.WARNING,
                message=f"Estimated SNR ({snr_db:.1f} dB) is low (< 6 dB).",
                recommendation="Signal is noise-dominated. Feature classification and bandwidth estimates may be impaired."
            )
        )
        score_deductions += 20.0

    # 5. DC Bias Flag
    dc_offset = abs(np.mean(samples))
    if dc_offset > 0.05:
        flags.append(
            QualityFlag(
                code="DC_BIAS_PRESENT",
                severity=QualitySeverity.INFO,
                message=f"Significant DC bias detected (|DC| = {dc_offset:.4f}).",
                recommendation="Enable 'Remove DC Offset' preprocessing filter."
            )
        )
        score_deductions += 5.0

    # 6. Sample Rate Sanity Flag
    if metadata.sample_rate_hz < 100.0 or metadata.sample_rate_hz > 10e9:
        flags.append(
            QualityFlag(
                code="UNUSUAL_SAMPLE_RATE",
                severity=QualitySeverity.WARNING,
                message=f"Configured sample rate ({metadata.sample_rate_hz:.1f} Hz) is outside typical SDR ranges.",
                recommendation="Verify the configured sample rate matches the hardware capture rate."
            )
        )
        score_deductions += 10.0

    overall_score = max(0.0, 100.0 - score_deductions)
    return flags, overall_score
