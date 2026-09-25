"""
Unit tests for DSP feature extraction (peak frequency, OBW, power metrics).
"""

import pytest
import numpy as np
import os

from signalscope.ingest import parse_raw_iq
from signalscope.metadata import IQFormatConfig, IQDType
from signalscope.features import (
    compute_fft_psd,
    compute_peak_frequency,
    compute_occupied_bandwidth_99,
    compute_xdb_bandwidth,
    extract_all_parameters
)

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_data")


def test_cw_tone_peak_frequency_accuracy():
    cw_path = os.path.join(SAMPLE_DIR, "cw_tone_100k.iq")
    config = IQFormatConfig(
        dtype=IQDType.FLOAT32,
        sample_rate=1.0e6,
        center_freq_hz=100.0e6  # 100 MHz RF
    )

    samples, meta = parse_raw_iq(cw_path, config)
    params, freqs, psd_db = extract_all_parameters(samples, meta)

    # Find peak baseband frequency parameter
    peak_bb_param = next(p for p in params if p.name == "peak_baseband_freq")
    peak_rf_param = next(p for p in params if p.name == "peak_rf_freq")

    peak_bb_val_khz = float(peak_bb_param.value)
    assert abs(peak_bb_val_khz - 100.0) < 2.0  # Within 2 kHz of 100 kHz target

    peak_rf_val_mhz = float(peak_rf_param.value)
    assert abs(peak_rf_val_mhz - 100.100) < 0.002  # 100 MHz + 100 kHz


def test_am_signal_obw_calculation():
    am_path = os.path.join(SAMPLE_DIR, "am_10khz_mod.iq")
    config = IQFormatConfig(dtype=IQDType.FLOAT32, sample_rate=2.0e6)

    samples, meta = parse_raw_iq(am_path, config)
    params, freqs, psd_db = extract_all_parameters(samples, meta)

    obw_param = next(p for p in params if p.name == "occupied_bw_99")
    obw_val_khz = float(obw_param.value)
    # AM 10 kHz audio mod should have sidebands at +-10 kHz -> ~20 kHz total OBW
    assert 15.0 <= obw_val_khz <= 30.0


def test_xdb_bandwidth():
    freqs = np.linspace(-500e3, 500e3, 1000)
    # Synthetic Gaussian PSD peak at 0 Hz with 10 kHz sigma
    psd_db = - (freqs / 10e3) ** 2

    bw_3db, f_low, f_high = compute_xdb_bandwidth(freqs, psd_db, x_db=3.0)
    # -3dB point for psd_db = -(f/10kHz)^2 occurs where -(f/10kHz)^2 = -3 -> f = sqrt(3)*10kHz ~ 17.32 kHz -> total BW ~ 34.64 kHz
    assert 30.0e3 <= bw_3db <= 38.0e3
