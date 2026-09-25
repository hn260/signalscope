"""
Unit tests for modulation feature extraction and classification.
"""

import pytest
import numpy as np
import os

from signalscope.ingest import parse_raw_iq
from signalscope.metadata import IQFormatConfig, IQDType
from signalscope.classification import extract_modulation_features, classify_modulation

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_data")


def test_cw_classification():
    cw_path = os.path.join(SAMPLE_DIR, "cw_tone_100k.iq")
    config = IQFormatConfig(dtype=IQDType.FLOAT32, sample_rate=1.0e6)
    samples, meta = parse_raw_iq(cw_path, config)

    feats = extract_modulation_features(samples, meta.sample_rate_hz)
    best_class, confidence, probs = classify_modulation(feats, is_complex=True)

    assert best_class == "CW / Unmodulated"
    assert confidence >= 0.50
    assert feats["sigma_aa"] < 0.05  # Near zero envelope variation


def test_am_classification():
    am_path = os.path.join(SAMPLE_DIR, "am_10khz_mod.iq")
    config = IQFormatConfig(dtype=IQDType.FLOAT32, sample_rate=2.0e6)
    samples, meta = parse_raw_iq(am_path, config)

    feats = extract_modulation_features(samples, meta.sample_rate_hz)
    best_class, confidence, probs = classify_modulation(feats, is_complex=True)

    assert best_class == "AM (Amplitude Modulated)"
    assert feats["sigma_aa"] > 0.05  # Significant envelope modulation
