"""
Unit tests for JSON and CSV report generation.
"""

import pytest
import json
import os

from signalscope.ingest import parse_raw_iq
from signalscope.metadata import IQFormatConfig, IQDType
from signalscope.features import extract_all_parameters
from signalscope.quality import evaluate_signal_quality
from signalscope.classification import extract_modulation_features, classify_modulation
from signalscope.report import generate_json_report, generate_csv_table

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_data")


def test_report_generation_json_and_csv():
    cw_path = os.path.join(SAMPLE_DIR, "cw_tone_100k.iq")
    config = IQFormatConfig(dtype=IQDType.FLOAT32, sample_rate=1.0e6)
    samples, meta = parse_raw_iq(cw_path, config)

    params, freqs, psd_db = extract_all_parameters(samples, meta)
    flags, score = evaluate_signal_quality(samples, meta, snr_db=20.0)
    feats = extract_modulation_features(samples, meta.sample_rate_hz)
    class_res = classify_modulation(feats)

    json_str = generate_json_report(meta, params, flags, score, class_res)
    report_dict = json.loads(json_str)

    assert "signalscope_version" in report_dict
    assert report_dict["metadata"]["filename"] == "cw_tone_100k.iq"
    assert len(report_dict["extracted_parameters"]) > 5
    assert "honesty_disclaimer" in report_dict

    csv_str = generate_csv_table(params)
    assert "Parameter,Value,Unit,Measurement Method" in csv_str
    assert "Sample Rate" in csv_str
