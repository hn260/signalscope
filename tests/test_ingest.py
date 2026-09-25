"""
Unit tests for signal ingestion module (WAV and raw IQ files).
"""

import pytest
import numpy as np
import io
import os

from signalscope.ingest import parse_raw_iq, parse_wav
from signalscope.metadata import IQFormatConfig, IQDType, IQOrdering, Endianness


def test_parse_raw_iq_float32():
    # 4 complex samples = 8 float32 values
    raw_data = np.array([1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0], dtype=np.float32).tobytes()
    config = IQFormatConfig(
        dtype=IQDType.FLOAT32,
        ordering=IQOrdering.IQ,
        endianness=Endianness.LITTLE,
        sample_rate=1000000.0,
        center_freq_hz=433.92e6
    )

    samples, meta = parse_raw_iq(raw_data, config, filename="test.iq")
    assert len(samples) == 4
    assert meta.num_samples == 4
    assert meta.duration_sec == 4e-6
    assert meta.center_freq_hz == 433.92e6
    assert meta.frequency_is_absolute is True
    assert samples[0] == 1.0 + 0.0j
    assert samples[1] == 0.0 + 1.0j


def test_parse_raw_iq_int16_qi_ordering():
    # Int16 values normalized by 32768
    raw_data = np.array([0, 32767, 32767, 0], dtype=np.int16).tobytes()
    config = IQFormatConfig(
        dtype=IQDType.INT16,
        ordering=IQOrdering.QI,  # Q first, then I
        endianness=Endianness.LITTLE,
        sample_rate=2000000.0
    )

    samples, meta = parse_raw_iq(raw_data, config, filename="test_qi.iq")
    assert len(samples) == 2
    # First sample: Q=0, I=32767/32768 -> I + 1j*Q = 0.999969 + 0j
    assert np.isclose(samples[0].real, 1.0, atol=1e-3)
    assert np.isclose(samples[0].imag, 0.0, atol=1e-3)


def test_parse_raw_iq_odd_length_error():
    # Odd number of float32 elements -> should raise ValueError
    raw_data = np.array([1.0, 0.0, 0.5], dtype=np.float32).tobytes()
    config = IQFormatConfig(dtype=IQDType.FLOAT32, sample_rate=1e6)

    with pytest.raises(ValueError, match="odd"):
        parse_raw_iq(raw_data, config)


def test_parse_raw_iq_empty_error():
    config = IQFormatConfig(dtype=IQDType.FLOAT32, sample_rate=1e6)
    with pytest.raises(ValueError, match="empty"):
        parse_raw_iq(b"", config)


def test_parse_synthetic_wav(tmp_path):
    import scipy.io.wavfile as wavfile
    
    fs = 44100
    t = np.linspace(0, 0.1, int(fs * 0.1), endpoint=False)
    sine_wave = (0.5 * np.sin(2.0 * np.pi * 440 * t) * 32767).astype(np.int16)
    
    wav_file = tmp_path / "test_tone.wav"
    wavfile.write(str(wav_file), fs, sine_wave)

    samples, meta = parse_wav(str(wav_file))
    assert meta.file_format == "WAV"
    assert meta.sample_rate_hz == 44100.0
    assert meta.num_samples == len(sine_wave)
    assert np.max(np.abs(samples)) <= 1.0
