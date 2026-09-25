"""
File ingestion module for WAV audio and raw complex baseband IQ recordings.
Handles byte parsing, dtype mapping, byte order, I/Q ordering, normalization, and validation.
"""

import os
import io
import numpy as np
from typing import Tuple, Union, Optional
from scipy.io import wavfile

from .metadata import (
    IQFormatConfig,
    IQDType,
    IQOrdering,
    Endianness,
    SignalMetadata,
    MetadataOrigin,
)


def _get_numpy_dtype(iq_config: IQFormatConfig) -> np.dtype:
    prefix = "<" if iq_config.endianness == Endianness.LITTLE else ">"
    dtype_map = {
        IQDType.INT8: np.dtype("i1"),
        IQDType.INT16: np.dtype(f"{prefix}i2"),
        IQDType.INT32: np.dtype(f"{prefix}i4"),
        IQDType.FLOAT32: np.dtype(f"{prefix}f4"),
        IQDType.FLOAT64: np.dtype(f"{prefix}f8"),
    }
    return dtype_map[iq_config.dtype]


def parse_raw_iq(
    source: Union[str, bytes, io.BytesIO],
    config: IQFormatConfig,
    filename: str = "raw_recording.iq"
) -> Tuple[np.ndarray, SignalMetadata]:
    """
    Ingests binary interleaved IQ samples into a complex64 1D NumPy array.
    
    Raises ValueError on invalid file length, non-even sample pairs, or corrupted values.
    """
    if isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(f"IQ file not found: {source}")
        filename = os.path.basename(source)
        file_size = os.path.getsize(source)
        with open(source, "rb") as f:
            raw_bytes = f.read()
    elif isinstance(source, io.BytesIO):
        raw_bytes = source.getvalue()
        file_size = len(raw_bytes)
    elif isinstance(source, bytes):
        raw_bytes = source
        file_size = len(raw_bytes)
    else:
        raise TypeError("Source must be file path, bytes, or BytesIO object")

    if file_size == 0:
        raise ValueError("Provided IQ file is empty (0 bytes)")

    np_dtype = _get_numpy_dtype(config)
    element_size = np_dtype.itemsize
    total_elements = len(raw_bytes) // element_size

    if len(raw_bytes) % element_size != 0:
        raise ValueError(
            f"File size ({file_size} bytes) is not a multiple of sample element size ({element_size} bytes)"
        )

    if total_elements % 2 != 0:
        raise ValueError(
            f"Total sample elements ({total_elements}) is odd. IQ signals require even numbers of interleaved I/Q samples."
        )

    samples_raw = np.frombuffer(raw_bytes, dtype=np_dtype)

    # Convert to float32 normalized range [-1.0, 1.0]
    if config.dtype == IQDType.INT8:
        samples_float = samples_raw.astype(np.float32) / 128.0
    elif config.dtype == IQDType.INT16:
        samples_float = samples_raw.astype(np.float32) / 32768.0
    elif config.dtype == IQDType.INT32:
        samples_float = samples_raw.astype(np.float32) / 2147483648.0
    else:
        samples_float = samples_raw.astype(np.float32)

    # Separate I and Q
    if config.ordering == IQOrdering.IQ:
        i_samples = samples_float[0::2]
        q_samples = samples_float[1::2]
    else:
        q_samples = samples_float[0::2]
        i_samples = samples_float[1::2]

    complex_samples = (i_samples + 1j * q_samples).astype(np.complex64)

    # Sanity checks
    if np.any(np.isnan(complex_samples)) or np.any(np.isinf(complex_samples)):
        raise ValueError("Ingested IQ signal contains NaN or Inf values")

    num_samples = len(complex_samples)
    duration = num_samples / config.sample_rate

    metadata = SignalMetadata(
        filename=filename,
        file_format="RAW_IQ",
        file_size_bytes=file_size,
        num_samples=num_samples,
        duration_sec=duration,
        sample_rate_hz=config.sample_rate,
        center_freq_hz=config.center_freq_hz,
        frequency_is_absolute=config.center_freq_hz is not None and config.center_freq_hz > 0,
        metadata_origin=MetadataOrigin.USER_INPUT,
        iq_config=config,
        num_channels=1
    )

    return complex_samples, metadata


def parse_wav(
    source: Union[str, bytes, io.BytesIO],
    filename: str = "audio_recording.wav",
    channel_mode: str = "mix"
) -> Tuple[np.ndarray, SignalMetadata]:
    """
    Ingests WAV audio files into a real or complex float32 NumPy array.
    """
    if isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(f"WAV file not found: {source}")
        filename = os.path.basename(source)
        file_size = os.path.getsize(source)
        read_target = source
    elif isinstance(source, (bytes, io.BytesIO)):
        bio = source if isinstance(source, io.BytesIO) else io.BytesIO(source)
        file_size = bio.getbuffer().nbytes
        read_target = bio
    else:
        raise TypeError("Source must be file path, bytes, or BytesIO object")

    try:
        sample_rate, raw_data = wavfile.read(read_target)
    except Exception as e:
        raise ValueError(f"Failed to parse WAV header/data: {str(e)}")

    if raw_data.size == 0:
        raise ValueError("WAV file contains 0 audio samples")

    # Determine channel format
    if raw_data.ndim == 1:
        audio_data = raw_data
        num_channels = 1
    else:
        num_channels = raw_data.shape[1]
        if channel_mode == "left" or channel_mode == "0":
            audio_data = raw_data[:, 0]
        elif channel_mode == "right" or channel_mode == "1":
            audio_data = raw_data[:, min(1, num_channels - 1)]
        else:  # mix / average
            audio_data = raw_data.mean(axis=1)

    # Normalize dtype to float32 [-1.0, +1.0]
    if raw_data.dtype == np.uint8:
        float_samples = (audio_data.astype(np.float32) - 128.0) / 128.0
    elif raw_data.dtype == np.int16:
        float_samples = audio_data.astype(np.float32) / 32768.0
    elif raw_data.dtype == np.int32:
        float_samples = audio_data.astype(np.float32) / 2147483648.0
    elif raw_data.dtype in (np.float32, np.float64):
        float_samples = audio_data.astype(np.float32)
    else:
        float_samples = audio_data.astype(np.float32)

    num_samples = len(float_samples)
    duration = num_samples / float(sample_rate)

    metadata = SignalMetadata(
        filename=filename,
        file_format="WAV",
        file_size_bytes=file_size,
        num_samples=num_samples,
        duration_sec=duration,
        sample_rate_hz=float(sample_rate),
        center_freq_hz=None,  # Audio baseband by default
        frequency_is_absolute=False,
        metadata_origin=MetadataOrigin.FILE_HEADER,
        num_channels=num_channels
    )

    return float_samples, metadata
