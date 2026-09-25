"""
Synthetic Sample Dataset Generator for SignalScope demo & verification.
Generates complex baseband IQ files and WAV audio files with known ground-truth parameters.
"""

import os
import json
import numpy as np
from scipy.io import wavfile

SAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_all_samples():
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    print(f"Generating synthetic signal samples in {SAMPLE_DIR}...")

    # 1. Single-Tone CW IQ (100 kHz, Fs=1 MS/s)
    fs_cw = 1.0e6
    t_cw = np.arange(10000) / fs_cw
    freq_cw = 100.0e3  # +100 kHz baseband tone
    s_cw = 0.7 * np.exp(1j * 2.0 * np.pi * freq_cw * t_cw).astype(np.complex64)
    
    cw_iq_path = os.path.join(SAMPLE_DIR, "cw_tone_100k.iq")
    cw_bytes = np.empty((len(s_cw) * 2,), dtype=np.float32)
    cw_bytes[0::2] = s_cw.real
    cw_bytes[1::2] = s_cw.imag
    with open(cw_iq_path, "wb") as f:
        f.write(cw_bytes.tobytes())

    cw_meta = {
        "filename": "cw_tone_100k.iq",
        "description": "Pure single-tone CW complex baseband IQ",
        "sample_rate_hz": fs_cw,
        "known_peak_baseband_freq_hz": freq_cw,
        "dtype": "float32",
        "ordering": "IQ",
        "expected_obw_99_hz": 1200.0,
        "expected_modulation": "CW / Unmodulated"
    }
    with open(os.path.join(SAMPLE_DIR, "cw_tone_100k.json"), "w") as f:
        json.dump(cw_meta, f, indent=2)

    # 2. Two-Tone IQ (50 kHz & 150 kHz, Fs=1 MS/s)
    fs_tt = 1.0e6
    t_tt = np.arange(10000) / fs_tt
    s_tt = (
        0.4 * np.exp(1j * 2.0 * np.pi * 50e3 * t_tt) +
        0.4 * np.exp(1j * 2.0 * np.pi * 150e3 * t_tt)
    ).astype(np.complex64)

    tt_iq_path = os.path.join(SAMPLE_DIR, "twotone_50k_150k.iq")
    tt_bytes = np.empty((len(s_tt) * 2,), dtype=np.float32)
    tt_bytes[0::2] = s_tt.real
    tt_bytes[1::2] = s_tt.imag
    with open(tt_iq_path, "wb") as f:
        f.write(tt_bytes.tobytes())

    tt_meta = {
        "filename": "twotone_50k_150k.iq",
        "description": "Dual tone (+50 kHz and +150 kHz) complex baseband IQ",
        "sample_rate_hz": fs_tt,
        "known_tones_hz": [50000.0, 150000.0],
        "dtype": "float32",
        "ordering": "IQ"
    }
    with open(os.path.join(SAMPLE_DIR, "twotone_50k_150k.json"), "w") as f:
        json.dump(tt_meta, f, indent=2)

    # 3. AM Signal (Carrier=200 kHz, Mod=10 kHz, Fs=2 MS/s)
    fs_am = 2.0e6
    t_am = np.arange(20000) / fs_am
    mod_signal = 0.8 * np.sin(2.0 * np.pi * 10e3 * t_am)
    s_am = ((1.0 + mod_signal) * np.exp(1j * 2.0 * np.pi * 200e3 * t_am)).astype(np.complex64)

    am_iq_path = os.path.join(SAMPLE_DIR, "am_10khz_mod.iq")
    am_bytes = np.empty((len(s_am) * 2,), dtype=np.float32)
    am_bytes[0::2] = s_am.real
    am_bytes[1::2] = s_am.imag
    with open(am_iq_path, "wb") as f:
        f.write(am_bytes.tobytes())

    am_meta = {
        "filename": "am_10khz_mod.iq",
        "description": "AM signal (200 kHz carrier, 10 kHz audio mod, m=0.8)",
        "sample_rate_hz": fs_am,
        "carrier_baseband_hz": 200000.0,
        "expected_obw_99_hz": 20000.0,
        "expected_modulation": "AM (Amplitude Modulated)"
    }
    with open(os.path.join(SAMPLE_DIR, "am_10khz_mod.json"), "w") as f:
        json.dump(am_meta, f, indent=2)

    # 4. FM Signal (Carrier=300 kHz, Mod=5 kHz, DeltaF=25 kHz, Fs=2 MS/s)
    fs_fm = 2.0e6
    t_fm = np.arange(20000) / fs_fm
    fm_audio = np.sin(2.0 * np.pi * 5e3 * t_fm)
    delta_f = 25e3
    phase_fm = 2.0 * np.pi * 300e3 * t_fm + (delta_f / 5e3) * np.sin(2.0 * np.pi * 5e3 * t_fm)
    s_fm = 0.7 * np.exp(1j * phase_fm).astype(np.complex64)

    fm_iq_path = os.path.join(SAMPLE_DIR, "fm_carrier_20khz.iq")
    fm_bytes = np.empty((len(s_fm) * 2,), dtype=np.float32)
    fm_bytes[0::2] = s_fm.real
    fm_bytes[1::2] = s_fm.imag
    with open(fm_iq_path, "wb") as f:
        f.write(fm_bytes.tobytes())

    fm_meta = {
        "filename": "fm_carrier_20khz.iq",
        "description": "FM signal (300 kHz carrier, 5 kHz audio, delta_f=25 kHz)",
        "sample_rate_hz": fs_fm,
        "carrier_baseband_hz": 300000.0,
        "expected_obw_99_hz": 60000.0,
        "expected_modulation": "FM (Frequency Modulated)"
    }
    with open(os.path.join(SAMPLE_DIR, "fm_carrier_20khz.json"), "w") as f:
        json.dump(fm_meta, f, indent=2)

    # 5. Audio WAV Sample (1 kHz sine tone, Fs=44.1 kHz, 1 sec)
    fs_wav = 44100
    t_wav = np.arange(fs_wav) / float(fs_wav)
    audio_tone = (0.6 * np.sin(2.0 * np.pi * 1000.0 * t_wav) * 32767).astype(np.int16)
    
    wav_path = os.path.join(SAMPLE_DIR, "audio_1khz_tone.wav")
    wavfile.write(wav_path, fs_wav, audio_tone)

    wav_meta = {
        "filename": "audio_1khz_tone.wav",
        "description": "1 kHz pure sine tone audio WAV file",
        "sample_rate_hz": fs_wav,
        "known_freq_hz": 1000.0,
        "duration_sec": 1.0
    }
    with open(os.path.join(SAMPLE_DIR, "audio_1khz_tone.json"), "w") as f:
        json.dump(wav_meta, f, indent=2)

    print("Sample dataset successfully generated.")


if __name__ == "__main__":
    generate_all_samples()
