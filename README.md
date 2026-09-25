# SignalScope — Offline-First RF & Audio Signal Analyzer

**Smart India Hackathon 2026 (SIH 2026)**  
**Problem Statement ID:** SIH26147  
**Title:** Automated Model for Analysis of .IQ and .wav Files Along with Signal Parameter Extraction  
**Organization:** NTRO (National Technical Research Organisation)  
**Theme:** Space Technology  
**Category:** Software  

---

## 📌 Executive Summary

**SignalScope** is an offline-first RF baseband and WAV audio signal analysis toolkit. It ingests IQ baseband recordings and audio files, visualizes waveforms, spectra, and spectrograms, extracts measurable signal parameters with explicit method transparency and confidence ratings, classifies candidate modulation types using statistical instantaneous features, and exports auditable JSON and CSV analysis reports.

---

## 🛡️ Honesty & Transparency Policy

SignalScope strictly adheres to the following domain limits:
1. **Relative vs. Absolute Frequency:** Baseband complex IQ recordings do not implicitly contain RF center frequency information. SignalScope explicitly labels frequency values as **Relative Baseband (Hz/kHz)** unless the user provides center frequency metadata.
2. **Occupied Bandwidth Methods:** SignalScope provides two distinct, documented bandwidth estimation methods:
   - **99% Occupied Power Bandwidth:** Spectral integration across the power density curve.
   - **X-dB Down Bandwidth:** Spectral width relative to the peak power bin (-3 dB, -20 dB).
3. **Explainable Modulation Classifier:** Modulation classification outputs are presented as **statistical model estimates** with candidate confidence indicators rather than ground-truth intelligence assertions.
4. **Deterministic Verification:** Includes bundled synthetic sample recordings with known ground-truth metrics for deterministic verification.

---

## 🚀 Quickstart Guide

### Prerequisites
- Python 3.11+
- Git

### Installation

```bash
# Clone repository
git clone https://github.com/Antigravity/signalscope.git
cd signalscope

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies and signalscope package
pip install -e .
```

### Running the Web Dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser to access the interactive SignalScope dashboard.

### Running Automated Test Suite

```bash
pytest -v
```

---

## 📁 Repository Structure

```text
signalscope/
├── app.py                       # Streamlit multi-tab dashboard entry point
├── pyproject.toml               # Package build metadata & dependencies
├── requirements.txt             # Dependency specification
├── README.md                    # Project documentation & SIH guide
├── sample_data/                 # Synthetic demo recordings & ground truth
│   ├── generate_samples.py      # Generator script for test signals
│   ├── README.md                # Sample dataset provenance table
│   ├── cw_tone_100k.iq          # 100 kHz CW tone sample
│   ├── twotone_50k_150k.iq      # Dual tone sample (+50 kHz, +150 kHz)
│   ├── am_10khz_mod.iq          # AM signal (200 kHz carrier, 10 kHz audio)
│   ├── fm_carrier_20khz.iq      # FM signal (300 kHz carrier, 25 kHz delta_f)
│   └── audio_1khz_tone.wav      # 1 kHz sine audio WAV recording
├── src/
│   └── signalscope/             # Core DSP & Analysis Engine
│       ├── __init__.py
│       ├── metadata.py          # Pydantic schemas & IQ configuration models
│       ├── ingest.py            # WAV & raw binary IQ byte parser
│       ├── preprocessing.py     # DC offset removal, normalization, filtering
│       ├── features.py          # DSP parameter extraction routines
│       ├── quality.py           # Signal quality diagnostic flags
│       ├── classification.py    # Instantaneous feature extractor & classifier
│       ├── visualization.py     # Plotly interactive charting engine
│       ├── report.py            # JSON & CSV report generators
│       └── utils.py             # Frequency & byte unit formatters
└── tests/                       # Automated pytest suite
    ├── test_ingest.py           # Ingestion & parser tests
    ├── test_features.py         # Sub-bin frequency & OBW tests
    ├── test_classification.py   # Modulation feature extraction tests
    └── test_report.py           # Report schema validation tests
```

---

## 🔬 DSP & Parameter Extraction Methodology

| Parameter | Display Unit | Measurement Method & Formula | Status |
| :--- | :--- | :--- | :--- |
| **Sample Rate** | MS/s / kHz | Header parsing or user configuration | `CONFIDENT` |
| **Duration** | seconds / ms | $T = N / F_s$ | `CONFIDENT` |
| **Peak Amplitude** | FS (Normalized) | $\max \|s[n]\|$ | `CONFIDENT` |
| **RMS Power** | dBFS | $20 \log_{10} \sqrt{\frac{1}{N} \sum \|s[n]\|^2}$ | `CONFIDENT` |
| **Crest Factor (PAR)** | ratio / dB | $\frac{\text{Peak Amplitude}}{\text{RMS Amplitude}}$ | `CONFIDENT` |
| **Peak Baseband Freq** | kHz | Welch PSD peak bin with Hann windowing | `CONFIDENT` |
| **Peak Absolute RF Freq** | MHz | $f_{RF} = f_{center} + f_{baseband}$ | `CONFIDENT` (if center freq provided) |
| **99% Occupied BW** | kHz | Spectral power cumulative integration interval | `CONFIDENT` |
| **-20 dB Down BW** | kHz | Frequency span where $PSD(f) \ge PSD_{peak} - 20\text{ dB}$ | `CONFIDENT` |
| **Estimated SNR** | dB | Percentile energy floor ($P_{90} - P_{25}$) | `EXPERIMENTAL` |

---

## 📻 Feature-Based Modulation Classification

Instantaneous signal features computed for candidate classification:
- **$\gamma_{max}$**: Maximum spectral density of normalized instantaneous amplitude.
- **$\sigma_{aa}^2$**: Variance of normalized envelope magnitude.
- **$\sigma_{ap}^2$**: Variance of non-linear unwrapped phase.
- **$\sigma_{af}^2$**: Variance of normalized instantaneous frequency.

Supported candidate classes: `CW / Unmodulated`, `AM`, `FM`, `2-FSK`, `BPSK / QPSK`.

---

## 🧪 Verification & Test Results

All 11 unit tests pass deterministically:
- Raw binary IQ ingestion verified for `int8`, `int16`, `int32`, `float32`, `IQ` / `QI` ordering.
- Peak frequency recovery verified on synthetic tones within sub-bin tolerance (< 1% error).
- 99% OBW and -20dB bandwidth calculation algorithms validated against analytical targets.
- JSON and CSV report schema generation verified.

---

## 📜 License & Compliance

Developed for SIH 2026. Built strictly using open-source Python libraries (`NumPy`, `SciPy`, `Plotly`, `Streamlit`, `Pydantic`, `pytest`). All sample recordings are synthetic and non-confidential.
