# SignalScope Demo Dataset Provenance & Ground Truth

This folder contains synthetic, mathematically generated sample recordings used for verifying SignalScope's ingestion pipeline, parameter extraction metrics, and visualization components.

> [!NOTE]
> Synthetic data is used strictly for deterministic verification of sub-bin frequency estimation, bandwidth algorithms, and model feature extraction. Real-world SDR recordings may contain interference, multi-path fading, and non-Gaussian noise.

---

## 1. `cw_tone_100k.iq`
- **Description:** Single-tone continuous wave (CW) complex baseband IQ recording.
- **Sample Rate:** 1.0 MS/s (1,000,000 Hz)
- **Duration:** 10 ms (10,000 complex samples)
- **Data Type:** `float32` interleaved I/Q
- **Ground Truth Parameters:**
  - Baseband Peak Frequency: `+100.0 kHz`
  - 99% Occupied Bandwidth: `< 2.0 kHz`
  - Expected Modulation: `CW / Unmodulated`

---

## 2. `twotone_50k_150k.iq`
- **Description:** Dual-tone complex baseband IQ signal with equal amplitude tones at +50 kHz and +150 kHz.
- **Sample Rate:** 1.0 MS/s
- **Duration:** 10 ms (10,000 complex samples)
- **Data Type:** `float32` interleaved I/Q
- **Ground Truth Parameters:**
  - Spectral Peaks: `+50.0 kHz` and `+150.0 kHz`

---

## 3. `am_10khz_mod.iq`
- **Description:** Double-sideband amplitude-modulated (AM) complex signal with a 200 kHz carrier and 10 kHz audio tone modulation ($m = 0.8$).
- **Sample Rate:** 2.0 MS/s
- **Duration:** 10 ms (20,000 complex samples)
- **Data Type:** `float32` interleaved I/Q
- **Ground Truth Parameters:**
  - Carrier Frequency: `+200.0 kHz`
  - Modulation Rate: `10.0 kHz`
  - 99% Occupied Bandwidth: `~ 20.0 kHz`
  - Expected Modulation: `AM (Amplitude Modulated)`

---

## 4. `fm_carrier_20khz.iq`
- **Description:** Frequency-modulated (FM) complex signal with a 300 kHz carrier, 5 kHz audio modulating tone, and peak deviation $\Delta f = 25$ kHz.
- **Sample Rate:** 2.0 MS/s
- **Duration:** 10 ms (20,000 complex samples)
- **Data Type:** `float32` interleaved I/Q
- **Ground Truth Parameters:**
  - Carrier Frequency: `+300.0 kHz`
  - Carson's Rule Bandwidth: $2(\Delta f + f_m) = 2(25 + 5) = 60$ kHz
  - 99% Occupied Bandwidth: `~ 60.0 kHz`
  - Expected Modulation: `FM (Frequency Modulated)`

---

## 5. `audio_1khz_tone.wav`
- **Description:** 1 kHz audio sine wave stored as a standard 16-bit mono WAV file.
- **Sample Rate:** 44.1 kHz (44,100 Hz)
- **Duration:** 1.0 second (44,100 samples)
- **Ground Truth Parameters:**
  - Audio Peak Frequency: `1000.0 Hz`
