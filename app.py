"""
SignalScope Streamlit Web Application
SIH 2026 | Problem Statement: SIH26147 (NTRO)
Automated Model for Analysis of .IQ and .wav Files Along with Signal Parameter Extraction
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st

from signalscope.metadata import (
    IQFormatConfig,
    IQDType,
    IQOrdering,
    Endianness,
    SignalMetadata,
    MetadataOrigin,
)
from signalscope.ingest import parse_raw_iq, parse_wav
from signalscope.preprocessing import remove_dc_offset, normalize_amplitude
from signalscope.features import extract_all_parameters
from signalscope.quality import evaluate_signal_quality
from signalscope.classification import extract_modulation_features, classify_modulation
from signalscope.visualization import plot_time_domain, plot_psd, plot_spectrogram
from signalscope.report import generate_json_report, generate_csv_table
from signalscope.utils import format_frequency, format_bytes

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_data")

# Streamlit Page Config
st.set_page_config(
    page_title="SignalScope — RF & Audio Analysis Engine",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title & Header
st.title("📡 SignalScope — RF & Audio Signal Analyzer")
st.caption("Smart India Hackathon 2026 | Problem Statement: SIH26147 (NTRO) | Offline-First Baseband & WAV Processor")

st.markdown("---")

# Sidebar Configuration
st.sidebar.header("📁 Input Recording & Settings")

input_mode = st.sidebar.radio(
    "Data Source",
    ["Load Bundled Synthetic Sample", "Upload Custom File"],
    index=0
)

samples_data = None
metadata = None
iq_config = None

if input_mode == "Load Bundled Synthetic Sample":
    sample_files = {
        "1. CW Tone 100 kHz (.iq)": ("cw_tone_100k.iq", "cw_tone_100k.json"),
        "2. Two-Tone 50 kHz & 150 kHz (.iq)": ("twotone_50k_150k.iq", "twotone_50k_150k.json"),
        "3. AM Modulated Signal 10 kHz (.iq)": ("am_10khz_mod.iq", "am_10khz_mod.json"),
        "4. FM Carrier 300 kHz (.iq)": ("fm_carrier_20khz.iq", "fm_carrier_20khz.json"),
        "5. Audio 1 kHz Sine Tone (.wav)": ("audio_1khz_tone.wav", "audio_1khz_tone.json"),
    }
    
    selected_sample_label = st.sidebar.selectbox("Select Synthetic Demo Recording", list(sample_files.keys()))
    iq_filename, json_filename = sample_files[selected_sample_label]
    
    file_path = os.path.join(SAMPLE_DIR, iq_filename)
    json_path = os.path.join(SAMPLE_DIR, json_filename)
    
    with open(json_path, "r") as f:
        ground_truth = json.load(f)

    st.sidebar.info(f"**Sample Ground Truth:**\n{ground_truth.get('description', '')}")

    if iq_filename.endswith(".iq"):
        center_freq_input = st.sidebar.number_input(
            "Optional RF Center Frequency (MHz)",
            value=100.0,
            step=1.0,
            help="Specify SDR tuned RF center frequency for absolute frequency scaling."
        )
        center_freq_hz = center_freq_input * 1e6 if center_freq_input > 0 else None

        iq_config = IQFormatConfig(
            dtype=IQDType(ground_truth.get("dtype", "float32")),
            ordering=IQOrdering(ground_truth.get("ordering", "IQ")),
            endianness=Endianness.LITTLE,
            sample_rate=float(ground_truth.get("sample_rate_hz", 1e6)),
            center_freq_hz=center_freq_hz
        )
        samples_data, metadata = parse_raw_iq(file_path, iq_config, filename=iq_filename)
    else:
        samples_data, metadata = parse_wav(file_path, filename=iq_filename)

else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload Recording (.iq, .raw, .bin, .wav, .dat)",
        type=["iq", "raw", "bin", "wav", "dat"]
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        filename = uploaded_file.name

        if filename.lower().endswith(".wav"):
            samples_data, metadata = parse_wav(file_bytes, filename=filename)
        else:
            st.sidebar.subheader("⚙️ Raw IQ Format Parameters")
            dtype_val = st.sidebar.selectbox("Numeric Data Type", [e.value for e in IQDType], index=3)
            ordering_val = st.sidebar.selectbox("I/Q Interleaving Order", [e.value for e in IQOrdering], index=0)
            endian_val = st.sidebar.selectbox("Byte Endianness", [e.value for e in Endianness], index=0)
            sample_rate_mhz = st.sidebar.number_input("Sample Rate (MS/s)", value=1.0, min_value=0.001, step=0.5)
            
            enable_center_freq = st.sidebar.checkbox("Specify Known RF Center Frequency", value=False)
            center_freq_mhz = st.sidebar.number_input(
                "RF Center Frequency (MHz)",
                value=433.92,
                disabled=not enable_center_freq
            ) if enable_center_freq else None

            iq_config = IQFormatConfig(
                dtype=IQDType(dtype_val),
                ordering=IQOrdering(ordering_val),
                endianness=Endianness(endian_val),
                sample_rate=sample_rate_mhz * 1e6,
                center_freq_hz=center_freq_mhz * 1e6 if center_freq_mhz else None
            )

            try:
                samples_data, metadata = parse_raw_iq(file_bytes, iq_config, filename=filename)
            except Exception as e:
                st.error(f"❌ Failed to parse raw IQ file: {str(e)}")
                st.stop()
    else:
        st.info("👈 Please select a demo recording or upload a file from the sidebar to begin analysis.")
        st.stop()

# Preprocessing Controls in Sidebar
st.sidebar.subheader("🔧 Preprocessing Filters")
remove_dc = st.sidebar.checkbox("Remove DC Offset", value=True)
normalize_amp = st.sidebar.checkbox("Normalize Peak Amplitude", value=False)

if samples_data is not None:
    if remove_dc:
        samples_data, dc_i, dc_q = remove_dc_offset(samples_data)
        metadata.preprocessing_applied.append("DC Offset Removal")
    if normalize_amp:
        samples_data, scale = normalize_amplitude(samples_data)
        metadata.preprocessing_applied.append("Peak Normalization")

    # Run Analysis Engine
    parameters, freqs, psd_db = extract_all_parameters(samples_data, metadata)
    snr_param = next(p for p in parameters if p.name == "estimated_snr")
    snr_val = float(snr_param.value)

    quality_flags, quality_score = evaluate_signal_quality(samples_data, metadata, snr_val)
    feats = extract_modulation_features(samples_data, metadata.sample_rate_hz)
    class_result = classify_modulation(feats, is_complex=np.iscomplexobj(samples_data))
    best_class, class_confidence, class_probs = class_result

    # Top Metric KPI Bar
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    kpi1.metric("Format", metadata.file_format)
    kpi2.metric("Duration", f"{metadata.duration_sec*1e3:.1f} ms")
    kpi3.metric("Sample Rate", format_frequency(metadata.sample_rate_hz) + "/s")
    
    freq_label = "RF Center Known" if metadata.has_center_frequency else "Relative Baseband"
    kpi4.metric("Frequency Reference", freq_label)
    
    score_color = "normal" if quality_score >= 80 else "inverse"
    kpi5.metric("Quality Rating", f"{quality_score:.0f}%", delta=f"{len(quality_flags)} flags")
    kpi6.metric("Predicted Modulation", best_class.split("(")[0].strip())

    st.markdown("---")

    # Main Tab Layout
    tab_vis, tab_params, tab_mod, tab_qual, tab_export, tab_about = st.tabs([
        "📊 Visualizations",
        "📋 Parameter Table",
        "📻 Modulation Analysis",
        "⚠️ Quality & Flags",
        "📥 Report & Export",
        "ℹ️ System Overview & Honesty Policy"
    ])

    with tab_vis:
        st.subheader("Signal Time & Frequency Domain Visualizations")

        col_time, col_fft = st.columns(2)
        with col_time:
            fig_wave = plot_time_domain(samples_data, metadata.sample_rate_hz)
            st.plotly_chart(fig_wave, use_container_width=True)

        with col_fft:
            peak_bb_param = next(p for p in parameters if p.name == "peak_baseband_freq")
            obw_param = next(p for p in parameters if p.name == "occupied_bw_99")
            
            peak_freq_hz = float(peak_bb_param.value) * 1e3
            obw_hz = float(obw_param.value) * 1e3
            f_low_hz = peak_freq_hz - obw_hz / 2.0
            f_high_hz = peak_freq_hz + obw_hz / 2.0

            fig_psd = plot_psd(
                freqs, psd_db,
                peak_freq_hz=peak_freq_hz,
                obw_99_hz=obw_hz,
                f_low_hz=f_low_hz,
                f_high_hz=f_high_hz,
                center_freq_hz=metadata.center_freq_hz
            )
            st.plotly_chart(fig_psd, use_container_width=True)

        st.markdown("#### STFT Spectrogram")
        fig_spec = plot_spectrogram(samples_data, metadata.sample_rate_hz, center_freq_hz=metadata.center_freq_hz)
        st.plotly_chart(fig_spec, use_container_width=True)

    with tab_params:
        st.subheader("Extracted Signal Parameters & Measurement Methods")
        st.caption("Each parameter includes standard measurement units, mathematical method description, confidence score, and underlying assumptions.")

        param_records = [
            {
                "Parameter Name": p.display_name,
                "Measured Value": p.value,
                "Unit": p.unit,
                "Measurement Method": p.method,
                "Status": p.status.value,
                "Confidence": f"{p.confidence*100:.0f}%",
                "Assumptions": p.assumptions
            }
            for p in parameters
        ]
        df_params = pd.DataFrame(param_records)
        st.dataframe(df_params, use_container_width=True, hide_index=True)

    with tab_mod:
        st.subheader("Feature-Based Modulation Candidate Classification")
        
        col_feats, col_probs = st.columns(2)
        with col_feats:
            st.markdown("#### Instantaneous Statistical Features")
            st.json({
                "γ_max (Max Spectral Density of Amplitude)": f"{feats['gamma_max']:.6f}",
                "σ_aa² (Normalized Envelope Variance)": f"{feats['sigma_aa']:.6f}",
                "σ_ap² (Phase Non-linear Variance)": f"{feats['sigma_ap']:.6f}",
                "σ_af² (Instantaneous Frequency Variance)": f"{feats['sigma_af']:.6f}"
            })

        with col_probs:
            st.markdown("#### Candidate Class Probability Distribution")
            df_probs = pd.DataFrame(
                list(class_probs.items()),
                columns=["Modulation Format", "Probability"]
            ).sort_values(by="Probability", ascending=False)
            
            st.bar_chart(df_probs.set_index("Modulation Format"))

        st.warning(
            "⚠️ **Scope & Model Limits Disclaimer:** Modulation classification outputs are model heuristic estimates "
            "derived from statistical instantaneous features. They should be treated as candidate indicators rather than ground-truth intelligence."
        )

    with tab_qual:
        st.subheader("Signal Quality Diagnostic Assessment")
        st.progress(quality_score / 100.0, text=f"Overall Quality Rating Score: {quality_score:.0f}%")

        if len(quality_flags) == 0:
            st.success("✅ Clean recording! No quality flags or hardware warnings triggered.")
        else:
            for flag in quality_flags:
                if flag.severity.value == "WARNING":
                    st.warning(f"**[{flag.code}]** {flag.message}\n\n👉 *Recommendation:* {flag.recommendation}")
                else:
                    st.info(f"**[{flag.code}]** {flag.message}\n\n👉 *Recommendation:* {flag.recommendation}")

    with tab_export:
        st.subheader("Download & Export Signal Reports")

        json_report_bytes = generate_json_report(metadata, parameters, quality_flags, quality_score, class_result)
        csv_table_bytes = generate_csv_table(parameters)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                label="📄 Download JSON Analysis Report",
                data=json_report_bytes,
                file_name=f"{os.path.splitext(metadata.filename)[0]}_signalscope_report.json",
                mime="application/json",
                use_container_width=True
            )
        with col_dl2:
            st.download_button(
                label="📊 Download CSV Parameter Table",
                data=csv_table_bytes,
                file_name=f"{os.path.splitext(metadata.filename)[0]}_parameters.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.markdown("#### Preview JSON Report Output")
        st.code(json_report_bytes, language="json")

    with tab_about:
        st.subheader("SIH 2026 Problem Statement: SIH26147 (NTRO)")
        st.markdown("""
        **Problem Statement Title:** Automated Model for Analysis of `.IQ` and `.wav` Files Along with Signal Parameter Extraction  
        **Organization:** NTRO  
        **Theme:** Space Technology  
        
        ### Key Project Capabilities
        1. **Multi-Format File Ingestion:** Ingests raw interleaved binary IQ files (`int8`, `int16`, `int32`, `float32`, `float64`) and WAV audio recordings.
        2. **Transparent Frequency Scaling:** Explicitly distinguishes relative baseband frequency from absolute RF frequency.
        3. **Dual Bandwidth Estimation:** Computes 99% Occupied Power Bandwidth and X-dB Down (-3dB, -20dB) Bandwidth.
        4. **Explainable Modulation Classifier:** Extract instantaneous amplitude, phase, and frequency statistics.
        5. **Auditable Reporting:** Standard JSON & CSV report export.
        """)
