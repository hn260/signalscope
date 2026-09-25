"""
Interactive Plotly visualizer module for SignalScope.
Provides time-domain waveforms/envelopes, Power Spectral Density (PSD) with OBW shading,
and STFT spectrograms.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional
from scipy.signal import spectrogram


def plot_time_domain(
    samples: np.ndarray,
    sample_rate: float,
    max_points: int = 4000
) -> go.Figure:
    """
    Creates interactive time-domain plot showing I & Q components and magnitude envelope.
    Downsamples for rendering speed if sample count > max_points.
    """
    n_total = len(samples)
    step = max(1, n_total // max_points)
    sampled = samples[::step]
    
    time_ms = (np.arange(len(sampled)) * step / sample_rate) * 1e3

    fig = go.Figure()

    if np.iscomplexobj(sampled):
        fig.add_trace(
            go.Scatter(
                x=time_ms, y=sampled.real,
                mode="lines", name="In-phase (I)",
                line=dict(color="#1f77b4", width=1.2)
            )
        )
        fig.add_trace(
            go.Scatter(
                x=time_ms, y=sampled.imag,
                mode="lines", name="Quadrature (Q)",
                line=dict(color="#ff7f0e", width=1.2)
            )
        )
        envelope = np.abs(sampled)
        fig.add_trace(
            go.Scatter(
                x=time_ms, y=envelope,
                mode="lines", name="Magnitude Envelope |s(t)|",
                line=dict(color="#2ca02c", width=1.8, dash="dot")
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=time_ms, y=sampled,
                mode="lines", name="Audio Amplitude s(t)",
                line=dict(color="#1f77b4", width=1.5)
            )
        )

    fig.update_layout(
        title="<b>Time Domain Waveform & Envelope</b>",
        xaxis_title="Time (ms)",
        yaxis_title="Normalized Amplitude (FS)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
        height=400,
    )
    return fig


def plot_psd(
    freqs: np.ndarray,
    psd_db: np.ndarray,
    peak_freq_hz: float,
    obw_99_hz: float,
    f_low_hz: float,
    f_high_hz: float,
    center_freq_hz: Optional[float] = None
) -> go.Figure:
    """
    Creates interactive Power Spectral Density (PSD) plot with OBW shading and peak marker.
    """
    is_absolute = center_freq_hz is not None and center_freq_hz > 0

    if is_absolute:
        freq_axis_mhz = (freqs + center_freq_hz) / 1e6
        peak_freq_plot = (peak_freq_hz + center_freq_hz) / 1e6
        f_low_plot = (f_low_hz + center_freq_hz) / 1e6
        f_high_plot = (f_high_hz + center_freq_hz) / 1e6
        x_label = "RF Frequency (MHz)"
    else:
        freq_axis_mhz = freqs / 1e3
        peak_freq_plot = peak_freq_hz / 1e3
        f_low_plot = f_low_hz / 1e3
        f_high_plot = f_high_hz / 1e3
        x_label = "Relative Baseband Frequency (kHz)"

    fig = go.Figure()

    # Main PSD Trace
    fig.add_trace(
        go.Scatter(
            x=freq_axis_mhz, y=psd_db,
            mode="lines", name="Power Spectral Density",
            line=dict(color="#0055ff", width=1.5)
        )
    )

    # 99% OBW Shaded Area
    obw_indices = np.where((freqs >= f_low_hz) & (freqs <= f_high_hz))[0]
    if len(obw_indices) > 0:
        fig.add_trace(
            go.Scatter(
                x=freq_axis_mhz[obw_indices],
                y=psd_db[obw_indices],
                fill="tozeroy",
                fillcolor="rgba(255, 165, 0, 0.25)",
                line=dict(color="rgba(255, 165, 0, 0.6)", width=1),
                name=f"99% OBW Band ({obw_99_hz/1e3:.2f} kHz)"
            )
        )

    # Peak Frequency Marker
    peak_db = float(np.max(psd_db))
    fig.add_trace(
        go.Scatter(
            x=[peak_freq_plot], y=[peak_db],
            mode="markers+text",
            marker=dict(symbol="triangle-down", size=12, color="red"),
            text=[f"Peak: {peak_freq_plot:.3f} {'MHz' if is_absolute else 'kHz'}"],
            textposition="top center",
            name="Peak Frequency"
        )
    )

    fig.update_layout(
        title=f"<b>Power Spectral Density (PSD) & 99% Occupied Bandwidth</b>",
        xaxis_title=x_label,
        yaxis_title="Power Density (dB/Hz)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
        height=420,
    )
    return fig


def plot_spectrogram(
    samples: np.ndarray,
    sample_rate: float,
    nfft: int = 512,
    center_freq_hz: Optional[float] = None
) -> go.Figure:
    """
    Creates STFT Spectrogram time-frequency heatmap.
    """
    is_complex = np.iscomplexobj(samples)
    is_absolute = center_freq_hz is not None and center_freq_hz > 0

    if is_complex:
        f, t, Sxx = spectrogram(
            samples,
            fs=sample_rate,
            window="hann",
            nperseg=nfft,
            noverlap=nfft // 2,
            return_onesided=False,
            scaling="density"
        )
        f = np.fft.fftshift(f)
        Sxx = np.fft.fftshift(Sxx, axes=0)
    else:
        f, t, Sxx = spectrogram(
            samples,
            fs=sample_rate,
            window="hann",
            nperseg=nfft,
            noverlap=nfft // 2,
            return_onesided=True,
            scaling="density"
        )

    Sxx_db = 10.0 * np.log10(np.maximum(Sxx, 1e-15))
    t_ms = t * 1e3

    if is_absolute:
        f_plot = (f + center_freq_hz) / 1e6
        y_label = "RF Frequency (MHz)"
    else:
        f_plot = f / 1e3
        y_label = "Relative Baseband Frequency (kHz)"

    fig = go.Figure(
        data=go.Heatmap(
            z=Sxx_db,
            x=t_ms,
            y=f_plot,
            colorscale="Viridis",
            colorbar=dict(title="dB/Hz"),
        )
    )

    fig.update_layout(
        title="<b>STFT Spectrogram (Time-Frequency Density)</b>",
        xaxis_title="Time (ms)",
        yaxis_title=y_label,
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
        height=450,
    )
    return fig
