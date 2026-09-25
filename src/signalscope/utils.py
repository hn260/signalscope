"""
Utility functions for unit formatting and display helpers.
"""


def format_frequency(freq_hz: float) -> str:
    """Formats frequency with appropriate scale unit (Hz, kHz, MHz, GHz)."""
    abs_f = abs(freq_hz)
    if abs_f >= 1e9:
        return f"{freq_hz / 1e9:.4f} GHz"
    elif abs_f >= 1e6:
        return f"{freq_hz / 1e6:.4f} MHz"
    elif abs_f >= 1e3:
        return f"{freq_hz / 1e3:.2f} kHz"
    else:
        return f"{freq_hz:.1f} Hz"


def format_bytes(bytes_count: int) -> str:
    """Formats byte counts into human readable sizes."""
    if bytes_count >= 1e9:
        return f"{bytes_count / 1e9:.2f} GB"
    elif bytes_count >= 1e6:
        return f"{bytes_count / 1e6:.2f} MB"
    elif bytes_count >= 1e3:
        return f"{bytes_count / 1e3:.1f} KB"
    else:
        return f"{bytes_count} B"
