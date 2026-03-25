"""
Core Fourier transform implementations and signal utilities.

Provides a from-scratch DFT, NumPy FFT wrapper, composite signal
generation, and frequency-domain filtering.
"""

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Discrete Fourier Transform (naive O(n²) — educational, not for production)
# ---------------------------------------------------------------------------

def dft(signal: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Compute the Discrete Fourier Transform of a 1-D signal.

    Uses the direct summation definition:
        X[k] = Σ_{n=0}^{N-1} x[n] · e^{-j·2π·k·n / N}

    Parameters
    ----------
    signal : 1-D real array

    Returns
    -------
    Complex spectrum of length N.
    """
    N = len(signal)
    n = np.arange(N)
    k = n.reshape(-1, 1)
    twiddle = np.exp(-2j * np.pi * k * n / N)
    return twiddle @ signal


def idft(spectrum: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Compute the Inverse DFT via direct summation.

    x[n] = (1/N) Σ_{k=0}^{N-1} X[k] · e^{j·2π·k·n / N}
    """
    N = len(spectrum)
    n = np.arange(N)
    k = n.reshape(-1, 1)
    twiddle = np.exp(2j * np.pi * k * n / N)
    return (twiddle @ spectrum) / N


# ---------------------------------------------------------------------------
# FFT wrapper (uses NumPy for speed, returns frequencies too)
# ---------------------------------------------------------------------------

def fft_analyse(signal: NDArray[np.float64], sample_rate: float):
    """Return the single-sided amplitude spectrum and frequency axis.

    Parameters
    ----------
    signal : 1-D real array
    sample_rate : samples per second (Hz)

    Returns
    -------
    freqs : frequency bins (Hz), length N//2
    magnitudes : amplitude at each bin, length N//2
    phases : phase angle (radians) at each bin, length N//2
    full_spectrum : the raw complex FFT output (length N)
    """
    N = len(signal)
    spectrum = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, d=1.0 / sample_rate)

    # single-sided
    half = N // 2
    freqs = freqs[:half]
    magnitudes = 2.0 / N * np.abs(spectrum[:half])
    phases = np.angle(spectrum[:half])

    return freqs, magnitudes, phases, spectrum


# ---------------------------------------------------------------------------
# Signal generation helpers
# ---------------------------------------------------------------------------

def generate_composite_signal(
    frequencies: list[float],
    amplitudes: list[float],
    duration: float = 1.0,
    sample_rate: float = 1000.0,
    noise_std: float = 0.0,
):
    """Build a signal by summing sinusoids, optionally adding Gaussian noise.

    Parameters
    ----------
    frequencies : list of component frequencies (Hz)
    amplitudes  : list of component amplitudes
    duration    : signal length in seconds
    sample_rate : samples per second
    noise_std   : standard deviation of additive Gaussian noise (0 = none)

    Returns
    -------
    t : time axis array
    signal : the composite waveform
    components : list of individual sinusoidal components
    """
    t = np.arange(0, duration, 1.0 / sample_rate)
    components = [a * np.sin(2 * np.pi * f * t) for f, a in zip(frequencies, amplitudes)]
    signal = np.sum(components, axis=0)
    if noise_std > 0:
        signal = signal + np.random.default_rng(42).normal(0, noise_std, len(t))
    return t, signal, components


# ---------------------------------------------------------------------------
# Frequency-domain filtering
# ---------------------------------------------------------------------------

def frequency_filter(
    spectrum: NDArray[np.complex128],
    sample_rate: float,
    cutoff: float,
    mode: str = "low",
) -> NDArray[np.complex128]:
    """Zero out frequency bins above or below a cutoff.

    Parameters
    ----------
    spectrum : full complex FFT (length N)
    sample_rate : Hz
    cutoff : cutoff frequency in Hz
    mode : "low" keeps frequencies <= cutoff,
           "high" keeps frequencies >= cutoff

    Returns
    -------
    Filtered spectrum (same shape).
    """
    N = len(spectrum)
    freqs = np.fft.fftfreq(N, d=1.0 / sample_rate)
    filtered = spectrum.copy()

    if mode == "low":
        filtered[np.abs(freqs) > cutoff] = 0
    elif mode == "high":
        filtered[np.abs(freqs) < cutoff] = 0
    else:
        raise ValueError(f"mode must be 'low' or 'high', got '{mode}'")

    return filtered
