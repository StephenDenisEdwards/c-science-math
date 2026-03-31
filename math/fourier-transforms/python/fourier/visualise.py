"""
Visualization helpers for Fourier transform demonstrations.

Every public function returns a matplotlib Figure so callers can
either display interactively or save to disk.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from .dft import fft_analyse, generate_composite_signal, frequency_filter


# ---------------------------------------------------------------------------
# Colour palette (consistent look across all plots)
# ---------------------------------------------------------------------------
_COLOURS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def _colour(i: int) -> str:
    return _COLOURS[i % len(_COLOURS)]


# ---------------------------------------------------------------------------
# 1.  Time-domain + frequency-domain side-by-side
# ---------------------------------------------------------------------------

def plot_time_and_frequency(
    t: NDArray,
    signal: NDArray,
    sample_rate: float,
    title: str = "Signal Analysis",
) -> plt.Figure:
    """Plot the waveform alongside its single-sided amplitude spectrum."""
    freqs, mags, _, _ = fft_analyse(signal, sample_rate)

    fig, (ax_time, ax_freq) = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Time domain
    ax_time.plot(t, signal, color=_colour(0), linewidth=0.8)
    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel("Amplitude")
    ax_time.set_title("Time Domain")
    ax_time.grid(True, alpha=0.3)

    # Frequency domain
    ax_freq.stem(freqs, mags, linefmt=_colour(1), markerfmt=" ", basefmt="k-")
    ax_freq.set_xlabel("Frequency (Hz)")
    ax_freq.set_ylabel("Magnitude")
    ax_freq.set_title("Frequency Domain")
    ax_freq.set_xlim(0, sample_rate / 2)
    ax_freq.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2.  Signal decomposition — show each component + the sum
# ---------------------------------------------------------------------------

def plot_signal_decomposition(
    t: NDArray,
    signal: NDArray,
    components: list[NDArray],
    frequencies: list[float],
    amplitudes: list[float],
) -> plt.Figure:
    """Show each sinusoidal component stacked above the composite signal."""
    n_components = len(components)
    fig, axes = plt.subplots(n_components + 1, 1, figsize=(14, 2.5 * (n_components + 1)),
                             sharex=True)
    fig.suptitle("Signal Decomposition", fontsize=14, fontweight="bold")

    for i, (comp, freq, amp) in enumerate(zip(components, frequencies, amplitudes)):
        axes[i].plot(t, comp, color=_colour(i), linewidth=0.8)
        axes[i].set_ylabel("Amplitude")
        axes[i].set_title(f"Component: {freq} Hz, amplitude {amp}", fontsize=10)
        axes[i].grid(True, alpha=0.3)

    axes[-1].plot(t, signal, color="black", linewidth=0.8)
    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_ylabel("Amplitude")
    axes[-1].set_title("Composite Signal (sum of all components)", fontsize=10)
    axes[-1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3.  Filtering comparison (before / after in both domains)
# ---------------------------------------------------------------------------

def plot_filter_comparison(
    t: NDArray,
    signal: NDArray,
    sample_rate: float,
    cutoff: float,
    mode: str = "low",
) -> plt.Figure:
    """Apply a frequency filter and plot original vs filtered in both domains."""
    freqs_orig, mags_orig, _, spectrum = fft_analyse(signal, sample_rate)
    filtered_spectrum = frequency_filter(spectrum, sample_rate, cutoff, mode)
    filtered_signal = np.fft.ifft(filtered_spectrum).real

    freqs_filt, mags_filt, _, _ = fft_analyse(filtered_signal, sample_rate)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        f"{mode.title()}-pass filter  (cutoff = {cutoff} Hz)",
        fontsize=14,
        fontweight="bold",
    )

    # Original time
    axes[0, 0].plot(t, signal, color=_colour(0), linewidth=0.8)
    axes[0, 0].set_title("Original — Time Domain")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, alpha=0.3)

    # Original freq
    axes[0, 1].stem(freqs_orig, mags_orig, linefmt=_colour(0), markerfmt=" ", basefmt="k-")
    axes[0, 1].set_title("Original — Frequency Domain")
    axes[0, 1].set_ylabel("Magnitude")
    axes[0, 1].set_xlim(0, sample_rate / 2)
    axes[0, 1].grid(True, alpha=0.3)

    # Filtered time
    axes[1, 0].plot(t, filtered_signal, color=_colour(2), linewidth=0.8)
    axes[1, 0].set_title("Filtered — Time Domain")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].grid(True, alpha=0.3)

    # Filtered freq
    axes[1, 1].stem(freqs_filt, mags_filt, linefmt=_colour(2), markerfmt=" ", basefmt="k-")
    axes[1, 1].set_title("Filtered — Frequency Domain")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Magnitude")
    axes[1, 1].set_xlim(0, sample_rate / 2)
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4.  2-D FFT on an image
# ---------------------------------------------------------------------------

def plot_image_fft(image: NDArray, title: str = "2-D Fourier Transform") -> plt.Figure:
    """Compute and display the 2-D FFT magnitude spectrum of a greyscale image."""
    spectrum_2d = np.fft.fft2(image)
    shifted = np.fft.fftshift(spectrum_2d)
    magnitude = np.log1p(np.abs(shifted))

    fig, (ax_img, ax_spec) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax_img.imshow(image, cmap="gray")
    ax_img.set_title("Original Image")
    ax_img.axis("off")

    ax_spec.imshow(magnitude, cmap="inferno")
    ax_spec.set_title("FFT Magnitude Spectrum (log scale)")
    ax_spec.axis("off")

    fig.tight_layout()
    return fig
