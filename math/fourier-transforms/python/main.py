"""
Fourier Transform Demo — Standalone Script

Runs all demonstrations and saves plots to an 'output/' directory.
Usage:
    cd math/fourier-transforms/python
    pip install -r requirements.txt
    python main.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving files
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fourier.dft import dft, fft_analyse, generate_composite_signal
from fourier.visualise import (
    plot_time_and_frequency,
    plot_signal_decomposition,
    plot_filter_comparison,
    plot_image_fft,
)

OUTPUT = Path(__file__).resolve().parent / "output"


def main():
    OUTPUT.mkdir(exist_ok=True)

    frequencies = [5, 20, 50]
    amplitudes = [1.0, 0.5, 0.3]
    sample_rate = 500.0

    # ---- 1. Signal decomposition ----
    t, signal, components = generate_composite_signal(
        frequencies, amplitudes, duration=1.0, sample_rate=sample_rate
    )
    fig = plot_signal_decomposition(t, signal, components, frequencies, amplitudes)
    fig.savefig(OUTPUT / "1_decomposition.png", dpi=150)
    plt.close(fig)
    print("Saved 1_decomposition.png")

    # ---- 2. Time + frequency domain ----
    fig = plot_time_and_frequency(t, signal, sample_rate, title="Composite Signal")
    fig.savefig(OUTPUT / "2_time_and_frequency.png", dpi=150)
    plt.close(fig)
    print("Saved 2_time_and_frequency.png")

    # ---- 3. Low-pass filter on noisy signal ----
    t_noisy, noisy, _ = generate_composite_signal(
        frequencies, amplitudes, duration=1.0, sample_rate=sample_rate, noise_std=0.4
    )
    fig = plot_filter_comparison(t_noisy, noisy, sample_rate, cutoff=30.0, mode="low")
    fig.savefig(OUTPUT / "3_lowpass_filter.png", dpi=150)
    plt.close(fig)
    print("Saved 3_lowpass_filter.png")

    # ---- 4. High-pass filter ----
    fig = plot_filter_comparison(t, signal, sample_rate, cutoff=15.0, mode="high")
    fig.savefig(OUTPUT / "4_highpass_filter.png", dpi=150)
    plt.close(fig)
    print("Saved 4_highpass_filter.png")

    # ---- 5. 2-D FFT on synthetic image ----
    size = 256
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)
    image = (
        np.sin(2 * np.pi * 8 * Y)
        + 0.5 * np.sin(2 * np.pi * 16 * (X + Y))
        + 0.3 * np.sin(2 * np.pi * 32 * X)
    )
    fig = plot_image_fft(image, title="Synthetic Grating — 2-D FFT")
    fig.savefig(OUTPUT / "5_image_fft.png", dpi=150)
    plt.close(fig)
    print("Saved 5_image_fft.png")

    print(f"\nAll plots saved to {OUTPUT}")


if __name__ == "__main__":
    main()
