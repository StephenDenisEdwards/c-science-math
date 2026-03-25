# Fourier Transforms

An interactive project that demonstrates Fourier transforms from first principles, with visualisations at every step.

## What's Inside

| Module | Description |
|--------|-------------|
| `fourier/dft.py` | Core implementations — DFT from scratch (O(N²)), NumPy FFT wrapper, composite signal generation, frequency-domain filtering |
| `fourier/visualise.py` | Plotting functions for time/frequency domain, signal decomposition, filter comparison, and 2-D image FFT |
| `demo.ipynb` | Interactive Jupyter notebook with explanations, LaTeX equations, and inline plots |
| `main.py` | Standalone script that runs all demos and saves plots to `output/` |

## Demos

### 1. DFT from Scratch
A naive O(N²) implementation of the Discrete Fourier Transform using the direct summation formula:

$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j \, 2\pi \, k \, n \, / \, N}$$

Useful for understanding exactly what the transform does before relying on optimised libraries.

### 2. FFT Comparison
Verifies the hand-written DFT matches NumPy's FFT output, then benchmarks both to show the O(N²) vs O(N log N) speed difference.

### 3. Signal Decomposition
Builds a composite signal from individual sinusoids and plots each component separately, demonstrating the core insight: any periodic signal is a sum of sine waves.

### 4. Frequency-Domain Filtering
Applies low-pass and high-pass filters by zeroing out frequency bins in the FFT output, then inverse-transforming back. Includes a noisy signal example showing how filtering cleans up unwanted components.

### 5. 2-D FFT on Images
Computes the 2-D Fourier transform of greyscale images, displaying the log-scaled magnitude spectrum. Works with synthetic gratings out of the box, and supports loading real images via Pillow.

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`:
  - numpy
  - scipy
  - matplotlib
  - Pillow
  - jupyter (for the notebook only)

## Getting Started

```bash
cd math/fourier-transforms

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the standalone script

```bash
python main.py
```

This generates five PNG plots in the `output/` directory:

| File | Content |
|------|---------|
| `1_decomposition.png` | Individual signal components and their sum |
| `2_time_and_frequency.png` | Time-domain waveform alongside its frequency spectrum |
| `3_lowpass_filter.png` | Noisy signal before and after low-pass filtering |
| `4_highpass_filter.png` | Signal before and after high-pass filtering |
| `5_image_fft.png` | Synthetic image and its 2-D FFT magnitude spectrum |

### Run the Jupyter notebook

```bash
jupyter notebook demo.ipynb
```

The notebook includes the same demos with inline explanations, LaTeX equations, and editable parameters so you can experiment interactively.

## Using the Library in Your Own Code

```python
from fourier import (
    dft, idft,
    fft_analyse,
    generate_composite_signal,
    frequency_filter,
    plot_time_and_frequency,
    plot_signal_decomposition,
    plot_filter_comparison,
    plot_image_fft,
)

# Build a signal with 10 Hz and 60 Hz components
t, signal, components = generate_composite_signal(
    frequencies=[10, 60],
    amplitudes=[1.0, 0.5],
    duration=1.0,
    sample_rate=500.0,
)

# Analyse and plot
fig = plot_time_and_frequency(t, signal, sample_rate=500.0)
fig.savefig("my_plot.png")
```

## Project Structure

```
fourier-transforms/
├── fourier/
│   ├── __init__.py
│   ├── dft.py
│   └── visualise.py
├── demo.ipynb
├── main.py
├── requirements.txt
└── README.md
```
