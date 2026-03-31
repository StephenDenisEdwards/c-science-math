# Fourier Transforms

An interactive project that demonstrates Fourier transforms from first principles, with visualisations at every step.

## What's Inside

| Module | Description |
|--------|-------------|
| `python/fourier/dft.py` | Core implementations — DFT from scratch (O(N²)), NumPy FFT wrapper, composite signal generation, frequency-domain filtering |
| `python/fourier/visualise.py` | Plotting functions for time/frequency domain, signal decomposition, filter comparison, and 2-D image FFT |
| `python/demo.ipynb` | Interactive Jupyter notebook with explanations, LaTeX equations, and inline plots |
| `python/main.py` | Standalone script that runs all demos and saves plots to `python/output/` |
| `c/` | C implementation with scalar and AVX SIMD FFT + benchmark |
| `csharp/` | C# implementation with scalar and `System.Runtime.Intrinsics` AVX/FMA SIMD FFT + benchmark |

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

## Prerequisites

### Python (for notebook and visualisations)

- **Python 3.10+** — [download](https://www.python.org/downloads/)
- Dependencies listed in `python/requirements.txt` (numpy, scipy, matplotlib, Pillow, jupyter)

### .NET SDK (for C# implementation)

- **.NET 8.0+ SDK** — [download](https://dotnet.microsoft.com/download/dotnet/8.0)
- Verify with: `dotnet --version`
- No additional NuGet packages required — `System.Runtime.Intrinsics` is built into the runtime

### C compiler (for C implementation)

You need a C compiler that supports AVX intrinsics. Pick one for your platform:

**Windows — Option A: MSYS2 + MinGW-w64 (recommended)**

1. Download and run the installer from [msys2.org](https://www.msys2.org/)
2. Accept the default install location (`C:\msys64`)
3. When the installer finishes, it opens an MSYS2 terminal. Close it.
4. Open **MSYS MINGW64** from the Start menu (not the plain "MSYS2" shortcut — look for the one that says **MINGW64**)
5. In that terminal, install GCC:
   ```bash
   pacman -S mingw-w64-x86_64-gcc
   ```
   Type `Y` when prompted.
6. Add GCC to your Windows PATH so it works from any terminal:
   - Press `Win + R`, type `sysdm.cpl`, press Enter
   - Go to the **Advanced** tab → click **Environment Variables**
   - Under **User variables**, select **Path** and click **Edit**
   - Click **New** and add: `C:\msys64\mingw64\bin`
   - Click **OK** on all dialogs
7. Open a **new** terminal (the old one won't see the PATH change) and verify:
   ```bash
   gcc --version
   ```

**Windows — Option B: Visual Studio Build Tools**

1. Download [Build Tools for Visual Studio](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
2. In the installer, select the **"Desktop development with C++"** workload and install
3. Open **"Developer Command Prompt for VS"** from the Start menu (this sets up the compiler paths automatically)
4. Compile with:
   ```bash
   cl /O2 /arch:AVX main.c
   ```

**macOS:**

```bash
# Install Xcode Command Line Tools (includes Clang)
xcode-select --install
```
Follow the dialog that appears. Verify with: `gcc --version`

**Linux (Debian/Ubuntu):**

```bash
sudo apt update && sudo apt install gcc
```
Verify with: `gcc --version`

## Getting Started

### Python

```bash
cd math/fourier-transforms/python

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

This generates five PNG plots in the `python/output/` directory:

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

## C Implementation (Scalar vs AVX SIMD)

A native C implementation that benchmarks a textbook radix-2 Cooley-Tukey FFT against an AVX-accelerated version using 256-bit SIMD intrinsics. The SIMD version processes 4 doubles per instruction in the butterfly inner loop.

### Build and run

There are two separate steps: **compiling** the code and then **running** the resulting binary.

**Step 1 — Compile:**

Using GCC (Linux, macOS, or Windows with MSYS2):

```bash
cd math/fourier-transforms/c
gcc -O2 -mavx -Wall -Wextra -lm -o fft_bench main.c
```

Using MSVC (Windows with Visual Studio Build Tools — run from a **Developer Command Prompt**):

```bash
cd math/fourier-transforms/c
cl /O2 /arch:AVX main.c /Fe:fft_bench.exe
```

> **Note:** Both compilers print nothing when compilation succeeds. No output means it worked. You should now have a `fft_bench` binary in the current directory.

**Step 2 — Run the benchmark:**

```bash
./fft_bench            # Linux / macOS / MSYS2
fft_bench.exe          # Windows Command Prompt or PowerShell
```

This prints a correctness check followed by a timing table comparing scalar and SIMD performance across FFT sizes from 1K to 1M samples. The larger sizes may take a few seconds to complete.

See [Prerequisites](#c-compiler-for-c-implementation) for compiler installation. Falls back to scalar-only if AVX is not available at compile time.

## C# Implementation (Scalar vs AVX/FMA SIMD)

A C# implementation using `System.Runtime.Intrinsics.X86` for explicit AVX and FMA vectorisation. Uses `ArrayPool<double>` to avoid GC pressure from twiddle factor allocation.

### Build and run

```bash
cd math/fourier-transforms/csharp
dotnet run -c Release
```

See [Prerequisites](#net-sdk-for-c-implementation) for SDK installation. AVX support is detected at runtime — falls back to scalar if unavailable.

### What the benchmarks show

Both the C and C# benchmarks run the same test: FFT of a composite signal at sizes from 1K to 1M samples, comparing scalar vs SIMD execution time. You'll typically see:

- **Small sizes (1K–4K):** SIMD overhead outweighs the gains — twiddle factor computation dominates
- **Medium sizes (16K–256K):** SIMD shines — butterfly operations are compute-bound and vectorise well
- **Large sizes (1M+):** Memory bandwidth becomes the bottleneck, limiting SIMD advantage

## Using the Python Library in Your Own Code

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
├── python/                  # Python implementation
│   ├── fourier/             # Python package
│   │   ├── __init__.py
│   │   ├── dft.py
│   │   └── visualise.py
│   ├── demo.ipynb
│   ├── main.py
│   └── requirements.txt
├── c/                       # C implementation
│   ├── fft.h
│   ├── main.c
│   └── Makefile
├── csharp/                  # C# implementation
│   ├── Fft.cs
│   ├── Program.cs
│   └── FourierTransform.csproj
└── README.md
```
