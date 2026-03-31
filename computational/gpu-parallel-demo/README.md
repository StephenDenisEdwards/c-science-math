# GPU Parallel Vector Addition

Three implementations of the same task — adding 16 million array elements — each using a different GPU programming model. All three compare CPU (serial) vs GPU (parallel) performance.

| Implementation | File | Compiler | GPU Vendor Support |
|---------------|------|----------|--------------------|
| **CUDA** | `vector_add.cu` | `nvcc` (CUDA Toolkit) | NVIDIA only |
| **OpenCL** | `vector_add_opencl.c` | `gcc` / `cl` | NVIDIA, AMD, Intel |
| **OpenMP** | `vector_add_openmp.c` | `gcc` | Any (with compiler support) |

## Why GPUs Are Fast at This

A CPU is designed for complex, branching, sequential work. A modern desktop CPU has 8-24 cores, each very powerful — deep pipelines, branch prediction, large caches, out-of-order execution. It handles one or two threads per core extremely well.

A GPU is designed for the opposite: simple, uniform, massively parallel work. An NVIDIA RTX 3080 has 8,704 CUDA cores. Each core is far simpler than a CPU core — no branch prediction, tiny cache — but there are thousands of them. When every element in an array needs the same operation applied independently, a GPU can process them all at once.

This is called **SIMT** (Single Instruction, Multiple Threads) — thousands of threads execute the same instruction on different data simultaneously. It's the GPU equivalent of SIMD (Single Instruction, Multiple Data), but at a much larger scale.

### The Memory Transfer Problem

GPUs have their own dedicated memory (VRAM), separate from the CPU's system RAM. Before the GPU can work on data, it must be copied across the PCIe bus from system RAM to VRAM. Results must be copied back afterward.

For our vector addition:
- **To GPU:** 2 arrays x 16M floats x 4 bytes = 128 MB uploaded
- **Compute:** 16M additions (trivial)
- **From GPU:** 1 array x 16M floats x 4 bytes = 64 MB downloaded

The PCIe 3.0 x16 bus tops out at ~12 GB/s, so transferring 192 MB takes ~16 ms. The actual addition on the GPU takes well under 1 ms. This is why the "kernel only" speedup is huge (50-100x) but the "round trip" speedup is modest (2-3x) — the transfers dominate.

Real GPU workloads (matrix multiplication, neural networks, physics simulations) have much higher compute-to-transfer ratios, so the transfer cost is amortised over far more work.

---

## CUDA

### What It Is

CUDA (Compute Unified Device Architecture) is NVIDIA's proprietary parallel computing platform, introduced in 2006. It extends C/C++ with a small set of keywords that let you write functions ("kernels") that run on the GPU. CUDA is the dominant GPU computing platform — most AI frameworks, scientific computing libraries, and GPU-accelerated applications use it.

### How It Works

CUDA introduces a hierarchy of parallelism:

- **Thread**: The smallest unit. Each thread runs the kernel function once, on one element.
- **Block**: A group of threads (typically 128-1024) that run on the same Streaming Multiprocessor (SM) and can share fast on-chip memory.
- **Grid**: The collection of all blocks needed to cover the entire dataset.

When you launch a kernel, you specify the grid and block dimensions:

```cuda
vector_add_gpu<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
```

The `<<<blocks, threads_per_block>>>` syntax is CUDA-specific — it tells the GPU runtime how many blocks to create and how many threads per block. The GPU scheduler distributes these blocks across its Streaming Multiprocessors.

### The Kernel Explained

```cuda
__global__ void vector_add_gpu(const float *a, const float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}
```

- `__global__` — marks this function as a kernel that runs on the GPU but is called from CPU code.
- `blockIdx.x` — the index of the current block within the grid (0, 1, 2, ...).
- `blockDim.x` — the number of threads per block (256 in our case).
- `threadIdx.x` — the index of the current thread within its block (0-255).
- The formula `blockIdx.x * blockDim.x + threadIdx.x` gives each thread a globally unique index. For example, thread 3 in block 5 with 256 threads per block gets index `5 * 256 + 3 = 1283`.
- The `if (i < n)` guard is needed because the total number of threads (blocks x threads_per_block) is rounded up and may exceed `n`.

### Memory Model

CUDA has an explicit memory model — you manually allocate GPU memory and copy data back and forth:

```c
cudaMalloc(&d_a, bytes);                              // allocate on GPU
cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);  // CPU → GPU
/* ... launch kernel ... */
cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);  // GPU → CPU
cudaFree(d_a);                                         // free GPU memory
```

This is verbose but gives you full control over when transfers happen, which is critical for performance in complex applications.

### The `nvcc` Compiler

CUDA code uses the `.cu` file extension. The `nvcc` compiler splits the code:
- GPU kernel code is compiled to PTX (a GPU assembly language), then to device-specific machine code (SASS).
- Host (CPU) code is passed to the system's C/C++ compiler (`gcc`, `cl`, etc.).

The final binary contains both CPU and GPU code.

### Timing with CUDA Events

The program uses CUDA events rather than wall-clock time to measure GPU performance:

```c
cudaEventRecord(start);
vector_add_gpu<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&gpu_ms, start, stop);
```

CUDA events use the GPU's own clock, so they measure exactly how long the GPU spent working — unaffected by CPU scheduling, other processes, or driver overhead. This is the standard way to benchmark CUDA kernels.

### Strengths and Limitations

**Strengths:**
- Best performance on NVIDIA hardware — CUDA is tuned specifically for NVIDIA's architecture
- Largest ecosystem — cuBLAS, cuDNN, cuFFT, Thrust, and thousands of libraries
- Best tooling — Nsight profiler, cuda-gdb debugger, compute sanitiser
- Most documentation, tutorials, and community support

**Limitations:**
- NVIDIA GPUs only — will not run on AMD or Intel hardware
- Requires the CUDA Toolkit (several GB download) and the `nvcc` compiler
- Vendor lock-in — code cannot be ported to other GPUs without a rewrite

---

## OpenCL

### What It Is

OpenCL (Open Computing Language) is an open standard maintained by the Khronos Group (the same organisation behind OpenGL and Vulkan). It was first released in 2008 as a vendor-neutral alternative to CUDA. OpenCL runs on GPUs from NVIDIA, AMD, and Intel, as well as FPGAs, DSPs, and even CPUs.

### How It Works

OpenCL has a fundamentally different compilation model from CUDA. Instead of compiling GPU code ahead of time, the kernel source is compiled **at runtime** by the GPU driver:

1. Your C program contains the kernel as a string literal.
2. At runtime, the program sends this string to the OpenCL driver.
3. The driver compiles it for whatever GPU is installed.
4. The compiled kernel is then executed.

This means the same binary can run on completely different GPU hardware — the driver handles the translation. The tradeoff is more boilerplate code and a small startup cost for runtime compilation.

### The Execution Model

OpenCL uses different terminology from CUDA but the concepts map directly:

| CUDA | OpenCL | Meaning |
|------|--------|---------|
| Thread | Work-item | One instance of the kernel |
| Block | Work-group | A group of work-items that can synchronise |
| Grid | NDRange | The full set of work-items |
| `threadIdx.x` | `get_local_id(0)` | Index within the work-group |
| `blockIdx.x` | `get_group_id(0)` | Index of the work-group |
| `blockIdx.x * blockDim.x + threadIdx.x` | `get_global_id(0)` | Global index across all work-items |

### The Kernel Explained

```c
__kernel void vector_add(__global const float *a,
                         __global const float *b,
                         __global float *c,
                         const int n)
{
    int i = get_global_id(0);
    if (i < n)
        c[i] = a[i] + b[i];
}
```

- `__kernel` — marks this as a GPU function (equivalent to CUDA's `__global__`).
- `__global` — indicates the pointer refers to GPU global memory.
- `get_global_id(0)` — returns the work-item's global index in dimension 0. This is the OpenCL equivalent of CUDA's `blockIdx.x * blockDim.x + threadIdx.x`, but computed for you.

### The Host-Side Boilerplate

OpenCL requires significantly more setup code than CUDA. Where CUDA needs a few lines to allocate and launch, OpenCL requires you to explicitly:

1. **Query platforms** — find available OpenCL implementations (NVIDIA, AMD, Intel, etc.)
2. **Select a device** — pick a specific GPU (or CPU, or FPGA)
3. **Create a context** — an environment that manages memory and kernels for that device
4. **Create a command queue** — a queue of operations (transfers, kernel launches) to execute
5. **Compile the kernel** — pass the source string to the driver, check for errors
6. **Create buffers** — allocate GPU memory
7. **Set kernel arguments** — bind each buffer/value to a kernel parameter (one call per argument)
8. **Enqueue the kernel** — submit it to the command queue for execution
9. **Read back results** — enqueue a buffer read to copy results to CPU memory

This verbosity is the main criticism of OpenCL. The flexibility it provides (choosing devices, querying capabilities, runtime compilation) comes at the cost of a lot of boilerplate for simple programs.

### Runtime Compilation

The kernel source is stored as a C string in the host code:

```c
static const char *kernel_source =
    "__kernel void vector_add(__global const float *a, ...)\n"
    "{ ... }\n";

cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, ...);
clBuildProgram(program, 1, &device, NULL, NULL, NULL);
```

If the kernel has a syntax error, you only find out at runtime — not at compile time. The build log can be retrieved to diagnose errors:

```c
char log[4096];
clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
```

### Strengths and Limitations

**Strengths:**
- Vendor-neutral — same code runs on NVIDIA, AMD, Intel, and others
- Compiles with standard `gcc` or `cl` — no special compiler needed
- Can also target CPUs, FPGAs, and DSPs — not limited to GPUs
- Open standard — no single vendor controls the specification

**Limitations:**
- Verbose — much more boilerplate than CUDA for the same result
- Typically slower than CUDA on NVIDIA hardware — NVIDIA optimises CUDA first
- Kernel errors are only caught at runtime, not compile time
- Smaller ecosystem — fewer GPU-accelerated libraries compared to CUDA
- Runtime kernel compilation adds startup overhead (though kernels can be cached)
- Apple deprecated OpenCL in favour of Metal, so macOS support is frozen at OpenCL 1.2

---

## OpenMP Target Offloading

### What It Is

OpenMP (Open Multi-Processing) is a long-established standard for parallel programming in C, C++, and Fortran. It has been used for CPU multithreading since 1997 — if you've seen `#pragma omp parallel for`, that's OpenMP.

Starting with OpenMP 4.0 (2013), the standard added **target offloading** — the ability to offload computation to accelerator devices like GPUs. The idea is that you annotate existing loop-based code with pragmas and the compiler handles all the GPU details: memory allocation, data transfers, kernel generation, and launch configuration.

### How It Works

The key directive is `#pragma omp target`, which tells the compiler to run the following code on a GPU (or other accelerator). Combined with threading directives, a single pragma line replaces all of CUDA's and OpenCL's boilerplate:

```c
#pragma omp target teams distribute parallel for \
    map(to: a[0:n], b[0:n]) map(from: c[0:n])
for (int i = 0; i < n; i++)
    c[i] = a[i] + b[i];
```

Breaking this down:

- `target` — offload the following region to the GPU.
- `teams` — create multiple thread teams on the GPU (analogous to CUDA blocks).
- `distribute` — spread iterations across the teams.
- `parallel for` — parallelise iterations within each team (analogous to CUDA threads within a block).
- `map(to: a[0:n], b[0:n])` — copy arrays `a` and `b` from CPU to GPU before the kernel runs. The `0:n` syntax specifies the array section to transfer.
- `map(from: c[0:n])` — copy array `c` from GPU to CPU after the kernel finishes.

### The Compilation Process

When you compile with `-fopenmp -foffload=nvptx-none`, GCC does two compilations:

1. **Host compilation** — the normal C code for the CPU, with the pragma regions replaced by runtime calls.
2. **Device compilation** — the pragma regions are compiled to PTX (NVIDIA GPU assembly) using a built-in offloading compiler.

The result is a single binary containing both CPU and GPU code, similar to what `nvcc` produces for CUDA.

### Graceful Fallback

The most practical advantage of OpenMP offloading is graceful degradation. If compiled without offloading support (just `-fopenmp`), the same code runs on the CPU using threads:

```bash
# GPU offloading (requires gcc-offload-nvptx)
gcc -O2 -fopenmp -foffload=nvptx-none -o vector_add_openmp vector_add_openmp.c

# CPU threads only (any gcc with OpenMP)
gcc -O2 -fopenmp -o vector_add_openmp vector_add_openmp.c
```

Both produce a working binary. The CPU-threads version won't be as fast as true GPU offloading, but it demonstrates parallelism and always works.

At runtime, `omp_get_num_devices()` reports how many accelerator devices are available. If the answer is zero, the `target` directive runs on the host instead.

### Strengths and Limitations

**Strengths:**
- Minimal code changes — add pragmas to existing loops, no rewrite needed
- Portable — same source compiles for CPU-only or GPU targets
- No vendor-specific syntax — works with NVIDIA (via PTX), AMD (via HSA/AMDGPU), and others
- Graceful fallback — always produces a working binary, even without GPU support
- Standard C/C++/Fortran — compiles with `gcc`, no proprietary compiler required

**Limitations:**
- Compiler support is still maturing — GCC and Clang offloading works but can be difficult to set up, especially on Windows
- Less control over GPU execution — you can't fine-tune block sizes, shared memory, or memory access patterns the way CUDA allows
- Typically lower performance than hand-tuned CUDA — the compiler makes good-enough decisions but not optimal ones
- Debugging is harder — fewer GPU-specific debugging tools compared to CUDA's Nsight
- Windows support for GPU offloading is limited — `gcc-offload-nvptx` is primarily a Linux package

---

## Comparison Summary

| | CUDA | OpenCL | OpenMP |
|-|------|--------|--------|
| **Vendor** | NVIDIA (proprietary) | Khronos Group (open standard) | OpenMP ARB (open standard) |
| **First released** | 2006 | 2008 | 1997 (GPU offloading: 2013) |
| **GPU support** | NVIDIA only | NVIDIA, AMD, Intel | Any (compiler-dependent) |
| **Compiler** | `nvcc` (special) | `gcc` / `cl` (standard) | `gcc` / `clang` (standard) |
| **Kernel language** | C/C++ with extensions | C-like (OpenCL C) | Standard C with pragmas |
| **Compilation** | Ahead of time | Runtime (by GPU driver) | Ahead of time |
| **Boilerplate** | Moderate | Heavy | Minimal |
| **Performance** | Best (on NVIDIA) | Good | Good (less tunable) |
| **Ecosystem** | Largest | Medium | Small (for GPU) |
| **Learning curve** | Moderate | Steep | Gentle |
| **Without GPU** | Won't compile | Falls back to CPU device | Falls back to CPU threads |

### Which Should You Use?

- **CUDA** if you have an NVIDIA GPU and want the best performance, tooling, and library ecosystem. This is what most AI, HPC, and scientific computing projects use.
- **OpenCL** if you need to support multiple GPU vendors or want to target non-GPU accelerators. It's more work to write but runs everywhere.
- **OpenMP** if you have existing CPU code and want to try GPU offloading with minimal changes. It's the lowest-effort path but offers less control.

---

## Prerequisites

### 1. CUDA (NVIDIA GPUs only)

Requires the CUDA Toolkit, which provides the `nvcc` compiler.

Download from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).

**Windows:**

1. Download and run the installer (select at least "CUDA Development" components)
2. The installer adds `nvcc` to your PATH automatically
3. Open a **new** terminal and verify:
   ```bash
   nvcc --version
   ```

**Linux (Debian/Ubuntu):**

```bash
sudo apt update && sudo apt install nvidia-cuda-toolkit
nvcc --version
```

### 2. OpenCL (NVIDIA, AMD, or Intel GPUs)

Requires the OpenCL SDK headers and a GPU driver with OpenCL support (most modern drivers include this).

**Windows (NVIDIA):**

The CUDA Toolkit includes OpenCL headers and libraries. If you have the CUDA Toolkit installed, OpenCL is already available. Otherwise, install the [NVIDIA GPU Computing Toolkit](https://developer.nvidia.com/cuda-downloads).

Compile with:
```bash
gcc -O2 -o vector_add_opencl vector_add_opencl.c -lOpenCL -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.x/include" -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.x/lib/x64"
```

**Linux (Debian/Ubuntu):**

```bash
sudo apt update && sudo apt install ocl-icd-opencl-dev
```

### 3. OpenMP (any GPU, with compiler support)

OpenMP GPU offloading requires a compiler built with offloading support. Without it, the code still compiles and runs correctly — it just uses CPU threads instead of the GPU.

**GCC with NVIDIA offloading (Linux):**

```bash
sudo apt install gcc-offload-nvptx
```

**CPU-only fallback (any platform with GCC):**

```bash
gcc -O2 -fopenmp -o vector_add_openmp vector_add_openmp.c
```

This uses OpenMP threads on the CPU, which still demonstrates parallelism (just not GPU parallelism).

## Build and Run

There are two separate steps: **compiling** the code and then **running** the resulting binary.

### CUDA

**Step 1 — Compile:**

```bash
cd computational/gpu-parallel-demo
nvcc -O2 -o vector_add vector_add.cu
```

Or use the Makefile: `make cuda`

**Step 2 — Run:**

```bash
./vector_add              # Linux / MSYS2
vector_add.exe            # Windows
```

### OpenCL

**Step 1 — Compile:**

```bash
gcc -O2 -o vector_add_opencl vector_add_opencl.c -lOpenCL
```

Or use the Makefile: `make opencl`

**Step 2 — Run:**

```bash
./vector_add_opencl       # Linux / MSYS2
vector_add_opencl.exe     # Windows
```

### OpenMP

**Step 1 — Compile:**

With GPU offloading (requires `gcc-offload-nvptx`):

```bash
gcc -O2 -fopenmp -foffload=nvptx-none -o vector_add_openmp vector_add_openmp.c
```

CPU-only fallback:

```bash
gcc -O2 -fopenmp -o vector_add_openmp vector_add_openmp.c
```

Or use the Makefile: `make openmp`

**Step 2 — Run:**

```bash
./vector_add_openmp       # Linux / MSYS2
vector_add_openmp.exe     # Windows
```

### Build all (Makefile)

```bash
make all
```

> **Note:** Compilers print nothing when compilation succeeds. No output means it worked.

**Custom array size** (all three accept an argument, default is 16M elements):

```bash
./vector_add 33554432     # 32M elements
```

## Example Output

```
Vector Addition — CPU vs GPU
Elements:  16777216 (64.0 MB per array)

GPU:       NVIDIA GeForce RTX 3080
SM count:  68
Compute:   8.6

CPU (serial loop):                42.000 ms
GPU (kernel only):                 0.524 ms
GPU (with memory transfers):      18.712 ms

Kernel speedup:      80.2x
Round-trip speedup:  2.2x

Verification: PASSED — all 16777216 elements match
```

The kernel-only speedup is dramatic (often 50-100x), but the round-trip speedup is more modest because copying 192 MB of data (three arrays) over the PCIe bus takes time. This is a key insight: **GPU parallelism pays off most when the compute-to-transfer ratio is high** — matrix multiplication, simulations, and image processing benefit far more than simple element-wise addition.

## Project Structure

```
gpu-parallel-demo/
├── vector_add.cu          # CUDA implementation
├── vector_add_opencl.c    # OpenCL implementation
├── vector_add_openmp.c    # OpenMP target offloading implementation
├── Makefile
└── README.md
```

---

## Appendix: NVIDIA GPUs vs Apple Silicon for GPU Compute

### Can MacBooks Use CUDA?

No. Apple stopped using NVIDIA GPUs in 2016. Modern MacBooks use Apple Silicon (M1–M4) with integrated Apple GPUs. These do not support CUDA and never will — CUDA is NVIDIA-proprietary and requires NVIDIA hardware.

Apple's GPU compute API is **Metal**. It is capable but has a completely different ecosystem and far less library support for scientific and ML workloads compared to CUDA.

### Inference Performance: NVIDIA GPU vs Apple Silicon

For a model that fits in the GPU's VRAM, an NVIDIA GPU will outperform Apple Silicon for inference:

- **Raw throughput** — Even a mid-range NVIDIA GPU (e.g. RTX 4070) has higher compute throughput and faster memory bandwidth for the matrix operations that dominate inference.
- **Software maturity** — CUDA-optimised inference runtimes (TensorRT, vLLM, ExLlamaV2) are heavily tuned and extract far more performance than Metal-based alternatives.
- **Quantisation support** — NVIDIA has better support for fast INT4/INT8 inference kernels.

### Where Apple Silicon Wins

Apple Silicon's advantage is **unified memory**. The CPU and GPU share the same RAM, so there is no VRAM limit separate from system memory. An M4 Max with 128 GB unified memory can load large language models (e.g. 70B parameter, ~35-40 GB quantised) that would not fit in a typical discrete GPU's 8-24 GB VRAM.

This means Apple Silicon can run models that an NVIDIA laptop GPU physically cannot load — but it runs them more slowly than an NVIDIA GPU would if the model did fit.

### Why MacBooks Are Recommended for "AI Work"

The recommendation usually applies to specific use cases where raw GPU compute is not the bottleneck:

- **Local inference** — running pre-trained models (via llama.cpp, Ollama) works well on Apple Silicon thanks to unified memory and Metal support.
- **Application development** — building apps that call APIs (Claude, OpenAI) where the GPU is irrelevant.
- **Data science and prototyping** — notebooks, data exploration, and small-scale experiments.
- **Portability and battery life** — practical advantages for development on the move.

For **model training**, **heavy GPU compute**, or workloads like the parallel computing examples in this project, NVIDIA GPUs with CUDA remain the clear choice.

### Summary

| | NVIDIA GPU | Apple Silicon |
|-|-----------|---------------|
| **CUDA support** | Yes | No |
| **Compute API** | CUDA, OpenCL | Metal |
| **Inference (model fits)** | Faster | Slower |
| **Large model loading** | Limited by VRAM (8-24 GB typical) | Up to 128 GB unified memory |
| **ML ecosystem** | Mature (PyTorch, TensorFlow, TensorRT) | Growing (PyTorch MPS, MLX) |
| **Training** | Strong | Impractical for serious workloads |
