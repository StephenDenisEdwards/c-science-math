/*
 * GPU Parallel Vector Addition
 *
 * Adds two large arrays element-by-element on both CPU (serial) and GPU
 * (massively parallel), then compares the execution time.
 *
 * Each GPU thread handles one element -- with millions of elements running
 * simultaneously, this is the simplest possible demonstration of GPU
 * parallelism via CUDA.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define DEFAULT_N (1 << 24)  /* 16 million elements */

/* ---------------------------------------------------------------------------
 * GPU kernel -- one thread per element
 * ------------------------------------------------------------------------ */
__global__ void vector_add_gpu(const float *a, const float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}

/* ---------------------------------------------------------------------------
 * CPU baseline -- plain serial loop
 * ------------------------------------------------------------------------ */
void vector_add_cpu(const float *a, const float *b, float *c, int n)
{
    for (int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

/* ---------------------------------------------------------------------------
 * Helpers
 * ------------------------------------------------------------------------ */
static void fill_random(float *arr, int n)
{
    for (int i = 0; i < n; i++)
        arr[i] = (float)rand() / RAND_MAX;
}

static int verify(const float *cpu, const float *gpu, int n)
{
    for (int i = 0; i < n; i++) {
        float diff = cpu[i] - gpu[i];
        if (diff < 0) diff = -diff;
        if (diff > 1e-5f) {
            printf("  MISMATCH at index %d: cpu=%.7f gpu=%.7f\n", i, cpu[i], gpu[i]);
            return 0;
        }
    }
    return 1;
}

/* Helper to check CUDA calls */
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d -- %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

/* ---------------------------------------------------------------------------
 * Main
 * ------------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    int n = DEFAULT_N;
    if (argc > 1)
        n = atoi(argv[1]);

    size_t bytes = (size_t)n * sizeof(float);

    printf("Vector Addition -- CPU vs GPU\n");
    printf("Elements:  %d (%.1f MB per array)\n\n", n, bytes / (1024.0 * 1024.0));

    /* Print GPU info */
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("GPU:       %s\n", prop.name);
    printf("SM count:  %d\n", prop.multiProcessorCount);
    printf("Compute:   %d.%d\n\n", prop.major, prop.minor);

    /* Allocate host memory using pinned (page-locked) memory.
       Pageable memory (malloc) forces cudaMemcpy to copy through a pinned
       staging buffer, roughly halving PCIe bandwidth.  Pinned memory allows
       DMA transfers directly between host RAM and the GPU. */
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    CUDA_CHECK(cudaHostAlloc(&h_a,     bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_b,     bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_c_cpu, bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_c_gpu, bytes, cudaHostAllocDefault));

    srand((unsigned)time(NULL));
    fill_random(h_a, n);
    fill_random(h_b, n);

    /* ----- CPU ----- */
    clock_t cpu_start = clock();
    vector_add_cpu(h_a, h_b, h_c_cpu, n);
    clock_t cpu_end = clock();
    double cpu_ms = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000.0;

    /* ----- GPU ----- */
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;

    /* Warmup -- first CUDA kernel launch pays a one-time cost for context
       initialisation and JIT compilation.  Run a throwaway launch so that
       the timed runs measure only the actual work. */
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    vector_add_gpu<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Use CUDA events for accurate GPU timing */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /* Copy data to device */
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    /* Launch kernel */
    CUDA_CHECK(cudaEventRecord(start));
    vector_add_gpu<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

    /* Also time the full round-trip (including memory transfers) */
    cudaEvent_t rt_start, rt_stop;
    CUDA_CHECK(cudaEventCreate(&rt_start));
    CUDA_CHECK(cudaEventCreate(&rt_stop));

    CUDA_CHECK(cudaEventRecord(rt_start));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    vector_add_gpu<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(rt_stop));
    CUDA_CHECK(cudaEventSynchronize(rt_stop));

    float roundtrip_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&roundtrip_ms, rt_start, rt_stop));

    /* Copy result back for first run too (for verification) */
    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost));

    /* ----- Results ----- */
    printf("%-30s %10.3f ms\n", "CPU (serial loop):", cpu_ms);
    printf("%-30s %10.3f ms\n", "GPU (kernel only):", gpu_ms);
    printf("%-30s %10.3f ms\n", "GPU (with memory transfers):", roundtrip_ms);
    printf("\n");
    printf("Kernel speedup:      %.1fx\n", cpu_ms / gpu_ms);
    printf("Round-trip speedup:  %.1fx\n", cpu_ms / roundtrip_ms);
    printf("\n");

    if (verify(h_c_cpu, h_c_gpu, n))
        printf("Verification: PASSED -- all %d elements match\n", n);
    else
        printf("Verification: FAILED\n");

    /* Cleanup */
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(rt_start));
    CUDA_CHECK(cudaEventDestroy(rt_stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFreeHost(h_a));
    CUDA_CHECK(cudaFreeHost(h_b));
    CUDA_CHECK(cudaFreeHost(h_c_cpu));
    CUDA_CHECK(cudaFreeHost(h_c_gpu));

    return 0;
}
