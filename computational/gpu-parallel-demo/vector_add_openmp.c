/*
 * GPU Parallel Vector Addition -- OpenMP Target Offloading
 *
 * Same demo as the CUDA and OpenCL versions but using OpenMP's
 * `target` directive to offload work to a GPU. This is the simplest
 * approach -- just add pragmas to standard C code.
 *
 * If no GPU is available (or the compiler doesn't support offloading),
 * OpenMP silently falls back to running on the CPU with threads,
 * so this always produces correct results.
 *
 * Compile (with GPU offloading -- requires compiler support):
 *   gcc -O2 -fopenmp -foffload=nvptx-none -o vector_add_openmp vector_add_openmp.c
 *
 * Compile (CPU-only fallback -- any compiler with OpenMP):
 *   gcc -O2 -fopenmp -o vector_add_openmp vector_add_openmp.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define DEFAULT_N (1 << 24)  /* 16 million elements */

/* ---------------------------------------------------------------------------
 * GPU kernel via OpenMP target -- one line of pragma does all the work
 * ------------------------------------------------------------------------ */
static void vector_add_gpu(const float *a, const float *b, float *c, int n)
{
    #pragma omp target teams distribute parallel for \
        map(to: a[0:n], b[0:n]) map(from: c[0:n])
    for (int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

/* ---------------------------------------------------------------------------
 * CPU baseline -- plain serial loop (no OpenMP)
 * ------------------------------------------------------------------------ */
static void vector_add_cpu(const float *a, const float *b, float *c, int n)
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
            printf("  MISMATCH at index %d: cpu=%.7f gpu=%.7f\n",
                   i, cpu[i], gpu[i]);
            return 0;
        }
    }
    return 1;
}

/* ---------------------------------------------------------------------------
 * Main
 * ------------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    int n = DEFAULT_N;
    if (argc > 1)
        n = atoi(argv[1]);

    size_t bytes = (size_t)n * sizeof(float);

    printf("Vector Addition -- CPU vs GPU (OpenMP Target Offloading)\n");
    printf("Elements:  %d (%.1f MB per array)\n\n", n, bytes / (1024.0 * 1024.0));

    /* Check if GPU offloading is available */
    int num_devices = omp_get_num_devices();
    int default_device = omp_get_default_device();
    printf("OpenMP devices: %d\n", num_devices);
    if (num_devices > 0)
        printf("Default device: %d\n\n", default_device);
    else
        printf("No GPU devices found -- will run on CPU with threads\n\n");

    /* Allocate */
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c_cpu = (float *)malloc(bytes);
    float *h_c_gpu = (float *)malloc(bytes);

    srand((unsigned)time(NULL));
    fill_random(h_a, n);
    fill_random(h_b, n);

    /* ----- CPU (serial) ----- */
    double cpu_start = omp_get_wtime();
    vector_add_cpu(h_a, h_b, h_c_cpu, n);
    double cpu_ms = (omp_get_wtime() - cpu_start) * 1000.0;

    /* ----- GPU (OpenMP target) ----- */
    /* Warm-up run to trigger device initialisation */
    float dummy_a = 1.0f, dummy_b = 2.0f, dummy_c;
    #pragma omp target map(to: dummy_a, dummy_b) map(from: dummy_c)
    { dummy_c = dummy_a + dummy_b; }
    (void)dummy_c;

    double gpu_start = omp_get_wtime();
    vector_add_gpu(h_a, h_b, h_c_gpu, n);
    double gpu_ms = (omp_get_wtime() - gpu_start) * 1000.0;

    /* ----- Results ----- */
    printf("%-30s %10.3f ms\n", "CPU (serial loop):", cpu_ms);
    printf("%-30s %10.3f ms\n", "GPU/OpenMP target:", gpu_ms);
    printf("\n");

    if (gpu_ms > 0)
        printf("Speedup: %.1fx\n\n", cpu_ms / gpu_ms);

    if (verify(h_c_cpu, h_c_gpu, n))
        printf("Verification: PASSED -- all %d elements match\n", n);
    else
        printf("Verification: FAILED\n");

    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);

    return 0;
}
