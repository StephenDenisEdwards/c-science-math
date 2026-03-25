/*
 * Fourier Transform Benchmark — Scalar vs SIMD (AVX)
 *
 * Builds a composite signal, runs both FFT implementations,
 * verifies correctness, and benchmarks across various sizes.
 *
 * Compile:
 *   gcc -O2 -mavx -lm -o fft_bench main.c
 *   (or use the provided Makefile)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

#include "fft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */

/* Generate a composite signal: sum of sinusoids */
static void generate_signal(double *re, double *im, int n, double sample_rate) {
    double freqs[]  = {5.0, 20.0, 50.0};
    double amps[]   = {1.0, 0.5,  0.3};
    int n_components = 3;

    memset(re, 0, n * sizeof(double));
    memset(im, 0, n * sizeof(double));

    for (int i = 0; i < n; i++) {
        double t = (double)i / sample_rate;
        for (int c = 0; c < n_components; c++) {
            re[i] += amps[c] * sin(2.0 * M_PI * freqs[c] * t);
        }
    }
}

/* High-resolution timer — platform-independent */
static double get_time(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
#endif
}

/* Returns elapsed time in seconds */
static double bench(void (*fn)(double*, double*, int, int),
                    double *re_src, double *im_src, int n, int iterations) {
    double *re = (double *)malloc(n * sizeof(double));
    double *im = (double *)malloc(n * sizeof(double));

    double start = get_time();

    for (int iter = 0; iter < iterations; iter++) {
        memcpy(re, re_src, n * sizeof(double));
        memcpy(im, im_src, n * sizeof(double));
        fn(re, im, n, 0);
    }

    double elapsed = get_time() - start;

    free(re);
    free(im);

    return elapsed / iterations;
}

/* ------------------------------------------------------------------ */
/*  Main                                                              */
/* ------------------------------------------------------------------ */

int main(void) {
    printf("=== Fourier Transform Benchmark: Scalar vs SIMD (AVX) ===\n\n");

    /* ---- 1. Correctness check ---- */
    {
        int n = 1024;
        double *re1 = (double *)malloc(n * sizeof(double));
        double *im1 = (double *)malloc(n * sizeof(double));
        double *re2 = (double *)malloc(n * sizeof(double));
        double *im2 = (double *)malloc(n * sizeof(double));

        generate_signal(re1, im1, n, 1000.0);
        memcpy(re2, re1, n * sizeof(double));
        memcpy(im2, im1, n * sizeof(double));

        fft_scalar(re1, im1, n, 0);

#ifdef __AVX__
        fft_simd(re2, im2, n, 0);

        double max_err = 0.0;
        for (int i = 0; i < n; i++) {
            double err_re = fabs(re1[i] - re2[i]);
            double err_im = fabs(im1[i] - im2[i]);
            if (err_re > max_err) max_err = err_re;
            if (err_im > max_err) max_err = err_im;
        }
        printf("Correctness check (N=1024):\n");
        printf("  Max error between scalar and SIMD: %.2e\n\n", max_err);
#else
        printf("Correctness check: AVX not available, skipping SIMD comparison.\n\n");
#endif

        free(re1); free(im1);
        free(re2); free(im2);
    }

    /* ---- 2. Benchmark ---- */
    {
        int sizes[] = {1024, 4096, 16384, 65536, 262144, 1048576};
        int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

        printf("%-12s  %15s", "N", "Scalar (ms)");
#ifdef __AVX__
        printf("  %15s  %10s", "SIMD (ms)", "Speedup");
#endif
        printf("\n");

        for (int i = 0; i < 60; i++) printf("-");
        printf("\n");

        for (int s = 0; s < n_sizes; s++) {
            int n = sizes[s];
            int iterations = n <= 16384 ? 1000 : (n <= 65536 ? 100 : 10);

            double *re = (double *)malloc(n * sizeof(double));
            double *im = (double *)malloc(n * sizeof(double));
            generate_signal(re, im, n, 1000.0);

            double t_scalar = bench(fft_scalar, re, im, n, iterations);

            printf("%-12d  %15.3f", n, t_scalar * 1000.0);

#ifdef __AVX__
            double t_simd = bench(fft_simd, re, im, n, iterations);
            printf("  %15.3f  %9.2fx", t_simd * 1000.0, t_scalar / t_simd);
#endif
            printf("\n");

            free(re);
            free(im);
        }
    }

    printf("\nDone.\n");
    return 0;
}
