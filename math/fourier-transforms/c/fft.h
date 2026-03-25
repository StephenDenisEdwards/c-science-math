/*
 * fft.h — Scalar and SIMD FFT implementations
 *
 * Provides:
 *   fft_scalar()  — textbook radix-2 Cooley-Tukey FFT
 *   fft_simd()    — same algorithm with AVX intrinsics for the butterfly
 *   ifft_scalar() — inverse FFT (scalar)
 */

#ifndef FFT_H
#define FFT_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ------------------------------------------------------------------ */
/*  Bit-reversal permutation (shared by both implementations)         */
/* ------------------------------------------------------------------ */

static inline void bit_reverse(double *re, double *im, int n) {
    int j = 0;
    for (int i = 0; i < n - 1; i++) {
        if (i < j) {
            double tmp;
            tmp = re[i]; re[i] = re[j]; re[j] = tmp;
            tmp = im[i]; im[i] = im[j]; im[j] = tmp;
        }
        int k = n >> 1;
        while (k <= j) {
            j -= k;
            k >>= 1;
        }
        j += k;
    }
}

/* ------------------------------------------------------------------ */
/*  Scalar radix-2 Cooley-Tukey FFT                                   */
/* ------------------------------------------------------------------ */

#ifdef __GNUC__
__attribute__((optimize("no-tree-vectorize")))
#endif
static void fft_scalar(double *re, double *im, int n, int inverse) {
    bit_reverse(re, im, n);

    double sign = inverse ? 1.0 : -1.0;

    for (int len = 2; len <= n; len <<= 1) {
        double angle = sign * 2.0 * M_PI / len;
        double w_re = cos(angle);
        double w_im = sin(angle);

        for (int i = 0; i < n; i += len) {
            double cur_re = 1.0, cur_im = 0.0;

            for (int j = 0; j < len / 2; j++) {
                int u = i + j;
                int v = u + len / 2;

                /* butterfly */
                double t_re = cur_re * re[v] - cur_im * im[v];
                double t_im = cur_re * im[v] + cur_im * re[v];

                re[v] = re[u] - t_re;
                im[v] = im[u] - t_im;
                re[u] = re[u] + t_re;
                im[u] = im[u] + t_im;

                /* advance twiddle */
                double next_re = cur_re * w_re - cur_im * w_im;
                double next_im = cur_re * w_im + cur_im * w_re;
                cur_re = next_re;
                cur_im = next_im;
            }
        }
    }

    if (inverse) {
        for (int i = 0; i < n; i++) {
            re[i] /= n;
            im[i] /= n;
        }
    }
}

static inline void ifft_scalar(double *re, double *im, int n) {
    fft_scalar(re, im, n, 1);
}

/* ------------------------------------------------------------------ */
/*  AVX SIMD radix-2 FFT                                              */
/*                                                                    */
/*  Uses 256-bit AVX to process 4 doubles per instruction in the      */
/*  butterfly inner loop.  Falls back to scalar for stages where      */
/*  the butterfly count is < 4.                                       */
/* ------------------------------------------------------------------ */

#ifdef __AVX__
#include <immintrin.h>

static void fft_simd(double *re, double *im, int n, int inverse) {
    bit_reverse(re, im, n);

    double sign = inverse ? 1.0 : -1.0;

    for (int len = 2; len <= n; len <<= 1) {
        int half = len / 2;
        double angle = sign * 2.0 * M_PI / len;

        if (half >= 4) {
            /* Compute the initial 4 twiddle factors w^0..w^3 and the
             * "step-by-4" multiplier w^4 for iterative advancement.
             * This avoids all cos/sin calls inside the butterfly loop. */
            double w1_re = cos(angle), w1_im = sin(angle);

            /* w^0, w^1, w^2, w^3 */
            double init_re[4], init_im[4];
            init_re[0] = 1.0;           init_im[0] = 0.0;
            init_re[1] = w1_re;         init_im[1] = w1_im;
            init_re[2] = w1_re * w1_re - w1_im * w1_im;
            init_im[2] = 2.0 * w1_re * w1_im;
            init_re[3] = init_re[2] * w1_re - init_im[2] * w1_im;
            init_im[3] = init_re[2] * w1_im + init_im[2] * w1_re;

            /* w^4 — the step multiplier to advance all 4 lanes by 4 */
            double w4_re = cos(angle * 4), w4_im = sin(angle * 4);
            __m256d step_re = _mm256_set1_pd(w4_re);
            __m256d step_im = _mm256_set1_pd(w4_im);

            for (int i = 0; i < n; i += len) {
                /* Reset twiddle to w^0..w^3 for each block */
                __m256d wr = _mm256_loadu_pd(init_re);
                __m256d wi = _mm256_loadu_pd(init_im);

                for (int j = 0; j < half; j += 4) {
                    int u = i + j;
                    int v = u + half;

                    __m256d ure = _mm256_loadu_pd(&re[u]);
                    __m256d uim = _mm256_loadu_pd(&im[u]);
                    __m256d vre = _mm256_loadu_pd(&re[v]);
                    __m256d vim = _mm256_loadu_pd(&im[v]);

                    /* complex multiply: t = w * v */
                    __m256d t_re = _mm256_sub_pd(
                        _mm256_mul_pd(wr, vre),
                        _mm256_mul_pd(wi, vim));
                    __m256d t_im = _mm256_add_pd(
                        _mm256_mul_pd(wr, vim),
                        _mm256_mul_pd(wi, vre));

                    /* butterfly */
                    _mm256_storeu_pd(&re[v], _mm256_sub_pd(ure, t_re));
                    _mm256_storeu_pd(&im[v], _mm256_sub_pd(uim, t_im));
                    _mm256_storeu_pd(&re[u], _mm256_add_pd(ure, t_re));
                    _mm256_storeu_pd(&im[u], _mm256_add_pd(uim, t_im));

                    /* Advance twiddle by w^4: w_next = w_cur * w^4 */
                    __m256d new_wr = _mm256_sub_pd(
                        _mm256_mul_pd(wr, step_re),
                        _mm256_mul_pd(wi, step_im));
                    __m256d new_wi = _mm256_add_pd(
                        _mm256_mul_pd(wr, step_im),
                        _mm256_mul_pd(wi, step_re));
                    wr = new_wr;
                    wi = new_wi;
                }
            }
        } else {
            /* Scalar fallback for small stages */
            for (int i = 0; i < n; i += len) {
                double cur_re = 1.0, cur_im = 0.0;
                double w_re = cos(angle), w_im = sin(angle);

                for (int j = 0; j < half; j++) {
                    int u = i + j;
                    int v = u + half;

                    double t_re = cur_re * re[v] - cur_im * im[v];
                    double t_im = cur_re * im[v] + cur_im * re[v];

                    re[v] = re[u] - t_re;
                    im[v] = im[u] - t_im;
                    re[u] += t_re;
                    im[u] += t_im;

                    double next_re = cur_re * w_re - cur_im * w_im;
                    double next_im = cur_re * w_im + cur_im * w_re;
                    cur_re = next_re;
                    cur_im = next_im;
                }
            }
        }
    }

    if (inverse) {
        if (n >= 4) {
            __m256d inv = _mm256_set1_pd(1.0 / n);
            for (int i = 0; i < n; i += 4) {
                _mm256_storeu_pd(&re[i], _mm256_mul_pd(_mm256_loadu_pd(&re[i]), inv));
                _mm256_storeu_pd(&im[i], _mm256_mul_pd(_mm256_loadu_pd(&im[i]), inv));
            }
        } else {
            for (int i = 0; i < n; i++) {
                re[i] /= n;
                im[i] /= n;
            }
        }
    }
}

#endif /* __AVX__ */

#endif /* FFT_H */
