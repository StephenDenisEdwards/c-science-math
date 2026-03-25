using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace FourierTransform;

/// <summary>
/// Radix-2 Cooley-Tukey FFT — scalar and SIMD implementations.
/// </summary>
public static class Fft
{
    // ----------------------------------------------------------------
    //  Bit-reversal permutation (shared)
    // ----------------------------------------------------------------

    private static void BitReverse(double[] re, double[] im)
    {
        int n = re.Length;
        int j = 0;
        for (int i = 0; i < n - 1; i++)
        {
            if (i < j)
            {
                (re[i], re[j]) = (re[j], re[i]);
                (im[i], im[j]) = (im[j], im[i]);
            }
            int k = n >> 1;
            while (k <= j)
            {
                j -= k;
                k >>= 1;
            }
            j += k;
        }
    }

    // ----------------------------------------------------------------
    //  Scalar FFT
    // ----------------------------------------------------------------

    public static void Scalar(double[] re, double[] im, bool inverse = false)
    {
        int n = re.Length;
        BitReverse(re, im);

        double sign = inverse ? 1.0 : -1.0;

        for (int len = 2; len <= n; len <<= 1)
        {
            double angle = sign * 2.0 * Math.PI / len;
            double wRe = Math.Cos(angle);
            double wIm = Math.Sin(angle);

            for (int i = 0; i < n; i += len)
            {
                double curRe = 1.0, curIm = 0.0;

                for (int j = 0; j < len / 2; j++)
                {
                    int u = i + j;
                    int v = u + len / 2;

                    double tRe = curRe * re[v] - curIm * im[v];
                    double tIm = curRe * im[v] + curIm * re[v];

                    re[v] = re[u] - tRe;
                    im[v] = im[u] - tIm;
                    re[u] += tRe;
                    im[u] += tIm;

                    double nextRe = curRe * wRe - curIm * wIm;
                    double nextIm = curRe * wIm + curIm * wRe;
                    curRe = nextRe;
                    curIm = nextIm;
                }
            }
        }

        if (inverse)
        {
            for (int i = 0; i < n; i++)
            {
                re[i] /= n;
                im[i] /= n;
            }
        }
    }

    // ----------------------------------------------------------------
    //  SIMD FFT using System.Runtime.Intrinsics (AVX)
    //
    //  Optimisations over scalar:
    //    - AVX 256-bit: 4 doubles per instruction in butterfly loop
    //    - FMA where available (fused multiply-add)
    //    - Iterative twiddle advancement (no cos/sin in inner loop)
    // ----------------------------------------------------------------

    public static bool SimdSupported => Avx.IsSupported;

    public static unsafe void Simd(double[] re, double[] im, bool inverse = false)
    {
        if (!Avx.IsSupported)
            throw new PlatformNotSupportedException(
                "AVX is not supported on this processor. Use Fft.Scalar() instead.");

        int n = re.Length;
        BitReverse(re, im);

        double sign = inverse ? 1.0 : -1.0;

        for (int len = 2; len <= n; len <<= 1)
        {
            int half = len / 2;
            double angle = sign * 2.0 * Math.PI / len;

            if (half >= 4)
            {
                // Compute initial twiddle factors w^0..w^3 and step multiplier w^4
                double w1Re = Math.Cos(angle), w1Im = Math.Sin(angle);
                double w2Re = w1Re * w1Re - w1Im * w1Im;
                double w2Im = 2.0 * w1Re * w1Im;
                double w3Re = w2Re * w1Re - w2Im * w1Im;
                double w3Im = w2Re * w1Im + w2Im * w1Re;

                var initWr = Vector256.Create(1.0, w1Re, w2Re, w3Re);
                var initWi = Vector256.Create(0.0, w1Im, w2Im, w3Im);

                // w^4 step multiplier
                double w4Re = Math.Cos(angle * 4), w4Im = Math.Sin(angle * 4);
                var stepRe = Vector256.Create(w4Re);
                var stepIm = Vector256.Create(w4Im);

                fixed (double* pRe = re, pIm = im)
                {
                    for (int i = 0; i < n; i += len)
                    {
                        // Reset twiddle to w^0..w^3 for each block
                        var wr = initWr;
                        var wi = initWi;

                        for (int j = 0; j < half; j += 4)
                        {
                            int u = i + j;
                            int v = u + half;

                            var uRe = Avx.LoadVector256(pRe + u);
                            var uIm = Avx.LoadVector256(pIm + u);
                            var vRe = Avx.LoadVector256(pRe + v);
                            var vIm = Avx.LoadVector256(pIm + v);

                            Vector256<double> tRe, tIm;

                            if (Fma.IsSupported)
                            {
                                tRe = Fma.MultiplyAddNegated(wi, vIm,
                                      Avx.Multiply(wr, vRe));
                                tIm = Fma.MultiplyAdd(wi, vRe,
                                      Avx.Multiply(wr, vIm));
                            }
                            else
                            {
                                tRe = Avx.Subtract(
                                    Avx.Multiply(wr, vRe),
                                    Avx.Multiply(wi, vIm));
                                tIm = Avx.Add(
                                    Avx.Multiply(wr, vIm),
                                    Avx.Multiply(wi, vRe));
                            }

                            Avx.Store(pRe + v, Avx.Subtract(uRe, tRe));
                            Avx.Store(pIm + v, Avx.Subtract(uIm, tIm));
                            Avx.Store(pRe + u, Avx.Add(uRe, tRe));
                            Avx.Store(pIm + u, Avx.Add(uIm, tIm));

                            // Advance twiddle by w^4
                            var newWr = Avx.Subtract(
                                Avx.Multiply(wr, stepRe),
                                Avx.Multiply(wi, stepIm));
                            var newWi = Avx.Add(
                                Avx.Multiply(wr, stepIm),
                                Avx.Multiply(wi, stepRe));
                            wr = newWr;
                            wi = newWi;
                        }
                    }
                }
            }
            else
            {
                // Scalar fallback for small stages (half < 4)
                for (int i = 0; i < n; i += len)
                {
                    double curRe = 1.0, curIm = 0.0;
                    double wRe = Math.Cos(angle), wIm = Math.Sin(angle);

                    for (int j = 0; j < half; j++)
                    {
                        int u = i + j;
                        int v = u + half;

                        double tRe = curRe * re[v] - curIm * im[v];
                        double tIm = curRe * im[v] + curIm * re[v];

                        re[v] = re[u] - tRe;
                        im[v] = im[u] - tIm;
                        re[u] += tRe;
                        im[u] += tIm;

                        double nextRe = curRe * wRe - curIm * wIm;
                        double nextIm = curRe * wIm + curIm * wRe;
                        curRe = nextRe;
                        curIm = nextIm;
                    }
                }
            }
        }

        if (inverse)
        {
            fixed (double* pRe = re, pIm = im)
            {
                var inv = Vector256.Create(1.0 / n);
                int i = 0;
                for (; i + 4 <= n; i += 4)
                {
                    Avx.Store(pRe + i, Avx.Multiply(Avx.LoadVector256(pRe + i), inv));
                    Avx.Store(pIm + i, Avx.Multiply(Avx.LoadVector256(pIm + i), inv));
                }
                for (; i < n; i++)
                {
                    re[i] /= n;
                    im[i] /= n;
                }
            }
        }
    }
}
