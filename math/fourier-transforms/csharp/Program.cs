using System;
using System.Diagnostics;

namespace FourierTransform;

class Program
{
    static void GenerateSignal(double[] re, double[] im, double sampleRate)
    {
        double[] freqs = { 5.0, 20.0, 50.0 };
        double[] amps = { 1.0, 0.5, 0.3 };

        Array.Clear(re);
        Array.Clear(im);

        for (int i = 0; i < re.Length; i++)
        {
            double t = i / sampleRate;
            for (int c = 0; c < freqs.Length; c++)
                re[i] += amps[c] * Math.Sin(2.0 * Math.PI * freqs[c] * t);
        }
    }

    static double Bench(Action<double[], double[], bool> fft,
                        double[] reSrc, double[] imSrc, int iterations)
    {
        int n = reSrc.Length;
        double[] re = new double[n];
        double[] im = new double[n];

        // Warm up
        Array.Copy(reSrc, re, n);
        Array.Copy(imSrc, im, n);
        fft(re, im, false);

        var sw = Stopwatch.StartNew();
        for (int iter = 0; iter < iterations; iter++)
        {
            Array.Copy(reSrc, re, n);
            Array.Copy(imSrc, im, n);
            fft(re, im, false);
        }
        sw.Stop();

        return sw.Elapsed.TotalMilliseconds / iterations;
    }

    static void Main()
    {
        Console.WriteLine("=== Fourier Transform Benchmark: Scalar vs SIMD (C#) ===");
        Console.WriteLine($"    AVX supported: {Fft.SimdSupported}");
        Console.WriteLine();

        // ---- Correctness check ----
        {
            int n = 1024;
            double[] re1 = new double[n], im1 = new double[n];
            double[] re2 = new double[n], im2 = new double[n];

            GenerateSignal(re1, im1, 1000.0);
            Array.Copy(re1, re2, n);
            Array.Copy(im1, im2, n);

            Fft.Scalar(re1, im1);

            if (Fft.SimdSupported)
            {
                Fft.Simd(re2, im2);

                double maxErr = 0;
                for (int i = 0; i < n; i++)
                {
                    maxErr = Math.Max(maxErr, Math.Abs(re1[i] - re2[i]));
                    maxErr = Math.Max(maxErr, Math.Abs(im1[i] - im2[i]));
                }
                Console.WriteLine($"Correctness check (N=1024):");
                Console.WriteLine($"  Max error between scalar and SIMD: {maxErr:E2}");
            }
            else
            {
                Console.WriteLine("Correctness check: AVX not available, skipping SIMD comparison.");
            }
            Console.WriteLine();
        }

        // ---- Benchmark ----
        {
            int[] sizes = { 1024, 4096, 16384, 65536, 262144, 1048576 };

            Console.Write($"{"N",-12}  {"Scalar (ms)",15}");
            if (Fft.SimdSupported)
                Console.Write($"  {"SIMD (ms)",15}  {"Speedup",10}");
            Console.WriteLine();
            Console.WriteLine(new string('-', 60));

            foreach (int n in sizes)
            {
                int iterations = n <= 16384 ? 1000 : (n <= 65536 ? 100 : 10);

                double[] re = new double[n];
                double[] im = new double[n];
                GenerateSignal(re, im, 1000.0);

                double tScalar = Bench(Fft.Scalar, re, im, iterations);
                Console.Write($"{n,-12}  {tScalar,15:F3}");

                if (Fft.SimdSupported)
                {
                    double tSimd = Bench(Fft.Simd, re, im, iterations);
                    Console.Write($"  {tSimd,15:F3}  {tScalar / tSimd,9:F2}x");
                }

                Console.WriteLine();
            }
        }

        Console.WriteLine();
        Console.WriteLine("Done.");
    }
}
