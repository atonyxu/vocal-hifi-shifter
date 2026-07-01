#pragma once

// Radix-2 Cooley-Tukey FFT matching JUCE's performRealOnlyForwardTransform
// behavior. JUCE's FFT(order) has size = 2^order. The input buffer holds
// 2*size floats (interleaved complex: data[2k]=Re, data[2k+1]=Im).
// performRealOnlyForwardTransform zeros the imaginary parts and runs a forward
// complex FFT in place, so the output reads as interleaved complex spectrum.

#include <vector>
#include <cmath>

class FFT
{
public:
    explicit FFT(int order) : size(1 << order)
    {
        // Precompute bit-reversal table.
        bitRev.resize(size);
        const int bits = order;
        for (int i = 0; i < size; ++i)
        {
            int x = i, r = 0;
            for (int b = 0; b < bits; ++b) { r = (r << 1) | (x & 1); x >>= 1; }
            bitRev[i] = r;
        }
        // Precompute twiddle factors: exp(-2*pi*i*k / size).
        twiddles.resize(size);
        for (int k = 0; k < size; ++k)
        {
            const float ang = -2.0f * 3.14159265358979323846f * k / size;
            twiddles[k] = { std::cos(ang), std::sin(ang) };
        }
    }

    int getSize() const { return size; }

    // Mirrors juce::dsp::FFT::performRealOnlyForwardTransform(float* d).
    // d must hold 2*size floats. Real input in d[0..size-1] (even indices of
    // the complex view); imaginary parts are zeroed, then a forward complex
    // FFT of length `size` is performed in place.
    void performRealOnlyForwardTransform(float *d) const
    {
        // Zero imaginary parts (odd indices of the interleaved buffer).
        for (int i = 0; i < size; ++i)
            d[2 * i + 1] = 0.0f;

        forwardComplex(d);
    }

private:
    struct Complex { float re, im; };

    void forwardComplex(float *d) const
    {
        // Bit-reversal permutation over the `size` complex samples.
        for (int i = 0; i < size; ++i)
        {
            const int j = bitRev[i];
            if (j > i)
            {
                std::swap(d[2 * i],     d[2 * j]);
                std::swap(d[2 * i + 1], d[2 * j + 1]);
            }
        }
        // Iterative decimation-in-time butterfly.
        for (int half = 1; half < size; half <<= 1)
        {
            const int step = half * 2;
            const int twStride = size / step;
            for (int i = 0; i < size; i += step)
            {
                for (int k = 0; k < half; ++k)
                {
                    const Complex &w = twiddles[k * twStride];
                    float tr = d[2 * (i + k + half)];
                    float ti = d[2 * (i + k + half) + 1];
                    const float xr = tr * w.re - ti * w.im;
                    const float xi = tr * w.im + ti * w.re;
                    tr = d[2 * (i + k)];
                    ti = d[2 * (i + k) + 1];
                    d[2 * (i + k)]            = tr + xr;
                    d[2 * (i + k) + 1]        = ti + xi;
                    d[2 * (i + k + half)]     = tr - xr;
                    d[2 * (i + k + half) + 1] = ti - xi;
                }
            }
        }
    }

    int size;
    std::vector<int> bitRev;
    std::vector<Complex> twiddles;
};
