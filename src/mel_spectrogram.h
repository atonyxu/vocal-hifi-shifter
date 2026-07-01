#pragma once

// Vocoder mel spectrogram. Matches HachiTune's MelSpectrogram exactly:
//  - sampleRate=44100, nFft=2048, hopSize=512, numMels=128, fMin=40, fMax=16000
//  - Periodic Hann window: 0.5 * (1 - cos(2*pi*i / nFft))
//  - Center reflect padding (pad = nFft/2 each side)
//  - Slaney mel scale (NOT HTK): piecewise-linear below 1000 Hz, log above
//  - Slaney area normalization: enorm = 2 / (fHigh - fLow)
//  - Sparse per-band filterbank
//  - Natural log magnitude: log(max(sum, 1e-10))

#include "fft.h"
#include <algorithm>
#include <cmath>
#include <vector>

class MelSpectrogram
{
public:
    MelSpectrogram(int sampleRate = 44100, int nFft = 2048, int hopSize = 512,
                   int numMels = 128, float fMin = 40.0f, float fMax = 16000.0f)
        : sampleRate(sampleRate), nFft(nFft), hopSize(hopSize),
          numMels(numMels), fMin(fMin), fMax(fMax), fft(int(std::log2(nFft)))
    {
        // Periodic Hann window (matches JUCE/librosa default).
        window.resize(nFft);
        for (int i = 0; i < nFft; ++i)
            window[i] = 0.5f * (1.0f - std::cos(2.0f * 3.14159265358979323846f * i / nFft));
        createMelFilterbank();
    }

    // Returns mel spectrogram [T, numMels] (frame-major), natural log scale.
    std::vector<std::vector<float>> compute(const float *audio, int numSamples) const
    {
        const int padLeft  = nFft / 2;
        const int padRight = nFft / 2;
        const int paddedLength = numSamples + padLeft + padRight;
        int numFrames = (paddedLength - nFft) / hopSize + 1;
        if (numFrames < 1) numFrames = 1;

        const int numBins = nFft / 2 + 1;
        std::vector<std::vector<float>> mel(numFrames, std::vector<float>(numMels));
        std::vector<float> frame(nFft * 2, 0.0f); // interleaved complex FFT buffer
        std::vector<float> mag(numBins);

        for (int i = 0; i < numFrames; ++i)
        {
            const int centerSample = i * hopSize;
            const int startSample  = centerSample - padLeft;

            std::fill(frame.begin(), frame.end(), 0.0f);
            for (int j = 0; j < nFft; ++j)
            {
                int srcIdx = startSample + j;
                if (srcIdx < 0)
                    frame[j] = audio[std::min(-srcIdx - 1, numSamples - 1)] * window[j];
                else if (srcIdx >= numSamples)
                {
                    int reflectIdx = numSamples - 1 - (srcIdx - numSamples);
                    frame[j] = audio[std::max(0, reflectIdx)] * window[j];
                }
                else
                    frame[j] = audio[srcIdx] * window[j];
            }

            fft.performRealOnlyForwardTransform(frame.data());

            for (int k = 0; k < numBins; ++k)
            {
                float re = frame[k * 2];
                float im = frame[k * 2 + 1];
                mag[k] = std::sqrt(re * re + im * im + 1e-9f);
            }

            for (int m = 0; m < numMels; ++m)
            {
                const auto &band = melFilterbank[m];
                float sum = 0.0f;
                for (int k = band.startBin; k < band.endBin; ++k)
                    sum += mag[k] * band.weights[k - band.startBin];
                mel[i][m] = std::log(std::max(sum, 1e-10f));
            }
        }
        return mel;
    }

private:
    struct MelBand
    {
        int startBin = 0;
        int endBin = 0;
        std::vector<float> weights;
    };

    void createMelFilterbank()
    {
        // Slaney mel scale (matches librosa default with htk=False).
        const float f_min_mel    = 0.0f;
        const float f_sp         = 200.0f / 3.0f; // ~66.67 Hz/mel below 1000 Hz
        const float min_log_hz   = 1000.0f;
        const float min_log_mel  = (min_log_hz - f_min_mel) / f_sp; // = 15.0
        const float logstep      = std::log(6.4f) / 27.0f;           // ~0.0687

        auto hzToMel = [=](float hz) -> float {
            if (hz < min_log_hz) return (hz - f_min_mel) / f_sp;
            return min_log_mel + std::log(hz / min_log_hz) / logstep;
        };
        auto melToHz = [=](float mel) -> float {
            if (mel < min_log_mel) return f_min_mel + f_sp * mel;
            return min_log_hz * std::exp(logstep * (mel - min_log_mel));
        };

        const float melMin = hzToMel(fMin);
        const float melMax = hzToMel(fMax);

        std::vector<float> melPoints(numMels + 2);
        for (int i = 0; i <= numMels + 1; ++i)
            melPoints[i] = melMin + (melMax - melMin) * i / (numMels + 1);

        std::vector<float> hzPoints(numMels + 2);
        for (int i = 0; i <= numMels + 1; ++i)
            hzPoints[i] = melToHz(melPoints[i]);

        const int numBins = nFft / 2 + 1;
        melFilterbank.resize(numMels);
        for (int m = 0; m < numMels; ++m)
        {
            float fLow = hzPoints[m];
            float fCenter = hzPoints[m + 1];
            float fHigh = hzPoints[m + 2];
            float enorm = 2.0f / (fHigh - fLow);

            int firstBin = numBins, lastBin = -1;
            for (int k = 0; k < numBins; ++k)
            {
                float freq = float(k) * sampleRate / nFft;
                if (freq >= fLow && freq <= fHigh)
                {
                    if (k < firstBin) firstBin = k;
                    if (k > lastBin)  lastBin = k;
                }
            }

            MelBand &band = melFilterbank[m];
            if (lastBin < firstBin) { band.startBin = 0; band.endBin = 0; continue; }
            band.startBin = firstBin;
            band.endBin = lastBin + 1;
            band.weights.assign(band.endBin - band.startBin, 0.0f);

            for (int k = firstBin; k <= lastBin; ++k)
            {
                float freq = float(k) * sampleRate / nFft;
                if (freq >= fLow && freq < fCenter)
                    band.weights[k - firstBin] = enorm * (freq - fLow) / (fCenter - fLow);
                else if (freq >= fCenter && freq <= fHigh)
                    band.weights[k - firstBin] = enorm * (fHigh - freq) / (fHigh - fCenter);
            }
        }
    }

    int sampleRate, nFft, hopSize, numMels;
    float fMin, fMax;
    std::vector<float> window;
    std::vector<MelBand> melFilterbank;
    FFT fft;
};
