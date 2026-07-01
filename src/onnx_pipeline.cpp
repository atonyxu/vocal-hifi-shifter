#include "onnx_pipeline.h"
#include "fft.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

#ifdef _WIN32
#include <Windows.h>
// Build a UTF-16 path for ONNX Runtime's wide-char session constructor on Windows.
static std::wstring widenPath(const std::string &s)
{
    if (s.empty()) return std::wstring();
    int needed = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring out(needed > 0 ? needed - 1 : 0, L'\0');
    if (needed > 1)
        MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &out[0], needed);
    return out;
}
#define ORT_MAKE_SESSION(env_var, path_str, opts_var) \
    new Ort::Session((env_var), widenPath(path_str).c_str(), (opts_var))
#else
#define ORT_MAKE_SESSION(env_var, path_str, opts_var) \
    new Ort::Session((env_var), (path_str).c_str(), (opts_var))
#endif

// ==================================================================
// FCPEPitchDetector
// ==================================================================

void FCPEPitchDetector::initMelFilterbank()
{
    // HTK mel formula (matches HachiTune's FCPE implementation).
    const int numBins = N_FFT / 2 + 1;
    auto hzToMel = [](float hz) -> float {
        return 2595.0f * std::log10(1.0f + hz / 700.0f);
    };
    auto melToHz = [](float mel) -> float {
        return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
    };

    const float melMin = hzToMel(MEL_FMIN);
    const float melMax = hzToMel(MEL_FMAX);

    std::vector<float> melPoints(N_MELS + 2);
    for (int i = 0; i <= N_MELS + 1; ++i)
        melPoints[i] = melMin + (melMax - melMin) * i / (N_MELS + 1);

    std::vector<float> hzPoints(N_MELS + 2);
    for (int i = 0; i <= N_MELS + 1; ++i)
        hzPoints[i] = melToHz(melPoints[i]);

    melFilterbank.resize(N_MELS);
    for (int m = 0; m < N_MELS; ++m)
    {
        float fLow = hzPoints[m], fCenter = hzPoints[m + 1], fHigh = hzPoints[m + 2];
        float enorm = 2.0f / (fHigh - fLow);
        int firstBin = numBins, lastBin = -1;
        for (int k = 0; k < numBins; ++k)
        {
            float freq = float(k) * FCPE_SAMPLE_RATE / N_FFT;
            if (freq >= fLow && freq <= fHigh) { if (k < firstBin) firstBin = k; if (k > lastBin) lastBin = k; }
        }
        MelBand &band = melFilterbank[m];
        if (lastBin < firstBin) { band.startBin = 0; band.endBin = 0; continue; }
        band.startBin = firstBin;
        band.endBin = lastBin + 1;
        band.weights.assign(band.endBin - band.startBin, 0.0f);
        for (int k = firstBin; k <= lastBin; ++k)
        {
            float freq = float(k) * FCPE_SAMPLE_RATE / N_FFT;
            if (freq >= fLow && freq < fCenter)
                band.weights[k - firstBin] = enorm * (freq - fLow) / (fCenter - fLow);
            else if (freq >= fCenter && freq <= fHigh)
                band.weights[k - firstBin] = enorm * (fHigh - freq) / (fHigh - fCenter);
        }
    }
}

void FCPEPitchDetector::initHannWindow()
{
    // Symmetric Hann (numpy.hanning style).
    hannWindow.resize(WIN_SIZE);
    for (int i = 0; i < WIN_SIZE; ++i)
        hannWindow[i] = 0.5f * (1.0f - std::cos(2.0f * 3.14159265358979323846f * i / (WIN_SIZE - 1)));
}

void FCPEPitchDetector::initCentTable()
{
    centTable.resize(OUT_DIMS);
    const float centMin = f0ToCent(F0_MIN);
    const float centMax = f0ToCent(F0_MAX);
    for (int i = 0; i < OUT_DIMS; ++i)
        centTable[i] = centMin + (centMax - centMin) * i / (OUT_DIMS - 1);
}

bool FCPEPitchDetector::loadModel(const std::string &modelPath,
                                   const std::string &melFilterbankPath,
                                   const std::string &centTablePath)
{
    initMelFilterbank();
    initHannWindow();
    initCentTable();

    // Override mel filterbank from file (dense N_MELS x numBins -> sparse).
    if (!melFilterbankPath.empty())
    {
        std::ifstream f(melFilterbankPath, std::ios::binary);
        if (f)
        {
            f.seekg(0, std::ios::end);
            const int numBins = N_FFT / 2 + 1;
            const size_t expect = size_t(N_MELS) * numBins * sizeof(float);
            if (size_t(f.tellg()) >= expect)
            {
                std::vector<float> data(N_MELS * numBins);
                f.seekg(0, std::ios::beg);
                f.read(reinterpret_cast<char *>(data.data()), expect);
                for (int m = 0; m < N_MELS; ++m)
                {
                    int firstBin = numBins, lastBin = -1;
                    for (int k = 0; k < numBins; ++k)
                        if (data[m * numBins + k] != 0.0f)
                        { if (k < firstBin) firstBin = k; lastBin = k; }
                    MelBand &band = melFilterbank[m];
                    if (lastBin < firstBin) { band.startBin = 0; band.endBin = 0; continue; }
                    band.startBin = firstBin;
                    band.endBin = lastBin + 1;
                    band.weights.resize(band.endBin - band.startBin);
                    for (int k = firstBin; k <= lastBin; ++k)
                        band.weights[k - firstBin] = data[m * numBins + k];
                }
                useFileFilterbank = true;
            }
        }
    }

    // Override cent table from file.
    if (!centTablePath.empty())
    {
        std::ifstream f(centTablePath, std::ios::binary);
        if (f)
        {
            f.seekg(0, std::ios::end);
            if (size_t(f.tellg()) >= OUT_DIMS * sizeof(float))
            {
                f.seekg(0, std::ios::beg);
                f.read(reinterpret_cast<char *>(centTable.data()), OUT_DIMS * sizeof(float));
            }
        }
    }

    try
    {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetInterOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session = ORT_MAKE_SESSION(env, modelPath, opts);

        // ONNX releases name buffers with the session; keep our own persistent
        // string copies and hand out raw pointers into them.
        Ort::AllocatorWithDefaultOptions alloc;
        inputNameStr.clear(); outputNameStr.clear();
        for (size_t i = 0; i < session->GetInputCount(); ++i)
        { auto p = session->GetInputNameAllocated(i, alloc); inputNameStr.push_back(p.get()); }
        for (size_t i = 0; i < session->GetOutputCount(); ++i)
        { auto p = session->GetOutputNameAllocated(i, alloc); outputNameStr.push_back(p.get()); }
        inputNames.clear(); outputNames.clear();
        for (auto &s : inputNameStr) inputNames.push_back(s.c_str());
        for (auto &s : outputNameStr) outputNames.push_back(s.c_str());

        loaded = true;
        return true;
    }
    catch (const Ort::Exception &e)
    {
        std::cerr << "[fcpe] failed to load model: " << e.what() << "\n";
        loaded = false;
        return false;
    }
}

std::vector<float> FCPEPitchDetector::resampleTo16k(const float *audio, int numSamples, int srcRate)
{
    if (srcRate == FCPE_SAMPLE_RATE)
        return std::vector<float>(audio, audio + numSamples);
    // Linear interpolation.
    const double ratio = double(FCPE_SAMPLE_RATE) / srcRate;
    const int outSamples = int(numSamples * ratio);
    std::vector<float> out(outSamples);
    for (int i = 0; i < outSamples; ++i)
    {
        double srcPos = i / ratio;
        int srcIdx = int(srcPos);
        double frac = srcPos - srcIdx;
        if (srcIdx + 1 < numSamples)
            out[i] = float(audio[srcIdx] * (1.0 - frac) + audio[srcIdx + 1] * frac);
        else if (srcIdx < numSamples)
            out[i] = audio[srcIdx];
    }
    return out;
}

std::vector<std::vector<float>> FCPEPitchDetector::extractMel(const std::vector<float> &audio)
{
    const int numBins = N_FFT / 2 + 1;
    const int padLeft = (WIN_SIZE - HOP_SIZE) / 2;
    const int padRight = std::max((WIN_SIZE - HOP_SIZE + 1) / 2,
                                  WIN_SIZE - int(audio.size()) - padLeft);

    std::vector<float> padded;
    if (padRight < int(audio.size()))
    {
        padded.reserve(padLeft + audio.size() + padRight);
        for (int i = padLeft; i > 0; --i)
            padded.push_back(audio[std::min(i, int(audio.size()) - 1)]);
        padded.insert(padded.end(), audio.begin(), audio.end());
        const int audioSize = int(audio.size());
        for (int i = 0; i < padRight; ++i)
        {
            int idx = audioSize - 2 - i;
            padded.push_back(audio[idx < 0 ? 0 : idx]);
        }
    }
    else
    {
        padded.resize(padLeft + audio.size() + padRight, 0.0f);
        std::copy(audio.begin(), audio.end(), padded.begin() + padLeft);
    }

    int numFrames = 1 + (int(padded.size()) - WIN_SIZE) / HOP_SIZE;
    if (numFrames < 1) numFrames = 1;

    std::vector<std::vector<float>> mel(numFrames, std::vector<float>(N_MELS, 0.0f));
    std::vector<float> fftBuf(N_FFT * 2, 0.0f);
    std::vector<float> mag(numBins);
    FFT fft(int(std::log2(N_FFT)));

    for (int frame = 0; frame < numFrames; ++frame)
    {
        int start = frame * HOP_SIZE;
        std::fill(fftBuf.begin(), fftBuf.end(), 0.0f);
        for (int i = 0; i < WIN_SIZE && start + i < int(padded.size()); ++i)
            fftBuf[i] = padded[start + i] * hannWindow[i];

        fft.performRealOnlyForwardTransform(fftBuf.data());

        for (int k = 0; k < numBins; ++k)
        {
            float re = fftBuf[k * 2], im = fftBuf[k * 2 + 1];
            mag[k] = std::sqrt(re * re + im * im + 1e-9f);
        }
        for (int m = 0; m < N_MELS; ++m)
        {
            const auto &band = melFilterbank[m];
            float sum = 0.0f;
            for (int k = band.startBin; k < band.endBin; ++k)
                sum += mag[k] * band.weights[k - band.startBin];
            mel[frame][m] = std::log(std::max(sum, CLIP_VAL));
        }
    }
    return mel;
}

std::vector<float> FCPEPitchDetector::decodeF0(const float *latent, int numFrames, float threshold)
{
    std::vector<float> f0(numFrames, 0.0f);
    for (int t = 0; t < numFrames; ++t)
    {
        const float *frame = latent + size_t(t) * OUT_DIMS;
        int maxIdx = 0;
        float maxVal = frame[0];
        for (int i = 1; i < OUT_DIMS; ++i)
            if (frame[i] > maxVal) { maxVal = frame[i]; maxIdx = i; }
        if (maxVal <= threshold) { f0[t] = 0.0f; continue; }
        int localStart = std::max(0, maxIdx - 4);
        int localEnd = std::min(OUT_DIMS - 1, maxIdx + 4);
        float ws = 0.0f, ww = 0.0f;
        for (int i = localStart; i <= localEnd; ++i)
        { ws += centTable[i] * frame[i]; ww += frame[i]; }
        f0[t] = ww > 1e-9f ? centToF0(ws / ww) : 0.0f;
    }
    return f0;
}

std::vector<float> FCPEPitchDetector::extractF0(const float *audio, int numSamples,
                                                int sampleRate, float threshold)
{
    if (!loaded) return {};
    try
    {
        auto audio16k = resampleTo16k(audio, numSamples, sampleRate);
        auto mel = extractMel(audio16k);
        if (mel.empty()) return {};

        const int numFrames = int(mel.size());
        std::vector<float> melInput(size_t(numFrames) * N_MELS);
        for (int t = 0; t < numFrames; ++t)
            for (int m = 0; m < N_MELS; ++m)
                melInput[size_t(t) * N_MELS + m] = mel[t][m];

        int64_t shape[3] = {1, numFrames, N_MELS};
        auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto inputTensor = Ort::Value::CreateTensor<float>(
            memInfo, melInput.data(), melInput.size(), shape, 3);

        Ort::RunOptions runOpts{nullptr};
        auto outputs = session->Run(runOpts, inputNames.data(), &inputTensor, 1,
                                   outputNames.data(), 1);

        float *out = outputs[0].GetTensorMutableData<float>();
        const size_t count = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
        if (count < size_t(OUT_DIMS)) return {};
        const int outFrames = int(count / OUT_DIMS);
        return decodeF0(out, outFrames, threshold);
    }
    catch (const Ort::Exception &e)
    {
        std::cerr << "[fcpe] inference failed: " << e.what() << "\n";
        return {};
    }
}

// ==================================================================
// Vocoder
// ==================================================================

bool Vocoder::loadModel(const std::string &modelPath)
{
    try
    {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetInterOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef HIFISHIFTER_USE_TENSORRT
        // TensorRT execution provider (defaults, matches HachiTune usage).
        // The TensorRT EP requires the CUDA EP as a fallback for unsupported
        // subgraphs, so register CUDA after TensorRT.
        OrtTensorRTProviderOptions trtOpts{};
        opts.AppendExecutionProvider_TensorRT(trtOpts);
        OrtCUDAProviderOptions cudaOpts{};
        cudaOpts.device_id = 0;
        opts.AppendExecutionProvider_CUDA(cudaOpts);
#endif

        session = ORT_MAKE_SESSION(env, modelPath, opts);

        Ort::AllocatorWithDefaultOptions alloc;
        inputNameStr.clear(); outputNameStr.clear();
        for (size_t i = 0; i < session->GetInputCount(); ++i)
        { auto p = session->GetInputNameAllocated(i, alloc); inputNameStr.push_back(p.get()); }
        for (size_t i = 0; i < session->GetOutputCount(); ++i)
        { auto p = session->GetOutputNameAllocated(i, alloc); outputNameStr.push_back(p.get()); }
        inputNames.clear(); outputNames.clear();
        for (auto &s : inputNameStr) inputNames.push_back(s.c_str());
        for (auto &s : outputNameStr) outputNames.push_back(s.c_str());

        loaded = true;
        return true;
    }
    catch (const Ort::Exception &e)
    {
        std::cerr << "[vocoder] failed to load model: " << e.what() << "\n";
        loaded = false;
        return false;
    }
}

std::vector<float> Vocoder::inferChunkLocked(const std::vector<std::vector<float>> &mel,
                                             const std::vector<float> &f0, size_t numFrames)
{
    constexpr float kMelMinClamp = -15.0f, kMelMaxClamp = 5.0f;
    constexpr float kF0MinValid = 20.0f, kF0MaxValid = 2000.0f;

    // Mel layout: [1, NUM_MELS, numFrames] (mel-major is frame-strided).
    std::vector<int64_t> melShape = {1, NUM_MELS, int64_t(numFrames)};
    std::vector<float> melBuf(NUM_MELS * numFrames);
    for (size_t frame = 0; frame < numFrames; ++frame)
    {
        const auto &src = mel[frame];
        const int melCount = std::min(NUM_MELS, int(src.size()));
        size_t dst = frame;
        int m = 0;
        for (; m < melCount; ++m, dst += numFrames)
            melBuf[dst] = std::clamp(src[m], kMelMinClamp, kMelMaxClamp);
        for (; m < NUM_MELS; ++m, dst += numFrames)
            melBuf[dst] = 0.0f;
    }

    // F0 layout: [1, numFrames].
    std::vector<int64_t> f0Shape = {1, int64_t(numFrames)};
    std::vector<float> f0Buf(numFrames);
    for (size_t i = 0; i < numFrames; ++i)
    {
        const float freq = f0[i];
        f0Buf[i] = (freq > 0.0f) ? std::clamp(freq, kF0MinValid, kF0MaxValid) : 0.0f;
    }

    auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> inputs;
    inputs.emplace_back(Ort::Value::CreateTensor<float>(memInfo, melBuf.data(), melBuf.size(),
                                                       melShape.data(), melShape.size()));
    inputs.emplace_back(Ort::Value::CreateTensor<float>(memInfo, f0Buf.data(), f0Buf.size(),
                                                       f0Shape.data(), f0Shape.size()));

    Ort::RunOptions runOpts{nullptr};
    auto outputs = session->Run(runOpts, inputNames.data(), inputs.data(), inputs.size(),
                                outputNames.data(), 1);
    if (outputs.empty() || !outputs[0].HasValue()) return {};

    float *out = outputs[0].GetTensorMutableData<float>();
    const size_t outSize = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> wav(outSize);
    for (size_t i = 0; i < outSize; ++i)
        wav[i] = std::clamp(out[i], -1.0f, 1.0f);
    return wav;
}

std::vector<float> Vocoder::infer(const std::vector<std::vector<float>> &mel,
                                   const std::vector<float> &f0)
{
    if (!loaded || mel.empty() || f0.empty()) return {};
    const size_t numFrames = std::min(mel.size(), f0.size());
    if (numFrames == 0) return {};

    try
    {
        // Short input: single pass.
        if (numFrames <= kMaxChunkFrames)
            return inferChunkLocked(mel, f0, numFrames);

        // Long input: chunked inference with crossfade.
        const size_t step = kMaxChunkFrames - kOverlapFrames;
        const size_t overlapSamples = kOverlapFrames * HOP_SIZE;
        const size_t totalSamples = numFrames * HOP_SIZE;

        std::vector<float> waveform(totalSamples, 0.0f);
        size_t writeEnd = 0;
        size_t chunkIdx = 0;
        for (size_t frameOff = 0; frameOff < numFrames; frameOff += step, ++chunkIdx)
        {
            const size_t chunkEnd = std::min(frameOff + kMaxChunkFrames, numFrames);
            const size_t chunkFrames = chunkEnd - frameOff;

            std::vector<std::vector<float>> chunkMel(mel.begin() + frameOff, mel.begin() + chunkEnd);
            std::vector<float> chunkF0(f0.begin() + frameOff, f0.begin() + chunkEnd);

            auto chunkWav = inferChunkLocked(chunkMel, chunkF0, chunkFrames);
            if (chunkWav.empty())
            {
                std::cerr << "[vocoder] chunk " << chunkIdx << " failed\n";
                return {};
            }

            const size_t dstOffset = frameOff * HOP_SIZE;
            const size_t chunkSamples = chunkWav.size();

            if (chunkIdx == 0)
            {
                size_t copyLen = std::min(chunkSamples, totalSamples);
                std::copy_n(chunkWav.begin(), copyLen, waveform.begin());
                writeEnd = copyLen;
            }
            else
            {
                size_t prevTail = (writeEnd > dstOffset) ? (writeEnd - dstOffset) : 0;
                size_t fade = std::min({overlapSamples, prevTail, chunkSamples});
                for (size_t i = 0; i < fade; ++i)
                {
                    float t = float(i) / float(fade);
                    waveform[dstOffset + i] = waveform[dstOffset + i] * (1.0f - t) + chunkWav[i] * t;
                }
                size_t tailStart = dstOffset + fade;
                size_t srcTail = (chunkSamples > fade) ? (chunkSamples - fade) : 0;
                size_t safeTail = std::min(srcTail, totalSamples - tailStart);
                if (safeTail > 0)
                    std::copy_n(chunkWav.begin() + fade, safeTail, waveform.begin() + tailStart);
                writeEnd = std::min(tailStart + safeTail, totalSamples);
            }
        }
        waveform.resize(writeEnd);
        for (auto &s : waveform) s = std::clamp(s, -1.0f, 1.0f);
        return waveform;
    }
    catch (const Ort::Exception &e)
    {
        std::cerr << "[vocoder] inference failed: " << e.what() << "\n";
        return {};
    }
}
