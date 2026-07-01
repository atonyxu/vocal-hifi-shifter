#pragma once

// ONNX-based pipeline: FCPE pitch detection + PC-NSF-HiFiGAN vocoder.
// Both follow HachiTune's implementation exactly.

#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

// ------------------------------------------------------------------
// FCPE pitch detector
// ------------------------------------------------------------------
class FCPEPitchDetector
{
public:
    static constexpr float F0_MIN = 32.7f;
    static constexpr float F0_MAX = 1975.5f;
    static constexpr int OUT_DIMS = 360;
    static constexpr int N_MELS = 128;
    static constexpr int N_FFT = 1024;
    static constexpr int WIN_SIZE = 1024;
    static constexpr int HOP_SIZE = 160;
    static constexpr int FCPE_SAMPLE_RATE = 16000;
    static constexpr float MEL_FMIN = 0.0f;
    static constexpr float MEL_FMAX = 8000.0f;
    static constexpr float CLIP_VAL = 1e-5f;

    bool loadModel(const std::string &modelPath,
                   const std::string &melFilterbankPath,
                   const std::string &centTablePath);
    ~FCPEPitchDetector() { if (session) delete session; }

    // Extract F0 at FCPE frame rate (100 fps). Returns F0 in Hz (0 = unvoiced).
    std::vector<float> extractF0(const float *audio, int numSamples,
                                int sampleRate, float threshold = 0.05f);

private:
    struct MelBand { int startBin = 0; int endBin = 0; std::vector<float> weights; };

    void initMelFilterbank();
    void initHannWindow();
    void initCentTable();
    std::vector<float> resampleTo16k(const float *audio, int numSamples, int srcRate);
    std::vector<std::vector<float>> extractMel(const std::vector<float> &audio);
    std::vector<float> decodeF0(const float *latent, int numFrames, float threshold);

    static float centToF0(float cent) { return 10.0f * std::pow(2.0f, cent / 1200.0f); }
    static float f0ToCent(float f0)   { return 1200.0f * std::log2(f0 / 10.0f); }

    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "fcpe"};
    Ort::Session *session = nullptr;
    std::vector<const char *> inputNames, outputNames;
    std::vector<std::string> inputNameStr, outputNameStr; // backing storage
    std::vector<MelBand> melFilterbank;
    std::vector<float> hannWindow;
    std::vector<float> centTable;
    bool loaded = false;
    bool useFileFilterbank = false;
};

// ------------------------------------------------------------------
// PC-NSF-HiFiGAN vocoder
// ------------------------------------------------------------------
class Vocoder
{
public:
    static constexpr int SAMPLE_RATE = 44100;
    static constexpr int HOP_SIZE = 512;
    static constexpr int NUM_MELS = 128;
    // Maximum frames per single ONNX inference (matches HachiTune).
    static constexpr size_t kMaxChunkFrames = 512;
    static constexpr size_t kOverlapFrames = 16;

    bool loadModel(const std::string &modelPath);
    ~Vocoder() { if (session) delete session; }
    bool isLoaded() const { return loaded; }

    // Synthesize waveform from mel [T, NUM_MELS] and F0 [T].
    std::vector<float> infer(const std::vector<std::vector<float>> &mel,
                             const std::vector<float> &f0);

private:
    std::vector<float> inferChunkLocked(const std::vector<std::vector<float>> &mel,
                                        const std::vector<float> &f0, size_t numFrames);

    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "vocoder"};
    Ort::Session *session = nullptr;
    std::vector<const char *> inputNames, outputNames;
    std::vector<std::string> inputNameStr, outputNameStr; // backing storage
    bool loaded = false;
};
