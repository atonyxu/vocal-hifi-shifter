// hifishifter - PC-NSF-HiFiGAN vocal pitch shifter (TensorRT-capable)
//
// Usage:
//   hifishifter -i <input.wav> -o <output.wav> -p <semitones> [options]
//
// Options:
//   -i, --input <path>      Input vocal WAV file (required)
//   -o, --output <path>     Output WAV file (required)
//   -p, --pitch <float>     Pitch shift in semitones (+12 = octave up) (required)
//   -m, --models <dir>      Models directory (default: ./models)
//   -h, --help              Show this help
//
// Models expected under --models dir:
//   pc_nsf_hifigan.onnx, fcpe.onnx, cent_table.bin, mel_filterbank.bin

#include "wav_io.h"
#include "mel_spectrogram.h"
#include "onnx_pipeline.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

namespace
{

void printHelp()
{
    std::cout <<
        "hifishifter - PC-NSF-HiFiGAN vocal pitch shifter\n\n"
        "Usage:\n"
        "  hifishifter -i <input.wav> -o <output.wav> -p <semitones> [options]\n\n"
        "Options:\n"
        "  -i, --input <path>      Input vocal WAV file (required)\n"
        "  -o, --output <path>      Output WAV file (required)\n"
        "  -p, --pitch <float>      Pitch shift in semitones (+12 = octave up) (required)\n"
        "  -m, --models <dir>       Models directory (default: ./models)\n"
        "  -h, --help               Show this help\n"
        "\n"
        "TensorRT is enabled when the binary is built with HIFISHIFTER_USE_TENSORRT.\n";
}

// Path join helper that works on both Windows and POSIX.
std::string joinPath(const std::string &a, const std::string &b)
{
    if (a.empty()) return b;
    if (b.empty()) return a;
    char sep =
#ifdef _WIN32
        '\\';
#else
        '/';
#endif
    if (a.back() == '/' || a.back() == '\\' || b.front() == '/' || b.front() == '\\')
        return a + b;
    return a + sep + b;
}

} // namespace

int main(int argc, char **argv)
{
    std::string inputPath, outputPath, modelsDir = "models";
    float semitones = 0.0f;
    bool havePitch = false;

    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        auto next = [&](const char *name) -> std::string {
            if (i + 1 >= argc) { std::cerr << "error: missing value for " << name << "\n"; std::exit(2); }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") { printHelp(); return 0; }
        else if (a == "-i" || a == "--input")      inputPath = next("-i");
        else if (a == "-o" || a == "--output")    outputPath = next("-o");
        else if (a == "-p" || a == "--pitch")     { semitones = std::stof(next("-p")); havePitch = true; }
        else if (a == "-m" || a == "--models")     modelsDir = next("-m");
        else { std::cerr << "error: unknown argument '" << a << "'\n"; return 2; }
    }

    if (inputPath.empty()) { std::cerr << "error: --input is required\n"; printHelp(); return 2; }
    if (outputPath.empty()) { std::cerr << "error: --output is required\n"; printHelp(); return 2; }
    if (!havePitch)         { std::cerr << "error: --pitch is required\n"; printHelp(); return 2; }

#ifndef HIFISHIFTER_USE_TENSORRT
    std::cout << "[info] TensorRT EP disabled (built without HIFISHIFTER_USE_TENSORRT)\n";
#endif

    // ---------------------------------------------------------------
    // 1. Read input WAV
    // ---------------------------------------------------------------
    auto t0 = std::chrono::high_resolution_clock::now();
    wav_io::AudioData audio;
    std::string err;
    if (!wav_io::readWav(inputPath, audio, err))
    {
        std::cerr << "error: " << err << "\n";
        return 1;
    }
    std::cout << "[info] loaded " << inputPath
              << " (" << audio.sampleRate << " Hz, " << audio.numChannels << " ch, "
              << double(audio.samples.size()) / audio.numChannels / audio.sampleRate
              << " s)\n";

    std::vector<float> mono = wav_io::toMono(audio);
    const int srcRate = audio.sampleRate;

    // ---------------------------------------------------------------
    // 2. Resample to 44100 Hz (vocoder native rate)
    //    Linear interpolation, matching HachiTune's resampler behavior.
    // ---------------------------------------------------------------
    std::vector<float> audio44k;
    const int vocoderRate = Vocoder::SAMPLE_RATE;
    if (srcRate == vocoderRate)
    {
        audio44k = std::move(mono);
    }
    else
    {
        const double ratio = double(vocoderRate) / srcRate;
        const int outSamples = int(mono.size() * ratio);
        audio44k.resize(outSamples);
        for (int i = 0; i < outSamples; ++i)
        {
            double srcPos = i / ratio;
            int srcIdx = int(srcPos);
            double frac = srcPos - srcIdx;
            if (srcIdx + 1 < int(mono.size()))
                audio44k[i] = float(mono[srcIdx] * (1.0 - frac) + mono[srcIdx + 1] * frac);
            else if (srcIdx < int(mono.size()))
                audio44k[i] = mono[srcIdx];
        }
    }
    std::cout << "[info] resampled to " << vocoderRate << " Hz (" << audio44k.size() << " samples)\n";

    // ---------------------------------------------------------------
    // 3. Compute vocoder mel spectrogram (Slaney, 44.1kHz)
    // ---------------------------------------------------------------
    MelSpectrogram melComputer(vocoderRate, 2048, Vocoder::HOP_SIZE,
                               Vocoder::NUM_MELS, 40.0f, 16000.0f);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto mel = melComputer.compute(audio44k.data(), int(audio44k.size()));
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "[info] mel spectrogram: " << mel.size() << " frames ("
              << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms)\n";

    // ---------------------------------------------------------------
    // 4. Extract F0 with FCPE (16kHz internally)
    // ---------------------------------------------------------------
    FCPEPitchDetector fcpe;
    if (!fcpe.loadModel(joinPath(modelsDir, "fcpe.onnx"),
                        joinPath(modelsDir, "mel_filterbank.bin"),
                        joinPath(modelsDir, "cent_table.bin")))
    {
        std::cerr << "error: failed to load FCPE model\n";
        return 1;
    }
    t1 = std::chrono::high_resolution_clock::now();
    auto fcpeF0 = fcpe.extractF0(audio44k.data(), int(audio44k.size()), vocoderRate);
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "[info] FCPE F0: " << fcpeF0.size() << " frames ("
              << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms)\n";

    // ---------------------------------------------------------------
    // 5. Resample F0 to vocoder frame rate (log-domain interpolation).
    //    FCPE frame time = 160/16000 = 0.01s; vocoder frame time = 512/44100.
    // ---------------------------------------------------------------
    const int targetFrames = int(mel.size());
    std::vector<float> f0(targetFrames, 0.0f);
    if (!fcpeF0.empty())
    {
        const double fcpeFrameTime = 160.0 / 16000.0;            // 0.01s
        const double vocoderFrameTime = double(Vocoder::HOP_SIZE) / vocoderRate;
        for (int i = 0; i < targetFrames; ++i)
        {
            double vocoderTime = i * vocoderFrameTime;
            double pos = vocoderTime / fcpeFrameTime;
            int srcIdx = int(pos);
            double frac = pos - srcIdx;
            if (srcIdx + 1 < int(fcpeF0.size()))
            {
                float a = fcpeF0[srcIdx], b = fcpeF0[srcIdx + 1];
                if (a > 0.0f && b > 0.0f)
                    f0[i] = std::exp(std::log(a) * (1.0 - frac) + std::log(b) * frac);
                else if (a > 0.0f) f0[i] = a;
                else if (b > 0.0f) f0[i] = b;
                else f0[i] = 0.0f;
            }
            else if (srcIdx < int(fcpeF0.size()))
                f0[i] = fcpeF0[srcIdx];
            else
                f0[i] = fcpeF0.back() > 0.0f ? fcpeF0.back() : 0.0f;
        }
    }

    // ---------------------------------------------------------------
    // 6. Pitch-shift F0: freq *= 2^(semitones/12), voiced frames only.
    // ---------------------------------------------------------------
    if (semitones != 0.0f)
    {
        const float ratio = std::pow(2.0f, semitones / 12.0f);
        for (auto &freq : f0)
            if (freq > 0.0f) freq *= ratio;
    }
    std::cout << "[info] pitch shift applied: " << semitones << " semitones\n";

    // ---------------------------------------------------------------
    // 7. Vocoder inference (PC-NSF-HiFiGAN)
    // ---------------------------------------------------------------
    Vocoder vocoder;
    if (!vocoder.loadModel(joinPath(modelsDir, "pc_nsf_hifigan.onnx")))
    {
        std::cerr << "error: failed to load vocoder model\n";
        return 1;
    }
    t1 = std::chrono::high_resolution_clock::now();
    auto waveform = vocoder.infer(mel, f0);
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "[info] vocoder inference: " << waveform.size() << " samples ("
              << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms)\n";

    if (waveform.empty())
    {
        std::cerr << "error: vocoder produced no output\n";
        return 1;
    }

    // ---------------------------------------------------------------
    // 8. Write output WAV (32-bit float, mono)
    // ---------------------------------------------------------------
    if (!wav_io::writeWavF32Mono(outputPath, waveform.data(), waveform.size(),
                                 vocoderRate, err))
    {
        std::cerr << "error: " << err << "\n";
        return 1;
    }
    auto tEnd = std::chrono::high_resolution_clock::now();
    std::cout << "[info] wrote " << outputPath << " (" << vocoderRate << " Hz, "
              << double(waveform.size()) / vocoderRate << " s, total "
              << std::chrono::duration<double, std::milli>(tEnd - t0).count() << " ms)\n";
    return 0;
}
