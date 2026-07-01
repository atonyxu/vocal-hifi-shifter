#pragma once

// Minimal WAV reader/writer (PCM 8/16/24/32-bit and 32-bit float, mono/stereo)
// Stereo input is downmixed to mono. Self-contained, no third-party deps.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace wav_io
{

struct AudioData
{
    std::vector<float> samples;   // interleaved if multi-channel
    int sampleRate = 0;
    int numChannels = 0;
};

// ------------------------------------------------------------------
// Little-endian helpers
// ------------------------------------------------------------------
static inline uint16_t rd16(const uint8_t *p)
{
    return uint16_t(p[0]) | (uint16_t(p[1]) << 8);
}
static inline uint32_t rd32(const uint8_t *p)
{
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) |
           (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
static inline void wr16(uint8_t *p, uint16_t v)
{
    p[0] = uint8_t(v & 0xFF); p[1] = uint8_t((v >> 8) & 0xFF);
}
static inline void wr32(uint8_t *p, uint32_t v)
{
    p[0] = uint8_t(v & 0xFF);         p[1] = uint8_t((v >> 8) & 0xFF);
    p[2] = uint8_t((v >> 16) & 0xFF); p[3] = uint8_t((v >> 24) & 0xFF);
}

// Read an entire WAV file into AudioData.
inline bool readWav(const std::string &path, AudioData &out, std::string &err)
{
    FILE *f = nullptr;
#ifdef _WIN32
    fopen_s(&f, path.c_str(), "rb");
#else
    f = fopen(path.c_str(), "rb");
#endif
    if (!f) { err = "cannot open file: " + path; return false; }

    std::vector<uint8_t> buf;
    {
        std::fseek(f, 0, SEEK_END);
        long sz = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        if (sz <= 0) { fclose(f); err = "empty file"; return false; }
        buf.resize(size_t(sz));
        if (std::fread(buf.data(), 1, buf.size(), f) != buf.size())
        { fclose(f); err = "read error"; return false; }
    }
    fclose(f);

    const uint8_t *p   = buf.data();
    const uint8_t *end = p + buf.size();
    if (p + 12 > end || std::memcmp(p, "RIFF", 4) != 0)
    { err = "not a RIFF/WAVE file"; return false; }
    p += 4; // "RIFF"
    p += 4; // overall size (ignored)
    if (std::memcmp(p, "WAVE", 4) != 0) { err = "not a WAVE file"; return false; }
    p += 4;

    uint16_t audioFormat = 1, numChannels = 0, bitsPerSample = 0;
    uint32_t sampleRate = 0;
    const uint8_t *dataPtr = nullptr;
    uint32_t dataSize = 0;
    bool fmtFound = false, dataFound = false;

    while (p + 8 <= end)
    {
        char id[5] = {0};
        std::memcpy(id, p, 4);
        uint32_t chunkSize = rd32(p + 4);
        p += 8;
        const uint8_t *chunkEnd = p + chunkSize;
        if (chunkEnd > end) { err = "chunk overruns file"; return false; }

        if (std::memcmp(id, "fmt ", 4) == 0)
        {
            if (chunkSize < 16) { err = "fmt chunk too small"; return false; }
            audioFormat    = rd16(p);
            numChannels    = rd16(p + 2);
            sampleRate     = rd32(p + 4);
            bitsPerSample  = rd16(p + 14);
            fmtFound = true;
        }
        else if (std::memcmp(id, "data", 4) == 0)
        {
            dataSize = chunkSize;
            dataPtr  = p;
            dataFound = true;
        }
        p = chunkEnd;
        if (chunkSize & 1) ++p; // word-align
    }

    if (!fmtFound) { err = "missing fmt chunk"; return false; }
    if (!dataFound || dataPtr == nullptr) { err = "missing data chunk"; return false; }
    if (numChannels == 0) { err = "invalid channel count"; return false; }
    if (sampleRate == 0) { err = "invalid sample rate"; return false; }

    const int bytesPerSample = (bitsPerSample + 7) / 8;
    if (bytesPerSample == 0) { err = "invalid bits per sample"; return false; }
    const size_t totalFrames = dataSize / (bytesPerSample * numChannels);

    out.sampleRate  = static_cast<int>(sampleRate);
    out.numChannels = numChannels;
    out.samples.resize(totalFrames * numChannels);

    if (audioFormat == 3 && bitsPerSample == 32)
    {
        for (size_t i = 0; i < out.samples.size(); ++i)
        {
            uint32_t v = rd32(dataPtr + i * 4);
            float fv; std::memcpy(&fv, &v, 4);
            out.samples[i] = fv;
        }
    }
    else if (audioFormat == 1 || audioFormat == 0xFFFE) // PCM or EXTENSIBLE
    {
        const uint8_t *q = dataPtr;
        for (size_t i = 0; i < out.samples.size(); ++i)
        {
            if (bitsPerSample == 8)
            {
                out.samples[i] = (int(q[i]) - 128) / 128.0f;
                q += 1;
            }
            else if (bitsPerSample == 16)
            {
                int16_t v = int16_t(rd16(q));
                out.samples[i] = v / 32768.0f;
                q += 2;
            }
            else if (bitsPerSample == 24)
            {
                int32_t v = int32_t(uint32_t(q[0]) | (uint32_t(q[1]) << 8) | (uint32_t(q[2]) << 16));
                if (v & 0x800000) v |= ~0xFFFFFF; // sign-extend
                out.samples[i] = v / 8388608.0f;
                q += 3;
            }
            else if (bitsPerSample == 32)
            {
                int32_t v = int32_t(rd32(q));
                out.samples[i] = v / 2147483648.0f;
                q += 4;
            }
            else { err = "unsupported bit depth"; return false; }
        }
    }
    else { err = "unsupported audio format"; return false; }

    return true;
}

// Write a mono 32-bit float WAV file.
inline bool writeWavF32Mono(const std::string &path, const float *samples,
                           size_t numSamples, int sampleRate, std::string &err)
{
    FILE *f = nullptr;
#ifdef _WIN32
    fopen_s(&f, path.c_str(), "wb");
#else
    f = fopen(path.c_str(), "wb");
#endif
    if (!f) { err = "cannot open output file: " + path; return false; }

    const uint32_t dataSize = uint32_t(numSamples) * 4u;
    const uint32_t byteRate = uint32_t(sampleRate) * 1u * 4u;
    uint8_t header[44];
    std::memcpy(header, "RIFF", 4);
    wr32(header + 4, 36 + dataSize);
    std::memcpy(header + 8, "WAVE", 4);
    std::memcpy(header + 12, "fmt ", 4);
    wr32(header + 16, 16);            // PCM chunk size
    wr16(header + 20, 3);             // IEEE float
    wr16(header + 22, 1);             // mono
    wr32(header + 24, uint32_t(sampleRate));
    wr32(header + 28, byteRate);
    wr16(header + 32, 4);             // block align
    wr16(header + 34, 32);            // bits per sample
    std::memcpy(header + 36, "data", 4);
    wr32(header + 40, dataSize);

    if (std::fwrite(header, 1, 44, f) != 44) { fclose(f); err = "header write failed"; return false; }

    std::vector<uint8_t> bytes(numSamples * 4);
    for (size_t i = 0; i < numSamples; ++i)
    {
        float v = samples[i];
        uint32_t u; std::memcpy(&u, &v, 4);
        wr32(bytes.data() + i * 4, u);
    }
    if (std::fwrite(bytes.data(), 1, bytes.size(), f) != bytes.size())
    { fclose(f); err = "data write failed"; return false; }

    fclose(f);
    return true;
}

// Downmix to mono (averages channels). No-op for mono input.
inline std::vector<float> toMono(const AudioData &ad)
{
    if (ad.numChannels <= 1) return ad.samples;
    const size_t frames = ad.samples.size() / ad.numChannels;
    std::vector<float> mono(frames);
    for (size_t i = 0; i < frames; ++i)
    {
        float acc = 0.0f;
        for (int c = 0; c < ad.numChannels; ++c)
            acc += ad.samples[i * ad.numChannels + c];
        mono[i] = acc / ad.numChannels;
    }
    return mono;
}

} // namespace wav_io
