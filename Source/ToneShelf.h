#pragma once
#include <cmath>

namespace ddsp
{
// Estanteria de agudos (RBJ, S = 1) en estereo. Tone -1..+1 -> -6..+6 dB por encima de ~3 kHz.
// Sin asignaciones de memoria: se puede recalcular en el hilo de audio.
class ToneShelf
{
public:
    void prepare (double sampleRate) { sr = sampleRate; reset(); setToneDb (0.0f); }
    void reset() { for (auto& c : z) c[0] = c[1] = 0.0f; }

    void setToneDb (float dB)
    {
        if (dB == lastDb) return;
        lastDb = dB;
        const double A = std::pow (10.0, (double) dB / 40.0);
        const double w0 = 2.0 * 3.14159265358979323846 * 3000.0 / sr;
        const double cw = std::cos (w0), alpha = std::sin (w0) * 0.5 * std::sqrt (2.0), sA = std::sqrt (A);
        const double a0 = (A + 1) - (A - 1) * cw + 2 * sA * alpha;
        b0 = (float) (A * ((A + 1) + (A - 1) * cw + 2 * sA * alpha) / a0);
        b1 = (float) (-2 * A * ((A - 1) + (A + 1) * cw) / a0);
        b2 = (float) (A * ((A + 1) + (A - 1) * cw - 2 * sA * alpha) / a0);
        a1 = (float) (2 * ((A - 1) - (A + 1) * cw) / a0);
        a2 = (float) (((A + 1) - (A - 1) * cw - 2 * sA * alpha) / a0);
    }

    void process (float* data, int n, int channel) noexcept
    {
        if (lastDb == 0.0f) return;
        float z1 = z[channel][0], z2 = z[channel][1];
        for (int i = 0; i < n; ++i)
        {
            const float x = data[i];
            const float y = b0 * x + z1;
            z1 = b1 * x - a1 * y + z2;
            z2 = b2 * x - a2 * y;
            data[i] = y;
        }
        z[channel][0] = z1; z[channel][1] = z2;
    }

private:
    double sr = 44100.0;
    float lastDb = 1.0e9f;
    float b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0;
    float z[2][2] {};
};
}
