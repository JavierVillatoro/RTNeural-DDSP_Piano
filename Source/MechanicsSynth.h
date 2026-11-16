#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#include "LevelTables.h"
#include "MechanicsTables.h"
#include "VoiceShaping.h"

namespace ddsp
{
// ==============================================================================
// Mecanica del piano sintetizada (sin muestras): rafagas de ruido de baja frecuencia con caida exponencial, con los niveles
// relativos a la nota medidos en Salamander V3 (ver Tools/make_mechanics_tables.py). Cuatro tipos:
//   Attack      golpe grave del martillo al pulsar (30-300 Hz; en grabaciones reales el ataque tiene 8-14 dB mas energia grave
//               que el DDSP), nivel ~ nota * velocity
//   Release     golpe del apagador y la tecla al soltar (casi todo < 500 Hz; -20 dB a ~140 ms)
//   PedalDown / PedalUp   golpes del pedal (y un rumor grave largo al pisar)
// 'amount' (0..2) escala todo; 0 = apagado. Salida estereo (ruido independiente por canal, reparto por registro), que se suma
// al audio del motor ANTES de la reverb.
// ==============================================================================
class MechanicsSynth
{
public:
    // Nivel del golpe de ataque respecto a la nota (dB). Calibrado con la metrica de ataque (10-80 ms, energia 30-200 Hz frente a
    // 300-4000 Hz) sobre notas aisladas de MAESTRO 2009: con 'amount' = 1 el DDSP pasa de -23,9 a -14,2 dB en MIDI 60-71 (real
    // -15,4) y de -26,4 a -19,6 en 72-83 (real -17,1). En 84-95 apenas mejora (-25,5 -> -22,6 frente a -11): esas notas son muy
    // bajas y el golpe se escala con ellas. Parte del hueco real puede ser sala/pedal y no golpe de martillo.
    static constexpr float kAttackOffsetDb = -6.0f;

    void prepare (double sampleRate)
    {
        sr = sampleRate;
        for (int k = 0; k < kKinds; ++k)
        {
            const auto& p = kParams[k];
            const double w0 = 2.0 * 3.14159265358979 * p.fc / sr, cw = std::cos (w0), al = std::sin (w0) / (2.0 * 0.70710678);
            const double a0 = 1.0 + al;
            lp[k] = { (float) ((1.0 - cw) * 0.5 / a0), (float) ((1.0 - cw) / a0), (float) ((1.0 - cw) * 0.5 / a0), (float) (-2.0 * cw / a0), (float) ((1.0 - al) / a0) };
            // ganancia de normalizacion: ruido uniforme de RMS 1 -> RMS 1 tras el filtro (se mide, no se calcula)
            uint32_t r = 0x9E3779B9u; float z1 = 0, z2 = 0; double acc = 0; const int N = (int) sr;
            for (int i = 0; i < N; ++i)
            {
                const float x = noise (r), y = lp[k].b0 * x + z1;
                z1 = lp[k].b1 * x - lp[k].a1 * y + z2; z2 = lp[k].b2 * x - lp[k].a2 * y;
                if (i > N / 8) acc += (double) y * y;
            }
            norm[k] = (float) (1.0 / std::sqrt (acc / (N - N / 8) + 1e-30));
            const double tau = p.tau;
            msFactor[k] = (float) (tau / 0.2 * (1.0 - std::exp (-0.2 / tau)));      // media de env^2 en los primeros 100 ms
            decay[k] = (float) std::exp (-1.0 / (tau * sr));
            rampCoef[k] = (float) (1.0 - std::exp (-1.0 / (p.ramp * sr)));
        }
        reset();
    }

    void reset() { for (auto& b : bursts) b.active = false; }
    void setAmount (float a) noexcept { amount = a; }
    void setModel (int m) noexcept { model = m == 0 ? 0 : 1; }

    void noteOn (int note, float vel) noexcept
    {
        if (amount <= 0.f || note < 21 || note > 108) return;
        const float* lv = model == 0 ? kLevelDbV6 : kLevelDbV8;
        const float db = keyLevel (note) + (levelAtVelocity (lv, vel) - levelAtVelocity (lv, 0.5f)) + kAttackOffsetDb;
        spawn (Attack, db, note);
    }

    void noteOff (int note) noexcept
    {
        if (amount <= 0.f || note < 21 || note > 108) return;
        spawn (Release, keyLevel (note) + kReleaseRatioDb[note - 21], note);
    }

    void pedal (bool down) noexcept
    {
        if (amount <= 0.f) return;
        const float ref = keyLevel (60);
        if (down) { spawn (PedalDown, ref + kPedalDownRatioDb, 64); spawn (PedalRumble, ref + kPedalDownRatioDb - 14.0f, 64); }
        else      spawn (PedalUp, ref + kPedalUpRatioDb, 64);
    }

    /** Suma 'n' muestras en L y R (ya existentes). */
    void render (float* L, float* R, int n) noexcept
    {
        for (auto& b : bursts)
        {
            if (! b.active) continue;
            const auto& f = lp[b.kind]; const float dec = decay[b.kind], rc = rampCoef[b.kind];
            for (int i = 0; i < n; ++i)
            {
                const float e = b.env * b.ramp;
                const float xl = noise (b.rng), xr = noise (b.rng);
                const float yl = f.b0 * xl + b.zl1, yr = f.b0 * xr + b.zr1;
                b.zl1 = f.b1 * xl - f.a1 * yl + b.zl2; b.zl2 = f.b2 * xl - f.a2 * yl;
                b.zr1 = f.b1 * xr - f.a1 * yr + b.zr2; b.zr2 = f.b2 * xr - f.a2 * yr;
                L[i] += b.amp * b.gl * e * yl;
                R[i] += b.amp * b.gr * e * yr;
                b.env *= dec; b.ramp += (1.0f - b.ramp) * rc;
                if (b.env < 1.0e-5f) { b.active = false; break; }
            }
        }
    }

private:
    enum Kind { Attack = 0, Release, PedalDown, PedalUp, PedalRumble, kKinds };
    struct Params { double fc, tau, ramp; };
    static constexpr Params kParams[kKinds] = {
        { 160.0, 0.035, 0.002 },   // Attack: knock calibrado en el proyecto Physical_modeling (f0 ~110 Hz, tau 35 ms)
        { 200.0, 0.085, 0.003 },   // Release: -20 dB a ~195 ms y -40 dB a ~390 ms (Salamander rel*.wav: 170-230 ms y 410-710 ms)
        { 200.0, 0.080, 0.004 },   // PedalDown
        { 200.0, 0.080, 0.004 },   // PedalUp
        { 100.0, 1.000, 0.020 },   // rumor grave largo al pisar el pedal
    };
    struct Biquad { float b0, b1, b2, a1, a2; };
    struct Burst
    {
        bool active = false; int kind = 0; float amp = 0, env = 0, ramp = 0, gl = 1, gr = 1;
        float zl1 = 0, zl2 = 0, zr1 = 0, zr2 = 0; uint32_t rng = 1; uint64_t order = 0;
    };

    static float noise (uint32_t& s) noexcept      // ruido uniforme de RMS 1
    {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        return ((float) (s >> 8) * (2.0f / 16777216.0f) - 1.0f) * 1.7320508f;
    }

    float keyLevel (int note) const noexcept { return (model == 0 ? kKeyLevelDbV6 : kKeyLevelDbV8)[note - 21]; }

    void spawn (int kind, float levelDb, int note) noexcept
    {
        int idx = -1; uint64_t oldest = UINT64_MAX;
        for (int i = 0; i < kMax; ++i)
        {
            if (! bursts[(size_t) i].active) { idx = i; break; }
            if (bursts[(size_t) i].order < oldest) { oldest = bursts[(size_t) i].order; idx = i; }
        }
        auto& b = bursts[(size_t) idx];
        b.active = true; b.kind = kind; b.order = ++counter;
        b.amp = amount * std::pow (10.0f, levelDb / 20.0f) / std::sqrt (msFactor[kind]) * norm[kind];
        b.env = 1.0f; b.ramp = 0.0f; b.zl1 = b.zl2 = b.zr1 = b.zr2 = 0.0f;
        b.rng = 0x1234567u + (uint32_t) (counter * 2654435761u);
        const float p = std::clamp (0.7f * ((float) note - 64.5f) / 43.5f, -1.0f, 1.0f);   // mismo reparto por registro que el motor
        const float th = (p + 1.0f) * 0.78539816f;
        b.gl = 1.41421356f * std::cos (th); b.gr = 1.41421356f * std::sin (th);
    }

    static constexpr int kMax = 24;
    double sr = 44100.0;
    float amount = 0.f;
    int model = 1;
    uint64_t counter = 0;
    std::array<Burst, kMax> bursts;
    Biquad lp[kKinds] {};
    float norm[kKinds] {}, msFactor[kKinds] {}, decay[kKinds] {}, rampCoef[kKinds] {};
};
}
