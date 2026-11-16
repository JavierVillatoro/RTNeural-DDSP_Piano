#pragma once
#include <algorithm>
#include <cmath>

#include "LevelTables.h"

namespace ddsp
{
// ==============================================================================
// Ajustes por nota que se deciden al pulsar la tecla (el motor solo recibe velocity y una ganancia lineal):
//
//  * Hardness (-1 blando ... +1 duro): la red asocia volumen y brillo a la misma velocity. Aqui se le da una velocity
//    distinta (v + 0,25*h) y se compensa el volumen con la tabla de nivel medido del modelo: la nota suena igual de
//    fuerte pero con el timbre de un golpe mas (o menos) fuerte. Idea sin validar de oido: se juzga tocando.
//  * Evenness (0..1): corrige el unico desvio de nivel por registro que se midio frente a grabaciones reales de MAESTRO
//    (2009, notas aisladas, velocity MIDI 55-75): los agudos altos (MIDI 84-95) salen ~8 dB mas bajos en el DDSP.
//    Entre 36 y 83 coinciden (+-2 dB), asi que ahi no se toca. Sin datos fiables por encima de 95 ni en graves.
// ==============================================================================
inline float levelAtVelocity (const float* tableDb, float v) noexcept
{
    const float x = std::clamp (v, 0.0f, 1.0f) * 10.0f - 1.0f;             // indice continuo: 0 = velocity 0.1
    if (x <= 0.0f)                                                          // por debajo de 0.1: se prolonga la pendiente del primer tramo
        return tableDb[0] + x * (tableDb[1] - tableDb[0]);
    const int i = std::min ((int) x, kLevelPoints - 2);
    const float t = std::min (x - (float) i, 1.0f);
    return tableDb[i] + t * (tableDb[i + 1] - tableDb[i]);
}

struct VoiceShaping
{
    float hardness = 0.0f;                 // -1..+1
    float evenness = 0.0f;                 // 0..1
    const float* levelDb = nullptr;        // tabla de nivel del modelo (kLevelDbV6 / kLevelDbV8)

    static float trebleLiftDb (int note) noexcept
    {
        const float t = std::clamp (((float) note - 80.0f) / 10.0f, 0.0f, 1.0f);   // 0 hasta MIDI 80, +8 dB desde MIDI 90
        return 8.0f * t * t * (3.0f - 2.0f * t);
    }

    void apply (int note, float vel, float& velNet, float& gainLin) const noexcept
    {
        velNet = vel;
        float gainDb = 0.0f;
        if (hardness != 0.0f && levelDb != nullptr && vel > 0.0f)
        {
            velNet = std::clamp (vel + 0.25f * hardness, 0.02f, 1.0f);
            gainDb += std::clamp (levelAtVelocity (levelDb, vel) - levelAtVelocity (levelDb, velNet), -12.0f, 12.0f);
        }
        gainDb += evenness * trebleLiftDb (note);
        gainLin = gainDb == 0.0f ? 1.0f : std::pow (10.0f, gainDb / 20.0f);
    }
};
}
