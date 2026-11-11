#pragma once
#include <algorithm>
#include <cmath>

namespace ddsp
{
// ==============================================================================
// Curva de velocity: traduce la velocity MIDI que manda el teclado (0..1) a la velocity
// que recibe el modelo. Cuatro pasos, en este orden:
//
//   1. Rango de entrada (velMax): la velocity con la que TU teclado llega a "fortissimo".
//      Muchos teclados pequenos no pasan de ~100-110; sin esto, la parte alta del recorrido
//      de la tecla no se usa nunca.
//   2. Touch (-1 suave ... +1 duro): potencia x^gamma, gamma = 2.5^touch. Duro exige mas
//      fuerza para sonar fuerte; suave llega a forte con poca fuerza.
//   3. Dynamics (0..1.5): ensancha o estrecha el rango alrededor del punto medio.
//      0 = todas las notas con la misma velocity; 1 = sin cambio; >1 = mas contraste.
//   4. Compensacion del modelo (opcional): el modelo "aplana" la zona alta (de velocity 100
//      a 127 solo gana ~3 dB, frente a ~4,5 dB por cada 13 pasos en el centro). Con una tabla del
//      nivel medido de cada modelo se invierte esa respuesta para que cada paso de velocity
//      de entrada sume los mismos dB.
// ==============================================================================
struct VelocityCurve
{
    static constexpr int kPoints = 10;   // la tabla de nivel tiene un punto en velocity 0.1, 0.2 ... 1.0

    float velMax = 1.0f;                 // 0..1
    float touch = 0.0f;                  // -1..+1
    float dynamics = 1.0f;               // 0..1.5
    const float* levelDb = nullptr;      // nivel (dB) medido del modelo; nullptr = sin compensacion

    static float gammaFromTouch (float t) noexcept { return std::pow (2.5f, t); }

    float apply (float in) const noexcept
    {
        float x = std::clamp (in / std::max (velMax, 0.05f), 0.0f, 1.0f);
        x = std::pow (x, gammaFromTouch (touch));
        x = std::clamp (0.5f + (x - 0.5f) * dynamics, 0.0f, 1.0f);
        if (levelDb != nullptr)
            x = linearise (x);
        return std::clamp (x, 1.0f / 127.0f, 1.0f);
    }

private:
    // Busca la velocity del modelo cuyo nivel cae en una recta de dB entre el nivel de velocity 0.1 y el de 1.0.
    float linearise (float x) const noexcept
    {
        if (x <= 0.1f)
            return x;
        const float lo = levelDb[0], hi = levelDb[kPoints - 1];
        const float target = lo + (hi - lo) * (x - 0.1f) / 0.9f;
        for (int i = 0; i < kPoints - 1; ++i)
        {
            if (target <= levelDb[i + 1])
            {
                const float t = (target - levelDb[i]) / std::max (levelDb[i + 1] - levelDb[i], 1.0e-3f);
                return 0.1f * (float) (i + 1) + 0.1f * std::clamp (t, 0.0f, 1.0f);
            }
        }
        return 1.0f;
    }
};
}
