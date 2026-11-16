#pragma once

namespace ddsp
{
// Nivel RMS medio (dB, 0,5 s tras el ataque; notas 36/48/60/72/84) de cada modelo para velocity 0.1, 0.2 ... 1.0.
// Medido con ddsp_render. Lo usan la compensacion de la curva de velocity (VelocityCurve.h) y Hardness (VoiceShaping.h).
// Si se reentrena o se anade un modelo hay que volver a medirlo.
constexpr int kLevelPoints = 10;
inline constexpr float kLevelDbV6[kLevelPoints] = { -76.3f, -70.5f, -65.0f, -59.8f, -54.9f, -50.4f, -46.7f, -43.6f, -40.8f, -38.1f };
inline constexpr float kLevelDbV8[kLevelPoints] = { -62.3f, -57.3f, -52.8f, -48.4f, -44.2f, -40.3f, -36.9f, -34.4f, -32.7f, -31.5f };
}
