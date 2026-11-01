#pragma once

// ==============================================================================
// DDSPEngine: motor de sintesis DDSP-Piano en tiempo real (sin dependencia de JUCE).
//
//  - Reproduce EXACTAMENTE la red entrenada en training/training_kaggle_*.py con RTNeural:
//      context : Dense(3,32)+LeakyReLU -> GRU(64) -> Dense(32)
//      core    : Dense(34,128)+LeakyReLU -> GRU(192) -> Dense(192)+LeakyReLU -> Dense(161)
//                (161 = amp 1 + armonicos 96 + filtros de ruido 64)
//  - La red corre a 250 Hz (un "frame neuronal" cada sr/250 muestras, con acumulador fraccional);
//    entre frames se interpola linealmente, como el upsampling bilineal del entrenamiento.
//  - Voces: pool de kPool voces que comparten pesos; cada una tiene su estado GRU.
//  - Semantica de notas (viene en el .bin, 'meta.semantics_version'):
//      v1 (training_kaggle_6): al soltar la tecla velocity=0 y el pitch persiste release_sec.
//      v2 (training_kaggle_7): igual, pero si el pedal esta pisado la nota sigue activa
//                              (velocity se mantiene) hasta levantar el pedal.
// ==============================================================================

#include <RTNeural/RTNeural.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ddsp
{
constexpr int kPool       = 28;   // voces fisicas (incluye las que se estan apagando)
constexpr int kHarm       = 96;
constexpr int kNoise      = 64;
constexpr int kCoreIn     = 34;   // pitch/127, velocity, contexto(32)
constexpr int kCoreOut    = 1 + kHarm + kNoise;
constexpr int kCtxIn      = 3;    // pedal, velocity global, pitch medio
constexpr int kCtxOut     = 32;
constexpr int kStrings    = 2;
constexpr int kPartials   = kStrings * kHarm;
constexpr int kModelRate  = 16000;
constexpr int kFrameRate  = 250;
constexpr int kHop        = kModelRate / kFrameRate;   // 64
constexpr int kWin        = 2 * kHop;                  // 128
constexpr int kRing       = 256;
constexpr int kTicks      = 4;     // el trabajo de las redes de cada frame se reparte en 4 trozos (voces escalonadas)

using ContextNet = RTNeural::ModelT<float, kCtxIn, kCtxOut,
    RTNeural::DenseT<float, kCtxIn, 32>,
    RTNeural::PReLUActivationT<float, 32>,
    RTNeural::GRULayerT<float, 32, 64>,
    RTNeural::DenseT<float, 64, kCtxOut>>;

using CoreNet = RTNeural::ModelT<float, kCoreIn, kCoreOut,
    RTNeural::DenseT<float, kCoreIn, 128>,
    RTNeural::PReLUActivationT<float, 128>,
    RTNeural::GRULayerT<float, 128, 192>,
    RTNeural::DenseT<float, 192, 192>,
    RTNeural::PReLUActivationT<float, 192>,
    RTNeural::DenseT<float, 192, kCoreOut>>;

struct Tensor
{
    std::vector<int> dims;
    std::vector<float> data;
};


/** Red "core" evaluada para TODAS las voces activas a la vez (los pesos se leen una sola vez por frame).
    Misma red y mismos pesos que CoreNet (RTNeural), pero mucho mas rapida con muchas voces porque la GRU
    deja de estar limitada por el ancho de banda de memoria. La prueba de oro compara las dos con Keras. */
class FastCore
{
public:
    static constexpr int In = kCoreIn, H1 = 128, U = 192, H2 = 192, Out = kCoreOut;
    static constexpr int kSlots = kPool + 1;   // el ultimo slot es la voz de reposo
    static constexpr int kIdle = kPool;

    bool load (const std::map<std::string, Tensor>& t, float leakyAlpha, std::string& err);
    void resetAll();
    void copyState (int dst, int src);
    /** slots[i] = indice de voz de la fila i; in: n*34 floats; out: n*161 floats */
    void forward (const int* slots, int n, const float* in, float* out);

private:
    std::vector<float> w1, b1, wx, bi, uh, br, w2, b2, w3, b3;
    float alpha = 0.2f;
    std::vector<float> state, h1, g, hh, hp, h2, zb, rb, cb;
};

class Engine
{
public:
    Engine();
    ~Engine();

    /** Carga el .bin generado por training/export_binary.py. */
    bool loadWeights (const void* data, size_t size, std::string* error = nullptr);
    bool isLoaded() const noexcept { return loaded; }

    void prepare (double sampleRate);
    void reset();

    void setPolyphonyLimit (int n) noexcept;
    void setMaxHoldSeconds (float s) noexcept { maxHoldSec = s; }
    /** Nivel global del ruido filtrado (1 = el del modelo) y reduccion extra en los extremos del teclado (0..1). */
    void setNoise (float gain, float extremes) noexcept { noiseGain = gain; noiseExtremes = extremes; }
    /** Voces "inactivas" que aportan ruido de fondo (solo modelos v1, que lo aprendieron asi). */
    void setIdleNoiseVoices (int n) noexcept { idleNoiseVoices = n; }
    int  getTrainedVoices() const noexcept { return trainedVoices; }

    void noteOn  (int note, float velocity01);
    void noteOff (int note);
    void setSustain (float value01);
    void allNotesOff();

    /** Escribe 'n' muestras MONO (sin reverb) en out. */
    void render (float* out, int n);

    int  numActiveVoices() const noexcept;
    bool isSustainDown() const noexcept { return pedalDown; }
    int  semanticsVersion() const noexcept { return semantics; }
    double getSampleRate() const noexcept { return sr; }

    // Profiling (segundos acumulados): red de contexto+voces, ruido FFT, osciladores
    double profNet = 0, profNoise = 0, profSynth = 0;

    // -- Solo para tests (prueba de oro): ejecuta las redes sobre secuencias dadas ---
    void runContextSequence (const float* in, float* out, int frames);
    void runCoreSequence (const float* in, float* out, int frames);      // ruta rapida (la del plugin)
    void runCoreSequenceRT (const float* in, float* out, int frames);    // RTNeural puro (referencia)

private:
    enum class State : uint8_t { Free, Held, Sustained, Releasing, Fading };

    struct Voice
    {
        State state = State::Free;
        bool  retire = false;      // en el proximo segmento baja a 0 y pasa a Free
        int   note = 0;
        float vel = 0.f;
        int   fadeLeft = 0;
        int   framesOnset = 0;     // frames desde el onset
        int   framesState = 0;     // frames desde que entro en Sustained/Releasing
        float velIn = 0.f;         // ultima velocity de entrada (la usa la red de contexto)
        bool  retireTicked = false;
        float noisePow[kNoise] {}; // potencia de ruido de su ultimo frame
        uint64_t order = 0;
        int   nAct = 0;            // armonicos por debajo de 8 kHz
        int   nPart = 0;           // nAct * n_strings
        int   nPartPad = 0;        // redondeado a multiplo de 4
        alignas (32) float in[kCoreIn] {};
        alignas (32) float cr[kPartials] {}, ci[kPartials] {}, rc[kPartials] {}, rs[kPartials] {};
        alignas (32) float g[kPartials] {}, dg[kPartials] {};
        alignas (32) float target[kPartials] {};
        alignas (32) float mask[kPartials] {};   // 0 si la parcial supera la frecuencia de corte
    };

    // -- pesos / meta
    bool loaded = false;
    std::map<std::string, Tensor> tensors;
    bool loadNetworks (std::string* error);
    float metaValue (const char* name, float def) const;

    // -- redes
    std::unique_ptr<ContextNet> context;
    FastCore fast;                        // red de las voces (todas a la vez); slot kIdle = voz de reposo (pitch 0, vel 0)
    std::unique_ptr<CoreNet> refCore;     // misma red con RTNeural: referencia para la prueba de oro
    int idleNoiseVoices = 0;
    int trainedVoices = 8;
    alignas (32) float ctxIn[kCtxIn] {};
    alignas (32) float ctxOut[kCtxOut] {};

    // -- fisica
    float alphaB = 0, betaB = 0, alphaT = 0, betaT = 0;
    float noiseGain = 1.f, noiseExtremes = 0.f;
    int   semantics = 2;
    int   nStrings = 1;
    float stringGain = 1.f;
    float detuneCents[kStrings] { 0.f, 0.f };
    float releaseSec = 1.f, maxSustainSec = 6.f, pedalThreshold = 0.5f;
    int   fadeFrames = 3;
    float maxHoldSec = 10.f;
    float cutoffHz = 8000.f;

    // -- estado de ejecucion
    double sr = 44100.0;
    double frameLen = 44100.0 / kFrameRate;
    int64_t frameIndex = 0;       // numero de frame neuronal
    int   segLeft = 0;            // muestras que quedan del segmento actual
    int   segLen = 0;
    int   polyLimit = kPool - 4;
    uint64_t orderCounter = 0;
    float pedalValue = 0.f;
    bool  pedalDown = false;
    std::array<Voice, kPool> voices;

    // -- ruido (16 kHz, OLA) compartido por todas las voces
    float ring[kRing] {};
    double noiseRead = 0.0;
    double noiseInc = 0.0;
    uint32_t rng = 0x12345678u;

    // FFT de 128 puntos
    float fftCos[kWin / 2] {}, fftSin[kWin / 2] {};
    float hann[kWin] {};
    void fft (float* re, float* im, bool inverse) const;
    float uniform() noexcept;

    // -- helpers
    float noiseScale (int note) const noexcept;
    int  findFreeVoice() const;
    int  allocateVoice();
    void startVoice (int idx, int note, float vel);
    void setupPartials (Voice& v);
    void neuralTick();
    int64_t tickBoundary (int64_t k) const noexcept;
    int64_t tickCount = 0;
    void synthSegment (float* out, int n);
    void retireVoice (Voice& v);
    float noiseSample (double pos) const noexcept;
    void zeroState();
};

} // namespace ddsp
