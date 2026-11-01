#include "DDSPEngine.h"

#include <algorithm>
#include <chrono>
#include <cstring>

namespace ddsp
{
namespace
{
constexpr double kPi = 3.14159265358979323846;

inline float sigmoidf (float x) noexcept { return 1.0f / (1.0f + std::exp (-x)); }

// 2 * sigmoid(x)^ln(10) + 1e-7  (get_modified_sigmoid del entrenamiento)
inline float modifiedSigmoid (float x) noexcept
{
    return 2.0f * std::pow (sigmoidf (x), 2.302585093f) + 1e-7f;
}

template <typename Layer>
bool setDense (Layer& layer, const Tensor& w, const Tensor& b, int in, int out, std::string& err, const char* name)
{
    if (w.dims.size() != 2 || w.dims[0] != in || w.dims[1] != out || (int) b.data.size() != out)
    {
        err = std::string ("Dimensiones inesperadas en ") + name;
        return false;
    }
    std::vector<std::vector<float>> ws ((size_t) out, std::vector<float> ((size_t) in));
    for (int i = 0; i < in; ++i)         // Keras: kernel[in][out]; RTNeural: [out][in]
        for (int o = 0; o < out; ++o)
            ws[(size_t) o][(size_t) i] = w.data[(size_t) i * (size_t) out + (size_t) o];
    layer.setWeights (ws);
    layer.setBias (b.data.data());
    return true;
}

template <typename Layer>
bool setGru (Layer& layer, const Tensor& w, const Tensor& u, const Tensor& b, int in, int units, std::string& err, const char* name)
{
    if (w.dims.size() != 2 || w.dims[0] != in || w.dims[1] != 3 * units
        || u.dims.size() != 2 || u.dims[0] != units || u.dims[1] != 3 * units
        || b.dims.size() != 2 || b.dims[0] != 2 || b.dims[1] != 3 * units)
    {
        err = std::string ("Dimensiones inesperadas en ") + name;
        return false;
    }
    const size_t k3 = (size_t) units * 3;
    std::vector<std::vector<float>> wv ((size_t) in, std::vector<float> (k3)),
                                    uv ((size_t) units, std::vector<float> (k3)),
                                    bv (2, std::vector<float> (k3));
    for (int i = 0; i < in; ++i)
        for (size_t k = 0; k < k3; ++k)
            wv[(size_t) i][k] = w.data[(size_t) i * k3 + k];
    for (int i = 0; i < units; ++i)
        for (size_t k = 0; k < k3; ++k)
            uv[(size_t) i][k] = u.data[(size_t) i * k3 + k];
    for (size_t r = 0; r < 2; ++r)
        for (size_t k = 0; k < k3; ++k)
            bv[r][k] = b.data[r * k3 + k];
    layer.setWVals (wv);
    layer.setUVals (uv);
    layer.setBVals (bv);
    return true;
}
} // namespace

// ==============================================================================
Engine::Engine()
{
    for (int k = 0; k < kWin / 2; ++k)
    {
        fftCos[k] = (float) std::cos (2.0 * kPi * k / kWin);
        fftSin[k] = (float) std::sin (2.0 * kPi * k / kWin);
    }
    for (int n = 0; n < kWin; ++n)
        hann[n] = (float) (0.5 - 0.5 * std::cos (2.0 * kPi * n / kWin));   // periodica, como tf.signal.hann_window
}

Engine::~Engine() = default;

// ------------------------------------------------------------------------------
// Carga de pesos
// ------------------------------------------------------------------------------
bool Engine::loadWeights (const void* data, size_t size, std::string* error)
{
    std::string err;
    loaded = false;
    tensors.clear();

    const auto* p = static_cast<const uint8_t*> (data);
    size_t pos = 0;
    auto need = [&] (size_t n) { return pos + n <= size; };
    auto rd = [&] (void* dst, size_t n) { std::memcpy (dst, p + pos, n); pos += n; };

    if (! need (12) || std::memcmp (p, "DDSW", 4) != 0)
    {
        if (error) *error = "Fichero de pesos invalido (cabecera)";
        return false;
    }
    pos = 4;
    uint32_t version = 0, count = 0;
    rd (&version, 4);
    rd (&count, 4);
    if (version != 1)
    {
        if (error) *error = "Version de formato de pesos no soportada";
        return false;
    }
    for (uint32_t t = 0; t < count; ++t)
    {
        uint16_t len = 0;
        if (! need (2)) { if (error) *error = "Fichero truncado"; return false; }
        rd (&len, 2);
        if (! need ((size_t) len + 1)) { if (error) *error = "Fichero truncado"; return false; }
        std::string name ((const char*) p + pos, len);
        pos += len;
        uint8_t nd = 0;
        rd (&nd, 1);
        Tensor tn;
        size_t total = 1;
        for (int d = 0; d < nd; ++d)
        {
            uint32_t dim = 0;
            if (! need (4)) { if (error) *error = "Fichero truncado"; return false; }
            rd (&dim, 4);
            tn.dims.push_back ((int) dim);
            total *= dim;
        }
        if (! need (total * 4)) { if (error) *error = "Fichero truncado"; return false; }
        tn.data.resize (total);
        rd (tn.data.data(), total * 4);
        tensors[name] = std::move (tn);
    }

    if (! loadNetworks (&err))
    {
        if (error) *error = err;
        return false;
    }
    loaded = true;
    return true;
}

float Engine::metaValue (const char* name, float def) const
{
    auto it = tensors.find (std::string ("meta.") + name);
    return (it == tensors.end() || it->second.data.empty()) ? def : it->second.data[0];
}

bool Engine::loadNetworks (std::string* error)
{
    std::string err;
    auto get = [&] (const char* n) -> const Tensor* {
        auto it = tensors.find (n);
        if (it == tensors.end()) { err = std::string ("Falta el tensor ") + n; return nullptr; }
        return &it->second;
    };

    const char* required[] = { "ctx.dense_in.w", "ctx.dense_in.b", "ctx.gru.w", "ctx.gru.u", "ctx.gru.b",
                               "ctx.dense_out.w", "ctx.dense_out.b", "core.dense_in.w", "core.dense_in.b",
                               "core.gru.w", "core.gru.u", "core.gru.b", "core.dense_hidden.w",
                               "core.dense_hidden.b", "core.heads.w", "core.heads.b", "phys.inharm" };
    for (auto* n : required)
        if (get (n) == nullptr) { if (error) *error = err; return false; }

    const float alpha = metaValue ("leaky_alpha", 0.2f);
    const std::vector<float> alphaV32 (32, alpha), alphaV128 (128, alpha), alphaV192 (192, alpha);

    context = std::make_unique<ContextNet>();
    bool ok = setDense (context->get<0>(), *get ("ctx.dense_in.w"), *get ("ctx.dense_in.b"), 3, 32, err, "ctx.dense_in")
              && setGru (context->get<2>(), *get ("ctx.gru.w"), *get ("ctx.gru.u"), *get ("ctx.gru.b"), 32, 64, err, "ctx.gru")
              && setDense (context->get<3>(), *get ("ctx.dense_out.w"), *get ("ctx.dense_out.b"), 64, 32, err, "ctx.dense_out");
    if (! ok) { if (error) *error = err; return false; }
    context->get<1>().setAlphaVals (alphaV32);

    refCore = std::make_unique<CoreNet>();
    {
        CoreNet& net = *refCore;
        ok = setDense (net.get<0>(), *get ("core.dense_in.w"), *get ("core.dense_in.b"), kCoreIn, 128, err, "core.dense_in")
             && setGru (net.get<2>(), *get ("core.gru.w"), *get ("core.gru.u"), *get ("core.gru.b"), 128, 192, err, "core.gru")
             && setDense (net.get<3>(), *get ("core.dense_hidden.w"), *get ("core.dense_hidden.b"), 192, 192, err, "core.dense_hidden")
             && setDense (net.get<5>(), *get ("core.heads.w"), *get ("core.heads.b"), 192, kCoreOut, err, "core.heads");
        if (! ok) { if (error) *error = err; return false; }
        net.get<1>().setAlphaVals (alphaV128);
        net.get<4>().setAlphaVals (alphaV192);
    }
    if (! fast.load (tensors, alpha, err)) { if (error) *error = err; return false; }

    const auto& inh = get ("phys.inharm")->data;
    alphaB = inh[0]; betaB = inh[1]; alphaT = inh[2]; betaT = inh[3];

    semantics     = (int) metaValue ("semantics_version", 2.f);
    nStrings      = std::clamp ((int) metaValue ("n_strings", 1.f), 1, kStrings);
    stringGain    = metaValue ("string_gain", 1.f);
    detuneCents[0] = metaValue ("detune_cents0", 0.f);
    detuneCents[1] = metaValue ("detune_cents1", 0.f);
    releaseSec    = metaValue ("release_sec", 1.f);
    maxSustainSec = metaValue ("max_sustain_sec", 6.f);
    pedalThreshold = metaValue ("pedal_threshold", 0.5f);
    fadeFrames    = std::max (1, (int) metaValue ("fade_frames", 3.f));
    trainedVoices = (int) metaValue ("n_voices_train", 8.f);
    idleNoiseVoices = semantics <= 1 ? trainedVoices : 0;   // v2 enmascara el ruido de voces inactivas
    return true;
}

// ------------------------------------------------------------------------------
// Ciclo de vida
// ------------------------------------------------------------------------------
void Engine::prepare (double sampleRate)
{
    sr = sampleRate;
    frameLen = sr / (double) kFrameRate;
    noiseInc = (double) kModelRate / sr;
    cutoffHz = (float) std::min (8000.0, 0.45 * sr);
    reset();
}

void Engine::zeroState()
{
    for (auto& v : voices)
    {
        v.state = State::Free;
        v.retire = false;
    }
    std::memset (ring, 0, sizeof (ring));
    noiseRead = 0.0;
    frameIndex = 0;
    tickCount = 0;
    segLeft = 0;
    segLen = 0;
    orderCounter = 0;
    pedalValue = 0.f;
    pedalDown = false;
}

void Engine::reset()
{
    zeroState();
    if (context) context->reset();
    fast.resetAll();
    if (refCore) refCore->reset();
}

void Engine::setPolyphonyLimit (int n) noexcept
{
    polyLimit = std::clamp (n, 1, kPool - 4);
}

int Engine::numActiveVoices() const noexcept
{
    int n = 0;
    for (auto& v : voices)
        if (v.state != State::Free) ++n;
    return n;
}

// ------------------------------------------------------------------------------
// MIDI
// ------------------------------------------------------------------------------
int Engine::findFreeVoice() const
{
    for (int i = 0; i < kPool; ++i)
        if (voices[(size_t) i].state == State::Free) return i;
    return -1;
}

int Engine::allocateVoice()
{
    int active = 0;
    for (auto& v : voices)
        if (v.state == State::Held || v.state == State::Sustained || v.state == State::Releasing) ++active;

    if (active >= polyLimit)
    {
        // Voice stealing como en el dataset: primero la tecla ya levantada mas antigua, si no la mas antigua.
        int victim = -1;
        for (int pass = 0; pass < 2 && victim < 0; ++pass)
        {
            uint64_t best = UINT64_MAX;
            for (int i = 0; i < kPool; ++i)
            {
                const auto& v = voices[(size_t) i];
                const bool match = pass == 0 ? (v.state == State::Sustained || v.state == State::Releasing)
                                             : (v.state == State::Held);
                if (match && v.order < best) { best = v.order; victim = i; }
            }
        }
        if (victim >= 0)
        {
            auto& v = voices[(size_t) victim];
            v.state = State::Fading;
            v.fadeLeft = fadeFrames;
            if (v.vel <= 0.f) v.retire = true;
        }
    }

    int idx = findFreeVoice();
    if (idx < 0)
    {
        // Pool agotado: se reutiliza la voz que se esta apagando desde hace mas tiempo (corte duro)
        uint64_t best = UINT64_MAX;
        for (int i = 0; i < kPool; ++i)
            if (voices[(size_t) i].order < best) { best = voices[(size_t) i].order; idx = i; }
    }
    return idx;
}

void Engine::setupPartials (Voice& v)
{
    const double p = (double) v.note;
    const double f0 = 440.0 * std::pow (2.0, (p - 69.0) / 12.0);
    const double B = std::exp ((double) alphaT * p + (double) betaT) + std::exp ((double) alphaB * p + (double) betaB);

    const double maxDetune = std::pow (2.0, std::max (detuneCents[0], detuneCents[1]) / 1200.0);
    int nAct = 0;
    for (int k = 1; k <= kHarm; ++k)
    {
        const double fk = f0 * k * std::sqrt (1.0 + B * k * k) * maxDetune;
        if (fk < (double) cutoffHz) nAct = k; else break;
    }
    v.nAct = nAct;
    v.nPart = nAct * nStrings;
    v.nPartPad = (v.nPart + 3) & ~3;

    for (int j = 0; j < kPartials; ++j)
    {
        v.cr[j] = 1.f; v.ci[j] = 0.f; v.rc[j] = 1.f; v.rs[j] = 0.f;
        v.g[j] = 0.f; v.dg[j] = 0.f; v.target[j] = 0.f; v.mask[j] = 0.f;
    }
    for (int s = 0; s < nStrings; ++s)
    {
        const double detune = std::pow (2.0, (double) detuneCents[s] / 1200.0);
        for (int k = 1; k <= nAct; ++k)
        {
            const int j = s * nAct + (k - 1);
            const double fk = f0 * k * std::sqrt (1.0 + B * k * k) * detune;
            const double w = 2.0 * kPi * fk / sr;
            v.rc[j] = (float) std::cos (w);
            v.rs[j] = (float) std::sin (w);
            v.mask[j] = fk < (double) cutoffHz ? 1.f : 0.f;
        }
    }
}

void Engine::startVoice (int idx, int note, float vel)
{
    auto& v = voices[(size_t) idx];
    v.state = State::Held;
    v.retire = false;
    v.note = note;
    v.vel = vel;
    v.fadeLeft = 0;
    v.framesOnset = 0;
    v.framesState = 0;
    v.velIn = vel;
    v.retireTicked = false;
    for (auto& p : v.noisePow) p = 0.f;
    v.order = ++orderCounter;
    fast.copyState (idx, FastCore::kIdle);   // estado GRU inicial = voz en reposo, como en el entrenamiento
    setupPartials (v);
}

void Engine::noteOn (int note, float velocity01)
{
    if (! loaded) return;
    // Una nota repetida corta a la anterior del mismo pitch (entra en release, como en el dataset)
    for (auto& v : voices)
        if (v.note == note && (v.state == State::Held || v.state == State::Sustained))
        {
            v.state = State::Releasing;
            v.framesState = 0;
        }
    const int idx = allocateVoice();
    if (idx >= 0)
        startVoice (idx, note, std::clamp (velocity01, 0.f, 1.f));
}

void Engine::noteOff (int note)
{
    int best = -1;
    uint64_t bestOrder = 0;
    for (int i = 0; i < kPool; ++i)
    {
        const auto& v = voices[(size_t) i];
        if (v.state == State::Held && v.note == note && v.order >= bestOrder) { best = i; bestOrder = v.order; }
    }
    if (best < 0) return;
    auto& v = voices[(size_t) best];
    v.framesState = 0;
    v.state = (semantics >= 2 && pedalDown) ? State::Sustained : State::Releasing;
}

void Engine::setSustain (float value01)
{
    pedalValue = std::clamp (value01, 0.f, 1.f);
    const bool down = pedalValue >= pedalThreshold;
    if (pedalDown && ! down)
        for (auto& v : voices)
            if (v.state == State::Sustained) { v.state = State::Releasing; v.framesState = 0; }
    pedalDown = down;
}

void Engine::allNotesOff()
{
    for (auto& v : voices)
        if (v.state == State::Held || v.state == State::Sustained) { v.state = State::Releasing; v.framesState = 0; }
}

// ------------------------------------------------------------------------------
// Ruido filtrado (compartido por todas las voces; suma de potencias)
// ------------------------------------------------------------------------------
float Engine::uniform() noexcept
{
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return (float) (rng >> 8) * (2.0f / 16777216.0f) - 1.0f;
}

void Engine::fft (float* re, float* im, bool inverse) const
{
    const int N = kWin;
    for (int i = 1, j = 0; i < N; ++i)
    {
        int bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { std::swap (re[i], re[j]); std::swap (im[i], im[j]); }
    }
    for (int len = 2; len <= N; len <<= 1)
    {
        const int half = len >> 1, step = N / len;
        for (int i = 0; i < N; i += len)
            for (int k = 0; k < half; ++k)
            {
                const float wr = fftCos[k * step];
                const float wi = (inverse ? 1.f : -1.f) * fftSin[k * step];
                const int a = i + k, b = a + half;
                const float tr = re[b] * wr - im[b] * wi, ti = re[b] * wi + im[b] * wr;
                re[b] = re[a] - tr; im[b] = im[a] - ti;
                re[a] += tr;        im[a] += ti;
            }
    }
}

float Engine::noiseSample (double pos) const noexcept
{
    const double fl = std::floor (pos);
    const float t = (float) (pos - fl);
    const int i = (int) fl;
    const float y0 = ring[(i - 1) & (kRing - 1)], y1 = ring[i & (kRing - 1)];
    const float y2 = ring[(i + 1) & (kRing - 1)], y3 = ring[(i + 2) & (kRing - 1)];
    // Catmull-Rom
    return y1 + 0.5f * t * (y2 - y0 + t * (2.f * y0 - 5.f * y1 + 4.f * y2 - y3 + t * (3.f * (y1 - y2) + y3 - y0)));
}

// ------------------------------------------------------------------------------
// Frame neuronal (250 Hz)
// ------------------------------------------------------------------------------
// Escala del ruido de una voz: nivel global y, opcionalmente, menos ruido en los extremos del teclado
// (por debajo de la nota 40 y por encima de la 80; el efecto completo se alcanza una octava mas alla).
float Engine::noiseScale (int note) const noexcept
{
    const int beyond = std::max (0, std::max (40 - note, note - 80));
    const float t = std::min (1.f, (float) beyond / 12.f);
    return noiseGain * (1.f - noiseExtremes * t);
}

void Engine::retireVoice (Voice& v)
{
    v.retire = true;
}

int64_t Engine::tickBoundary (int64_t k) const noexcept
{
    return (int64_t) std::llround ((double) k * frameLen / (double) kTicks);
}

// Cada "tick" (4 por frame neuronal) procesa 1/kTicks de las voces. Cada voz sigue siendo actualizada a 250 Hz y
// su rampa dura exactamente un frame; solo cambia EN QUE MOMENTO del frame le toca. Asi ningun bloque de audio
// paga el calculo de todas las voces a la vez.
void Engine::neuralTick()
{
    const auto tNet0 = std::chrono::steady_clock::now();
    const int64_t k = tickCount++;
    const int g = (int) (k % kTicks);
    segLen = std::max (1, (int) (tickBoundary (k + 1) - tickBoundary (k)));
    segLeft = segLen;
    const float period = (float) std::max<int64_t> (1, tickBoundary (k + kTicks) - tickBoundary (k));

    const int maxHoldFrames = (int) (maxHoldSec * kFrameRate);
    const int maxSustainFrames = (int) (maxSustainSec * kFrameRate);
    const int releaseFrames = (int) (releaseSec * kFrameRate);

    // 1) red de contexto (una vez por frame, en el tick 0)
    if (g == 0)
    {
        float gv = 0.f, gp = 0.f;
        int cnt = 0;
        for (const auto& v : voices)
            if (v.state != State::Free && ! v.retire && v.velIn > 0.f) { gv += v.velIn; gp += (float) v.note / 127.f; ++cnt; }
        ctxIn[0] = pedalValue;
        ctxIn[1] = gv;
        ctxIn[2] = gp / ((float) cnt + 1e-7f);
        context->forward (ctxIn);
        std::memcpy (ctxOut, context->getOutputs(), sizeof (ctxOut));
    }

    // 2) voces de este grupo: temporizadores, rampa que termina y fila para la red
    int slots[FastCore::kSlots];
    int rowOf[kPool];
    alignas (32) float ins[FastCore::kSlots * kCoreIn];
    alignas (32) float outs[FastCore::kSlots * kCoreOut];
    for (int i = 0; i < kPool; ++i) rowOf[i] = -1;
    int nRows = 0;

    auto rampToZero = [&] (Voice& v)
    {
        for (int j = 0; j < v.nPartPad; ++j) { v.target[j] = 0.f; v.dg[j] = (0.f - v.g[j]) / period; }
        v.retireTicked = true;
    };

    for (int i = g; i < kPool; i += kTicks)
    {
        auto& v = voices[(size_t) i];
        if (v.state == State::Free) continue;

        // la rampa anterior acaba justo ahora: se ajusta al objetivo y se renormaliza el rotador
        for (int q = 0; q < v.nPartPad; ++q)
        {
            const float r2 = v.cr[q] * v.cr[q] + v.ci[q] * v.ci[q];
            const float kk = 0.5f * (3.f - r2);
            v.cr[q] *= kk; v.ci[q] *= kk;
            v.g[q] = v.target[q];
            v.dg[q] = 0.f;
        }
        if (v.retire)
        {
            if (v.retireTicked) { v.state = State::Free; v.retire = false; v.retireTicked = false; continue; }
            rampToZero (v);
            continue;
        }

        ++v.framesOnset;
        ++v.framesState;
        float vin = 0.f;
        switch (v.state)
        {
            case State::Held:
                vin = v.vel;
                if (v.framesOnset > maxHoldFrames) { v.state = State::Releasing; v.framesState = 0; vin = 0.f; }
                break;
            case State::Sustained:
                vin = v.vel;
                if (v.framesOnset > maxHoldFrames || v.framesState > maxSustainFrames)
                {
                    v.state = State::Releasing; v.framesState = 0; vin = 0.f;
                }
                break;
            case State::Releasing:
                if (v.framesState > releaseFrames) retireVoice (v);
                break;
            case State::Fading:
                --v.fadeLeft;
                vin = v.fadeLeft > 0 ? v.vel * (float) v.fadeLeft / (float) fadeFrames : 0.f;
                if (v.fadeLeft <= 0) retireVoice (v);
                break;
            default: break;
        }
        v.velIn = vin;
        if (v.retire) { rampToZero (v); continue; }

        float* in = ins + nRows * kCoreIn;
        in[0] = (float) v.note / 127.f;
        in[1] = vin;
        std::memcpy (in + 2, ctxOut, sizeof (ctxOut));
        slots[nRows] = i;
        rowOf[i] = nRows++;
    }
    int idleRow = -1;
    if (g == 0)
    {
        float* in = ins + nRows * kCoreIn;
        in[0] = 0.f; in[1] = 0.f;
        std::memcpy (in + 2, ctxOut, sizeof (ctxOut));
        slots[nRows] = FastCore::kIdle;
        idleRow = nRows++;
    }

    // 3) redes de las voces de este grupo, a la vez
    if (nRows > 0)
        fast.forward (slots, nRows, ins, outs);

    for (int i = g; i < kPool; i += kTicks)
    {
        if (rowOf[i] < 0) continue;
        auto& v = voices[(size_t) i];
        const float* o = outs + rowOf[i] * kCoreOut;

        const float amp = modifiedSigmoid (o[0]);
        float harm[kHarm];
        float sum = 0.f;
        for (int kk = 0; kk < kHarm; ++kk) { harm[kk] = modifiedSigmoid (o[1 + kk]); sum += harm[kk]; }
        const float inv = 1.f / sum;
        for (int s = 0; s < nStrings; ++s)
            for (int kk = 0; kk < v.nAct; ++kk)
            {
                const int j = s * v.nAct + kk;
                v.target[j] = amp * harm[kk] * inv * stringGain * v.mask[j];
            }
        for (int j = 0; j < v.nPartPad; ++j) v.dg[j] = (v.target[j] - v.g[j]) / period;

        for (int kk = 0; kk < kNoise; ++kk)
        {
            const float m = modifiedSigmoid (o[1 + kHarm + kk]);
            v.noisePow[kk] = m * m * noiseScale (v.note) * noiseScale (v.note);
        }
    }

    const auto tNet1 = std::chrono::steady_clock::now();
    profNet += std::chrono::duration<double> (tNet1 - tNet0).count();

    // 4) ruido (una vez por frame, en el tick 0): suma de las potencias de ruido de las voces
    if (g == 0)
    {
        float noisePower[kNoise] {};
        bool anyNoise = false;
        if (idleNoiseVoices > 0 && idleRow >= 0)
        {
            const float* o = outs + idleRow * kCoreOut;
            const int nIdle = std::max (0, idleNoiseVoices - numActiveVoices());
            for (int kk = 0; kk < kNoise; ++kk)
            {
                const float m = modifiedSigmoid (o[1 + kHarm + kk]);
                noisePower[kk] += (float) nIdle * m * m * noiseGain * noiseGain;
            }
            anyNoise = nIdle > 0;
        }
        for (const auto& v : voices)
        {
            if (v.state == State::Free || v.retire) continue;
            for (int kk = 0; kk < kNoise; ++kk) noisePower[kk] += v.noisePow[kk];
            anyNoise = true;
        }

        const int64_t t = frameIndex++;
        const int64_t base = t * kHop;
        for (int n = 0; n < kHop; ++n) ring[(int) ((base + kHop + n) & (kRing - 1))] = 0.f;
        if (anyNoise)
        {
            float re[kWin], im[kWin];
            for (int n = 0; n < kWin; ++n) { re[n] = uniform(); im[n] = 0.f; }
            fft (re, im, false);
            for (int kk = 0; kk <= kWin / 2; ++kk)
            {
                const float m = (kk < kNoise) ? std::sqrt (noisePower[kk]) : 0.f;   // el bin de Nyquist va a 0 (padding del entrenamiento)
                re[kk] *= m; im[kk] *= m;
                if (kk > 0 && kk < kWin / 2) { re[kWin - kk] *= m; im[kWin - kk] *= m; }
            }
            fft (re, im, true);
            for (int n = 0; n < kWin; ++n)
            {
                const float x = re[(n + kHop) & (kWin - 1)] / (float) kWin;   // irfft + roll(window/2)
                ring[(int) ((base + n) & (kRing - 1))] += x * hann[n];
            }
        }
        noiseRead = (double) base - 3.0;   // 3 muestras de retardo para que el interpolador cubico vea datos finales
        profNoise += std::chrono::duration<double> (std::chrono::steady_clock::now() - tNet1).count();
    }
}

// ------------------------------------------------------------------------------
// Sintesis de un tramo del segmento actual
// ------------------------------------------------------------------------------
void Engine::synthSegment (float* out, int n)
{
    const auto tS0 = std::chrono::steady_clock::now();
    std::fill (out, out + n, 0.f);

    for (auto& v : voices)
    {
        if (v.state == State::Free || v.nPartPad == 0) continue;
        const int M = v.nPartPad;
        for (int i = 0; i < n; ++i)
        {
            float acc[4] = { 0.f, 0.f, 0.f, 0.f };
            for (int j = 0; j < M; j += 4)
                for (int l = 0; l < 4; ++l)
                {
                    const int q = j + l;
                    const float c = v.cr[q], s = v.ci[q];
                    const float ns = s * v.rc[q] + c * v.rs[q];
                    v.cr[q] = c * v.rc[q] - s * v.rs[q];
                    v.ci[q] = ns;
                    v.g[q] += v.dg[q];
                    acc[l] += ns * v.g[q];
                }
            out[i] += (acc[0] + acc[1]) + (acc[2] + acc[3]);
        }
    }

    for (int i = 0; i < n; ++i)
    {
        out[i] += noiseSample (noiseRead);
        noiseRead += noiseInc;
    }

    profSynth += std::chrono::duration<double> (std::chrono::steady_clock::now() - tS0).count();
    segLeft -= n;
}

void Engine::render (float* out, int n)
{
    if (! loaded) { std::fill (out, out + n, 0.f); return; }
    int pos = 0;
    while (pos < n)
    {
        if (segLeft <= 0) neuralTick();
        const int todo = std::min (segLeft, n - pos);
        synthSegment (out + pos, todo);
        pos += todo;
    }
}

// ------------------------------------------------------------------------------
// Tests (prueba de oro)
// ------------------------------------------------------------------------------
void Engine::runContextSequence (const float* in, float* out, int frames)
{
    context->reset();
    for (int t = 0; t < frames; ++t)
    {
        std::memcpy (ctxIn, in + (size_t) t * kCtxIn, sizeof (ctxIn));
        context->forward (ctxIn);
        std::memcpy (out + (size_t) t * kCtxOut, context->getOutputs(), sizeof (float) * kCtxOut);
    }
}

void Engine::runCoreSequence (const float* in, float* out, int frames)
{
    fast.resetAll();
    const int slot = 0;
    for (int t = 0; t < frames; ++t)
        fast.forward (&slot, 1, in + (size_t) t * kCoreIn, out + (size_t) t * kCoreOut);
}

void Engine::runCoreSequenceRT (const float* in, float* out, int frames)
{
    auto& net = *refCore;
    net.reset();
    alignas (32) float buf[kCoreIn];
    for (int t = 0; t < frames; ++t)
    {
        std::memcpy (buf, in + (size_t) t * kCoreIn, sizeof (buf));
        net.forward (buf);
        std::memcpy (out + (size_t) t * kCoreOut, net.getOutputs(), sizeof (float) * kCoreOut);
    }
}

// ------------------------------------------------------------------------------
// FastCore
// ------------------------------------------------------------------------------
namespace
{
// C[v][n] = bias[n] + sum_k A[v][k] * B[k][n]    (B en el layout de Keras: [entradas][salidas])
void gemmBias (const float* A, int V, int K, const float* B, int N, const float* bias, float* C)
{
    constexpr int NT = 64;
    for (int n0 = 0; n0 < N; n0 += NT)
    {
        const int nt = std::min (NT, N - n0);
        for (int v = 0; v < V; ++v)
            for (int j = 0; j < nt; ++j)
                C[(size_t) v * N + n0 + j] = bias[n0 + j];
        for (int k = 0; k < K; ++k)
        {
            const float* b = B + (size_t) k * N + n0;
            for (int v = 0; v < V; ++v)
            {
                const float a = A[(size_t) v * K + k];
                float* c = C + (size_t) v * N + n0;
                for (int j = 0; j < nt; ++j)
                    c[j] += a * b[j];
            }
        }
    }
}

inline void leaky (float* x, int n, float alpha) noexcept
{
    for (int i = 0; i < n; ++i)
        x[i] = x[i] > 0.f ? x[i] : alpha * x[i];
}
} // namespace

bool FastCore::load (const std::map<std::string, Tensor>& t, float leakyAlpha, std::string& err)
{
    auto get = [&] (const char* n) -> const std::vector<float>* {
        auto it = t.find (n);
        if (it == t.end()) { err = std::string ("Falta el tensor ") + n; return nullptr; }
        return &it->second.data;
    };
    const auto *W1 = get ("core.dense_in.w"), *B1 = get ("core.dense_in.b"), *WX = get ("core.gru.w"), *UH = get ("core.gru.u"),
               *BG = get ("core.gru.b"), *W2 = get ("core.dense_hidden.w"), *B2 = get ("core.dense_hidden.b"),
               *W3 = get ("core.heads.w"), *B3 = get ("core.heads.b");
    if (! (W1 && B1 && WX && UH && BG && W2 && B2 && W3 && B3)) return false;
    if ((int) W1->size() != In * H1 || (int) WX->size() != H1 * 3 * U || (int) UH->size() != U * 3 * U
        || (int) BG->size() != 2 * 3 * U || (int) W2->size() != U * H2 || (int) W3->size() != H2 * Out)
    {
        err = "Dimensiones inesperadas en la red core";
        return false;
    }
    w1 = *W1; b1 = *B1; wx = *WX; uh = *UH; w2 = *W2; b2 = *B2; w3 = *W3; b3 = *B3;
    bi.assign (BG->begin(), BG->begin() + 3 * U);
    br.assign (BG->begin() + 3 * U, BG->end());
    alpha = leakyAlpha;
    state.assign ((size_t) kSlots * U, 0.f);
    h1.assign ((size_t) kSlots * H1, 0.f);
    g.assign ((size_t) kSlots * 3 * U, 0.f);
    hh.assign ((size_t) kSlots * 3 * U, 0.f);
    hp.assign ((size_t) kSlots * U, 0.f);
    h2.assign ((size_t) kSlots * H2, 0.f);
    zb.assign (U, 0.f); rb.assign (U, 0.f); cb.assign (U, 0.f);
    return true;
}

void FastCore::resetAll() { std::fill (state.begin(), state.end(), 0.f); }

void FastCore::copyState (int dst, int src)
{
    std::copy (state.begin() + (size_t) src * U, state.begin() + (size_t) (src + 1) * U, state.begin() + (size_t) dst * U);
}

void FastCore::forward (const int* slots, int n, const float* in, float* out)
{
    using ArrMap = Eigen::Map<Eigen::Array<float, Eigen::Dynamic, 1>>;
    using CArrMap = Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, 1>>;

    gemmBias (in, n, In, w1.data(), H1, b1.data(), h1.data());
    leaky (h1.data(), n * H1, alpha);
    gemmBias (h1.data(), n, H1, wx.data(), 3 * U, bi.data(), g.data());
    for (int v = 0; v < n; ++v)
        std::copy (state.begin() + (size_t) slots[v] * U, state.begin() + (size_t) (slots[v] + 1) * U, hp.begin() + (size_t) v * U);
    gemmBias (hp.data(), n, U, uh.data(), 3 * U, br.data(), hh.data());

    ArrMap z (zb.data(), U), r (rb.data(), U), c (cb.data(), U);
    for (int v = 0; v < n; ++v)
    {
        const float* gv = g.data() + (size_t) v * 3 * U;
        const float* hv = hh.data() + (size_t) v * 3 * U;
        ArrMap hn (hp.data() + (size_t) v * U, U);
        z = 0.5f * ((CArrMap (gv, U) + CArrMap (hv, U)) * 0.5f).tanh() + 0.5f;                 // sigmoid
        r = 0.5f * ((CArrMap (gv + U, U) + CArrMap (hv + U, U)) * 0.5f).tanh() + 0.5f;
        c = (CArrMap (gv + 2 * U, U) + r * CArrMap (hv + 2 * U, U)).tanh();
        hn = c + z * (hn - c);                                                                  // h = (1-z)*c + z*h
        std::copy (hp.begin() + (size_t) v * U, hp.begin() + (size_t) (v + 1) * U, state.begin() + (size_t) slots[v] * U);
    }

    gemmBias (hp.data(), n, U, w2.data(), H2, b2.data(), h2.data());
    leaky (h2.data(), n * H2, alpha);
    gemmBias (h2.data(), n, H2, w3.data(), Out, b3.data(), out);
}

} // namespace ddsp
