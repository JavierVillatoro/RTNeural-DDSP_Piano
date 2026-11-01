// ddsp_render: herramienta de linea de comandos para verificar el motor C++ sin abrir un DAW.
//
//   ddsp_render render <eventos.txt> <ddsp_weights.bin> <salida.wav> [sampleRate=16000] [vocesInactivasRuido]
//        Renderiza (mono, SIN reverb) una lista de eventos MIDI con el MISMO motor que el plugin.
//   ddsp_render golden <golden.bin> <ddsp_weights.bin>
//        Prueba de oro: compara las salidas de RTNeural con las de Keras (generadas por
//        training/make_golden.py). Devuelve 0 si todo coincide.
//
// Formato de eventos (texto, tiempos en segundos):   on t nota vel01 | off t nota | ped t valor01 | end t

#include "DDSPEngine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined(__SSE__) || defined(_M_X64)
 #include <xmmintrin.h>
#endif

static std::vector<char> readFile (const std::string& path)
{
    std::ifstream f (path, std::ios::binary);
    if (! f) return {};
    return std::vector<char> ((std::istreambuf_iterator<char> (f)), std::istreambuf_iterator<char>());
}

static bool writeWavFloat (const std::string& path, const std::vector<float>& x, uint32_t sr)
{
    std::ofstream f (path, std::ios::binary);
    if (! f) return false;
    auto w32 = [&] (uint32_t v) { f.write ((const char*) &v, 4); };
    auto w16 = [&] (uint16_t v) { f.write ((const char*) &v, 2); };
    const uint32_t dataBytes = (uint32_t) (x.size() * 4);
    f.write ("RIFF", 4); w32 (36 + dataBytes); f.write ("WAVEfmt ", 8);
    w32 (16); w16 (3); w16 (1); w32 (sr); w32 (sr * 4); w16 (4); w16 (32);
    f.write ("data", 4); w32 (dataBytes);
    f.write ((const char*) x.data(), dataBytes);
    return true;
}

struct Event { double t; int type; int note; float value; };   // 0 on, 1 off, 2 pedal

static int cmdRender (int argc, char** argv)
{
    if (argc < 5) { std::cerr << "uso: ddsp_render render eventos.txt pesos.bin salida.wav [sr]\n"; return 2; }
    const double sr = argc > 5 ? std::atof (argv[5]) : 16000.0;

    auto wbytes = readFile (argv[3]);
    ddsp::Engine engine;
    std::string err;
    if (! engine.loadWeights (wbytes.data(), wbytes.size(), &err)) { std::cerr << "Error pesos: " << err << "\n"; return 1; }
    engine.prepare (sr);
    engine.setPolyphonyLimit (24);
    if (argc > 9) engine.setNoise ((float) std::atof (argv[8]), (float) std::atof (argv[9]));   // ruido: nivel y reduccion en extremos
    if (argc > 6) engine.setIdleNoiseVoices (std::atoi (argv[6]));   // p.ej. 24 para igualar el render de referencia de Python

    std::vector<Event> ev;
    double endT = 0.0;
    std::ifstream in (argv[2]);
    for (std::string line; std::getline (in, line);)
    {
        std::istringstream ss (line);
        std::string k; double t = 0; ss >> k >> t;
        if (k == "on")  { int n; float v; ss >> n >> v; ev.push_back ({ t, 0, n, v }); }
        else if (k == "off") { int n; ss >> n; ev.push_back ({ t, 1, n, 0.f }); }
        else if (k == "ped") { float v; ss >> v; ev.push_back ({ t, 2, 0, v }); }
        else if (k == "end") endT = t;
    }
    std::stable_sort (ev.begin(), ev.end(), [] (const Event& a, const Event& b) { return a.t < b.t; });
    if (endT <= 0.0 && ! ev.empty()) endT = ev.back().t + 2.0;

    const size_t total = (size_t) std::llround (endT * sr);
    std::vector<float> out (total, 0.f);
    size_t pos = 0;
    int peakVoices = 0;
    const int blockSize = argc > 7 ? std::atoi (argv[7]) : 512;
    std::vector<double> blockMs;
    const auto t0 = std::chrono::steady_clock::now();
    auto renderTo = [&] (size_t target)
    {
        target = std::min (target, total);
        while (pos < target)
        {
            const int n = (int) std::min<size_t> ((size_t) blockSize, target - pos);   // bloques como un DAW
            const auto b0 = std::chrono::steady_clock::now();
            engine.render (out.data() + pos, n);
            if (n == blockSize) blockMs.push_back (std::chrono::duration<double, std::milli> (std::chrono::steady_clock::now() - b0).count());
            peakVoices = std::max (peakVoices, engine.numActiveVoices());
            pos += (size_t) n;
        }
    };
    for (auto& e : ev)
    {
        renderTo ((size_t) std::llround (e.t * sr));
        if (e.type == 0) engine.noteOn (e.note, e.value);
        else if (e.type == 1) engine.noteOff (e.note);
        else engine.setSustain (e.value);
    }
    renderTo (total);
    const double secs = std::chrono::duration<double> (std::chrono::steady_clock::now() - t0).count();

    if (! writeWavFloat (argv[4], out, (uint32_t) sr)) { std::cerr << "No se pudo escribir " << argv[4] << "\n"; return 1; }
    std::printf ("[OK] %s  (%.1f s de audio en %.2f s -> %.1fx tiempo real, voces pico=%d, semantica v%d)\n",
                 argv[4], endT, secs, endT / secs, peakVoices, engine.semanticsVersion());
    if (! blockMs.empty())
    {
        std::vector<double> v = blockMs;
        std::sort (v.begin(), v.end());
        const double budget = 1000.0 * blockSize / sr;
        std::printf ("      bloques de %d muestras (presupuesto %.2f ms): mediana=%.3f ms  p99=%.3f ms  p99.9=%.3f ms  MAX=%.3f ms  -> peor caso usa %.0f%% del presupuesto\n",
                     blockSize, budget, v[v.size() / 2], v[(size_t) (v.size() * 0.99)], v[(size_t) (v.size() * 0.999)], v.back(), 100.0 * v.back() / budget);
    }
    std::printf ("      tiempo: redes=%.2fs  ruido(FFT)=%.2fs  osciladores=%.2fs  (total %.2fs)\n", engine.profNet, engine.profNoise, engine.profSynth, secs);
    return 0;
}

static int cmdGolden (int argc, char** argv)
{
    if (argc < 4) { std::cerr << "uso: ddsp_render golden golden.bin pesos.bin\n"; return 2; }
    auto g = readFile (argv[2]);
    auto wbytes = readFile (argv[3]);
    if (g.size() < 8 || std::memcmp (g.data(), "GOLD", 4) != 0) { std::cerr << "golden.bin invalido\n"; return 1; }
    uint32_t F = 0; std::memcpy (&F, g.data() + 4, 4);
    const float* p = (const float*) (g.data() + 8);
    const float* ctxIn = p;  p += (size_t) F * ddsp::kCtxIn;
    const float* ctxRef = p; p += (size_t) F * ddsp::kCtxOut;
    const float* coreIn = p; p += (size_t) F * ddsp::kCoreIn;
    const float* coreRef = p;

    ddsp::Engine engine;
    std::string err;
    if (! engine.loadWeights (wbytes.data(), wbytes.size(), &err)) { std::cerr << "Error pesos: " << err << "\n"; return 1; }
    engine.prepare (16000.0);

    std::vector<float> ctxOut ((size_t) F * ddsp::kCtxOut), coreOut ((size_t) F * ddsp::kCoreOut), coreOutRT ((size_t) F * ddsp::kCoreOut);
    engine.runContextSequence (ctxIn, ctxOut.data(), (int) F);
    engine.runCoreSequence (coreIn, coreOut.data(), (int) F);        // ruta rapida del plugin
    engine.runCoreSequenceRT (coreIn, coreOutRT.data(), (int) F);    // RTNeural puro

    auto compare = [&] (const char* name, const std::vector<float>& a, const float* b, size_t n, int width)
    {
        double maxErr = 0, sumRef = 0; size_t worst = 0;
        for (size_t i = 0; i < n; ++i)
        {
            const double e = std::fabs ((double) a[i] - (double) b[i]);
            if (e > maxErr) { maxErr = e; worst = i; }
            sumRef += std::fabs ((double) b[i]);
        }
        std::printf ("  %-8s max|err| = %.3e  (en frame %zu, canal %zu)   |ref| medio = %.3f\n",
                     name, maxErr, worst / (size_t) width, worst % (size_t) width, sumRef / (double) n);
        return maxErr;
    };
    std::printf ("Prueba de oro RTNeural vs Keras (%u frames)\n", F);
    const double e1 = compare ("context", ctxOut, ctxRef, (size_t) F * ddsp::kCtxOut, ddsp::kCtxOut);
    const double e2 = compare ("core RT", coreOutRT, coreRef, (size_t) F * ddsp::kCoreOut, ddsp::kCoreOut);
    const double e3 = compare ("core fast", coreOut, coreRef, (size_t) F * ddsp::kCoreOut, ddsp::kCoreOut);
    const bool ok = e1 < 2e-3 && e2 < 2e-3 && e3 < 2e-3;
    std::printf ("%s\n", ok ? "[OK] Los pesos estan SINCRONIZADOS con RTNeural." : "[FALLO] Hay diferencias: revisa el orden de pesos/activaciones.");
    return ok ? 0 : 1;
}

int main (int argc, char** argv)
{
#if defined(__SSE__) || defined(_M_X64)
    _mm_setcsr (_mm_getcsr() | 0x8040);   // flush-to-zero + denormals-are-zero, como ScopedNoDenormals
#endif
    if (argc >= 2 && std::strcmp (argv[1], "render") == 0) return cmdRender (argc, argv);
    if (argc >= 2 && std::strcmp (argv[1], "golden") == 0) return cmdGolden (argc, argv);
    std::cerr << "uso: ddsp_render render|golden ...\n";
    return 2;
}
