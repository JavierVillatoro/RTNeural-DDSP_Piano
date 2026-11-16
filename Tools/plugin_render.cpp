// ddsp_plugin_render: renderiza un MIDI con el PLUGIN REAL (el mismo PluginProcessor que carga el DAW: curva de velocity, motor,
// mecanica, reverb de JUCE, tono, anchura y limitador), sin abrir ninguna ventana ni DAW.
//
//   ddsp_plugin_render eventos.txt salida.wav [nombre=valor ...]
//
// Formato de eventos (el mismo que ddsp_render, tiempos en segundos):  on t nota vel01 | off t nota | ped t valor01 | end t
// Ajustes: cualquier parametro del plugin por su identificador, p. ej. mechanics=0 humanize=0.2 model=1 gain=24 width=1
//          (ids: gain reverb voices model noise noiseext velmax touch dynamics velcomp transpose humanize unison width bassb
//           damper evenness tone hardness mechanics) y las especiales sr=44100 block=512.
#include <JuceHeader.h>
#include "PluginProcessor.h"

#include <fstream>
#include <sstream>

struct Ev { double t; int type; int note; float value; };   // 0 on, 1 off, 2 pedal

int main (int argc, char** argv)
{
    if (argc < 3)
    {
        std::fprintf (stderr, "uso: ddsp_plugin_render eventos.txt salida.wav [nombre=valor ...]\n");
        return 2;
    }
    juce::ScopedJuceInitialiser_GUI juceInit;

    std::vector<Ev> events;
    double endT = 0.0;
    {
        std::ifstream in (argv[1]);
        for (std::string line; std::getline (in, line);)
        {
            std::istringstream ss (line);
            std::string k; double t = 0; ss >> k >> t;
            if (k == "on")        { int n; float v; ss >> n >> v; events.push_back ({ t, 0, n, v }); }
            else if (k == "off")  { int n; ss >> n; events.push_back ({ t, 1, n, 0.f }); }
            else if (k == "ped")  { float v; ss >> v; events.push_back ({ t, 2, 0, v }); }
            else if (k == "end")  endT = t;
        }
    }
    std::stable_sort (events.begin(), events.end(), [] (const Ev& a, const Ev& b) { return a.t < b.t; });
    if (endT <= 0.0 && ! events.empty()) endT = events.back().t + 3.0;

    double sr = 44100.0;
    int block = 512;
    std::vector<std::pair<std::string, double>> settings;
    for (int i = 3; i < argc; ++i)
    {
        const std::string a = argv[i]; const auto eq = a.find ('=');
        if (eq == std::string::npos) continue;
        const std::string key = a.substr (0, eq); const double val = std::atof (a.c_str() + eq + 1);
        if (key == "sr") sr = val; else if (key == "block") block = (int) val; else settings.emplace_back (key, val);
    }

    DDSPianoAudioProcessor proc;
    for (auto& [key, val] : settings)
    {
        if (auto* p = proc.apvts.getParameter (key))
            p->setValueNotifyingHost (p->convertTo0to1 ((float) val));
        else
            std::fprintf (stderr, "parametro desconocido: %s\n", key.c_str());
    }
    proc.setPlayConfigDetails (0, 2, sr, block);
    proc.prepareToPlay (sr, block);
    juce::Thread::sleep (1500);   // la reverb (convolucion) carga su respuesta al impulso en otro hilo

    const int total = (int) std::llround (endT * sr);
    juce::AudioBuffer<float> result (2, total);
    juce::AudioBuffer<float> buf (2, block);
    size_t nextEv = 0;
    for (int pos = 0; pos < total; pos += block)
    {
        const int n = std::min (block, total - pos);
        juce::MidiBuffer midi;
        while (nextEv < events.size() && (int) std::llround (events[nextEv].t * sr) < pos + n)
        {
            const auto& e = events[nextEv++];
            const int off = juce::jlimit (0, n - 1, (int) std::llround (e.t * sr) - pos);
            if (e.type == 0)      midi.addEvent (juce::MidiMessage::noteOn (1, e.note, e.value), off);
            else if (e.type == 1) midi.addEvent (juce::MidiMessage::noteOff (1, e.note), off);
            else                  midi.addEvent (juce::MidiMessage::controllerEvent (1, 64, juce::jlimit (0, 127, (int) std::lround (e.value * 127.0f))), off);
        }
        buf.setSize (2, n, false, false, true);
        buf.clear();
        proc.processBlock (buf, midi);
        for (int ch = 0; ch < 2; ++ch)
            result.copyFrom (ch, pos, buf, ch, 0, n);
    }

    juce::File outFile (argv[2]);
    outFile.deleteFile();
    auto stream = std::make_unique<juce::FileOutputStream> (outFile);
    if (! stream->openedOk()) { std::fprintf (stderr, "no se pudo escribir %s\n", argv[2]); return 1; }
    juce::WavAudioFormat wav;
    std::unique_ptr<juce::AudioFormatWriter> writer (wav.createWriterFor (stream.get(), sr, 2, 24, {}, 0));
    if (writer == nullptr) { std::fprintf (stderr, "no se pudo crear el escritor WAV\n"); return 1; }
    stream.release();
    writer->writeFromAudioSampleBuffer (result, 0, total);
    writer.reset();

    float peak = 0.f;
    for (int ch = 0; ch < 2; ++ch) peak = std::max (peak, result.getMagnitude (ch, 0, total));
    std::printf ("[OK] %s  (%.1f s a %.0f Hz, pico %.1f dBFS)\n", argv[2], endT, sr, 20.0 * std::log10 (peak + 1e-12f));
    return 0;
}
