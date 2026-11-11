#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <BinaryData.h>

namespace
{
struct ModelData
{
    const char* weights; int weightsSize;
    const char* ir;      int irSize;
    const char* name;
};

// Ajuste de nivel por modelo para comparar a igual volumen: medido con notas sueltas, el v8 sale
// ~9-10 dB mas fuerte que el v6 a velocity media (5 dB a fortissimo, 12 dB a pianissimo).
const float kModelGainDb[DDSPianoAudioProcessor::kModels] = { 0.0f, -10.0f };

// Nivel RMS medio (dB, 0,5 s tras el ataque; notas 36/48/60/72/84) de cada modelo para velocity 0.1, 0.2 ... 1.0.
// Medido con ddsp_render; sirve para la "compensacion del modelo" de la curva de velocity (ver VelocityCurve.h).
// Si se reentrena o se anade un modelo hay que volver a medirlo.
const float kLevelDb[DDSPianoAudioProcessor::kModels][ddsp::VelocityCurve::kPoints] =
{
    { -76.3f, -70.5f, -65.0f, -59.8f, -54.9f, -50.4f, -46.7f, -43.6f, -40.8f, -38.1f },   // v6
    { -62.3f, -57.3f, -52.8f, -48.4f, -44.2f, -40.3f, -36.9f, -34.4f, -32.7f, -31.5f },   // v8
};

const ModelData kModelData[DDSPianoAudioProcessor::kModels] =
{
    { BinaryData::ddsp_weights_v6_bin, BinaryData::ddsp_weights_v6_binSize, BinaryData::reverb_ir_v6_st_wav, BinaryData::reverb_ir_v6_st_wavSize, "v6" },
    { BinaryData::ddsp_weights_v8_bin, BinaryData::ddsp_weights_v8_binSize, BinaryData::reverb_ir_v8_st_wav, BinaryData::reverb_ir_v8_st_wavSize, "v8" },
};
}

// ==============================================================================
juce::AudioProcessorValueTreeState::ParameterLayout DDSPianoAudioProcessor::createLayout()
{
    using namespace juce;
    AudioProcessorValueTreeState::ParameterLayout layout;
    // El modelo se entreno con audio de nivel bajo (~ -30 dBFS): la ganancia por defecto lo compensa.
    layout.add (std::make_unique<AudioParameterFloat> (ParameterID { "gain", 1 }, "Gain",
                                                       NormalisableRange<float> (-24.0f, 48.0f, 0.1f), 24.0f,
                                                       AudioParameterFloatAttributes().withLabel ("dB")));
    layout.add (std::make_unique<AudioParameterFloat> (ParameterID { "reverb", 1 }, "Room",
                                                       NormalisableRange<float> (0.0f, 1.0f, 0.001f), 1.0f));
    layout.add (std::make_unique<AudioParameterInt> (ParameterID { "voices", 1 }, "Max Voices", 4, 24, 24));
    layout.add (std::make_unique<AudioParameterFloat> (ParameterID { "noise", 1 }, "Noise",
                                                       NormalisableRange<float> (0.0f, 1.5f, 0.01f), 1.0f));
    layout.add (std::make_unique<AudioParameterFloat> (ParameterID { "noiseext", 1 }, "Noise Extremes",
                                                       NormalisableRange<float> (0.0f, 1.0f, 0.01f), 0.0f));
    // Curva de velocity: Vel Max (la velocity con la que tu teclado llega a fortissimo), Touch (suave/duro),
    // Dynamics (contraste) y compensacion del modelo (reparte los dB por igual a lo largo de la tecla).
    layout.add (std::make_unique<AudioParameterInt> (ParameterID { "velmax", 1 }, "Vel Max", 40, 127, 127));
    layout.add (std::make_unique<AudioParameterFloat> (ParameterID { "touch", 1 }, "Touch",
                                                       NormalisableRange<float> (-1.0f, 1.0f, 0.01f), 0.0f));
    layout.add (std::make_unique<AudioParameterFloat> (ParameterID { "dynamics", 1 }, "Dynamics",
                                                       NormalisableRange<float> (0.0f, 1.5f, 0.01f), 1.0f));
    layout.add (std::make_unique<AudioParameterBool> (ParameterID { "velcomp", 1 }, "Velocity Compensation", true));
    // Transposicion (semitonos; mueve las notas MIDI, no cambia el sonido de cada nota), variacion entre notas y 2ª cuerda
    layout.add (std::make_unique<AudioParameterInt> (ParameterID { "transpose", 1 }, "Transpose", -24, 24, 0,
                                                     AudioParameterIntAttributes().withLabel ("st")));
    layout.add (std::make_unique<AudioParameterFloat> (ParameterID { "humanize", 1 }, "Humanize",
                                                       NormalisableRange<float> (0.0f, 1.0f, 0.01f), 0.4f));
    layout.add (std::make_unique<AudioParameterFloat> (ParameterID { "unison", 1 }, "Unison",
                                                       NormalisableRange<float> (0.0f, 1.0f, 0.01f), 0.4f));
    // Anchura estereo (mezcla M/S del resultado final): 0 = mono, 1 = reparto por registro + reverb estereo, >1 = mas ancho
    layout.add (std::make_unique<AudioParameterFloat> (ParameterID { "width", 1 }, "Width",
                                                       NormalisableRange<float> (0.0f, 1.5f, 0.01f), 1.0f));
    layout.add (std::make_unique<AudioParameterChoice> (ParameterID { "model", 1 }, "Model",
                                                        StringArray { "v6 (anterior)", "v8 (nuevo)" }, 1));
    return layout;
}

DDSPianoAudioProcessor::DDSPianoAudioProcessor()
    : AudioProcessor (BusesProperties().withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
      apvts (*this, nullptr, "STATE", createLayout())
{
    pGain = apvts.getRawParameterValue ("gain");
    pReverb = apvts.getRawParameterValue ("reverb");
    pVoices = apvts.getRawParameterValue ("voices");
    pModel = apvts.getRawParameterValue ("model");
    pNoise = apvts.getRawParameterValue ("noise");
    pNoiseExt = apvts.getRawParameterValue ("noiseext");
    pVelMax = apvts.getRawParameterValue ("velmax");
    pTouch = apvts.getRawParameterValue ("touch");
    pDynamics = apvts.getRawParameterValue ("dynamics");
    pVelComp = apvts.getRawParameterValue ("velcomp");
    pTranspose = apvts.getRawParameterValue ("transpose");
    pHumanize = apvts.getRawParameterValue ("humanize");
    pUnison = apvts.getRawParameterValue ("unison");
    pWidth = apvts.getRawParameterValue ("width");
    clearSoundingNotes();

    for (int m = 0; m < kModels; ++m)
    {
        engines[(size_t) m] = std::make_unique<ddsp::Engine>();
        std::string err;
        const auto& d = kModelData[m];
        const bool ok = engines[(size_t) m]->loadWeights (d.weights, (size_t) d.weightsSize, &err);
        loaded[(size_t) m] = ok;
        if (ok)
            infos[(size_t) m] = juce::String (d.name) + (engines[(size_t) m]->semanticsVersion() >= 2 ? " · pedal de sustain real" : " · pedal solo como contexto");
        else
            infos[(size_t) m] = juce::String (d.name) + " · ERROR cargando pesos: " + juce::String (err);

        reverb[(size_t) m].loadImpulseResponse (d.ir, (size_t) d.irSize,
                                                juce::dsp::Convolution::Stereo::yes, juce::dsp::Convolution::Trim::no, 0,
                                                juce::dsp::Convolution::Normalise::no);   // JUCE reescala por sr_original/sr_host
    }

    activeModel = juce::jlimit (0, kModels - 1, (int) std::lround (pModel->load()));
    activeModelUI.store (activeModel);
}

int DDSPianoAudioProcessor::getSelectedModel() const noexcept
{
    return juce::jlimit (0, kModels - 1, (int) std::lround (pModel->load()));
}

void DDSPianoAudioProcessor::setSelectedModel (int index)
{
    if (auto* p = apvts.getParameter ("model"))
    {
        p->beginChangeGesture();
        p->setValueNotifyingHost (p->convertTo0to1 ((float) juce::jlimit (0, kModels - 1, index)));
        p->endChangeGesture();
    }
}

ddsp::VelocityCurve DDSPianoAudioProcessor::currentVelocityCurve (int model) const noexcept
{
    ddsp::VelocityCurve c;
    c.velMax = pVelMax->load() / 127.0f;
    c.touch = pTouch->load();
    c.dynamics = pDynamics->load();
    c.levelDb = pVelComp->load() > 0.5f ? kLevelDb[juce::jlimit (0, kModels - 1, model)] : nullptr;
    return c;
}

bool DDSPianoAudioProcessor::calibrateVelMax()
{
    const int peak = peakVelIn.load();
    if (peak < 40)
        return false;   // aun no ha tocado nada fuerte (o nada)
    if (auto* p = apvts.getParameter ("velmax"))
    {
        p->beginChangeGesture();
        p->setValueNotifyingHost (p->convertTo0to1 ((float) juce::jlimit (40, 127, peak)));
        p->endChangeGesture();
    }
    peakVelIn.store (0);
    return true;
}

void DDSPianoAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    sampleRateHz = sampleRate;
    const int maxBlock = std::max (samplesPerBlock, 1);
    for (auto& e : engines)
        e->prepare (sampleRate);
    mono.assign ((size_t) maxBlock, 0.0f);
    monoR.assign ((size_t) maxBlock, 0.0f);
    widthSmooth.reset (sampleRate, 0.05);
    widthSmooth.setCurrentAndTargetValue (pWidth->load());
    revBuf.setSize (2, maxBlock, false, true, false);
    wetSum.setSize (2, maxBlock, false, true, false);
    fadeTotal = std::max (1, (int) (0.010 * sampleRate));   // fundido de entrada de 10 ms al cambiar de modelo
    fadeLeft = 0;
    lastSustain = 0.f;

    juce::dsp::ProcessSpec spec { sampleRate, (juce::uint32) maxBlock, 2 };
    for (auto& r : reverb)
    {
        r.prepare (spec);
        r.reset();
    }
    mixer.prepare (spec);
    mixer.reset();
}

bool DDSPianoAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    return layouts.getMainOutputChannelSet() == juce::AudioChannelSet::stereo();
}

void DDSPianoAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi)
{
    juce::ScopedNoDenormals noDenormals;
    const int numSamples = buffer.getNumSamples();
    for (int ch = getTotalNumInputChannels(); ch < getTotalNumOutputChannels(); ++ch)
        buffer.clear (ch, 0, numSamples);

    if (! loaded[0] && ! loaded[1])
    {
        buffer.clear();
        return;
    }
    if ((int) mono.size() < numSamples)
    {
        mono.resize ((size_t) numSamples);   // solo si el host supera el maximo anunciado
        monoR.resize ((size_t) numSamples);
    }
    if (revBuf.getNumSamples() < numSamples)
    {
        revBuf.setSize (2, numSamples, false, true, true);
        wetSum.setSize (2, numSamples, false, true, true);
    }

    // --- cambio de modelo (v6 <-> v8): el anterior se calla y el nuevo arranca limpio ---
    const int requested = getSelectedModel();
    if (requested != activeModel && loaded[(size_t) requested])
    {
        engines[(size_t) activeModel]->reset();
        activeModel = requested;
        engines[(size_t) activeModel]->reset();
        clearSoundingNotes();
        engines[(size_t) activeModel]->setSustain (lastSustain);   // si el pedal sigue pisado, el nuevo modelo lo sabe
        fadeLeft = fadeTotal;
        activeModelUI.store (activeModel);
    }
    auto& engine = *engines[(size_t) activeModel];
    engine.setPolyphonyLimit ((int) pVoices->load());
    engine.setNoise (pNoise->load(), pNoiseExt->load());
    engine.setHumanize (pHumanize->load());
    engine.setUnison (pUnison->load());
    const int transpose = (int) std::lround (pTranspose->load());

    // Pico de velocity del teclado fisico (antes de mezclar el teclado en pantalla): sirve para calibrar "Vel Max"
    for (const auto metadata : midi)
    {
        const auto m = metadata.getMessage();
        if (m.isNoteOn() && (int) m.getVelocity() > peakVelIn.load (std::memory_order_relaxed))
            peakVelIn.store ((int) m.getVelocity(), std::memory_order_relaxed);
    }
    const auto velCurve = currentVelocityCurve (activeModel);

    // Teclado en pantalla (y teclas del ordenador) + pedal de sustain virtual
    keyboardState.processNextMidiBuffer (midi, 0, numSamples, true);
    const bool vs = virtualSustain.load();
    if (vs != lastVirtualSustain)
    {
        midi.addEvent (juce::MidiMessage::controllerEvent (1, 64, vs ? 127 : 0), 0);
        lastVirtualSustain = vs;
    }

    // --- MIDI con precision de muestra: se sintetiza hasta cada evento y luego se aplica ---
    int done = 0;
    for (const auto metadata : midi)
    {
        const auto msg = metadata.getMessage();
        const int pos = juce::jlimit (0, numSamples, metadata.samplePosition);
        if (pos > done)
        {
            engine.renderStereo (mono.data() + done, monoR.data() + done, pos - done);
            done = pos;
        }
        if (msg.isNoteOn())
        {
            const float out = velCurve.apply (msg.getFloatVelocity());
            lastVelIn.store ((int) msg.getVelocity(), std::memory_order_relaxed);
            lastVelOut.store ((int) std::lround (out * 127.0f), std::memory_order_relaxed);
            const int key = msg.getNoteNumber();
            if (soundingNote[(size_t) key] >= 0)
                engine.noteOff (soundingNote[(size_t) key]);   // tecla repetida sin soltar: se libera la anterior
            const int n = key + transpose;
            const bool inRange = n >= 21 && n <= 108;   // el rango del piano con el que se entreno el modelo
            soundingNote[(size_t) key] = inRange ? n : -1;
            if (inRange)
                engine.noteOn (n, out);
        }
        else if (msg.isNoteOff())
        {
            const int key = msg.getNoteNumber();
            if (soundingNote[(size_t) key] >= 0)
                engine.noteOff (soundingNote[(size_t) key]);
            soundingNote[(size_t) key] = -1;
        }
        else if (msg.isController() && msg.getControllerNumber() == 64)
        {
            lastSustain = (float) msg.getControllerValue() / 127.0f;
            engine.setSustain (lastSustain);
        }
        else if (msg.isAllNotesOff() || msg.isAllSoundOff())
        {
            engine.allNotesOff();
            clearSoundingNotes();
        }
    }
    if (done < numSamples)
        engine.renderStereo (mono.data() + done, monoR.data() + done, numSamples - done);

    // --- ganancia (con fundido de entrada tras un cambio de modelo) ---
    const float gain = juce::Decibels::decibelsToGain (pGain->load() + kModelGainDb[activeModel]);
    auto* l = buffer.getWritePointer (0);
    auto* r = buffer.getWritePointer (1);
    for (int i = 0; i < numSamples; ++i)
    {
        float g = gain;
        if (fadeLeft > 0)
        {
            g *= (float) (fadeTotal - fadeLeft) / (float) fadeTotal;
            --fadeLeft;
        }
        l[i] = mono[(size_t) i] * g;
        r[i] = monoR[(size_t) i] * g;
    }

    // --- reverb entrenada: una convolucion por modelo; solo la del modelo activo recibe senal,
    //     la otra deja caer su cola, asi el cambio de sala es suave ---
    juce::dsp::AudioBlock<float> block (buffer);
    mixer.setWetMixProportion (pReverb->load());
    mixer.pushDrySamples (block);

    wetSum.clear (0, numSamples);
    wetSum.clear (1, numSamples);
    for (int k = 0; k < kModels; ++k)
    {
        revBuf.clear (0, numSamples);
        revBuf.clear (1, numSamples);
        if (k == activeModel)
        {
            revBuf.copyFrom (0, 0, buffer, 0, 0, numSamples);
            revBuf.copyFrom (1, 0, buffer, 1, 0, numSamples);
        }
        juce::dsp::AudioBlock<float> rb (revBuf);
        auto sub = rb.getSubBlock (0, (size_t) numSamples);
        juce::dsp::ProcessContextReplacing<float> rctx (sub);
        reverb[(size_t) k].process (rctx);
        wetSum.addFrom (0, 0, revBuf, 0, 0, numSamples);
        wetSum.addFrom (1, 0, revBuf, 1, 0, numSamples);
    }
    buffer.copyFrom (0, 0, wetSum, 0, 0, numSamples);
    buffer.copyFrom (1, 0, wetSum, 1, 0, numSamples);
    mixer.mixWetSamples (block);

    // --- anchura estereo: mezcla M/S del resultado (seco + reverb). 0 = mono, 1 = tal cual, >1 = mas ancho ---
    widthSmooth.setTargetValue (pWidth->load());
    {
        auto* wl = buffer.getWritePointer (0);
        auto* wr = buffer.getWritePointer (1);
        for (int i = 0; i < numSamples; ++i)
        {
            const float wd = widthSmooth.getNextValue();
            const float m = 0.5f * (wl[i] + wr[i]);
            const float s = 0.5f * (wl[i] - wr[i]) * wd;
            wl[i] = m + s;
            wr[i] = m - s;
        }
    }

    // --- limitador suave ---
    for (int ch = 0; ch < 2; ++ch)
    {
        auto* d = buffer.getWritePointer (ch);
        for (int i = 0; i < numSamples; ++i)
            d[i] = std::tanh (d[i]);
    }

    activeVoices.store (engine.numActiveVoices(), std::memory_order_relaxed);
    sustainDown.store (engine.isSustainDown(), std::memory_order_relaxed);
}

void DDSPianoAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    if (auto xml = apvts.copyState().createXml())
        copyXmlToBinary (*xml, destData);
}

void DDSPianoAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    if (auto xml = getXmlFromBinary (data, sizeInBytes))
        if (xml->hasTagName (apvts.state.getType()))
            apvts.replaceState (juce::ValueTree::fromXml (*xml));
}

juce::AudioProcessorEditor* DDSPianoAudioProcessor::createEditor() { return new DDSPianoAudioProcessorEditor (*this); }
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new DDSPianoAudioProcessor(); }
