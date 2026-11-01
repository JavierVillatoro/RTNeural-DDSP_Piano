#pragma once

#include <JuceHeader.h>
#include <array>
#include <atomic>
#include <memory>
#include <vector>

#include "DDSPEngine.h"

// ==============================================================================
// Procesador del plugin: solo traduce MIDI/parametros de JUCE al motor ddsp::Engine
// (que es C++ puro y se prueba aparte con la herramienta ddsp_render).
//
// Lleva DOS modelos embebidos para compararlos tocando: 0 = v6 (entrenamiento anterior),
// 1 = v8 (pedal real y notas largas). Cada uno tiene su motor y su reverb aprendida.
// ==============================================================================
class DDSPianoAudioProcessor : public juce::AudioProcessor
{
public:
    static constexpr int kModels = 2;

    DDSPianoAudioProcessor();
    ~DDSPianoAudioProcessor() override = default;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "DDSPiano"; }
    bool acceptsMidi() const override { return true; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 2.5; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    juce::AudioProcessorValueTreeState apvts;

    // Para probar sin teclado MIDI ni pedal: teclado en pantalla/ordenador y sustain virtual
    juce::MidiKeyboardState keyboardState;
    std::atomic<bool> virtualSustain { false };

    // Estado para el editor (se lee desde el hilo de GUI)
    std::atomic<int>  activeVoices { 0 };
    std::atomic<bool> sustainDown { false };
    std::atomic<int>  activeModelUI { 1 };

    /** Modelo elegido en el selector (0 = v6, 1 = v8). */
    int getSelectedModel() const noexcept;
    void setSelectedModel (int index);

    bool modelLoaded() const noexcept { return loaded[(size_t) activeModelUI.load()]; }
    juce::String modelInfo() const { return infos[(size_t) activeModelUI.load()]; }

private:
    static juce::AudioProcessorValueTreeState::ParameterLayout createLayout();

    std::array<std::unique_ptr<ddsp::Engine>, kModels> engines;
    std::array<bool, kModels> loaded {};
    std::array<juce::String, kModels> infos;

    std::array<juce::dsp::Convolution, kModels> reverb;
    juce::dsp::DryWetMixer<float> mixer;
    juce::AudioBuffer<float> revBuf, wetSum;
    std::vector<float> mono;

    // solo hilo de audio
    bool lastVirtualSustain = false;
    int  activeModel = 1;
    float lastSustain = 0.f;
    int  fadeTotal = 1, fadeLeft = 0;
    double sampleRateHz = 44100.0;

    std::atomic<float>* pGain = nullptr;
    std::atomic<float>* pReverb = nullptr;
    std::atomic<float>* pVoices = nullptr;
    std::atomic<float>* pModel = nullptr;
    std::atomic<float>* pNoise = nullptr;
    std::atomic<float>* pNoiseExt = nullptr;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DDSPianoAudioProcessor)
};
