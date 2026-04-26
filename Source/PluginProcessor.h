#pragma once

#include <JuceHeader.h>
#include <RTNeural/RTNeural.h>
#include <array>
#include <cmath>

// ==============================================================================
// 1. ARQUITECTURA: CEREBRO GLOBAL (Context Network)
// ==============================================================================
using RedContexto = RTNeural::ModelT<double, 3, 32,
    RTNeural::DenseT<double, 3, 32>,
    RTNeural::GRULayerT<double, 32, 64>,
    RTNeural::DenseT<double, 64, 32>
>;

// ==============================================================================
// 2. ARQUITECTURA: VOCES POLIFÓNICAS (DDSP Core)
// ==============================================================================
using TroncoDDSP = RTNeural::ModelT<double, 34, 192,
    RTNeural::DenseT<double, 34, 128>,
    RTNeural::GRULayerT<double, 128, 192>,
    RTNeural::DenseT<double, 192, 192>
>;

// ENVOLVEMOS LAS CABEZAS EN ModelT PARA DARLES MEMORIA INTERNA
using CapaAmplitud = RTNeural::ModelT<double, 192, 1, RTNeural::DenseT<double, 192, 1>>;
using CapaArmonicos = RTNeural::ModelT<double, 192, 96, RTNeural::DenseT<double, 192, 96>>;
using CapaRuido = RTNeural::ModelT<double, 192, 64, RTNeural::DenseT<double, 192, 64>>;
using RedDetuner = RTNeural::ModelT<double, 1, 1, RTNeural::DenseT<double, 1, 1>>;

// ==============================================================================

class DDSPianoAudioProcessor  : public juce::AudioProcessor
{
public:
    DDSPianoAudioProcessor();
    ~DDSPianoAudioProcessor() override;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }
    const juce::String getName() const override { return "DDSPiano"; }
    bool acceptsMidi() const override { return true; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int index) override {}
    const juce::String getProgramName (int index) override { return {}; }
    void changeProgramName (int index, const juce::String& newName) override {}
    void getStateInformation (juce::MemoryBlock& destData) override {}
    void setStateInformation (const void* data, int sizeInBytes) override {}
    
private:
    static constexpr int MAX_VOICES = 8; 

    // --- REDES NEURONALES ---
    RedContexto redContextoGlobal;
    std::array<TroncoDDSP, MAX_VOICES> vocesTronco;
    std::array<CapaAmplitud, MAX_VOICES> vocesAmp;
    std::array<CapaArmonicos, MAX_VOICES> vocesHarm;
    std::array<CapaRuido, MAX_VOICES> vocesNoise;
    std::array<RedDetuner, MAX_VOICES> vocesDetuner;
    
    // --- VARIABLES DE SINCRONIZACIÓN Y SÍNTESIS ---
    double muestrasPorFrameNeuronal = 0.0;
    double acumuladorFase = 0.0;
    double sampleRateC = 44100.0;

    std::array<int, MAX_VOICES> voiceMidiNote;
    std::array<double, MAX_VOICES> voiceVelocity;
    
    double globalPedal = 0.0;
    std::array<double, MAX_VOICES> currentAmp;
    std::array<double, MAX_VOICES> lastAmp;
    std::array<std::array<double, 96>, MAX_VOICES> currentHarmonics;

    // --- VARIABLES PARA INTERPOLACIÓN Y DETUNE ---
    std::array<double, MAX_VOICES> targetAmp;
    std::array<std::array<double, 96>, MAX_VOICES> targetHarmonics;
    std::array<std::array<double, 96>, MAX_VOICES> lastHarmonics;
    std::array<std::array<double, 96>, MAX_VOICES> oscPhases; 
    std::array<std::array<double, 96>, MAX_VOICES> oscPhases2;

    // --- VARIABLES PARA RUIDO FILTRADO ---
    std::array<std::array<double, 64>, MAX_VOICES> currentNoiseMags;
    juce::Random randomSource;
    std::array<double, MAX_VOICES> noiseFilterState;

    // --- MÁQUINA DE ESTADOS MIDI ---
    enum class VoiceState { Off, NoteOn, Releasing };
    std::array<VoiceState, MAX_VOICES> voiceState;

    // --- FÍSICA Y ESPACIO (INHARMONICITY & REVERB) ---
    double alpha_B = 0.0, beta_B = 0.0, alpha_T = 0.0, beta_T = 0.0;
    juce::dsp::Convolution reverb;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DDSPianoAudioProcessor)
};