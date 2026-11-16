#pragma once
#include <JuceHeader.h>
#include "PluginProcessor.h"

// Grafica de la curva de velocity: eje X = velocity del teclado, eje Y = velocity que recibe el modelo.
// La diagonal es "sin curva"; el punto marca la ultima nota tocada.
class VelocityCurveView : public juce::Component
{
public:
    explicit VelocityCurveView (DDSPianoAudioProcessor& p) : processor (p) {}
    void paint (juce::Graphics&) override;

private:
    DDSPianoAudioProcessor& processor;
};

class DDSPianoAudioProcessorEditor : public juce::AudioProcessorEditor,
                                     private juce::Timer
{
public:
    explicit DDSPianoAudioProcessorEditor (DDSPianoAudioProcessor&);
    ~DDSPianoAudioProcessorEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;

private:
    void timerCallback() override;

    DDSPianoAudioProcessor& processor;

    juce::TextButton v6Btn { "v6  ·  anterior" }, v8Btn { "v8  ·  nuevo" };
    juce::MidiKeyboardComponent keyboard;
    juce::ToggleButton sustainBtn { "Sustain (pedal virtual)" };
    juce::Slider gain, room, voices, noise, noiseExt;
    juce::Label gainLabel, roomLabel, voicesLabel, noiseLabel, noiseExtLabel, status;
    juce::AudioProcessorValueTreeState::SliderAttachment gainAtt, roomAtt, voicesAtt, noiseAtt, noiseExtAtt;

    // Curva de velocity
    VelocityCurveView curveView;
    juce::Slider touch, velMax, dynamics;
    juce::Label touchLabel, velMaxLabel, dynamicsLabel, velInfo;
    juce::ToggleButton velCompBtn { "Compensar modelo" };
    juce::TextButton calibrateBtn { "Calibrar Vel Max" };
    juce::AudioProcessorValueTreeState::SliderAttachment touchAtt, velMaxAtt, dynamicsAtt;
    juce::AudioProcessorValueTreeState::ButtonAttachment velCompAtt;

    // Transposicion y "vida" del sonido
    juce::Slider transpose, humanize, unison, width, bassTuning, damper;
    juce::Label transposeLabel, humanizeLabel, unisonLabel, widthLabel, bassTuningLabel, damperLabel;
    juce::AudioProcessorValueTreeState::SliderAttachment transposeAtt, humanizeAtt, unisonAtt, widthAtt, bassTuningAtt, damperAtt;

    // Fase 2 (uniformidad y timbre) y fase 3 (mecanica)
    juce::Slider evenness, tone, hardness, mechanics;
    juce::Label evennessLabel, toneLabel, hardnessLabel, mechanicsLabel;
    juce::AudioProcessorValueTreeState::SliderAttachment evennessAtt, toneAtt, hardnessAtt, mechanicsAtt;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DDSPianoAudioProcessorEditor)
};
