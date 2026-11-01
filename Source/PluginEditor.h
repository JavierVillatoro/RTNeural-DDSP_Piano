#pragma once
#include <JuceHeader.h>
#include "PluginProcessor.h"

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

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DDSPianoAudioProcessorEditor)
};
