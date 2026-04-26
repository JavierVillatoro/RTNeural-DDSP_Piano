#pragma once
#include <JuceHeader.h>
#include "PluginProcessor.h"

class DDSPianoAudioProcessorEditor  : public juce::AudioProcessorEditor
{
public:
    DDSPianoAudioProcessorEditor (DDSPianoAudioProcessor&);
    ~DDSPianoAudioProcessorEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;

private:
    DDSPianoAudioProcessor& audioProcessor;
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DDSPianoAudioProcessorEditor)
};