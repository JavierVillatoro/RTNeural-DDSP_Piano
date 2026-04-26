#include "PluginProcessor.h"
#include "PluginEditor.h"

DDSPianoAudioProcessorEditor::DDSPianoAudioProcessorEditor (DDSPianoAudioProcessor& p)
    : AudioProcessorEditor (&p), audioProcessor (p)
{
    setSize (400, 300);
}

DDSPianoAudioProcessorEditor::~DDSPianoAudioProcessorEditor() {}

void DDSPianoAudioProcessorEditor::paint (juce::Graphics& g)
{
    g.fillAll (juce::Colours::black);
    g.setColour (juce::Colours::white);
    g.setFont (15.0f);
    g.drawFittedText ("DDSP-Piano VST Cargado y Listo!", getLocalBounds(), juce::Justification::centred, 1);
}

void DDSPianoAudioProcessorEditor::resized() {}