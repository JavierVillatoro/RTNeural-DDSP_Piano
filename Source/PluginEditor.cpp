#include "PluginEditor.h"

namespace
{
void setupKnob (juce::Component& parent, juce::Slider& s, juce::Label& l, const juce::String& text)
{
    s.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
    s.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 80, 20);
    parent.addAndMakeVisible (s);
    l.setText (text, juce::dontSendNotification);
    l.setJustificationType (juce::Justification::centred);
    parent.addAndMakeVisible (l);
}

void setupModelButton (juce::Component& parent, juce::TextButton& b, juce::Colour on)
{
    b.setClickingTogglesState (true);
    b.setRadioGroupId (4242);
    b.setColour (juce::TextButton::buttonColourId, juce::Colour (0xff2a2e35));
    b.setColour (juce::TextButton::buttonOnColourId, on);
    b.setColour (juce::TextButton::textColourOffId, juce::Colours::lightgrey);
    b.setColour (juce::TextButton::textColourOnId, juce::Colours::white);
    parent.addAndMakeVisible (b);
}
}

DDSPianoAudioProcessorEditor::DDSPianoAudioProcessorEditor (DDSPianoAudioProcessor& p)
    : AudioProcessorEditor (&p), processor (p),
      keyboard (p.keyboardState, juce::MidiKeyboardComponent::horizontalKeyboard),
      gainAtt (p.apvts, "gain", gain), roomAtt (p.apvts, "reverb", room), voicesAtt (p.apvts, "voices", voices),
      noiseAtt (p.apvts, "noise", noise), noiseExtAtt (p.apvts, "noiseext", noiseExt)
{
    setupModelButton (*this, v6Btn, juce::Colour (0xffb8860b));
    setupModelButton (*this, v8Btn, juce::Colour (0xff2e8b57));
    v6Btn.onClick = [this] { if (v6Btn.getToggleState()) processor.setSelectedModel (0); };
    v8Btn.onClick = [this] { if (v8Btn.getToggleState()) processor.setSelectedModel (1); };

    setupKnob (*this, gain, gainLabel, "Gain");
    setupKnob (*this, room, roomLabel, "Room");
    setupKnob (*this, voices, voicesLabel, "Max voices");
    setupKnob (*this, noise, noiseLabel, "Noise");
    setupKnob (*this, noiseExt, noiseExtLabel, "Noise extremos");
    status.setJustificationType (juce::Justification::centred);
    addAndMakeVisible (status);
    keyboard.setAvailableRange (36, 96);
    keyboard.setVelocity (0.75f, false);
    keyboard.setKeyWidth (22.0f);
    keyboard.setWantsKeyboardFocus (true);
    addAndMakeVisible (keyboard);
    sustainBtn.onClick = [this] { processor.virtualSustain.store (sustainBtn.getToggleState()); };
    addAndMakeVisible (sustainBtn);
    setSize (700, 470);
    startTimerHz (10);
    timerCallback();
}

DDSPianoAudioProcessorEditor::~DDSPianoAudioProcessorEditor() = default;

void DDSPianoAudioProcessorEditor::timerCallback()
{
    const int sel = processor.getSelectedModel();
    v6Btn.setToggleState (sel == 0, juce::dontSendNotification);
    v8Btn.setToggleState (sel == 1, juce::dontSendNotification);

    status.setText (processor.modelInfo() + "   |   voces: " + juce::String (processor.activeVoices.load())
                        + "   |   pedal: " + (processor.sustainDown.load() ? "ON" : "off"),
                    juce::dontSendNotification);
    status.setColour (juce::Label::textColourId, processor.modelLoaded() ? juce::Colours::lightgreen : juce::Colours::orangered);
}

void DDSPianoAudioProcessorEditor::paint (juce::Graphics& g)
{
    g.fillAll (juce::Colour (0xff15171c));
    g.setColour (juce::Colours::white);
    g.setFont (juce::FontOptions (22.0f, juce::Font::bold));
    g.drawText ("DDSP-Piano", getLocalBounds().removeFromTop (50), juce::Justification::centred);
}

void DDSPianoAudioProcessorEditor::resized()
{
    auto area = getLocalBounds().reduced (16);
    area.removeFromTop (38);

    auto modelRow = area.removeFromTop (46);
    const int half = (modelRow.getWidth() - 12) / 2;
    v6Btn.setBounds (modelRow.removeFromLeft (half));
    modelRow.removeFromLeft (12);
    v8Btn.setBounds (modelRow);
    area.removeFromTop (10);

    keyboard.setBounds (area.removeFromBottom (96));
    area.removeFromBottom (6);
    auto bottom = area.removeFromBottom (30);
    sustainBtn.setBounds (bottom.removeFromRight (200));
    status.setBounds (bottom);
    const int w = area.getWidth() / 5;
    juce::Slider* sliders[] = { &gain, &room, &voices, &noise, &noiseExt };
    juce::Label* labels[] = { &gainLabel, &roomLabel, &voicesLabel, &noiseLabel, &noiseExtLabel };
    for (int i = 0; i < 5; ++i)
    {
        auto col = area.removeFromLeft (w);
        labels[i]->setBounds (col.removeFromTop (22));
        sliders[i]->setBounds (col);
    }
}
