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
      noiseAtt (p.apvts, "noise", noise), noiseExtAtt (p.apvts, "noiseext", noiseExt),
      curveView (p),
      touchAtt (p.apvts, "touch", touch), velMaxAtt (p.apvts, "velmax", velMax), dynamicsAtt (p.apvts, "dynamics", dynamics),
      velCompAtt (p.apvts, "velcomp", velCompBtn),
      transposeAtt (p.apvts, "transpose", transpose), humanizeAtt (p.apvts, "humanize", humanize), unisonAtt (p.apvts, "unison", unison),
      widthAtt (p.apvts, "width", width), bassTuningAtt (p.apvts, "bassb", bassTuning),
      damperAtt (p.apvts, "damper", damper),
      evennessAtt (p.apvts, "evenness", evenness), toneAtt (p.apvts, "tone", tone),
      hardnessAtt (p.apvts, "hardness", hardness), mechanicsAtt (p.apvts, "mechanics", mechanics)
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
    setupKnob (*this, touch, touchLabel, "Touch (suave - duro)");
    setupKnob (*this, velMax, velMaxLabel, "Vel Max (tu teclado)");
    setupKnob (*this, dynamics, dynamicsLabel, "Dynamics");
    touch.setDoubleClickReturnValue (true, 0.0);
    dynamics.setDoubleClickReturnValue (true, 1.0);
    velMax.setDoubleClickReturnValue (true, 127.0);
    setupKnob (*this, transpose, transposeLabel, "Transpose (semitonos)");
    setupKnob (*this, humanize, humanizeLabel, "Humanize (variacion)");
    setupKnob (*this, unison, unisonLabel, "Unison (2ª cuerda)");
    setupKnob (*this, evenness, evennessLabel, "Evenness (agudos)");
    evenness.setDoubleClickReturnValue (true, 0.5);
    evenness.setTooltip ("Fase 2. Sube el nivel de los agudos altos (MIDI 84+) hacia el de las grabaciones reales (en el DDSP salian ~8 dB bajos). De 36 a 83 no toca nada. 0 = desactivado.");
    setupKnob (*this, tone, toneLabel, "Tone (brillo)");
    tone.setDoubleClickReturnValue (true, 0.0);
    tone.setTooltip ("Fase 2. Brillo global: -6..+6 dB por encima de ~3 kHz. 0 = neutro.");
    setupKnob (*this, hardness, hardnessLabel, "Hardness (martillo)");
    hardness.setDoubleClickReturnValue (true, 0.0);
    hardness.setTooltip ("Fase 2. Timbre de martillo blando (-) o duro (+) manteniendo el volumen de cada nota. 0 = neutro.");
    setupKnob (*this, mechanics, mechanicsLabel, "Mechanics (mecanica)");
    mechanics.setDoubleClickReturnValue (true, 0.0);
    mechanics.setTooltip ("Fase 3. Golpe grave del martillo, golpe al soltar la tecla y ruidos de pedal (sintetizados, sin muestras). 1 = calibrado con grabaciones reales; 0 = apagado.");
    setupKnob (*this, damper, damperLabel, "Damper (al soltar)");
    damper.setDoubleClickReturnValue (true, 1.0);
    damper.setTooltip ("Al soltar la tecla, los armonicos agudos y el ruido mueren mas rapido (como un apagador real). 0 = como antes: queda un brillo metalico que crece al soltar, sobre todo en los graves.");
    setupKnob (*this, bassTuning, bassTuningLabel, "Bass Tuning (graves)");
    bassTuning.setDoubleClickReturnValue (true, 1.0);
    bassTuning.setTooltip ("Inarmonicidad de los graves: 0 = formula original del entrenamiento, 1 = medida en pianos reales (parciales altos mas cerca de su sitio). Afecta a las notas nuevas.");
    setupKnob (*this, width, widthLabel, "Width (estereo)");
    width.setDoubleClickReturnValue (true, 1.0);
    width.setTooltip ("Anchura estereo: graves a la izquierda y agudos a la derecha + reverb estereo. 0 = mono, 1 = normal, >1 = mas ancho.");
    transpose.setDoubleClickReturnValue (true, 0.0);
    transpose.setTooltip ("Mueve todas las notas MIDI N semitonos (no cambia el sonido de cada nota). Fuera del rango del piano (A0-C8) no suenan.");
    humanize.setTooltip ("Cada nota sale un poco distinta: velocity, afinacion y color de armonicos al azar. 0 = desactivado.");
    unison.setTooltip ("Segunda cuerda desafinada unos cents: batido lento como un piano real. 0 = desactivado.");
    addAndMakeVisible (curveView);
    addAndMakeVisible (velCompBtn);
    calibrateBtn.setTooltip ("Toca unas notas lo mas fuerte que puedas y pulsa: Vel Max pasa a ser la velocity mas alta que ha mandado tu teclado.");
    calibrateBtn.onClick = [this]
    {
        if (! processor.calibrateVelMax())
            velInfo.setText ("Toca primero unas notas fuertes en el teclado y vuelve a pulsar", juce::dontSendNotification);
    };
    addAndMakeVisible (calibrateBtn);
    velInfo.setJustificationType (juce::Justification::centredLeft);
    addAndMakeVisible (velInfo);
    status.setJustificationType (juce::Justification::centred);
    addAndMakeVisible (status);
    keyboard.setAvailableRange (36, 96);
    keyboard.setVelocity (0.75f, false);
    keyboard.setKeyWidth (22.0f);
    keyboard.setWantsKeyboardFocus (true);
    addAndMakeVisible (keyboard);
    sustainBtn.onClick = [this] { processor.virtualSustain.store (sustainBtn.getToggleState()); };
    addAndMakeVisible (sustainBtn);
    setSize (940, 650);
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

    const int vin = processor.lastVelIn.load(), peak = processor.peakVelIn.load();
    if (vin > 0 || peak > 0)
        velInfo.setText ("Ultima nota: teclado " + juce::String (vin) + "  ->  modelo " + juce::String (processor.lastVelOut.load())
                             + "     |  pico del teclado: " + juce::String (peak),
                         juce::dontSendNotification);
    curveView.repaint();
}

void VelocityCurveView::paint (juce::Graphics& g)
{
    auto r = getLocalBounds().toFloat().reduced (4.0f);
    g.setColour (juce::Colour (0xff1e2128));
    g.fillRoundedRectangle (r, 6.0f);
    g.setColour (juce::Colour (0xff3a3f4a));
    g.drawRoundedRectangle (r, 6.0f, 1.0f);
    r = r.reduced (8.0f);

    auto toXY = [&] (float vin, float vout)
    { return juce::Point<float> (r.getX() + vin * r.getWidth(), r.getBottom() - vout * r.getHeight()); };

    g.setColour (juce::Colour (0xff3a3f4a));   // cuadricula y diagonal "sin curva"
    for (int i = 1; i < 4; ++i)
    {
        g.drawLine (r.getX() + r.getWidth() * i / 4.0f, r.getY(), r.getX() + r.getWidth() * i / 4.0f, r.getBottom(), 0.5f);
        g.drawLine (r.getX(), r.getY() + r.getHeight() * i / 4.0f, r.getRight(), r.getY() + r.getHeight() * i / 4.0f, 0.5f);
    }
    g.drawLine (toXY (0, 0).x, toXY (0, 0).y, toXY (1, 1).x, toXY (1, 1).y, 1.0f);

    const auto curve = processor.currentVelocityCurve (processor.activeModelUI.load());
    juce::Path path;
    for (int i = 0; i <= 127; ++i)
    {
        const float vin = (float) i / 127.0f;
        const auto pt = toXY (vin, curve.apply (vin));
        if (i == 0) path.startNewSubPath (pt); else path.lineTo (pt);
    }
    g.setColour (juce::Colour (0xff4fc3f7));
    g.strokePath (path, juce::PathStrokeType (2.0f));

    const int vin = processor.lastVelIn.load();
    if (vin > 0)
    {
        const auto pt = toXY ((float) vin / 127.0f, curve.apply ((float) vin / 127.0f));
        g.setColour (juce::Colours::orange);
        g.fillEllipse (pt.x - 4.5f, pt.y - 4.5f, 9.0f, 9.0f);
    }
    g.setColour (juce::Colours::grey);
    g.setFont (juce::FontOptions (10.0f));
    g.drawText ("teclado", r.toNearestInt().removeFromBottom (12), juce::Justification::bottomRight);
    g.drawText ("modelo", r.toNearestInt().removeFromTop (12), juce::Justification::topLeft);
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

    keyboard.setBounds (area.removeFromBottom (80));
    area.removeFromBottom (6);
    auto bottom = area.removeFromBottom (30);
    sustainBtn.setBounds (bottom.removeFromRight (200));
    status.setBounds (bottom);
    area.removeFromBottom (6);
    auto velRow = area.removeFromBottom (30);
    calibrateBtn.setBounds (velRow.removeFromLeft (140));
    velRow.removeFromLeft (8);
    velCompBtn.setBounds (velRow.removeFromLeft (150));
    velInfo.setBounds (velRow);
    area.removeFromBottom (6);

    // A la derecha, la grafica de la curva de velocity; a la izquierda, una rejilla de 3 filas x 6 mandos:
    //   fila 1 = sonido; fila 2 = vida y mecanica (fases 1 y 3); fila 3 = timbre y dinamica (fase 2 + curva de velocity)
    curveView.setBounds (area.removeFromRight (190).removeFromTop (190));
    area.removeFromRight (8);
    struct Cell { juce::Slider* s; juce::Label* l; };
    const Cell grid[3][6] = {
        { { &gain, &gainLabel }, { &room, &roomLabel }, { &width, &widthLabel }, { &noise, &noiseLabel }, { &noiseExt, &noiseExtLabel }, { &voices, &voicesLabel } },
        { { &transpose, &transposeLabel }, { &humanize, &humanizeLabel }, { &unison, &unisonLabel }, { &bassTuning, &bassTuningLabel }, { &damper, &damperLabel }, { &mechanics, &mechanicsLabel } },
        { { &evenness, &evennessLabel }, { &tone, &toneLabel }, { &hardness, &hardnessLabel }, { &touch, &touchLabel }, { &velMax, &velMaxLabel }, { &dynamics, &dynamicsLabel } },
    };
    const int rowH = area.getHeight() / 3, colW = area.getWidth() / 6;
    for (int r = 0; r < 3; ++r)
    {
        auto row = area.removeFromTop (rowH);
        for (int c = 0; c < 6; ++c)
        {
            auto col = row.removeFromLeft (colW);
            grid[r][c].l->setBounds (col.removeFromTop (20));
            grid[r][c].s->setBounds (col);
        }
    }
}
