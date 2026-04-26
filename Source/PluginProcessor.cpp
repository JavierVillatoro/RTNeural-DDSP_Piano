#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <fstream>
#include "../modules/RTNeural/modules/json/json.hpp" 

// ==============================================================================
// UTILIDADES TÉCNICAS
// ==============================================================================
std::vector<std::vector<double>> transposeToRTNeural(const std::vector<double>& flat, int in_size, int out_size) {
    std::vector<std::vector<double>> out(out_size, std::vector<double>(in_size));
    for(int i = 0; i < in_size; ++i) {
        for(int j = 0; j < out_size; ++j) { out[j][i] = flat[i * out_size + j]; }
    }
    return out;
}

std::vector<std::vector<double>> to2DNormal(const std::vector<double>& flat, int rows, int cols) {
    std::vector<std::vector<double>> out(rows, std::vector<double>(cols));
    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) { out[i][j] = flat[i * cols + j]; }
    }
    return out;
}

double modifiedSigmoid(double x) {
    double sig = 1.0 / (1.0 + std::exp(-x));
    return 2.0 * std::pow(sig, 2.302585) + 1e-7; 
}

// ==============================================================================
// CONSTRUCTOR: CARGA TOTAL DE PESOS Y AUDIO IR
// ==============================================================================
DDSPianoAudioProcessor::DDSPianoAudioProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
     : AudioProcessor (BusesProperties().withOutput ("Output", juce::AudioChannelSet::stereo(), true))
#endif
{
    for(int i=0; i<MAX_VOICES; ++i) {
        voiceState[i] = VoiceState::Off;
        smoothAmp[i] = 0.0;
        targetAmp[i] = 0.0;
        noiseFilterState[i] = 0.0;
        for(int h=0; h<96; ++h) { 
            oscPhases[i][h] = 0.0; 
            oscPhases2[i][h] = 0.0; 
            currentHarmonics[i][h] = 0.0;
            targetHarmonics[i][h] = 0.0;
            lastHarmonics[i][h] = 0.0;
        }
    }

    juce::String folderPath = "C:/Users/franc/Desktop/proyectos/DDSP/training/rtneural_export/";

    try {
        // 1. CONTEXTO GLOBAL
        std::ifstream ctxS(folderPath.toStdString() + "context_weights.json");
        if (ctxS.is_open()) {
            nlohmann::json j; ctxS >> j;
            redContextoGlobal.get<0>().setWeights(transposeToRTNeural(j.at("dense_in").at("kernel"), 3, 32));
            redContextoGlobal.get<0>().setBias(j.at("dense_in").at("bias").get<std::vector<double>>().data());
            redContextoGlobal.get<1>().setWVals(to2DNormal(j.at("gru").at("kernel"), 32, 192));
            redContextoGlobal.get<1>().setUVals(to2DNormal(j.at("gru").at("recurrent_kernel"), 64, 192));
            redContextoGlobal.get<2>().setWeights(transposeToRTNeural(j.at("dense_out").at("kernel"), 64, 32));
            redContextoGlobal.get<2>().setBias(j.at("dense_out").at("bias").get<std::vector<double>>().data());
        }

        // 2. VOCES DDSP (CORE)
        std::ifstream coreS(folderPath.toStdString() + "ddsp_core_weights.json");
        if (coreS.is_open()) {
            nlohmann::json j; coreS >> j;
            auto wIn = transposeToRTNeural(j.at("dense_in").at("kernel"), 34, 128);
            auto bIn = j.at("dense_in").at("bias").get<std::vector<double>>();
            auto wGru = to2DNormal(j.at("gru").at("kernel"), 128, 576);
            auto uGru = to2DNormal(j.at("gru").at("recurrent_kernel"), 192, 576);
            auto bGru = j.at("gru").at("bias").get<std::vector<double>>();
            std::vector<std::vector<double>> bGru2D = { std::vector<double>(bGru.begin(), bGru.begin()+576) };
            if(bGru.size() > 576) bGru2D.push_back(std::vector<double>(bGru.begin()+576, bGru.end()));
            auto wHid = transposeToRTNeural(j.at("dense_hidden").at("kernel"), 192, 192);
            auto bHid = j.at("dense_hidden").at("bias").get<std::vector<double>>();
            auto wAmp = transposeToRTNeural(j.at("amp_out").at("kernel"), 192, 1);
            auto bAmp = j.at("amp_out").at("bias").get<std::vector<double>>();
            auto wHarm = transposeToRTNeural(j.at("harm_out").at("kernel"), 192, 96);
            auto bHarm = j.at("harm_out").at("bias").get<std::vector<double>>();
            auto wNoise = transposeToRTNeural(j.at("noise_out").at("kernel"), 192, 64);
            auto bNoise = j.at("noise_out").at("bias").get<std::vector<double>>();

            for (int v = 0; v < MAX_VOICES; ++v) {
                vocesTronco[v].get<0>().setWeights(wIn);
                vocesTronco[v].get<0>().setBias(bIn.data());
                vocesTronco[v].get<1>().setWVals(wGru);
                vocesTronco[v].get<1>().setUVals(uGru);
                vocesTronco[v].get<1>().setBVals(bGru2D);
                vocesTronco[v].get<2>().setWeights(wHid);
                vocesTronco[v].get<2>().setBias(bHid.data());
                vocesAmp[v].get<0>().setWeights(wAmp);
                vocesAmp[v].get<0>().setBias(bAmp.data());
                vocesHarm[v].get<0>().setWeights(wHarm);
                vocesHarm[v].get<0>().setBias(bHarm.data());
                vocesNoise[v].get<0>().setWeights(wNoise);
                vocesNoise[v].get<0>().setBias(bNoise.data());
            }
        }

        // 3. DETUNER
        std::ifstream detS(folderPath.toStdString() + "detuner_weights.json");
        if (detS.is_open()) {
            nlohmann::json j; detS >> j;
            auto wDet = transposeToRTNeural(j.at("dense").at("kernel"), 1, 1);
            for (int v = 0; v < MAX_VOICES; ++v) vocesDetuner[v].get<0>().setWeights(wDet);
        }

        // 4. INHARMONICITY (Pesos reales)
        std::ifstream inharmStream(folderPath.toStdString() + "inharmonicity_weights.json");
        if (inharmStream.is_open()) {
            nlohmann::json j; inharmStream >> j;
            alpha_B = j["alpha_B"]; beta_B = j["beta_B"];
            alpha_T = j["alpha_T"]; beta_T = j["beta_T"];
        }

        // 5. REVERB IR (Convolución)
        juce::File irFile(folderPath + "reverb_ir_44100.wav"); //Probar con reverb_ir_16k
        if (irFile.existsAsFile()) {
            reverb.loadImpulseResponse(irFile, juce::dsp::Convolution::Stereo::yes, juce::dsp::Convolution::Trim::yes, 0);
        }

    } catch(...) { juce::Logger::writeToLog("Error critico en carga de JSON/WAV"); }
}

DDSPianoAudioProcessor::~DDSPianoAudioProcessor() {}

void DDSPianoAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock) {
    sampleRateC = sampleRate;
    muestrasPorFrameNeuronal = 250.0 / sampleRate;
    acumuladorFase = 0.0;
    redContextoGlobal.reset();
    
    for (int v = 0; v < MAX_VOICES; ++v) {
        vocesTronco[v].reset();
        noiseFilterState[v] = 0.0;
    }
    
    // Preparar el motor de convolución (Reverb)
    juce::dsp::ProcessSpec spec;
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = samplesPerBlock;
    spec.numChannels = getTotalNumOutputChannels();
    reverb.prepare(spec);
}

void DDSPianoAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    const int numSamples = buffer.getNumSamples();
    auto* channelDataL = buffer.getWritePointer(0);
    auto* channelDataR = buffer.getWritePointer(1);

    // 1. GESTIÓN MIDI CON ESTADOS
    for (const auto metadata : midiMessages) {
        auto msg = metadata.getMessage();
        if (msg.isNoteOn()) {
            for (int v = 0; v < MAX_VOICES; ++v) {
                if (voiceState[v] == VoiceState::Off) {
                    voiceMidiNote[v] = msg.getNoteNumber();
                    voiceVelocity[v] = msg.getFloatVelocity();
                    voiceState[v] = VoiceState::NoteOn;
                    vocesTronco[v].reset(); 
                    break;
                }
            }
        } else if (msg.isNoteOff()) {
            for (int v = 0; v < MAX_VOICES; ++v) {
                if (voiceState[v] == VoiceState::NoteOn && voiceMidiNote[v] == msg.getNoteNumber()) {
                    voiceState[v] = VoiceState::Releasing; // Nota soltada
                }
            }
        } else if (msg.isController() && msg.getControllerNumber() == 64) {
            globalPedal = msg.getControllerValue() / 127.0;
        }
    }

    // 2. BUCLE PRINCIPAL DE SÍNTESIS
    for (int i = 0; i < numSamples; ++i) {
        acumuladorFase += muestrasPorFrameNeuronal;
        
        if (acumuladorFase >= 1.0) {
            acumuladorFase -= 1.0;
            
            double sV=0, sP=0; int cnt=0;
            for(int v=0; v<MAX_VOICES; v++) {
                if(voiceState[v] != VoiceState::Off) { 
                    sV += voiceVelocity[v]; sP += (voiceMidiNote[v]/127.0); cnt++; 
                }
            }
            alignas(32) double cIn[3] = {globalPedal, sV, (cnt>0?sP/cnt:0.0)};
            redContextoGlobal.forward(cIn);
            const double* cOut = redContextoGlobal.getOutputs();

            for (int v = 0; v < MAX_VOICES; ++v) {
                if (voiceState[v] != VoiceState::Off) {
                    lastHarmonics[v] = targetHarmonics[v];

                    alignas(32) double in34[34];
                    in34[0] = voiceMidiNote[v]/127.0; 
                    
                    // Decaimiento natural al soltar la tecla
                    if (voiceState[v] == VoiceState::Releasing) voiceVelocity[v] *= 0.95; 
                    
                    in34[1] = voiceVelocity[v];
                    for(int c=0; c<32; c++) in34[2+c] = cOut[c];

                    vocesTronco[v].forward(in34);
                    const double* tOut = vocesTronco[v].getOutputs();

                    vocesAmp[v].forward(tOut);
                    targetAmp[v] = modifiedSigmoid(vocesAmp[v].getOutputs()[0]);

                    // Kill switch si la nota ya se apagó
                    if (voiceState[v] == VoiceState::Releasing && targetAmp[v] < 0.001) {
                        voiceState[v] = VoiceState::Off;
                        targetAmp[v] = 0.0;
                    }

                    vocesHarm[v].forward(tOut);
                    const double* rH = vocesHarm[v].getOutputs();
                    double sH = 0;
                    for(int h=0; h<96; h++){ targetHarmonics[v][h] = modifiedSigmoid(rH[h]); sH += targetHarmonics[v][h]; }
                    for(int h=0; h<96; h++) targetHarmonics[v][h] /= (sH + 1e-7);

                    vocesNoise[v].forward(tOut);
                    const double* rN = vocesNoise[v].getOutputs();
                    for(int n=0; n<64; n++) currentNoiseMags[v][n] = modifiedSigmoid(rN[n]);
                }
            }
        }

        double outMono = 0.0;
        double alpha = acumuladorFase;

        for (int v = 0; v < MAX_VOICES; ++v) {
            if (voiceState[v] != VoiceState::Off) {
                smoothAmp[v] += 0.1 * (targetAmp[v] - smoothAmp[v]);

                double pM = (double)voiceMidiNote[v];
                double f0 = 440.0 * std::pow(2.0, (pM - 69.0) / 12.0);
                double B = std::exp(alpha_T * pM + beta_T) + std::exp(alpha_B * pM + beta_B);

                double vAcc = 0;
                for (int k = 1; k <= 96; ++k) {
                    double fk = f0 * k * std::sqrt(1.0 + B*k*k);
                    // Límite de 8kHz (Nyquist de tu modelo en Python)
                    if (fk < 8000.0) {
                        oscPhases[v][k-1] = std::fmod(oscPhases[v][k-1] + (fk*6.283185/sampleRateC), 6.283185);
                        double h = (1.0 - alpha) * lastHarmonics[v][k-1] + alpha * targetHarmonics[v][k-1];
                        vAcc += std::sin(oscPhases[v][k-1]) * h;
                    }
                }
                
                // Ruido Filtrado (IIR)
                double white = randomSource.nextDouble() * 2.0 - 1.0;
                double noiseEnergy = 0.0;
                for(int n=0; n<64; n++) noiseEnergy += currentNoiseMags[v][n];
                
                double cutoff = 0.05; 
                if (currentNoiseMags[v][0] > 0.001) cutoff = currentNoiseMags[v][10] / currentNoiseMags[v][0];
                if (cutoff > 0.8) cutoff = 0.8;
                if (cutoff < 0.01) cutoff = 0.01;

                noiseFilterState[v] += cutoff * (white - noiseFilterState[v]);
                
                // Mezcla de osciladores y ruido
                outMono += (vAcc + noiseFilterState[v] * (noiseEnergy / 64.0) * 2.0) * smoothAmp[v];
            }
        }

        // Ganancia Pre-Reverb
        outMono *= 0.15; 
        outMono = std::tanh(outMono);
        channelDataL[i] = (float)outMono; 
        channelDataR[i] = (float)outMono;
    }

    // --- 3. PROCESAMIENTO DE REVERB (Convolución JUCE) ---
    juce::dsp::AudioBlock<float> block(buffer);
    juce::dsp::ProcessContextReplacing<float> context(block);
    reverb.process(context);

    // Ajuste maestro post-reverb
    buffer.applyGain(0.7f);
}

void DDSPianoAudioProcessor::releaseResources() {}
bool DDSPianoAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const { 
    return layouts.getMainOutputChannelSet() == juce::AudioChannelSet::stereo(); 
}
juce::AudioProcessorEditor* DDSPianoAudioProcessor::createEditor() { return new DDSPianoAudioProcessorEditor (*this); }
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new DDSPianoAudioProcessor(); }