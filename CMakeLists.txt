cmake_minimum_required(VERSION 3.22)

# ==============================================================================
# DDSPiano - piano DDSP en tiempo real (JUCE + RTNeural). Windows / macOS / Linux.
#
#   Configurar y compilar (cualquier sistema):
#       cmake -B build -DCMAKE_BUILD_TYPE=Release          (en Windows: cmake -B build)
#       cmake --build build --config Release
#   Solo el renderizador offline + prueba de oro (sin JUCE, compila en 1 min):
#       cmake -B build_tools -DDDSP_BUILD_PLUGIN=OFF
#       cmake --build build_tools --config Release --target ddsp_render
# ==============================================================================

# macOS: binario universal (Apple Silicon + Intel). Debe ir ANTES de project().
set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "Arquitecturas de macOS")
set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Version minima de macOS")

project(DDSPiano VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(DDSP_BUILD_PLUGIN "Compilar el plugin JUCE (VST3/AU/Standalone)" ON)
option(DDSP_BUILD_TOOLS  "Compilar ddsp_render (renderizador offline y prueba de oro)" ON)
option(DDSP_ENABLE_AVX2  "Usar AVX2+FMA en PCs x86-64 (Intel desde 2013 / AMD desde 2015). Apagalo para CPUs muy antiguas" ON)

# ==============================================================================
# 1. RTNeural (backend Eigen)
# ==============================================================================
set(RTNEURAL_EIGEN ON  CACHE BOOL "Usa Eigen backend" FORCE)
set(RTNEURAL_XSIMD OFF CACHE BOOL "Usa XSIMD backend" FORCE)
set(RTNEURAL_STL   OFF CACHE BOOL "Usa STL backend"   FORCE)
add_subdirectory(modules/RTNeural)

# ==============================================================================
# 2. Motor DDSP (C++ puro, sin JUCE): lo usan el plugin y las herramientas
# ==============================================================================
add_library(ddsp_engine STATIC Source/DDSPEngine.cpp)
target_include_directories(ddsp_engine PUBLIC Source)
target_link_libraries(ddsp_engine PUBLIC RTNeural)
target_compile_features(ddsp_engine PUBLIC cxx_std_17)
# Eigen: permite las matrices grandes de la GRU como miembros de clase
target_compile_definitions(ddsp_engine PUBLIC EIGEN_STACK_ALLOCATION_LIMIT=0)
set_target_properties(ddsp_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)
if (MSVC)
    target_compile_options(ddsp_engine PRIVATE /W3 /fp:precise)
else()
    target_compile_options(ddsp_engine PRIVATE -Wall -Wextra -O3)
endif()
# SIMD: el coste dominante son las GRU (RTNeural/Eigen) y los osciladores; AVX2+FMA los acelera mucho.
# En macOS (ARM64 usa NEON siempre; el slice x86_64 se queda en SSE) no se toca nada.
if (DDSP_ENABLE_AVX2 AND NOT APPLE)
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _ddsp_arch)
    if (_ddsp_arch MATCHES "amd64|x86_64|x64")
        if (MSVC)
            target_compile_options(ddsp_engine PRIVATE /arch:AVX2)
        else()
            target_compile_options(ddsp_engine PRIVATE -mavx2 -mfma)
        endif()
    endif()
endif()

if (DDSP_BUILD_TOOLS)
    add_executable(ddsp_render Tools/ddsp_render.cpp)
    target_link_libraries(ddsp_render PRIVATE ddsp_engine)
endif()

# ==============================================================================
# 3. Plugin JUCE
# ==============================================================================
if (DDSP_BUILD_PLUGIN)
    add_subdirectory(modules/JUCE)

    juce_add_plugin(DDSPiano
        IS_SYNTH TRUE
        NEEDS_MIDI_INPUT TRUE
        NEEDS_MIDI_OUTPUT FALSE
        IS_MIDI_EFFECT FALSE
        EDITOR_WANTS_KEYBOARD_FOCUS FALSE
        COPY_PLUGIN_AFTER_BUILD FALSE
        FORMATS VST3 AU Standalone
        VST3_CATEGORIES Instrument Synth
        AU_MAIN_TYPE kAudioUnitType_MusicDevice
        COMPANY_NAME "Javier Villatoro"
        PRODUCT_NAME "DDSPiano"
        PLUGIN_CODE "Ddsp"
        PLUGIN_MANUFACTURER_CODE "Jvil"
        VST3_AUTO_ASSIGN_VST2_IDS FALSE)

    juce_generate_juce_header(DDSPiano)

    # Pesos y respuesta al impulso EMBEBIDOS en el binario: el plugin no depende de ninguna ruta del disco
    juce_add_binary_data(DDSPianoData SOURCES
        Resources/ddsp_weights_v6.bin
        Resources/reverb_ir_v6.wav
        Resources/ddsp_weights_v8.bin
        Resources/reverb_ir_v8.wav)
    set_target_properties(DDSPianoData PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_sources(DDSPiano PRIVATE
        Source/PluginProcessor.cpp
        Source/PluginEditor.cpp)

    target_compile_definitions(DDSPiano PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0)

    target_link_libraries(DDSPiano
        PRIVATE
            DDSPianoData
            ddsp_engine
            juce::juce_audio_utils
            juce::juce_dsp
        PUBLIC
            juce::juce_recommended_config_flags)

    if (MSVC)
        target_compile_options(DDSPiano PRIVATE /W3)
    endif()
endif()
