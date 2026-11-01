# Herramientas de verificación (motor C++ ↔ modelo Python)

`ddsp_render` usa **el mismo motor (`Source/DDSPEngine.*`) que el plugin**, sin JUCE ni DAW.

```
cmake -B build_tools -DDDSP_BUILD_PLUGIN=OFF
cmake --build build_tools --config Release --target ddsp_render
```

## 1. Prueba de oro: ¿los pesos están sincronizados con RTNeural?
```
# en DDSP/training (Python):
python make_golden.py --h5 checkpoints_ddsp8 --out ../../DDSPiano/Resources/golden_v8.bin
python export_binary.py --h5 checkpoints_ddsp8 --out ../../DDSPiano/Resources

# en DDSPiano:
build_tools/Release/ddsp_render golden Resources/golden_v8.bin Resources/ddsp_weights_v8.bin
```
Debe imprimir `[OK]` con errores ≈ 1e-5 para `context`, `core RT` (RTNeural) y `core fast` (la ruta que usa el plugin).
Si falla, el `.bin` no corresponde a la red (orden de pesos, activaciones, tamaños).

## 2. Comparar audio con Python (mismo MIDI)
```
# Python (referencia, seco). Semántica v2 (training_kaggle_8):
python inference_07.py --midi Mompou.mid --h5 checkpoints_ddsp8 --seconds 20 --events --out tmp/v8
# Semántica v1 (training_kaggle_6): python reference_render.py --midi Mompou.mid --h5 checkpoints_descargados_06_2 --seconds 20 --out tmp/ref

# C++ (mismo MIDI, a 16 kHz para comparar muestra a muestra):
build_tools/Release/ddsp_render render tmp/v8_events.txt Resources/ddsp_weights_v8.bin tmp/v8_cpp.wav 16000

python compare_renders.py tmp/v8.wav tmp/v8_cpp.wav [--baseline otro_render_python_con_otra_semilla.wav]
```
Referencia de lo que es "bien": nivel ≤ ±0,5 dB, correlación de envolvente > 0,93 y distancia espectral
a ~1 dB del suelo debido solo al ruido aleatorio (`--baseline`).

## 3. Pesos nuevos de Kaggle → plugin
1. Descomprime `pesos_ddsp8.zip` en `DDSP/training/checkpoints_ddsp8/`.
2. `python export_binary.py --h5 checkpoints_ddsp8 --out ../../DDSPiano/Resources` (genera `ddsp_weights.bin` y `reverb_ir.wav`).
3. `python make_golden.py ...` y la prueba de oro (punto 1).
4. Recompila el plugin: `cmake --build build --config Release`.

Los pesos van **embebidos** en el binario (`juce_add_binary_data`): el plugin no lee nada del disco.

## Modelos embebidos en el plugin
El plugin lleva dos modelos y un selector en la ventana: **v6** (entrenamiento anterior, `*_v6.*`) y **v8**
(pedal real y notas largas, `*_v8.*`). Para añadir uno nuevo: exportar con `export_binary.py --h5 ... --out <tmp>`,
copiar `ddsp_weights.bin` y `reverb_ir.wav` a `Resources/` con el sufijo de la version, y añadirlo a
`juce_add_binary_data` (CMakeLists.txt) y a `kModelData` (PluginProcessor.cpp).
