#!/usr/bin/env python
"""Genera Source/MechanicsTables.h: niveles relativos de los ruidos mecanicos (suelta de tecla y pedal) respecto a las notas.

  python Tools/make_mechanics_tables.py --salamander <carpeta 48khz24bit de Salamander>

Calibracion (solo NUMEROS, no se incrusta audio de Salamander; licencia CC BY 3.0, Alexander Holm):
  * nivel de cada ruido = RMS de sus 100 ms mas fuertes (rel1..rel88.wav, pedalD*.wav, pedalU*.wav);
  * nivel de la nota de referencia = RMS 0,15-0,65 s tras el ataque de la capa v8 (velocity 57-64 ~ 0,5) de las 30 notas
    de Salamander, interpolado a las 88 teclas;
  * mezcla DEL AUTOR (SalamanderGrandPianoV3.sfz): los rel* se tocan con volume=-37 dB y amp_veltrack=82 (a velocity 0,5 =
    0,18 + 0,82 * 0,25 = -8,3 dB); pedalD con -20 dB y pedalU con -19 dB;
  * 'nivel por tecla' del DDSP: RMS 0,15-0,65 s de cada tecla a velocity 0,5 (humanize y unison a 0) con ddsp_render.
En el motor: nivel del ruido = nivel_DDSP(tecla, vel 0,5) + cociente medido en Salamander.
"""
import argparse, os, subprocess, tempfile
import numpy as np, soundfile as sf
import scipy.io.wavfile as wav

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
SA_NOTES = [21 + 3 * i for i in range(30)]                       # A0, C1, D#1, ... C8
NAMES = ["A", "C", "D#", "F#"]


def sa_name(midi):
    # MIDI 21 = A0, 24 = C1 ... (4 notas por octava: A, C, D#, F#)
    i = (midi - 21) // 3
    octave, k = divmod(i + 0, 4)           # A0 | C1 D#1 F#1 A1 | C2 ...
    # la secuencia real es A0, C1, D#1, F#1, A1, C2, ...: indice 0 = A0, luego grupos de 4 empezando en C
    if i == 0: return "A0"
    j = i - 1
    return f"{['C', 'D#', 'F#', 'A'][j % 4]}{1 + j // 4}"


def mono(path):
    x, sr = sf.read(path, always_2d=True)
    return x.mean(axis=1), sr


def env10(x, sr):
    n = int(0.01 * sr); m = len(x) // n
    return 10 * np.log10(np.mean(x[:m * n].reshape(m, n) ** 2, axis=1) + 1e-24)


def note_rms(path):
    x, sr = mono(path); e = env10(x, sr)
    i0 = int(np.argmax(e > e.max() - 20)); a = int((i0 * 0.01 + 0.15) * sr)
    return 10 * np.log10(np.mean(x[a:a + int(0.5 * sr)] ** 2) + 1e-24)


def noise_rms100(path):
    x, sr = mono(path); e = env10(x, sr)
    i0 = int(np.argmax(e > e.max() - 12)); a = max(0, int((i0 * 0.01 - 0.01) * sr))
    return 10 * np.log10(np.mean(x[a:a + int(0.1 * sr)] ** 2) + 1e-24)


def ddsp_key_levels(exe, weights, tmp):
    keys = list(range(21, 109)); SLOT = 2.0
    ev = "".join(f"on {i * SLOT + 0.1:.3f} {k} 0.5\noff {i * SLOT + 1.2:.3f} {k}\n" for i, k in enumerate(keys)) + f"end {len(keys) * SLOT}\n"
    open(os.path.join(tmp, "e.txt"), "w").write(ev)
    subprocess.run([exe, "render", os.path.join(tmp, "e.txt"), weights, os.path.join(tmp, "o.wav"), "16000", "0", "512", "1", "0", "0", "0"], check=True, capture_output=True)
    x = wav.read(os.path.join(tmp, "o.wav"))[1].astype(np.float64)
    return [10 * np.log10(np.mean(x[int((i * SLOT + 0.25) * 16000):int((i * SLOT + 0.75) * 16000)] ** 2) + 1e-24) for i in range(len(keys))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salamander", required=True)
    ap.add_argument("--exe", default=os.path.join(ROOT, "build_tools", "Release", "ddsp_render.exe"))
    a = ap.parse_args()
    D = a.salamander
    sa = np.array([note_rms(os.path.join(D, f"{sa_name(m)}v8.wav")) for m in SA_NOTES])
    sa88 = np.interp(np.arange(21, 109), SA_NOTES, sa)
    rel = np.array([noise_rms100(os.path.join(D, f"rel{n}.wav")) for n in range(1, 89)])
    VELTRACK = 20 * np.log10(0.18 + 0.82 * 0.5 ** 2)            # amp_veltrack=82 a velocity 0,5
    t_rel = rel - 37.0 + VELTRACK - sa88                         # nivel de la suelta respecto a una nota de velocity 0,5 (dB)
    ref60 = sa88[60 - 21]
    ped_d = np.mean([noise_rms100(os.path.join(D, f"pedalD{i}.wav")) for i in (1, 2)]) - 20.0 - ref60
    ped_u = np.mean([noise_rms100(os.path.join(D, f"pedalU{i}.wav")) for i in (1, 2)]) - 19.0 - ref60
    with tempfile.TemporaryDirectory() as tmp:
        lv6 = ddsp_key_levels(a.exe, os.path.join(ROOT, "Resources", "ddsp_weights_v6.bin"), tmp)
        lv8 = ddsp_key_levels(a.exe, os.path.join(ROOT, "Resources", "ddsp_weights_v8.bin"), tmp)

    def arr(name, v):
        body = ",\n    ".join(", ".join(f"{x:.2f}f" for x in v[i:i + 8]) for i in range(0, len(v), 8))
        return f"inline constexpr float {name}[88] = {{\n    {body} }};\n"
    out = ("#pragma once\n// GENERADO por Tools/make_mechanics_tables.py. No editar a mano.\n"
           "// Niveles (dB) de los ruidos mecanicos respecto a una nota de velocity 0,5 (calibracion sobre Salamander V3, CC BY 3.0,\n"
           "// Alexander Holm: solo numeros) y nivel RMS por tecla de cada modelo DDSP a velocity 0,5.\nnamespace ddsp\n{\n"
           + arr("kKeyLevelDbV6", lv6) + arr("kKeyLevelDbV8", lv8) + arr("kReleaseRatioDb", t_rel)
           + f"inline constexpr float kPedalDownRatioDb = {ped_d:.2f}f;   // respecto a una nota MIDI 60 a velocity 0,5\n"
           + f"inline constexpr float kPedalUpRatioDb = {ped_u:.2f}f;\n}}\n")
    open(os.path.join(ROOT, "Source", "MechanicsTables.h"), "w", encoding="utf-8").write(out)
    print(f"Salamander: nota ref (v8) {sa.min():.1f}..{sa.max():.1f} dBFS | rel RMS100 {rel.min():.1f}..{rel.max():.1f} dBFS")
    print("suelta de tecla respecto a una nota de vel 0,5 (dB), cada 8 teclas desde A0: " + " ".join(f"{x:.1f}" for x in t_rel[::8]))
    print(f"pedal abajo {ped_d:.1f} dB, pedal arriba {ped_u:.1f} dB (respecto a nota 60 a vel 0,5)")
    print("DDSP v8 nivel por tecla a vel 0,5 (dBFS), cada 8 teclas: " + " ".join(f"{x:.1f}" for x in lv8[::8]))
    print("escrito Source/MechanicsTables.h")


if __name__ == "__main__":
    main()
