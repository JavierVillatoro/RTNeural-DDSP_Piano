"""Crea una IR estereo a partir de la IR mono aprendida (reverb_ir_vX.wav).
Canal L = IR original (no se toca el sonido entrenado). Canal R = version con la FASE aleatorizada por tramos
(misma magnitud espectral y misma envolvente temporal, pero decorrelada), mezclada con L en potencia constante:
al principio (sonido directo) R ~ L, y hacia 40 ms R es independiente de L. Asi el centro queda estable y la cola se ensancha."""
import os, numpy as np, soundfile as sf

# Uso: python Tools/make_stereo_ir.py   (lee Resources/reverb_ir_v6.wav y _v8.wav, escribe Resources/*_st.wav)
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Resources")
rng = np.random.default_rng(1234)


def decorrelate(x, frame=512):
    """Fase aleatoria por tramos: STFT con ventana raiz-de-Hann al 50 % (suma de potencia = 1)."""
    hop = frame // 2
    win = np.sqrt(np.hanning(frame + 1)[:-1])
    pad = np.concatenate([np.zeros(hop), x, np.zeros(frame)])
    out = np.zeros_like(pad)
    for s in range(0, len(pad) - frame, hop):
        spec = np.fft.rfft(pad[s:s + frame] * win)
        ph = np.exp(1j * rng.uniform(0, 2 * np.pi, len(spec)))
        ph[0] = 1; ph[-1] = 1                       # DC y Nyquist reales
        out[s:s + frame] += np.fft.irfft(spec * ph, frame) * win
    return out[hop:hop + len(x)]


def build(name, sr_expected=16000):
    x, sr = sf.read(f"{SRC}/{name}.wav", always_2d=False)
    assert x.ndim == 1 and sr == sr_expected, (x.shape, sr)
    n = len(x)
    d = decorrelate(x)
    # igualar la envolvente de energia de la version decorrelada a la original (tramos de 4 ms)
    blk = int(0.004 * sr)
    for s in range(0, n, blk):
        e0 = np.sqrt(np.mean(x[s:s + blk] ** 2) + 1e-20); e1 = np.sqrt(np.mean(d[s:s + blk] ** 2) + 1e-20)
        d[s:s + blk] *= e0 / e1
    t = np.arange(n) / sr
    a = np.clip((t - 0.002) / 0.038, 0, 1)          # 0 hasta 2 ms, 1 a partir de 40 ms
    r = np.cos(a * np.pi / 2) * x + np.sin(a * np.pi / 2) * d
    st = np.stack([x, r], axis=1).astype(np.float32)
    sf.write(f"{SRC}/{name}_st.wav", st, sr, subtype='FLOAT')
    return x, r, sr


def coherence_report(name, x, r, sr):
    print(f"\n{name}: energia L = {np.sum(x ** 2):.4e}  R = {np.sum(r ** 2):.4e}  (R/L = {10 * np.log10(np.sum(r ** 2) / np.sum(x ** 2)):.2f} dB)")
    print("   tramo (ms)   | correlacion L/R | nivel R-L (dB)")
    for lo, hi in ((0, 2), (2, 10), (10, 20), (20, 40), (40, 80), (80, 200), (200, 600), (600, 1500)):
        a, b = int(lo * sr / 1000), int(hi * sr / 1000)
        l, rr = x[a:b], r[a:b]
        if np.sum(l ** 2) < 1e-14:
            continue
        c = np.sum(l * rr) / np.sqrt(np.sum(l ** 2) * np.sum(rr ** 2) + 1e-30)
        print(f"   {lo:4d}-{hi:4d}    | {c:15.3f} | {10 * np.log10((np.sum(rr ** 2) + 1e-30) / (np.sum(l ** 2) + 1e-30)):13.2f}")
    # magnitud espectral por bandas (suavizada): R debe parecerse a L
    n = len(x); f = np.fft.rfftfreq(n, 1 / sr)
    XL, XR = np.abs(np.fft.rfft(x)) ** 2, np.abs(np.fft.rfft(r)) ** 2
    print("   banda (Hz)   | nivel R-L (dB)")
    for lo, hi in ((50, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 7500)):
        m = (f >= lo) & (f < hi)
        print(f"   {lo:4d}-{hi:4d}    | {10 * np.log10(XR[m].sum() / XL[m].sum()):13.2f}")


for nm in ("reverb_ir_v6", "reverb_ir_v8"):
    x, r, sr = build(nm)
    coherence_report(nm, x, r, sr)
