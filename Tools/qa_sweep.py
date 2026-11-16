#!/usr/bin/env python
"""QA automatico del motor DDSP: barre las 88 teclas en varias situaciones y busca artefactos con metricas objetivas.

  python Tools/qa_sweep.py [--weights Resources/ddsp_weights_v8.bin] [--label v8] [--legacy] [--set clave=valor ...]

Escenarios (todo a 16 kHz, mono, sin reverb, con los valores por defecto del plugin):
  S1  mantener 1,5 s y soltar, 88 teclas x 3 velocities   -> rebrote tras soltar, cortes bruscos, clics, nivel por tecla
  S2  repetir la misma nota sin soltar                     -> clics
  S3  nota sostenida por el pedal y pedal levantado        -> rebrote / cortes / clics del release por pedal
  S4  40 notas a la vez (robo de voces) y soltar todas      -> clics, NaN, picos
  S5  velocities extremas (0,01 y 1,0)                      -> NaN y monotonia del nivel con la velocity
Umbrales: rebrote > 6 dB, corte > 20 dB (por encima de -90 dBFS), clic = salto 12x mayor que el entorno, salto de nivel
entre teclas vecinas > 4 dB, residuo respecto al nivel suavizado > 3 dB.
--legacy desactiva las correcciones de release (damper, ruido al soltar): es el CONTROL POSITIVO, debe encontrar alertas.
"""
import argparse, datetime, os, subprocess, sys, tempfile
import numpy as np
import scipy.io.wavfile as wav
from scipy.ndimage import uniform_filter1d

SR = 16000
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
KEYS = list(range(21, 109))
BANDS = [(20, 150), (150, 400), (400, 1000), (1000, 2500), (2500, 5000), (5000, 7800)]
THR = dict(regrow=6.0, cut=20.0, jump=4.0, resid=3.0)
FLOOR_DBFS = -90.0   # umbral de audibilidad para 'corte' (antes -100)


def render(exe, weights, ev_text, extra, tmp, name):
    ev, out = os.path.join(tmp, name + ".txt"), os.path.join(tmp, name + ".wav")
    open(ev, "w").write(ev_text)
    subprocess.run([exe, "render", ev, weights, out, str(SR), "0", "512"] + extra, check=True, capture_output=True)
    return wav.read(out)[1].astype(np.float64)


def db(x):
    return 20 * np.log10(np.abs(x) + 1e-12)


def window_rms_db(x, t0, t1, win=0.01):
    n = int(win * SR); a = int(t0 * SR); b = min(int(t1 * SR), len(x))
    seg = x[a:b]; m = len(seg) // n
    if m < 2: return np.array([-200.0])
    return 10 * np.log10(np.mean(seg[:m * n].reshape(m, n) ** 2, axis=1) + 1e-24)


def tail_metrics(x, t_off):
    """Rebrote por banda (dB), corte brusco (dB) y clics en el release que empieza en t_off."""
    regrow = 0.0
    times = np.arange(0.1, 1.2, 0.1)
    for lo, hi in BANDS:
        lv, ok = [], []
        for dt in times:
            a = x[int((t_off + dt) * SR):int((t_off + dt + 0.1) * SR)]
            if len(a) < 1600:
                lv.append(-300.0); ok.append(False); continue
            sp = np.abs(np.fft.rfft(a * np.hanning(len(a)), 4096)) ** 2; f = np.fft.rfftfreq(4096, 1 / SR)
            lv.append(10 * np.log10(sp[(f >= lo) & (f < hi)].sum() + 1e-30))
            ok.append(10 * np.log10(np.mean(a ** 2) + 1e-24) > -100)
        for i in range(1, len(lv)):
            if ok[i] and ok[i - 1]:
                regrow = max(regrow, lv[i] - min(lv[:i]))
    w = window_rms_db(x, t_off + 0.1, t_off + 1.3)   # se ignora la caida inmediata tras soltar (el apagador actuando)
    cut = 0.0
    for i in range(1, len(w)):
        if w[i - 1] > FLOOR_DBFS:   # por debajo de este nivel un corte es inaudible (con +24 dB de ganancia del plugin queda en ~-66 dBFS)
            cut = max(cut, w[i - 1] - w[i])
    tot = [10 * np.log10(np.mean(x[int((t_off + d) * SR):int((t_off + d + 0.1) * SR)] ** 2) + 1e-24) for d in np.arange(0.1, 1.0, 0.1)]
    wob = max([tot[i] - tot[i - 1] for i in range(1, len(tot)) if tot[i - 1] > -100] + [0.0])
    return regrow, cut, wob


def clicks(x, skip):
    """Numero de saltos anomalos (muestra a muestra) fuera de las ventanas 'skip' = [(t0, t1), ...] (ataques)."""
    if len(x) < 100: return 0
    d = np.abs(np.diff(x)); peak = np.abs(x).max() + 1e-12
    env = uniform_filter1d(d, size=161)
    spike = (d > 12 * env + 1e-12) & (d > 2e-4 * peak)
    mask = np.ones(len(d), bool)
    for t0, t1 in skip:
        mask[int(t0 * SR):int(t1 * SR)] = False
    return int(np.sum(spike & mask))


def scenario_s1(exe, weights, extra, tmp, vels):
    SLOT, T_ON, T_OFF = 4.0, 0.2, 1.7
    out = {}
    for v in vels:
        ev = "".join(f"on {i * SLOT + T_ON:.3f} {k} {v}\noff {i * SLOT + T_OFF:.3f} {k}\n" for i, k in enumerate(KEYS)) + f"end {len(KEYS) * SLOT}\n"
        x = render(exe, weights, ev, extra, tmp, f"s1_{v}")
        rows = []
        for i, k in enumerate(KEYS):
            s = x[int(i * SLOT * SR):int((i + 1) * SLOT * SR)]
            rg, ct, wb = tail_metrics(s, T_OFF)
            lvl = 10 * np.log10(np.mean(s[int((T_ON + 0.15) * SR):int((T_ON + 0.65) * SR)] ** 2) + 1e-24)
            rows.append((k, rg, ct, clicks(s, [(T_ON - 0.01, T_ON + 0.03)]), lvl, np.isfinite(s).all(), wb))
        out[v] = rows
    return out


def scenario_s2(exe, weights, extra, tmp):
    SLOT = 3.2
    ev = "".join(f"on {i * SLOT + 0.2:.3f} {k} 0.6\non {i * SLOT + 0.7:.3f} {k} 0.6\noff {i * SLOT + 1.5:.3f} {k}\n" for i, k in enumerate(KEYS)) + f"end {len(KEYS) * SLOT}\n"
    x = render(exe, weights, ev, extra, tmp, "s2")
    bad = []
    for i, k in enumerate(KEYS):
        s = x[int(i * SLOT * SR):int((i + 1) * SLOT * SR)]
        c = clicks(s, [(0.19, 0.23), (0.69, 0.73)])
        if c: bad.append((k, c))
    return bad


def scenario_s3(exe, weights, extra, tmp):
    SLOT = 5.0; ks = KEYS[::3]
    ev = ""
    for i, k in enumerate(ks):
        t0 = i * SLOT
        ev += f"ped {t0:.3f} 1\non {t0 + 0.2:.3f} {k} 0.6\noff {t0 + 1.2:.3f} {k}\nped {t0 + 2.2:.3f} 0\n"
    ev += f"end {len(ks) * SLOT}\n"
    x = render(exe, weights, ev, extra, tmp, "s3")
    rows = []
    for i, k in enumerate(ks):
        s = x[int(i * SLOT * SR):int((i + 1) * SLOT * SR)]
        rg, ct, _ = tail_metrics(s, 2.2)
        rows.append((k, rg, ct, clicks(s, [(0.19, 0.23)])))
    return rows


def scenario_s4(exe, weights, extra, tmp):
    rng = np.random.default_rng(5)
    ks = rng.choice(KEYS, 40, replace=False); vs = rng.uniform(0.3, 0.9, 40)
    ev = "".join(f"on {0.2 + 0.025 * i:.3f} {k} {v:.3f}\noff {3.0 + 0.01 * i:.3f} {k}\n" for i, (k, v) in enumerate(zip(ks, vs))) + "end 6.0\n"
    x = render(exe, weights, ev, extra, tmp, "s4")
    skip = [(0.2 + 0.025 * i - 0.01, 0.2 + 0.025 * i + 0.03) for i in range(40)]
    rg, ct, _ = tail_metrics(x, 3.4)
    return dict(finite=bool(np.isfinite(x).all()), peak=float(db(np.abs(x).max())), clicks=clicks(x, skip), regrow=rg, cut=ct)


def scenario_s5(exe, weights, extra, tmp, lvl_by_v):
    SLOT = 2.0; res = {}
    for v in (0.01, 1.0):
        ev = "".join(f"on {i * SLOT + 0.1:.3f} {k} {v}\noff {i * SLOT + 1.2:.3f} {k}\n" for i, k in enumerate(KEYS)) + f"end {len(KEYS) * SLOT}\n"
        x = render(exe, weights, ev, extra, tmp, f"s5_{v}")
        res[v] = [(k, 10 * np.log10(np.mean(x[int((i * SLOT + 0.25) * SR):int((i * SLOT + 0.75) * SR)] ** 2) + 1e-24)) for i, k in enumerate(KEYS)]
        res[v].append(np.isfinite(x).all())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=os.path.join(ROOT, "build_tools", "Release", "ddsp_render.exe"))
    ap.add_argument("--weights", default=os.path.join(ROOT, "Resources", "ddsp_weights_v8.bin"))
    ap.add_argument("--label", default="v8")
    ap.add_argument("--legacy", action="store_true", help="sin damper ni ruido que decae (control positivo)")
    ap.add_argument("--set", nargs="*", default=[], help="ajustes extra del motor, p. ej. evenness=1 hardness=0.5")
    ap.add_argument("--out", default=os.path.join(HERE, "qa_reports"))
    a = ap.parse_args()
    tau, damp = ("0", "0") if a.legacy else ("0.15", "1")
    # noiseGain noiseExt humanize unison modo bassTuning releaseNoiseTau releaseDamping  (valores por defecto del plugin)
    extra = ["1", "0", "0.4", "0.4", "mono", "1", tau, damp] + a.set
    os.makedirs(a.out, exist_ok=True)
    lines = []
    P = lambda s="": (print(s), lines.append(s))
    with tempfile.TemporaryDirectory() as tmp:
        vels = (0.2, 0.5, 0.85)
        s1 = scenario_s1(a.exe, a.weights, extra, tmp, vels)
        s2 = scenario_s2(a.exe, a.weights, extra, tmp)
        s3 = scenario_s3(a.exe, a.weights, extra, tmp)
        s4 = scenario_s4(a.exe, a.weights, extra, tmp)
        s5 = scenario_s5(a.exe, a.weights, extra, tmp, s1)

    P(f"# QA del motor ({a.label}{' · LEGACY (control positivo)' if a.legacy else ''}) · {datetime.date.today()}")
    P(f"\nAjustes: {' '.join(extra)}\n")
    P("## S1 · mantener y soltar (88 teclas x 3 velocities)")
    alerts = {"rebrote": [], "corte": [], "clics": [], "NaN": []}
    for v, rows in s1.items():
        for k, rg, ct, cl, lvl, fin, wb in rows:
            if rg > THR['regrow']: alerts["rebrote"].append((k, v, rg))
            if ct > THR['cut']: alerts["corte"].append((k, v, ct))
            if cl: alerts["clics"].append((k, v, cl))
            if not fin: alerts["NaN"].append((k, v, 0))
    for name, lst in alerts.items():
        top = sorted(lst, key=lambda t: -t[2])[:6]
        P(f"- **{name}**: {len(lst)} alertas de {len(KEYS) * len(vels)}" + (" · peores (tecla, vel, valor): " + ", ".join(f"({k}, {v}, {x:.1f})" for k, v, x in top) if lst else ""))
    wob = [(r[0], v, r[6]) for v, rows in s1.items() for r in rows if r[6] > 6.0]
    P(f"- **vaiven de la cola** (el nivel SUBE > 6 dB entre ventanas de 100 ms durante el release): {len(wob)} alertas" + (" · teclas: " + ", ".join(sorted({str(k) for k, _, _ in wob}, key=int)[:20]) if wob else ""))
    L = np.array([[r[4] for r in s1[v]] for v in vels])   # nivel (dB) [vel, tecla]
    P("\n### Nivel por tecla (RMS 0,15-0,65 s tras el ataque, dBFS)")
    P("tecla: " + " ".join(f"{k:6d}" for k in KEYS[::8]))
    for v, row in zip(vels, L):
        P(f"vel {v:4}: " + " ".join(f"{x:6.1f}" for x in row[::8]))
    jumps, resid = [], []
    from scipy.signal import medfilt
    for v, row in zip(vels, L):
        sm = medfilt(row, 7)
        for i in range(1, len(KEYS)):
            if abs(row[i] - row[i - 1]) > THR['jump']: jumps.append((KEYS[i], v, row[i] - row[i - 1]))
        for i in range(len(KEYS)):
            if abs(row[i] - sm[i]) > THR['resid']: resid.append((KEYS[i], v, row[i] - sm[i]))
    P(f"- saltos de nivel entre teclas vecinas > {THR['jump']} dB: **{len(jumps)}**" + (" · peores: " + ", ".join(f"({k}, {v}, {x:+.1f})" for k, v, x in sorted(jumps, key=lambda t: -abs(t[2]))[:6]) if jumps else ""))
    P(f"- teclas > {THR['resid']} dB fuera de su nivel suavizado: **{len(resid)}**" + (" · peores: " + ", ".join(f"({k}, {v}, {x:+.1f})" for k, v, x in sorted(resid, key=lambda t: -abs(t[2]))[:6]) if resid else ""))
    nonmono = [KEYS[i] for i in range(len(KEYS)) if not (L[0, i] < L[1, i] < L[2, i])]
    P(f"- teclas donde el nivel NO crece con la velocity (0,2 < 0,5 < 0,85): **{len(nonmono)}** {nonmono[:12]}")
    P("\n## S2 · repetir nota sin soltar")
    P(f"- teclas con clics: **{len(s2)}**" + (f" · {s2[:8]}" if s2 else ""))
    P("\n## S3 · pedal levantado")
    bad3 = [r for r in s3 if r[1] > THR['regrow'] or r[2] > THR['cut'] or r[3]]
    P(f"- alertas (rebrote, corte o clic): **{len(bad3)}** de {len(s3)} teclas" + (" · " + ", ".join(f"(tecla {k}: rebrote {rg:.1f}, corte {ct:.1f}, clics {c})" for k, rg, ct, c in sorted(bad3, key=lambda r: -r[1])[:6]) if bad3 else ""))
    P("\n## S4 · 40 notas a la vez (robo de voces)")
    P(f"- sin NaN/inf: **{s4['finite']}** · pico {s4['peak']:.1f} dBFS · clics: **{s4['clicks']}** · rebrote final {s4['regrow']:.1f} dB · corte final {s4['cut']:.1f} dB")
    P("\n## S5 · velocities extremas")
    lo, hi = s5[0.01], s5[1.0]
    P(f"- sin NaN/inf: **{lo[-1] and hi[-1]}** · nivel medio vel 0,01: {np.mean([x[1] for x in lo[:-1]]):.1f} dB, vel 1,0: {np.mean([x[1] for x in hi[:-1]]):.1f} dB · teclas con nivel(1,0) <= nivel(0,85): **{sum(1 for i in range(len(KEYS)) if hi[i][1] <= L[2, i])}**")
    total = sum(len(v) for v in alerts.values()) + len(jumps) + len(s2) + len(bad3) + s4['clicks'] + (0 if s4['finite'] else 1)
    P(f"\n**Alertas criticas totales (rebrote+corte+clics+NaN+S2+S3+S4): {sum(len(v) for v in alerts.values()) + len(s2) + len(bad3) + s4['clicks'] + (0 if s4['finite'] else 1)}**  ·  saltos de nivel: {len(jumps)}  ·  residuo de nivel: {len(resid)}")
    path = os.path.join(a.out, f"qa_{a.label}{'_legacy' if a.legacy else ''}_{datetime.date.today()}.md")
    open(path, "w", encoding="utf-8").write("\n".join(lines))
    print(f"\ninforme: {path}")


if __name__ == "__main__":
    main()
