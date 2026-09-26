import sys, wave
import numpy as np
from scipy import signal
sys.path.insert(0, ".")
from timeline2 import *

SR = 44100
N = int(SR * DUR)
rng = np.random.default_rng(5)


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


CH = {"Dsus2": (2, [2, 4, 9]), "Bm": (11, [11, 2, 6, 9]), "G": (7, [7, 11, 2, 9]), "D": (2, [2, 6, 9, 4]),
      "A": (9, [9, 1, 4, 11]), "F#m": (6, [6, 9, 1, 4]), "Asus4": (9, [9, 2, 4]), "Em7": (4, [4, 7, 11, 2]),
      "A7sus": (9, [9, 2, 4, 7]), "E": (4, [4, 8, 11, 6]), "B": (11, [11, 3, 6, 1]), "C#m": (1, [1, 4, 8, 11]),
      "Aadd9": (9, [9, 1, 4, 11]), "Eadd9": (4, [4, 8, 11, 6])}
BARS = (["Dsus2"] * 6 + ["Bm", "G", "D", "A"] * 2 + ["D", "A", "Bm", "G"] * 2 +
        ["G", "A", "F#m", "Bm", "G", "A", "Bm", "Asus4"] + ["G", "A", "Bm", "G", "Em7", "A7sus"] +
        ["E", "B", "C#m", "Aadd9", "E", "B", "Aadd9", "B"] + ["Eadd9"] * 4)


def chord(b):
    return CH[BARS[min(max(b, 0), len(BARS) - 1)]]


def voicing(pcs, lo, hi):
    return [m for m in range(lo, hi + 1) if m % 12 in pcs]


def bass_note(pc):
    return 31 + ((pc - 7) % 12)


def bus():
    return np.zeros((2, N))


def place(b, x, t0, gain=1.0, pan=0.0):
    i0 = int(round(t0 * SR))
    if x.ndim == 1:
        gl, gr = np.cos((pan + 1) * np.pi / 4) * 1.4142, np.sin((pan + 1) * np.pi / 4) * 1.4142
        x = np.stack([x * gl, x * gr])
    if i0 < 0:
        x, i0 = x[:, -i0:], 0
    n = min(x.shape[1], N - i0)
    if n > 0:
        b[:, i0:i0 + n] += x[:, :n] * gain


def env_adr(n, att, hold, rel):
    t = np.arange(n) / SR
    e = np.minimum(1, t / max(att, 1e-4))
    e = np.where(t > hold, np.cos(np.clip((t - hold) / rel, 0, 1) * np.pi / 2) ** 2, e)
    return e


def blep_saw(f, n, ph0=0.0):
    dt = f / SR
    ph = (ph0 + dt * np.arange(n)) % 1.0
    y = 2 * ph - 1
    m1 = ph < dt
    t1 = ph[m1] / dt
    y[m1] -= t1 + t1 - t1 * t1 - 1
    m2 = ph > 1 - dt
    t2 = (ph[m2] - 1) / dt
    y[m2] -= t2 * t2 + t2 + t2 + 1
    return y


def supersaw(m, dur, fc, att=0.01, rel=0.25, voices=7, spread=24.0, vib=0.0):
    n = int((dur + rel) * SR)
    f = mtof(m)
    L, R = np.zeros(n), np.zeros(n)
    t = np.arange(n) / SR
    for v in range(voices):
        det = ((v - (voices - 1) / 2) / ((voices - 1) / 2)) * spread
        ff = f * 2 ** (det / 1200)
        if vib > 0:
            fm = ff * (1 + vib * np.sin(2 * np.pi * 5.5 * t) * np.minimum(1, t / 0.4))
            ph = (rng.random() + np.cumsum(fm) / SR) % 1.0
            y = 2 * ph - 1
        else:
            y = blep_saw(ff, n, rng.random())
        pan = (v / (voices - 1)) * 2 - 1
        L += y * np.cos((pan + 1) * np.pi / 4)
        R += y * np.sin((pan + 1) * np.pi / 4)
    sos = signal.butter(2, min(fc, 17000) / (SR / 2), output="sos")
    out = np.stack([signal.sosfilt(sos, L), signal.sosfilt(sos, R)])
    return out * env_adr(n, att, dur, rel) / np.sqrt(voices)


_pc = {}


def pluck(m, bright, dur=0.3):
    k = (m, round(bright, 2))
    if k in _pc:
        return _pc[k]
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = mtof(m)
    out = np.zeros(n)
    h = 1
    while h * f < 15000 and h < 30:
        out += (1.0 / h) * (1 if h % 2 else 0.5) * np.exp(-t * (9 + h * bright)) * np.sin(2 * np.pi * h * f * t)
        h += 1
    out *= np.minimum(1, t / 0.0015)
    _pc[k] = out
    return out


def bell(m, dur=2.2):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = mtof(m)
    out = np.zeros(n)
    for r, a, d in [(1, 1, 1.6), (2, .45, .9), (3, .25, .55), (4.16, .18, .35), (5.43, .12, .22)]:
        if f * r < 16000:
            out += a * np.exp(-t / d) * np.sin(2 * np.pi * f * r * t)
    return out * np.minimum(1, t / 0.001)


def bass_saw(m, dur, fc=700):
    n = int((dur + 0.05) * SR)
    f = mtof(m)
    y = blep_saw(f, n) * 0.6 + np.sin(2 * np.pi * f * np.arange(n) / SR)
    y = signal.sosfilt(signal.butter(2, fc / (SR / 2), output="sos"), y)
    return y * env_adr(n, 0.004, dur, 0.05)


def sub(m, dur, att=0.3, rel=0.8):
    n = int((dur + rel) * SR)
    t = np.arange(n) / SR
    f = mtof(m)
    return (np.sin(2 * np.pi * f * t) + 0.2 * np.sin(4 * np.pi * f * t)) * env_adr(n, att, dur, rel)


def nbp(n, lo, hi):
    return signal.sosfilt(signal.butter(2, [lo / (SR / 2), hi / (SR / 2)], "band", output="sos"), rng.standard_normal(n))


def nhp(n, fc):
    return signal.sosfilt(signal.butter(2, fc / (SR / 2), "high", output="sos"), rng.standard_normal(n))


def kick(big=1.0):
    n = int(0.5 * SR)
    t = np.arange(n) / SR
    fr = 46 + 150 * np.exp(-t * 30)
    body = np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t * 6 / big)
    clk = signal.sosfilt(signal.butter(2, 6000 / (SR / 2), output="sos"), rng.standard_normal(n)) * np.exp(-t * 800) * 0.4
    return np.tanh(2.0 * (body + clk)) * 0.9


def snare(v=1.0):
    n = int(0.28 * SR)
    t = np.arange(n) / SR
    x = nbp(n, 1200, 8000) * np.exp(-t * (18 if v > 0.5 else 30))
    tone = (np.sin(2 * np.pi * 185 * t) + 0.5 * np.sin(2 * np.pi * 330 * t)) * np.exp(-t * 28)
    return np.tanh(1.5 * (x * 0.9 + tone * 0.7)) * 0.7


def clap():
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    env = sum((t >= d) * np.exp(-np.clip(t - d, 0, None) * (70 if d < 0.02 else 13)) for d in (0, .01, .021))
    return nbp(n, 900, 4000) * env * 0.7


def hat(d=80):
    n = int(0.1 * SR)
    t = np.arange(n) / SR
    return nhp(n, 8000) * np.exp(-t * d) * 0.5


def crash(dur=2.5):
    n = int(dur * SR)
    t = np.arange(n) / SR
    return nhp(n, 3000) * np.exp(-t * 4.4 / dur) * 0.5


def riser(dur, f0=250, f1=9000):
    n = int(dur * SR)
    x = rng.standard_normal(n)
    out = np.zeros(n)
    w = np.hanning(2048)
    for i in range(0, n - 2048, 1024):
        c = f0 * (f1 / f0) ** (i / n)
        sos = signal.butter(2, [max(40, c * .7) / (SR / 2), min(19000, c * 1.4) / (SR / 2)], "band", output="sos")
        out[i:i + 2048] += signal.sosfilt(sos, x[i:i + 2048]) * w
    t = np.arange(n) / SR
    sweep = np.sin(2 * np.pi * np.cumsum(150 * 16 ** (t / dur)) / SR) * 0.25
    return (out * 0.8 + sweep) * (t / dur) ** 2


CLK = []
for i in range(12):
    n = int(0.03 * SR)
    t = np.arange(n) / SR
    x = nbp(n, 1500, 7500) * np.exp(-t * rng.uniform(350, 550))
    x += 0.5 * np.sin(2 * np.pi * rng.uniform(2500, 3800) * t) * np.exp(-t * 700)
    x += 0.4 * np.sin(2 * np.pi * rng.uniform(700, 1200) * t) * np.exp(-t * 260)
    CLK.append(x / np.max(np.abs(x)))


def blip(f, d=0.03):
    n = int(d * SR)
    t = np.arange(n) / SR
    return np.sign(np.sin(2 * np.pi * f * t)) * np.exp(-t * 60) * 0.3


def formant_choir(pcs, dur, vowel="ah", lo=62, hi=81):
    notes = voicing(pcs, lo, hi)[-5:]
    n = int((dur + 1.0) * SR)
    t = np.arange(n) / SR
    L, R = np.zeros(n), np.zeros(n)
    for m in notes:
        for v in range(4):
            f = mtof(m) * 2 ** (rng.uniform(-9, 9) / 1200)
            vib = 1 + 0.005 * np.sin(2 * np.pi * rng.uniform(4.8, 6) * t + rng.uniform(0, 6)) * np.minimum(1, t / 0.5)
            ph = 2 * np.pi * np.cumsum(f * vib) / SR
            src = np.zeros(n)
            k = 1
            while k * f < 6000:
                src += np.sin(k * ph) / k
                k += 1
            p = rng.uniform(-0.8, 0.8)
            L += src * np.cos((p + 1) * np.pi / 4)
            R += src * np.sin((p + 1) * np.pi / 4)
    form = {"ah": [(800, 80, 1), (1150, 90, .55), (2900, 120, .25), (3900, 130, .12)],
            "oh": [(450, 70, 1), (800, 80, .6), (2830, 100, .18), (3800, 120, .08)]}[vowel]
    outs = []
    for x in (L, R):
        y = np.zeros(n)
        for fc, bw, g in form:
            b, a = signal.iirpeak(fc / (SR / 2), fc / bw)
            y += g * signal.lfilter(b, a, x)
        outs.append(y)
    out = np.stack(outs) * env_adr(n, 0.35, dur, 0.9)
    return out / (np.max(np.abs(out)) + 1e-9)


def load(path):
    with wave.open(path) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768
        sr0 = w.getframerate()
    x = signal.resample(x, int(len(x) * SR / sr0))
    x = signal.sosfilt(signal.butter(2, 80 / (SR / 2), "high", output="sos"), x)
    return x / (np.max(np.abs(x)) + 1e-9)


def bitcrush(x, hold=5, bits=5):
    y = np.repeat(x[::hold], hold)[:len(x)]
    q = 2 ** (bits - 1)
    return np.round(y * q) / q


def synth():
    B = {k: bus() for k in ["saw", "arp", "lead", "bass", "drums", "clk", "choir", "fx", "bells", "voice", "vfx", "coda"]}
    # --- supersaw chords
    for b in range(len(BARS)):
        t0 = b * BAR
        if t0 >= SEC["hello"]:
            break
        root, pcs = chord(b)
        if t0 < SEC["lens"]:
            fc, g = 500 + 250 * b, 0.10
        elif t0 < SEC["doors"]:
            fc, g = 2200, 0.16
        elif t0 < SEC["loom"]:
            fc, g = 3500, 0.18
        elif t0 < SEC["iface"]:
            fc, g = 4200, 0.2
        elif t0 < SEC["apo"]:
            fc, g = 2500 + 900 * (b - 30), 0.2
        else:
            fc, g = 8000, 0.24
        notes = voicing(pcs, 55, 76)
        dur = BAR * (1.25 if b == 43 else 1.0)
        if b == 43:
            dur = SEC["hello"] - 0.4 - t0
        for m in notes:
            place(B["saw"], supersaw(m, dur, fc, att=0.3 if t0 < SEC["lens"] else 0.01, rel=0.3), t0, g / np.sqrt(len(notes)))
    # --- arps (16ths, 32nd bursts later)
    for b in range(4, 44):
        t0 = b * BAR
        root, pcs = chord(b)
        tones = voicing(pcs, 64, 91)
        seq = tones + tones[-2:0:-1]
        steps = 32 if (t0 >= SEC["iface"] and b % 2 == 1) or t0 >= SEC["apo"] + 4 * BAR else 16
        for s in range(steps):
            t = t0 + s * BAR / steps
            br = 2.2 if t0 < SEC["lens"] else (1.1 if t0 < SEC["apo"] else 0.6)
            g = (0.05 + 0.04 * (b - 4) if t0 < SEC["lens"] else 0.15) * (1.0 if s % 4 == 0 else 0.7)
            place(B["arp"], pluck(seq[(s + b * 5) % len(seq)], br), t, g, pan=0.45 * np.sin(s * 1.7))
    # --- lead (apotheosis)
    MEL = [[(0, 76, 1), (1, 78, .5), (1.5, 80, .5), (2, 83, 2)], [(0, 78, 1), (1, 75, 1), (2, 78, 1), (3, 80, 1)],
           [(0, 80, 1.5), (1.5, 78, .5), (2, 76, 1), (3, 73, 1)], [(0, 76, 2), (2, 73, 1), (3, 71, 1)],
           [(0, 76, 1), (1, 78, .5), (1.5, 80, .5), (2, 83, 1), (3, 85, 1)], [(0, 87, 1.5), (1.5, 85, .5), (2, 83, 2)],
           [(0, 85, 1), (1, 83, 1), (2, 80, 1), (3, 81, 1)], [(0, 83, 4)]]
    for i, bar in enumerate(MEL):
        t0 = SEC["apo"] + i * BAR
        for (bt, m, d) in bar:
            dd = d * BEAT if i < 7 else (SEC["hello"] - 0.4 - t0)
            x = supersaw(m, dd, 7000, att=0.01, rel=0.2, voices=5, spread=14, vib=0.004)
            place(B["lead"], x, t0 + bt * BEAT, 0.16)
            place(B["lead"], supersaw(m - 12, dd, 4000, att=0.01, rel=0.2, voices=3, spread=10), t0 + bt * BEAT, 0.08)
    # --- bass
    place(B["bass"], sub(26, SEC["lens"] - 0.3, att=4.0, rel=0.3), 0.0, 0.45)
    for b in range(6, 44):
        t0 = b * BAR
        root, pcs = chord(b)
        m = bass_note(root)
        if t0 >= SEC["apo"]:
            for s in range(16):
                if s % 4 == 0:
                    continue
                place(B["bass"], bass_saw(m + (12 if s % 4 == 3 else 0), BEAT / 4 * 0.8, 900), t0 + s * BEAT / 4, 0.32)
        else:
            for s in (2, 3, 6, 7, 10, 11, 14, 15) if t0 >= SEC["doors"] else (2, 6, 10, 14):
                place(B["bass"], bass_saw(m + (12 if s in (7, 15) else 0), BEAT / 4 * 0.85, 700), t0 + s * BEAT / 4, 0.34)
    place(B["bass"], sub(26, 1.2, att=0.005, rel=1.0), SEC["lens"], 0.55)
    place(B["bass"], sub(28, 1.4, att=0.005, rel=1.2), SEC["apo"], 0.7)
    # --- drums from shared events
    for t, k in EVENTS:
        if k == "K":
            place(B["drums"], kick(), t, 0.8)
        elif k == "F":
            place(B["drums"], kick(1.2), t, 0.55)
        elif k == "S":
            place(B["drums"], snare(1.0), t, 0.55, rng.uniform(-.1, .1))
        elif k == "g":
            place(B["drums"], snare(0.3), t, 0.16, rng.uniform(-.3, .3))
        elif k == "H":
            place(B["drums"], hat(), t + BEAT / 4 * 0 + 0.0, 0.14, 0.25)
        elif k == "P":
            place(B["drums"], clap(), t, 0.45)
        elif k == "C":
            place(B["drums"], crash(), t, 0.3, rng.uniform(-.4, .4))
    # breakcore stutters in the interface section
    d = B["drums"]
    for b in range(30, 36):
        for s in range(0, 16, 2):
            if rng.random() < 0.22:
                i0 = int((b * BAR + s * BEAT / 4) * SR)
                sl = int(BEAT / 8 * SR)
                chunk = d[:, i0:i0 + sl].copy()
                for r in range(1, 4):
                    d[:, i0 + r * sl:i0 + (r + 1) * sl] = chunk * (1 - 0.15 * r)
    # --- teletype clicks
    for (s, txt, cps) in TYPED_BOOT + HELLO_T:
        for i, ch in enumerate(txt):
            if ch != " ":
                tgt = B["coda"] if s >= SEC["hello"] else B["clk"]
                place(tgt, CLK[rng.integers(0, 12)], s + i / cps, 0.16 * rng.uniform(.7, 1), -0.7 + 1.4 * i / max(1, len(txt) - 1))
        tgt = B["coda"] if s >= SEC["hello"] else B["bells"]
        place(tgt, bell(98, 1.2) * 0.6, s + len(txt) / cps + 0.05, 0.08, 0.5)
    for b in range(0, 44):
        t0 = b * BAR
        dens = 16 if t0 < SEC["iface"] else 32
        for s in range(dens):
            p = 0.35 if t0 < SEC["lens"] else (0.5 if t0 < SEC["iface"] else 0.8)
            if rng.random() < p:
                place(B["clk"], CLK[rng.integers(0, 12)], t0 + s * BAR / dens, 0.07 * rng.uniform(.5, 1), rng.uniform(-.8, .8))
        if t0 < SEC["lens"] or SEC["iface"] <= t0 < SEC["apo"]:
            for s in range(8):
                if rng.random() < 0.4:
                    place(B["fx"], blip(rng.choice([1320, 1760, 2349, 2637, 3520])), t0 + s * BEAT / 2, 0.08, rng.uniform(-.7, .7))
    # --- choir
    for b in range(0, 6):
        place(B["choir"], formant_choir(chord(b)[1], BAR, "oh", 50, 69), b * BAR, 0.10)
    for b in range(22, 30):
        place(B["choir"], formant_choir(chord(b)[1], BAR), b * BAR, 0.20)
    for b in range(36, 44):
        dd = BAR if b < 43 else SEC["hello"] - 0.4 - b * BAR
        place(B["choir"], formant_choir(chord(b)[1], dd, "ah", 64, 84), b * BAR, 0.42)
    # --- fx
    for end, ln, g in [(SEC["lens"], 2 * BAR, .3), (SEC["doors"], 2 * BAR, .3), (SEC["loom"], 2 * BAR, .3),
                       (SEC["iface"], 2 * BAR, .35), (SEC["apo"], 4 * BAR, .5), (TAPESTOP[0], 2.2, .5)]:
        place(B["fx"], riser(ln), end - ln, g)
    for t in (SEC["lens"], SEC["doors"], SEC["loom"], SEC["iface"], SEC["apo"]):
        place(B["fx"], crash(3.5), t, 0.45)
    rc = crash(2.0)[::-1] * 0.8
    place(B["fx"], rc, SEC["apo"] - 2.0, 0.5)
    # --- bells sparkle
    for b in range(22, 44):
        root, pcs = chord(b)
        tones = voicing(pcs, 84, 100)
        for s in range(8):
            if rng.random() < (0.25 if b < 36 else 0.45):
                place(B["bells"], bell(int(rng.choice(tones))), b * BAR + s * BEAT / 2, 0.08, rng.uniform(-.7, .7))
    # --- coda (hello, world)
    for i, m in enumerate([64, 71, 76, 78, 80, 83, 88]):
        place(B["coda"], bell(m, 3.5), SEC["hello"] + 0.5 + i * 0.42, 0.12, -0.6 + 0.2 * i)
    place(B["coda"], formant_choir(CH["Eadd9"][1], 4.2, "oh", 56, 76), SEC["hello"] + 0.3, 0.16)
    place(B["coda"], sub(40, 4.0, att=1.0, rel=1.5), SEC["hello"] + 0.3, 0.2)
    # --- voices
    for k, t in VOICE_T.items():
        who, text, dur = LINES[k]
        xs = []
        if "F" in who:
            xs.append(load(f"./tts/{k}_F.wav"))
        if "M" in who:
            m = load(f"./tts/{k}_M.wav")
            if xs:
                m = signal.resample(m, len(xs[0]))
                m /= np.max(np.abs(m)) + 1e-9
            xs.append(m)
        x = xs[0] if len(xs) == 1 else xs[0] * 0.75 + xs[1] * 0.65
        pan = PAN.get(k, 0.0)
        tgt = B["coda"] if t >= SEC["hello"] else B["voice"]
        place(tgt, x, t, 0.62, pan)
        if k in ("u2", "a1", "a3", "d1", "r4"):
            place(B["vfx"], bitcrush(x), t, 0.12, -pan)
        if k == "u2":
            sa = xs[0][:int(0.16 * SR)] * np.hanning(int(0.16 * SR))
            for j, dt in enumerate([-0.5, -0.4, -0.3, -0.2, -0.15, -0.1]):
                place(B["voice"], sa, t + dt, 0.25 + 0.06 * j, 0.6 * (-1) ** j)
        # comb 'singing' layer tuned to the current chord
        root, pcs = chord(int(t / BAR))
        xx = np.pad(x, (0, SR))
        comb = np.zeros((2, len(xx)))
        for j, mm in enumerate(voicing(pcs, 50, 69)[:4]):
            D = int(round(SR / mtof(mm)))
            a = np.zeros(D + 1)
            a[0], a[-1] = 1, -0.92
            comb[j % 2] += signal.lfilter([1.0], a, xx)
        comb = signal.sosfilt(signal.butter(2, 4500 / (SR / 2), output="sos"), comb)
        comb /= np.max(np.abs(comb)) + 1e-9
        place(B["vfx"] if t < SEC["hello"] else B["coda"], comb, t, 0.16)
    np.savez("./stems2.npz", **{k: v.astype(np.float32) for k, v in B.items()})
    print("stems saved")


def make_ir(rt60, predelay, lp, seed):
    L = int(SR * rt60 * 1.1)
    r = np.random.default_rng(seed)
    t = np.arange(L) / SR
    irs = []
    for c in range(2):
        nz = r.standard_normal(L)
        lo = signal.sosfilt(signal.butter(2, lp / (SR / 2), output="sos"), nz)
        ir = lo * np.exp(-6.91 * t / rt60) + 0.3 * (nz - lo) * np.exp(-6.91 * t / (rt60 * .35))
        ir *= np.minimum(1, t / 0.015)
        ir = np.concatenate([np.zeros(int(SR * (predelay + 0.007 * c))), ir])
        irs.append(ir / np.sqrt(np.sum(ir ** 2)))
    m = max(len(i) for i in irs)
    return np.stack([np.pad(i, (0, m - len(i))) for i in irs])


def rev(x, ir):
    return np.stack([signal.fftconvolve(x[c], ir[c])[:N] for c in range(2)])


def mix():
    S = np.load("./stems2.npz")
    G = dict(saw=1.0, arp=1.0, lead=1.0, bass=1.0, drums=1.0, clk=1.0, choir=1.6, fx=0.9, bells=1.0, voice=2.1, vfx=1.4, coda=1.6)
    for a in sys.argv[2:]:
        k, v = a.split("=")
        G[k] = float(v)
    b = {k: S[k].astype(np.float64) * G[k] for k in G}
    cath, hall, room = make_ir(4.8, .04, 4200, 3), make_ir(2.0, .02, 6000, 5), make_ir(0.5, .005, 8000, 7)
    sc = np.ones(N)
    for tk in KICKS:
        i0, L = int(tk * SR), int(0.28 * SR)
        n = min(L, N - i0)
        sc[i0:i0 + n] = np.minimum(sc[i0:i0 + n], (1 - 0.55 * (1 - np.linspace(0, 1, L)) ** 2)[:n])
    venv = np.abs(b["voice"]).sum(0)
    venv = signal.sosfiltfilt(signal.butter(1, 5 / (SR / 2), output="sos"), venv)
    duck = 1 - 0.5 * np.clip(venv / (np.max(venv) + 1e-9) * 2.5, 0, 1)
    wet_h = rev(0.3 * b["saw"] + 0.3 * b["arp"] + 0.35 * b["lead"] + 0.4 * b["bells"] + 0.06 * b["drums"] + 0.3 * b["fx"], hall)
    wet_c = rev(0.8 * b["choir"] + 0.45 * b["voice"] + 0.7 * b["vfx"] + 0.2 * b["bells"], cath)
    wet_r = rev(b["clk"], room)
    music = (b["saw"] + b["arp"] + b["bass"] + b["lead"]) * sc + b["choir"] * (.7 + .3 * sc) + b["drums"] + b["fx"] + b["bells"] + 0.45 * wet_h * sc
    main = music * duck + b["clk"] + 0.3 * wet_r + b["voice"] + b["vfx"] * 0.6 + 0.5 * wet_c
    # tape stop into silence
    i0, i1 = int(TAPESTOP[0] * SR), int(TAPESTOP[1] * SR)
    n = i1 - i0
    u = np.arange(n) / n
    pos = i0 + np.cumsum((1 - u) ** 1.7)
    for c in range(2):
        main[c, i0:i1] = np.interp(pos, np.arange(N), main[c]) * (1 - u ** 3)
    main[:, i1:] = 0
    coda = b["coda"] + 0.8 * rev(b["coda"], cath)
    out = main + coda
    out = signal.sosfilt(signal.butter(2, 28 / (SR / 2), "high", output="sos"), out)
    pre = out.copy()
    sc_ = 1 / (np.percentile(np.abs(out[:, :i0]), 99.95) + 1e-9)
    out *= sc_
    out = np.tanh(1.15 * out) / np.tanh(1.15)
    t = np.arange(N) / SR
    out *= np.clip((DUR - t) / 1.5, 0, 1)[None] ** 1.5
    out *= np.minimum(1, t / 0.01)[None]
    out = out / (np.max(np.abs(out)) + 1e-9) * 0.93
    with wave.open("./score2.wav", "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((out.T * 32767).astype(np.int16).tobytes())

    def rms(x, a, z):
        return np.sqrt(np.mean(x[:, int(a * SR):int(z * SR)] ** 2)) * sc_
    voice = b["voice"]
    print("win         music  voice  total")
    for k in ["b2", "s1", "d1", "d3", "l1", "u1", "u2", "a1", "a3"]:
        a, z = line_span(k)
        print(f"{k:4s} {a:5.1f}  {rms(music * duck, a, z):.3f}  {rms(voice, a, z):.3f}  {rms(pre, a, z):.3f}")
    for nm, (a, z) in {"boot": (0, 9.6), "lens": (9.6, 22.4), "doors": (22.4, 35.2), "loom": (35.2, 48), "iface": (48, 57.6), "apo": (57.6, 70), "hello": (70.4, 76)}.items():
        print(f"{nm:6s} total {rms(pre, a, z):.3f}")


if __name__ == "__main__":
    synth() if sys.argv[1] == "synth" else mix()
