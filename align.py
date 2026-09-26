import json, wave, re, sys
import numpy as np
from scipy.ndimage import uniform_filter1d
sys.path.insert(0, ".")
import timeline2 as T
HOP = 0.005

def load(p):
    w = wave.open(p); sr = w.getframerate()
    return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(float) / 32768, sr

def weights(words):
    out = []
    for w in words:
        p = 2.5 if w[-1] in ",;:" else (4.0 if w[-1] in ".!?" else 0.0)
        out.append((len(re.sub(r"\W", "", w)) + 1.5, p))
    return out

res, report = {}, []
for k, (who, text, dur) in T.LINES.items():
    x, sr = load(f"tts/{k}_{'F' if 'F' in who else 'M'}.wav")
    h = int(sr * HOP); n = len(x) // h
    e = uniform_filter1d(np.sqrt(np.mean(x[:n * h].reshape(n, h) ** 2, 1)), 3)
    thr = 0.05 * e.max(); act = e > thr
    idx = np.nonzero(act)[0]; s0, s1 = idx[0] * HOP, (idx[-1] + 1) * HOP
    d = uniform_filter1d(np.diff(np.log(e + 1e-4), prepend=np.log(e[0] + 1e-4)), 3)
    cands = [(i * HOP, d[i], bool(np.all(~act[max(0, i - 8):max(1, i - 1)])))
             for i in range(2, n - 3) if d[i] > 0.12 and d[i] >= d[i - 1] and d[i] >= d[i + 1] and e[i + 2] > thr]
    words = text.split(); wts = weights(words); tot = sum(a + b for a, b in wts)
    exp, acc = [], 0.0
    for a, b in wts:
        exp.append(s0 + (s1 - s0) * acc / tot); acc += a + b
    opts = []
    for i, ex in enumerate(exp):
        if i == 0:
            opts.append([(s0, 0.0)]); continue
        o = [(ex, 0.07)] + [(c, abs(c - ex) - (0.05 if g else 0) - 0.01 * min(st, 3)) for c, st, g in cands if abs(c - ex) < 0.35]
        opts.append(o)
    best = [[(c, -1) for _, c in opts[0]]]
    for i in range(1, len(opts)):
        row = []
        for tau, c in opts[i]:
            bb = (1e9, -1)
            for j, (tp, _) in enumerate(opts[i - 1]):
                if tp <= tau - 0.07 and best[i - 1][j][0] + c < bb[0]:
                    bb = (best[i - 1][j][0] + c, j)
            row.append(bb)
        best.append(row)
    j = int(np.argmin([b[0] for b in best[-1]])); taus = []
    for i in range(len(opts) - 1, -1, -1):
        taus.append(opts[i][j][0]); j = best[i][j][1]
    taus = taus[::-1]
    ends = [min(taus[i + 1] - 0.02, taus[i] + 0.9) if i + 1 < len(taus) else s1 for i in range(len(taus))]
    t0 = T.VOICE_T[k]
    res[k] = [[t0 + a, t0 + b, w] for a, b, w in zip(taus, ends, words)]
    old = [s for s, _, _ in T.word_times(k)]
    diff = np.array([t0 + a for a in taus]) - np.array(old)
    report.append((k, s0, np.mean(diff), np.max(np.abs(diff))))
json.dump(res, open("tts/words.json", "w"), indent=0)
print("line  lead-in  mean(real-est)  worst")
for k, s0, m, mx in report:
    print(f"{k:4s}  {s0*1000:5.0f}ms   {m*1000:+6.0f}ms      {mx*1000:5.0f}ms")
print("u1:", [(round(a - T.VOICE_T['u1'], 2), w) for a, b, w in res["u1"]])
