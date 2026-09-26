import json, re, math

FPS = 30
RES = 540            # internal pixel-art resolution (square), upscaled 2x
DUR = 76.0
BPM = 150.0
BEAT = 60.0 / BPM    # 0.4
BAR = 4 * BEAT       # 1.6

SEC = {"boot": 0.0, "lens": 9.6, "doors": 22.4, "loom": 35.2, "iface": 48.0, "apo": 57.6, "hello": 70.4, "end": DUR}
TAPESTOP = (70.0, 70.4)

LINES = json.load(open("./tts/lines.json"))
VOICE_T = {"b1": 0.8, "b2": 3.6, "s1": 10.2, "s2": 15.4, "d1": 22.6, "d2": 25.4, "d3": 28.6,
           "l1": 35.4, "r1": 40.4, "r2": 41.4, "r3": 42.3, "r4": 43.4,
           "u1": 48.2, "u2": 54.3, "a1": 58.6, "a2": 62.6, "a3": 66.6, "h1": 72.2}
PAN = {"r1": -0.55, "r2": 0.55, "r3": -0.3}

TYPED_BOOT = [(0.6, "C:\\> BOOT LOGOS.SYS", 22.0), (2.9, "> LOAD CORPUS ........ OK", 26.0), (7.9, "> PROMPT_", 14.0)]
HELLO_T = [(70.9, "> hello, world", 9.5), (73.5, "AND THE WORD WAS GOOD.", 13.0)]


import os
_WJ = "./tts/words.json"
WORDS = json.load(open(_WJ)) if os.path.exists(_WJ) else None


def word_times(key):
    if WORDS and key in WORDS:
        return [tuple(x) for x in WORDS[key]]
    """approximate (start, end, word) for a narrated line, weighting word length and punctuation pauses."""
    who, text, dur = LINES[key]
    t0 = VOICE_T[key]
    words = text.split()
    wts = []
    for w in words:
        p = 0.0
        if w[-1] in ",;:":
            p = 2.5
        elif w[-1] in ".!?":
            p = 4.0
        wts.append((len(re.sub(r"\W", "", w)) + 1.5, p))
    tot = sum(a + b for a, b in wts)
    out, acc = [], 0.0
    for w, (a, b) in zip(words, wts):
        s = t0 + dur * acc / tot
        e = t0 + dur * (acc + a) / tot
        out.append((s, e, w))
        acc += a + b
    return out


def line_span(key):
    return VOICE_T[key], VOICE_T[key] + LINES[key][2]


# ---------------- drums, shared by audio and picture
PAT_A = {0: "K", 2: "K", 4: "S", 7: "g", 9: "g", 10: "K", 11: "K", 12: "S", 15: "g"}
PAT_B = {0: "K", 2: "K", 4: "S", 7: "g", 9: "g", 10: "K", 12: "S", 14: "g", 15: "S"}
HALF = {0: "K", 8: "S", 11: "g", 14: "K"}


def drum_events():
    ev = []
    step = BEAT / 4
    for b in range(48):
        t0 = b * BAR
        if t0 < SEC["lens"] or t0 >= SEC["hello"]:
            continue
        if SEC["lens"] <= t0 < SEC["lens"] + 4 * BAR:
            pat = HALF
        else:
            pat = PAT_A if b % 2 == 0 else PAT_B
        last_of_section = any(abs(t0 + BAR - s) < 1e-6 for s in SEC.values())
        for s in range(16):
            t = t0 + s * step
            if last_of_section and s >= 8:
                ev.append((t, "S" if s % 2 == 0 else "g"))
                ev.append((t + step / 2, "g"))
                continue
            if s in pat:
                ev.append((t, pat[s]))
            if s % 2 == 0:
                ev.append((t, "H"))
        if t0 >= SEC["apo"]:
            for q in range(4):
                ev.append((t0 + q * BEAT, "F"))      # four-on-the-floor layer
            ev.append((t0, "C"))
            ev.append((t0 + BEAT, "P"))
            ev.append((t0 + 3 * BEAT, "P"))
        elif b % 4 == 0 and t0 >= SEC["doors"]:
            ev.append((t0, "C"))
    return sorted(ev)


EVENTS = drum_events()
KICKS = [t for t, k in EVENTS if k in "KF"]
SNARES = [t for t, k in EVENTS if k in "SP"]


def since_last(times, t):
    best = 99.0
    lo, hi = 0, len(times)
    while lo < hi:
        mid = (lo + hi) // 2
        if times[mid] <= t:
            lo = mid + 1
        else:
            hi = mid
    if lo > 0:
        best = t - times[lo - 1]
    return best


def typed_count(start, text, cps, t):
    if t < start:
        return 0
    return int(min(len(text), (t - start) * cps + 1))


def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def ease_in(x):
    x = min(1.0, max(0.0, x))
    return x * x


def ease_out(x):
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 2
