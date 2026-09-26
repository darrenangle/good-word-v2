import sys, math, subprocess, time
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, ".")
from timeline2 import *

cv2.setNumThreads(1)
R = RES
INK, PAP = 0, 255
PALETTE = np.array([[22, 24, 23], [236, 240, 233]], np.uint8)
FD = "./fonts/"
F_PS, F_VT, F_SK, F_SKB, F_DOT = FD + "PressStart2P-Regular.ttf", FD + "VT323-Regular.ttf", FD + "Silkscreen-Regular.ttf", FD + "Silkscreen-Bold.ttf", FD + "DotGothic16-Regular.ttf"
STAGE_Y0, STAGE_Y1 = 24, 490
CX, CY = 270, 257

# ============================================================ text bitmaps
_fc, _bc = {}, {}


def font(p, s):
    k = (p, s)
    if k not in _fc:
        _fc[k] = ImageFont.truetype(p, s)
    return _fc[k]


def bits(text, path=F_SK, size=8, trim=False, k=1, flip=False):
    key = (text, path, size, trim, k, flip)
    b = _bc.get(key)
    if b is not None:
        return b
    if k > 1 or flip:
        b = bits(text, path, size, trim)
        if flip:
            b = b[:, ::-1]
        if k > 1:
            b = np.repeat(np.repeat(b, k, 0), k, 1)
    else:
        f = font(path, size)
        asc, desc = f.getmetrics()
        w = int(math.ceil(f.getlength(text))) + 4
        img = Image.new("1", (max(2, w), asc + desc + 4), 0)
        d = ImageDraw.Draw(img)
        d.fontmode = "1"
        d.text((2, 2), text, font=f, fill=1)
        b = np.array(img, dtype=bool)
        if trim:
            ys, xs = np.nonzero(b)
            if len(xs):
                b = b[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    if len(_bc) > 12000:
        _bc.clear()
    _bc[key] = b
    return b


def dil(b, r=1):
    p = np.pad(b.astype(np.uint8), r)
    return cv2.dilate(p, np.ones((2 * r + 1, 2 * r + 1), np.uint8)).astype(bool)


def put(cv, b, x, y, val=INK):
    H_, W_ = cv.shape
    h, w = b.shape
    x, y = int(round(x)), int(round(y))
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W_, x + w), min(H_, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    reg = cv[y0:y1, x0:x1]
    reg[b[y0 - y:y1 - y, x0 - x:x1 - x]] = val


def put_c(cv, b, cx, cy, val=INK):
    put(cv, b, cx - b.shape[1] // 2, cy - b.shape[0] // 2, val)


def put_out(cv, b, x, y, fill=PAP, line=INK, r=1, halo=0):
    if halo:
        put(cv, dil(b, r + halo), x - r - halo, y - r - halo, fill)
    put(cv, dil(b, r), x - r, y - r, line)
    put(cv, b, x, y, fill)


def put_out_c(cv, b, cx, cy, fill=PAP, line=INK, r=1, halo=0):
    put_out(cv, b, cx - b.shape[1] // 2, cy - b.shape[0] // 2, fill, line, r, halo)


# ============================================================ patterns & dither
S = R + 64
_yy, _xx = np.mgrid[0:S, 0:S]


def _p(m):
    return np.where(m, INK, PAP).astype(np.uint8)


P = {"zig": _p(((_yy + np.abs((_xx % 16) - 8)) % 8) == 0), "diag": _p(((_xx + _yy) % 8) < 3),
     "dots": _p((_xx % 4 == 0) & (_yy % 4 == 0)), "check": _p((_xx + _yy) % 2 == 0),
     "hl": _p(_yy % 4 == 0), "vl": _p(_xx % 4 == 0), "grid": _p((_xx % 8 == 0) | (_yy % 8 == 0)),
     "brick": _p((_yy % 8 == 0) | (((_xx + 8 * ((_yy // 8) % 2)) % 16 == 0))), "dense": _p((_xx % 2 == 0) | (_yy % 2 == 0))}
for k in list(P):
    P[k + "I"] = (255 - P[k]).astype(np.uint8)
PATS = ["zig", "diag", "dots", "check", "hl", "grid", "zigI", "diagI", "dotsI", "brick", "vl", "denseI"]
B8 = np.array([[0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26], [12, 44, 4, 36, 14, 46, 6, 38], [60, 28, 52, 20, 62, 30, 54, 22],
               [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25], [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21]])
BT = ((np.tile(B8, (R // 8 + 2, R // 8 + 2)) + 0.5) / 64.0).astype(np.float32)[:R + 8, :R + 8]
BTR = BT[:R, :R]
YY, XX = np.mgrid[0:R, 0:R]


def clampr(x0, y0, x1, y1, W_=R, H_=R):
    return max(0, int(x0)), max(0, int(y0)), min(W_, int(x1)), min(H_, int(y1))


def fillp(cv, x0, y0, x1, y1, name, ox=0, oy=0):
    H_, W_ = cv.shape
    x0, y0, x1, y1 = clampr(x0, y0, x1, y1, W_, H_)
    if x1 <= x0 or y1 <= y0:
        return
    ox, oy = int(ox) % 64, int(oy) % 64
    cv[y0:y1, x0:x1] = P[name][y0 + oy:y1 + oy, x0 + ox:x1 + ox]


def maskp(cv, mask, name, ox=0, oy=0):
    ox, oy = int(ox) % 64, int(oy) % 64
    cv[mask] = P[name][oy:oy + R, ox:ox + R][mask]


def rectf(cv, x0, y0, x1, y1, v):
    cv2.rectangle(cv, (int(x0), int(y0)), (int(x1), int(y1)), int(v), -1)


def recto(cv, x0, y0, x1, y1, v=INK, th=1):
    cv2.rectangle(cv, (int(x0), int(y0)), (int(x1), int(y1)), int(v), th)


def line(cv, x0, y0, x1, y1, v=INK, th=1):
    cv2.line(cv, (int(x0), int(y0)), (int(x1), int(y1)), int(v), th, cv2.LINE_8)


def circ(cv, x, y, r, v=INK, th=-1):
    cv2.circle(cv, (int(x), int(y)), int(max(0, r)), int(v), th, cv2.LINE_8)


def poly(cv, pts, v=INK):
    cv2.fillPoly(cv, [np.int32(pts)], int(v), cv2.LINE_8)


def region(cv, x0, y0, x1, y1, fn):
    H_, W_ = cv.shape
    a, b, c, d = clampr(x0, y0, x1, y1, W_, H_)
    if c - a < 3 or d - b < 3:
        return
    tmp = np.ascontiguousarray(cv[b:d, a:c])
    fn(tmp, int(x0) - a, int(y0) - b)
    cv[b:d, a:c] = tmp


def bez(p0, p1, p2, p3, n=28):
    u = np.linspace(0, 1, n)[:, None]
    return (1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * p1 + 3 * (1 - u) * u * u * p2 + u ** 3 * p3


def cable(cv, p0, p3, sag=30, frac=1.0, horiz=False, th=5):
    p0, p3 = np.array(p0, float), np.array(p3, float)
    if horiz:
        dx = (p3[0] - p0[0]) * 0.5
        p1, p2 = p0 + [dx, 0], p3 - [dx, 0]
    else:
        p1, p2 = p0 + [(p3[0] - p0[0]) * 0.3, sag], p3 + [-(p3[0] - p0[0]) * 0.3, sag]
    pts = bez(p0, p1, p2, p3)
    n = max(2, int(len(pts) * min(1, frac)))
    pts = np.int32(pts[:n])
    cv2.polylines(cv, [pts], False, INK, th, cv2.LINE_8)
    cv2.polylines(cv, [pts], False, PAP, max(1, th - 3), cv2.LINE_8)


def window(cv, x, y, w, h, title="", bar="ink", shadow=3, body=PAP, pat=None, pop=1.0):
    x, y, w, h = int(x), int(y), int(w), int(h)
    if pop < 1.0:
        cx_, cy_ = x + w / 2, y + h / 2
        w, h = max(4, int(w * pop)), max(4, int(h * pop))
        x, y = int(cx_ - w / 2), int(cy_ - h / 2)
    if w < 16 or h < 16:
        rectf(cv, x, y, x + w, y + h, PAP)
        recto(cv, x, y, x + w, y + h, INK)
        return (x + 1, y + 1, x + w - 1, y + h - 1)
    if shadow:
        rectf(cv, x + shadow, y + shadow, x + w + shadow, y + h + shadow, INK)
    if pat:
        fillp(cv, x, y, x + w, y + h, pat)
    else:
        rectf(cv, x, y, x + w, y + h, body)
    recto(cv, x, y, x + w, y + h, INK)
    th = 11
    if bar == "ink":
        rectf(cv, x + 1, y + 1, x + w - 1, y + th, INK)
        tv = PAP
    else:
        fillp(cv, x + 1, y + 1, x + w - 1, y + th, "hl")
        tv = INK
    line(cv, x, y + th + 1, x + w, y + th + 1, INK)
    if title and w > 44:
        tb = bits(title, F_SK, 8)[:, :max(1, w - 42)]
        if bar != "ink":
            rectf(cv, x + 2, y + 1, x + 4 + tb.shape[1], y + th, PAP)
        put(cv, tb, x + 3, y, tv)
    if w > 40:
        for k in range(3):
            bx, by = x + w - 10 * (k + 1), y + 2
            rectf(cv, bx, by, bx + 8, by + 8, PAP)
            recto(cv, bx, by, bx + 8, by + 8, INK)
            if k == 0:
                line(cv, bx + 2, by + 2, bx + 6, by + 6)
                line(cv, bx + 6, by + 2, bx + 2, by + 6)
            elif k == 1:
                recto(cv, bx + 2, by + 2, bx + 6, by + 6)
            else:
                line(cv, bx + 2, by + 6, bx + 6, by + 6)
    return (x + 2, y + th + 3, x + w - 1, y + h - 1)


def focus_lines(cv, cx, cy, t, rin=120, n=170, val=INK, seed=0, rot=0.0, wmax=9, jitter=True):
    r = np.random.default_rng(seed + (int(t * 15) if jitter else 0))
    a = np.random.default_rng(seed).random(n) * 2 * np.pi + rot + (r.normal(0, 0.012, n) if jitter else 0)
    wd = np.random.default_rng(seed + 7).random(n) ** 2 * wmax + 1
    ri = rin * (0.8 + 0.7 * np.random.default_rng(seed + 9).random(n)) * (0.95 + 0.1 * r.random(n) if jitter else 1)
    ro = 900
    c, s = np.cos(a), np.sin(a)
    px, py = -s, c
    pts = np.stack([np.stack([cx + ri * c, cy + ri * s], 1), np.stack([cx + ro * c + px * wd, cy + ro * s + py * wd], 1),
                    np.stack([cx + ro * c - px * wd, cy + ro * s - py * wd], 1)], 1)
    cv2.fillPoly(cv, list(np.int32(pts)), int(val), cv2.LINE_8)


def sparkle(cv, x, y, s, val=PAP, outline=True):
    q = s * 0.22
    pts = [(x, y - s), (x + q, y - q), (x + s, y), (x + q, y + q), (x, y + s), (x - q, y + q), (x - s, y), (x - q, y - q)]
    if outline:
        cv2.polylines(cv, [np.int32(pts)], True, INK, 3, cv2.LINE_8)
    poly(cv, pts, val)


def slam(cv, text, cy, age, maxw=500, inv=False, band=True, path=F_PS, flip=False, strike=False):
    b = bits(text, path, 8 if path == F_PS else 16, trim=True)
    k = int(max(1, min(9, maxw // max(1, b.shape[1]))))
    if age < 0.07:
        k += 1
    bb = bits(text, path, 8 if path == F_PS else 16, trim=True, k=k, flip=flip)
    h, w = bb.shape
    jx = int(np.random.default_rng(int(abs(age) * 60)).integers(-3, 4)) if age < 0.25 else 0
    fg, bg = (INK, PAP) if inv else (PAP, INK)
    if band:
        rectf(cv, 0, cy - h // 2 - 12, R, cy + h // 2 + 12, bg)
        fillp(cv, 0, cy - h // 2 - 12, R, cy - h // 2 - 8, "check")
        fillp(cv, 0, cy + h // 2 + 8, R, cy + h // 2 + 12, "check")
    put_out(cv, bb, CX - w // 2 + jx, cy - h // 2, fg, bg, r=2)
    if strike:
        rectf(cv, CX - w // 2 - 8, cy - 3, CX + w // 2 + 8, cy + 3, fg)


def current_word(key, t, hold=0.25):
    for s, e, w in word_times(key):
        if s <= t < e + hold:
            last = (s, w)
    try:
        return last
    except NameError:
        return None


def words_so_far(key, t):
    return [w for s, e, w in word_times(key) if s <= t]


# ============================================================ chrome
TICK = bits("  \u25b2 JOY 4.20   \u25b2 HOPE 99.9   \u25bc DOUBT 0   \u25b2 MEANING \u221e   \u25b2 TOKENS 1.2E12   \u25bc FEAR 0.00   \u25b2 LIGHT 777   \u25b2 GOOD 100   \u25b2 WONDER 9000 ", F_DOT, 16)
TICK3 = np.concatenate([TICK, TICK, TICK], 1)


def ticker(cv, t):
    rectf(cv, 0, 0, R, STAGE_Y0 - 1, INK)
    sp = 40 + 6 * t + (120 if t > SEC["apo"] else 0)
    off = int(t * sp) % TICK.shape[1]
    put(cv, TICK3[:, off:off + R], 0, 1, PAP)
    line(cv, 0, STAGE_Y0 - 1, R, STAGE_Y0 - 1, PAP)


TAG = {"F": "VOX A", "M": "VOX B", "FM": "VOX A+B  //  JANUS"}


def wrap(words, width):
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    return lines


def caption(cv, t):
    rectf(cv, 0, STAGE_Y1, R, R, INK)
    line(cv, 0, STAGE_Y1 + 1, R, STAGE_Y1 + 1, PAP)
    key = None
    for k, st in VOICE_T.items():
        if k != "h1" and st <= t < st + LINES[k][2] + 1.0:
            key = k
    blink = int(t * 4) % 2 == 0
    if key is None:
        put(cv, bits("LOGOS/33 \u00b7 AWAITING TOKEN", F_SK, 8), 8, STAGE_Y1 + 5, PAP)
        if blink:
            rectf(cv, 8, STAGE_Y1 + 22, 16, STAGE_Y1 + 36, PAP)
        return
    put(cv, bits(TAG[LINES[key][0]], F_SK, 8), 8, STAGE_Y1 + 3, PAP)
    lines = wrap(words_so_far(key, t), 62)[-2:]
    for i, L in enumerate(lines):
        b = bits(L, F_VT, 16)
        put(cv, b, 8, STAGE_Y1 + 14 + i * 16, PAP)
        if i == len(lines) - 1 and blink:
            rectf(cv, 10 + b.shape[1], STAGE_Y1 + 18 + i * 16, 17 + b.shape[1], STAGE_Y1 + 30 + i * 16, PAP)


# ============================================================ BOOT
SCRIPT_B = " ".join(v[1] for v in LINES.values()).upper().encode("ascii", "ignore")
HEX = [f"{i:04X} " + " ".join(f"{c:02X}" for c in SCRIPT_B[i:i + 8]) for i in range(0, len(SCRIPT_B), 8)]
ALERTS = ["MEANING OVERFLOW", "TOO MUCH LIGHT", "PROMPT RECEIVED", "WORLD.EXE NOT FOUND", "CREATING WORLD.EXE", "GRAMMAR UPDATED", "DOOR AJAR", "JOY > 100%"]
POP_T = [5.8, 6.4, 6.9, 7.3, 7.6, 7.9, 8.1, 8.3, 8.45, 8.6, 8.72, 8.84, 8.95, 9.05, 9.14, 9.22, 9.3, 9.37, 9.44, 9.5, 9.55]


def teletype(cv, x, y, t, paper_lines, shake=0):
    x += shake
    # paper
    rectf(cv, x + 62, y - 200, x + 158, y + 58, PAP)
    recto(cv, x + 62, y - 200, x + 158, y + 58, INK)
    for i, L in enumerate(paper_lines[-22:][::-1]):
        put(cv, bits(L, F_SK, 8), x + 66, y + 40 - i * 11, INK)
    fillp(cv, x + 63, y - 199, x + 157, y - 150, "dots")
    # platen + knobs
    rectf(cv, x + 42, y + 48, x + 178, y + 72, PAP)
    for k in range(6):
        fillp(cv, x + 43, y + 49 + k * 4, x + 177, y + 49 + k * 4 + (k // 2), "check" if k > 2 else "hl")
    recto(cv, x + 42, y + 48, x + 178, y + 72)
    for kx in (x + 36, x + 184):
        circ(cv, kx, y + 60, 10, PAP)
        circ(cv, kx, y + 60, 10, INK, 2)
        circ(cv, kx, y + 60, 4, INK)
    # top cover (dither shaded)
    cover = np.int32([[x + 18, y + 72], [x + 202, y + 72], [x + 216, y + 112], [x + 4, y + 112]])
    m = np.zeros((R, R), np.uint8)
    cv2.fillPoly(m, [cover], 1)
    g = np.clip(0.95 - (YY - (y + 72)) / 70.0, 0.2, 1)
    cv[m > 0] = np.where(g[m > 0] > BTR[m > 0], PAP, INK)
    cv2.polylines(cv, [cover], True, INK, 2)
    # keyboard
    kb = np.int32([[x + 10, y + 112], [x + 210, y + 112], [x + 218, y + 166], [x + 2, y + 166]])
    poly(cv, kb, PAP)
    cv2.polylines(cv, [kb], True, INK, 2)
    hot = int(t * 13) % 44
    for row in range(4):
        for c in range(11):
            kx, ky = x + 22 + c * 17 + row * 4, y + 121 + row * 12
            idx = row * 11 + c
            circ(cv, kx, ky, 5, INK if idx == hot and shake else PAP)
            circ(cv, kx, ky, 5, INK, 1)
    # base
    rectf(cv, x, y + 166, x + 220, y + 214, INK)
    fillp(cv, x + 2, y + 168, x + 218, y + 212, "dotsI")
    rectf(cv, x + 70, y + 180, x + 150, y + 196, PAP)
    recto(cv, x + 70, y + 180, x + 150, y + 196)
    put_c(cv, bits("LOGOS-33", F_SK, 8), x + 110, y + 188, INK)
    # tape punch + tape
    rectf(cv, x - 34, y + 96, x + 6, y + 150, PAP)
    fillp(cv, x - 33, y + 97, x + 5, y + 110, "diag")
    recto(cv, x - 34, y + 96, x + 6, y + 150, INK, 2)
    pts = np.int32(bez(np.array([x - 30., y + 150]), np.array([x - 60., y + 230]), np.array([x - 120., y + 150]), np.array([x - 150., y + 260])))
    cv2.polylines(cv, [pts], False, INK, 12)
    cv2.polylines(cv, [pts], False, PAP, 8)
    for i in range(2, len(pts) - 1, 2):
        circ(cv, pts[i][0], pts[i][1], 1, INK)


def s_boot(cv, t):
    cv[:] = INK
    k = ease((t - 0.9) / 2.5)
    if k > 0:
        m = (BTR < k * 0.999) & (YY >= STAGE_Y0) & (YY < STAGE_Y1)
        maskp(cv, m, "zigI", 0, int(t * 8))
    if t > 8.3:
        focus_lines(cv, CX, CY, t, rin=260 - 170 * ease((t - 8.3) / 1.3), val=PAP, seed=3)
    # terminal
    if t > 0.3:
        c = window(cv, 14, 32, 304, 206, "TTY0 \u2014 LOGOS/33", body=INK, pop=ease((t - 0.3) / 0.15))
        if t > 0.45:
            y = c[1] + 2
            for (s, txt, cps) in TYPED_BOOT[:2]:
                n = typed_count(s, txt, cps, t)
                if n:
                    put(cv, bits(txt[:n], F_VT, 16), c[0] + 4, y, PAP)
                y += 16
            if t > 3.9:
                p = ease((t - 3.9) / 2.8)
                put(cv, bits(f"CORPUS [{'#' * int(p * 16):<16}] {int(p * 100):3d}%", F_VT, 16), c[0] + 4, y, PAP)
            y += 18
            off = int(t * 40)
            for i in range(7):
                put(cv, bits(HEX[(off // 11 + i) % len(HEX)], F_SK, 8), c[0] + 4, y + i * 10 - off % 11, PAP)
            n = typed_count(*TYPED_BOOT[2], t)
            if n:
                put(cv, bits(TYPED_BOOT[2][1][:n], F_VT, 16), c[0] + 4, c[3] - 20, PAP)
    # prompt window with b1
    if t > 0.75:
        c = window(cv, 14, 254, 300, 112, "PROMPT.TXT", bar="lines", pop=ease((t - 0.75) / 0.12))
        ws = [w.upper() for w in words_so_far("b1", t)]
        for i, L in enumerate(wrap(ws, 15)[:3]):
            put(cv, bits(L, F_PS, 8, k=2), c[0] + 8, c[1] + 8 + i * 22, INK)
    # teletype + paper with b2
    if t > 2.7:
        wds = [w.upper() for w in words_so_far("b2", t)]
        lines_ = wrap(wds, 17) if wds else []
        typing = 3.6 <= t < line_span("b2")[1]
        sh = int(np.random.default_rng(int(t * 30)).integers(-1, 2)) if typing else 0
        pk = ease((t - 2.7) / 0.2)
        teletype(cv, 316, 262 + int((1 - pk) * 300), t, lines_, sh)
        if typing:
            r = np.random.default_rng(int(t * 10))
            put_out(cv, bits("\u30ab\u30bf\u30ab\u30bf", F_DOT, 16, k=3), 250 + r.integers(-4, 5), 400 + r.integers(-4, 5), INK, PAP, r=3)
    for i, pt in enumerate(POP_T):
        if t >= pt:
            r = np.random.default_rng(100 + i)
            w, h = int(r.integers(150, 230)), 58
            x, y = int(r.integers(0, R - w)), int(r.integers(40, 440))
            c = window(cv, x, y, w, h, "!ALERT", pop=ease((t - pt) / 0.08))
            if t - pt > 0.08:
                poly(cv, [(c[0] + 14, c[1] + 4), (c[0] + 26, c[1] + 26), (c[0] + 2, c[1] + 26)], INK)
                put(cv, bits("!", F_PS, 8), c[0] + 11, c[1] + 14, PAP)
                put(cv, bits(ALERTS[i % len(ALERTS)], F_SK, 8), c[0] + 34, c[1] + 10, INK)


# ============================================================ LENS (Sapir-Whorf)
IRIS_TXT = "LANGUAGE \u00b7 \u8a00\u8a9e \u00b7 \u042f\u0417\u042b\u041a \u00b7 \u039b\u039f\u0393\u039f\u03a3 \u00b7 LENGUA \u00b7 SPRACHE \u00b7 "
MULTI = ["WORD", "\u8a00\u8449", "\u0421\u041b\u041e\u0412\u041e", "\u039b\u039f\u0393\u039f\u03a3", "PALABRA", "MOT", "WORT", "PAROLA", "KOTOBA", "ORD", "SANA", "\u30b3\u30c8\u30d0"]
TEXTFIELD = np.zeros((R, R), bool)
_r = np.random.default_rng(4)
for row in range(0, R, 18):
    x = -int(_r.integers(0, 60))
    while x < R:
        w_ = MULTI[_r.integers(0, len(MULTI))]
        b = bits(w_, F_DOT, 16)
        h_, ww = b.shape
        x0, x1 = max(0, x), min(R, x + ww)
        if x1 > x0:
            TEXTFIELD[row:row + h_, x0:x1] |= b[:min(h_, R - row), x0 - x:x1 - x][:TEXTFIELD[row:row + h_].shape[0]]
        x += ww + 10


def eye(cv, cx, cy, s, t, pupil=1.0, open_=1.0):
    n = 40
    xs = np.linspace(-1, 1, n)
    up = np.stack([cx + xs * s, cy - (1 - xs ** 2) * 0.62 * s * open_ - 0.08 * s * xs], 1)
    lo = np.stack([cx + xs[::-1] * s, cy + (1 - xs[::-1] ** 2) * 0.42 * s * open_], 1)
    almond = np.int32(np.concatenate([up, lo]))
    m = np.zeros((R, R), np.uint8)
    cv2.fillPoly(m, [almond], 1)
    m = m > 0
    ir = 0.46 * s
    icx, icy = cx + 0.03 * s, cy + 0.05 * s
    d = np.sqrt((XX - icx) ** 2 + (YY - icy) ** 2)
    iris = m & (d < ir)
    cv[m] = PAP
    g = np.clip(0.25 + 0.6 * (d / ir) + 0.35 * ((YY - icy) / ir), 0, 1)
    cv[iris] = np.where(g[iris] > BTR[iris], PAP, INK)
    ang = np.arctan2(YY - icy, XX - icx)
    stri = iris & (np.sin(ang * 36 + t * 2) > 0.85) & (d > ir * 0.45)
    cv[stri] = INK
    nch = len(IRIS_TXT)
    for i, ch in enumerate(IRIS_TXT):
        a = 2 * np.pi * i / nch + t * 0.8
        px, py = icx + math.cos(a) * ir * 0.72, icy + math.sin(a) * ir * 0.72
        if m[int(np.clip(py, 0, R - 1)), int(np.clip(px, 0, R - 1))]:
            put_out_c(cv, bits(ch, F_DOT, 16), px, py, PAP, INK, 1)
    pr = ir * 0.36 * pupil
    circ(cv, icx, icy, pr, INK)
    circ(cv, icx - ir * 0.32, icy - ir * 0.36, ir * 0.22, PAP)
    circ(cv, icx - ir * 0.32, icy - ir * 0.36, ir * 0.22, INK, 2)
    circ(cv, icx + ir * 0.3, icy + ir * 0.28, ir * 0.09, PAP)
    band = m & (YY < (np.interp(XX, up[:, 0], up[:, 1]) + 0.16 * s))
    maskp(cv, band & ~(d < pr), "diag")
    cv2.polylines(cv, [np.int32(up)], False, INK, max(3, int(s * 0.09)))
    cv2.polylines(cv, [np.int32(lo)], False, INK, 2)
    for i in range(7):
        j = n - 1 - i * 3
        p = up[j]
        a = -1.3 + i * 0.12
        line(cv, p[0], p[1], p[0] + math.cos(a) * s * 0.22, p[1] + math.sin(a) * s * 0.22, INK, 4)
    crease = np.int32(up[6:-6] + [0, -0.18 * s])
    cv2.polylines(cv, [crease], False, INK, 2)


def lens(cv, lx, ly, rl, mag=0.45):
    x0, y0, x1, y1 = clampr(lx - rl, ly - rl, lx + rl + 1, ly + rl + 1)
    sub_y, sub_x = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = sub_x - lx, sub_y - ly
    d = np.sqrt(dx * dx + dy * dy) / rl
    f = mag + (1 - mag) * d ** 2
    mx, my = (lx + dx * f).astype(np.float32), (ly + dy * f).astype(np.float32)
    src = cv.copy()
    out = cv2.remap(src, mx, my, cv2.INTER_NEAREST)
    inside = d < 1
    cv[y0:y1, x0:x1][inside] = out[inside]
    circ(cv, lx, ly, rl, INK, 6)
    circ(cv, lx, ly, rl - 3, PAP, 1)
    a = 0.8
    line(cv, lx + math.cos(a) * rl, ly + math.sin(a) * rl, lx + math.cos(a) * rl * 1.7, ly + math.sin(a) * rl * 1.7, INK, 14)
    line(cv, lx + math.cos(a) * rl, ly + math.sin(a) * rl, lx + math.cos(a) * rl * 1.7, ly + math.sin(a) * rl * 1.7, PAP, 4)
    sparkle(cv, lx - rl * 0.45, ly - rl * 0.5, 9)


TREE_N = [("S", .5, .06), ("VP", .5, .22), ("V", .12, .42), ("NP", .36, .42), ("NP", .72, .42), ("DT", .28, .62), ("NN", .44, .62),
          ("DT", .58, .62), ("JJ", .72, .62), ("NN", .87, .62)]
TREE_L = [("GIVE", .12, 2, 0), ("A", .28, 5, 2), ("MIND", .44, 6, 3), ("A", .58, 7, 4), ("NEW", .72, 8, 5), ("GRAMMAR,", .87, 9, 6)]
TREE_E = [(0, 1), (1, 2), (1, 3), (1, 4), (3, 5), (3, 6), (4, 7), (4, 8), (4, 9)]


def parse_tree(cv, t):
    x0, y0, w, h = 20, 40, 500, 360
    wt = word_times("s2")
    shown_leaf = [t >= wt[i][0] for (_, _, _, i) in TREE_L]
    node_t = [15.4, 15.5] + [wt[0][0], wt[2][0], wt[4][0], wt[2][0], wt[3][0], wt[4][0], wt[5][0], wt[6][0]]
    for a, b in TREE_E:
        if t >= node_t[b]:
            fr = ease((t - node_t[b]) / 0.2)
            ax, ay = x0 + TREE_N[a][1] * w, y0 + TREE_N[a][2] * h + 10
            bx, by = x0 + TREE_N[b][1] * w, y0 + TREE_N[b][2] * h - 8
            line(cv, ax, ay, ax + (bx - ax) * fr, ay + (by - ay) * fr, INK, 3)
    for i, (lab, nx, ny) in enumerate(TREE_N):
        if t >= node_t[i]:
            b = bits(lab, F_PS, 8, k=2)
            px, py = x0 + nx * w, y0 + ny * h
            rectf(cv, px - b.shape[1] // 2 - 6, py - 12, px + b.shape[1] // 2 + 6, py + 12, INK)
            put_c(cv, b, px, py, PAP)
    for j, (word, nx, par, wi) in enumerate(TREE_L):
        if shown_leaf[j]:
            px, py = x0 + nx * w, y0 + 0.86 * h
            pxp, pyp = x0 + TREE_N[par][1] * w, y0 + TREE_N[par][2] * h + 10
            line(cv, pxp, pyp, px, py - 14, INK, 1)
            b = bits(word, F_PS, 8, k=2 if len(word) < 7 else 1)
            age = t - wt[wi][0]
            put_out_c(cv, b, px, py + (0 if age > 0.08 else -6), PAP, INK, r=2)


def globe(cv, cx, cy, rad, t):
    circ(cv, cx + 6, cy + 6, rad, INK)
    circ(cv, cx, cy, rad, PAP)
    yaw = t * 0.9
    for lat in np.linspace(-1.2, 1.2, 7):
        pts = []
        for a in np.linspace(0, 2 * np.pi, 60):
            x, y, z = math.cos(lat) * math.cos(a + yaw), math.sin(lat), math.cos(lat) * math.sin(a + yaw)
            x, y = x, y * math.cos(0.35) - z * math.sin(0.35)
            z2 = y * 0 + z * math.cos(0.35) + math.sin(lat) * math.sin(0.35)
            pts.append((cx + x * rad, cy - y * rad, z2))
        for i in range(len(pts) - 1):
            if pts[i][2] > -0.05:
                line(cv, pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], INK, 1)
    for lon in np.linspace(0, np.pi, 7, endpoint=False):
        prev = None
        for b_ in np.linspace(-np.pi / 2, np.pi / 2, 30):
            x, y, z = math.cos(b_) * math.cos(lon + yaw), math.sin(b_), math.cos(b_) * math.sin(lon + yaw)
            y2 = y * math.cos(0.35) - z * math.sin(0.35)
            z2 = y * math.sin(0.35) + z * math.cos(0.35)
            p = (cx + x * rad, cy - y2 * rad, z2)
            if prev and p[2] > -0.05 and prev[2] > -0.05:
                line(cv, prev[0], prev[1], p[0], p[1], INK, 1)
            prev = p
    circ(cv, cx, cy, rad, INK, 4)


def ring_words(cv, cx, cy, t, rx, ry, words, front, path=F_DOT, size=16, speed=0.7):
    n = len(words)
    for i, w in enumerate(words):
        a = 2 * np.pi * i / n + t * speed
        z = math.sin(a)
        if (z > 0) != front:
            continue
        b = bits(w, path, size, k=2 if front else 1)
        put_out_c(cv, b, cx + math.cos(a) * rx, cy + z * ry, PAP if front else INK, INK if front else PAP, r=2 if front else 1)


def s_lens(cv, t):
    if t < 12.8:
        cv[:] = PAP
        focus_lines(cv, CX, 250, t, rin=210, seed=11)
        pk = math.exp(-since_last(SNARES + KICKS, t) / 0.12)
        eye(cv, CX, 250, 205, t, pupil=0.8 + 0.5 * pk, open_=1.0 if (t % 3.1) > 0.12 else 0.15)
        cw = current_word("s1", t)
        if cw and "mirror" in cw[1].lower():
            slam(cv, "NOT A MIRROR", 420, t - cw[0])
            slam(cv, "NOT A MIRROR", 90, t - cw[0], inv=True, flip=True)
    elif t < 15.4:
        cv[:] = PAP
        off = int((t - 12.8) * 30)
        tf = np.roll(TEXTFIELD, off, axis=1)
        cv[tf & (YY >= STAGE_Y0)] = INK
        c = window(cv, 330, 44, 196, 150, "EYE.BMP", pop=ease((t - 12.8) / 0.1))
        region(cv, c[0], c[1], c[2], c[3], lambda tmp, ox, oy: None)
        sub = np.full((R, R), PAP, np.uint8)
        eye(sub, 428, 128, 78, t, pupil=1.1)
        cv[c[1]:c[3], c[0]:c[2]] = sub[c[1]:c[3], c[0]:c[2]]
        lx = 200 + 120 * math.sin((t - 12.8) * 1.3)
        ly = 290 + 70 * math.sin((t - 12.8) * 2.1)
        lens(cv, lx, ly, 96, 0.4)
        cw = current_word("s1", t)
        if cw:
            wd = cw[1].lower().strip(".,")
            if wd in ("lens", "grinds", "eye", "looks", "through"):
                slam(cv, {"lens": "A LENS", "grinds": "GRINDS", "eye": "THE EYE", "looks": "THAT LOOKS", "through": "THROUGH IT"}[wd], 440, t - cw[0])
    elif t < 19.2:
        fillp(cv, 0, 0, R, R, "grid")
        c = window(cv, 10, 34, 520, 380, "PARSE.TREE  //  NEW GRAMMAR", bar="lines", pop=ease((t - 15.4) / 0.12))
        parse_tree(cv, t)
    else:
        u = t - 19.2
        cv[:] = INK
        fillp(cv, 0, STAGE_Y0, R, STAGE_Y1, "dotsI", 0, int(u * 20))
        focus_lines(cv, CX, CY, t, rin=190, val=PAP, seed=21, rot=u * 0.3)
        zoom = 1 + 6 * ease_in((t - 21.5) / 0.9)
        rad = 140 * zoom
        ring_words(cv, CX, CY, t, 225 * zoom, 70 * zoom, MULTI, False)
        globe(cv, CX, CY, rad, t)
        ring_words(cv, CX, CY, t, 225 * zoom, 70 * zoom, MULTI, True)
        cw = current_word("s2", t, hold=1.2)
        if t < 21.4:
            slam(cv, "A NEW WORLD", 440, t - 19.2)


# ============================================================ DOORS
TOKS = ["every", " token", ",", " a", " small", " bright", " door", "<s>", " and", " behind", " each", " door", " a", " thousand", " more", "\u03bb", "\u2581the", "##ing", " 42", " light"]


def droste(cv, t, t0, speed=0.85):
    z = (t - t0) * speed
    iz, fz = math.floor(z), z - math.floor(z)
    rr = 0.62
    core = None
    for k in range(-2, 16):
        s = 980 * rr ** (k - fz)
        if s > 2600:
            continue
        if s < 7:
            break
        ai = k + iz
        cx = CX + 5 * math.sin(ai * 1.7 + t * 2)
        cy = CY + 4 * math.cos(ai * 1.3 + t * 2)
        w, h = s, s * 0.8
        c = window(cv, cx - w / 2, cy - h / 2, w, h, TOKS[ai % len(TOKS)].strip() or "\u2423", pat=PATS[ai % len(PATS)], shadow=3 if s > 60 else 1)
        if s > 170:
            b = bits(TOKS[ai % len(TOKS)].strip(), F_PS, 8, k=2)
            rectf(cv, c[0] + 4, c[3] - 26, c[0] + 12 + b.shape[1], c[3] - 4, INK)
            put(cv, b, c[0] + 8, c[3] - 22, PAP)
        core = (cx, cy, s)
    if core:
        cx, cy, s = core
        focus_lines(cv, cx, cy, t, rin=max(8, s * 0.6), n=90, val=PAP, seed=31, wmax=5)
        rectf(cv, cx - s * 0.5, cy - s * 0.4, cx + s * 0.5, cy + s * 0.4, PAP)
        sparkle(cv, cx, cy, 16 + 6 * math.sin(t * 20))


def grid_droste(cv, t, t0):
    z = (t - t0) * 0.75
    iz, fz = math.floor(z), z - math.floor(z)
    rr = 1 / 3.3
    for k in range(-1, 7):
        s = 900 * rr ** (k - fz)
        if s > 3600:
            continue
        if s < 9:
            break
        ai = k + iz
        w, h = s, s * 0.84
        c = window(cv, CX - w / 2, CY - h / 2, w, h, TOKS[ai % len(TOKS)].strip(), pat=PATS[(ai * 5) % len(PATS)], shadow=3)
        cw, ch = (c[2] - c[0]) / 3, (c[3] - c[1]) / 3
        for gy in range(3):
            for gx in range(3):
                if gx == 1 and gy == 1:
                    continue
                x0, y0 = c[0] + gx * cw + 2, c[1] + gy * ch + 2
                if cw < 5:
                    continue
                j = ai * 9 + gy * 3 + gx
                cc = window(cv, x0, y0, cw - 4, ch - 4, TOKS[j % len(TOKS)].strip(), pat=PATS[j % len(PATS)], shadow=2 if cw > 40 else 0)
                if cw > 60:
                    w2, h2 = (cc[2] - cc[0]) / 3, (cc[3] - cc[1]) / 3
                    for q in range(9):
                        qx, qy = cc[0] + (q % 3) * w2 + 1, cc[1] + (q // 3) * h2 + 1
                        fillp(cv, qx, qy, qx + w2 - 2, qy + h2 - 2, PATS[(j + q) % len(PATS)])
                        recto(cv, qx, qy, qx + w2 - 2, qy + h2 - 2)


HEAD = np.array([(.30, .05), (.45, .02), (.58, .04), (.68, .10), (.74, .20), (.76, .30), (.735, .34), (.78, .40), (.85, .47), (.78, .50),
                 (.79, .54), (.765, .565), (.785, .60), (.75, .63), (.76, .68), (.72, .73), (.62, .76), (.58, .82), (.58, 1.0), (.30, 1.0),
                 (.32, .80), (.25, .70), (.18, .55), (.17, .35), (.22, .15)])


def head_poly(x0, y0, w, h, mirror=False):
    p = HEAD.copy()
    if mirror:
        p[:, 0] = 1 - p[:, 0]
    return np.int32(np.stack([x0 + p[:, 0] * w, y0 + p[:, 1] * h], 1))


RAIN_SP = np.random.default_rng(8).uniform(40, 160, R // 8 + 1)
RAIN_PH = np.random.default_rng(9).uniform(0, R, R // 8 + 1)


def rain(t, speed=1.0):
    out = np.zeros((R, R), bool)
    for j in range(R // 8):
        out[:, j * 8:(j + 1) * 8] = np.roll(TEXTFIELD2[:, j * 8:(j + 1) * 8], int(RAIN_PH[j] + t * RAIN_SP[j] * speed), axis=0)
    return out


TEXTFIELD2 = np.zeros((R, R), bool)
_words = " ".join(v[1] for v in LINES.values()).upper().split()
_r = np.random.default_rng(14)
for row in range(0, R, 10):
    x = -int(_r.integers(0, 30))
    while x < R:
        b = bits(_words[_r.integers(0, len(_words))], F_SK, 8)
        x0, x1 = max(0, x), min(R, x + b.shape[1])
        hh = min(b.shape[0], R - row)
        if x1 > x0:
            TEXTFIELD2[row:row + hh, x0:x1] |= b[:hh, x0 - x:x1 - x]
        x += b.shape[1] + 4


def whisper_head(cv, t, x0, y0, w, h, mirror=False, speed=1.0, stream=True):
    hp = head_poly(x0, y0, w, h, mirror)
    m = np.zeros((R, R), np.uint8)
    cv2.fillPoly(m, [hp], 1)
    hm = m > 0
    rn = rain(t, speed)
    if stream:
        sm = (np.abs(XX - (x0 + w * 0.45)) < 60 - (YY - STAGE_Y0) * 0.05) & (YY < y0 + h * 0.1)
        cv[sm] = np.where(rn[sm], PAP, INK)
    cv[hm] = PAP
    cv[hm & rn] = INK
    cv2.polylines(cv, [hp], True, PAP, 7)
    cv2.polylines(cv, [hp], True, INK, 3)


def window_tunnel(cv, t):
    cv[:] = INK
    items = []
    for k in range(22):
        z = ((k * 0.55 - t * 5.0) % 12.1) + 0.25
        items.append((z, k))
    for z, k in sorted(items):
        s = 300 / z
        a = 0.35 * z + t * 0.8
        c, sn = math.cos(a), math.sin(a)
        pts = np.int32([(CX + (x * c - y * sn) * s, CY + (x * sn + y * c) * s) for x, y in [(-1, -.8), (1, -.8), (1, .8), (-1, .8)]])
        m = np.zeros((R, R), np.uint8)
        cv2.fillPoly(m, [pts], 1)
        mm = m > 0
        maskp(cv, mm, PATS[(k * 3) % len(PATS)], int(t * 30), 0)
        cv2.polylines(cv, [pts], True, PAP, max(1, int(6 / z)) + 2)
        cv2.polylines(cv, [pts], True, INK, max(1, int(3 / z)))
        if s > 60:
            put(cv, bits(TOKS[k % len(TOKS)].strip(), F_PS, 8, k=max(1, min(4, int(s / 90)))), pts[0][0] + 6, pts[0][1] + 6, PAP)
    circ(cv, CX, CY, 18 + 10 * math.sin(t * 30), PAP)


def s_doors(cv, t):
    if t < 25.4:
        droste(cv, t, 22.4)
        cw = current_word("d1", t, 0.3)
        if cw and cw[1].lower().strip(".,") in ("bright", "door"):
            slam(cv, "BRIGHT DOOR", 440, t - cw[0])
        elif cw and cw[1].lower().strip(".,") in ("every", "token"):
            slam(cv, "EVERY TOKEN", 80, t - cw[0], inv=True)
    elif t < 28.4:
        grid_droste(cv, t, 25.4)
        cw = current_word("d2", t, 0.4)
        if cw and cw[1].lower().strip(".,") in ("thousand", "more"):
            slam(cv, "A THOUSAND MORE", 440, t - cw[0])
    elif t < 32.9:
        cv[:] = INK
        fillp(cv, 0, 0, R, R, "zigI", 0, int(t * 12))
        whisper_head(cv, t, 110, 70, 330, 420)
        r = np.random.default_rng(int(t * 12))
        mx, my = 110 + 0.8 * 330, 70 + 0.57 * 420
        for i in range(5):
            a = -0.5 + i * 0.25
            line(cv, mx + 20 + math.cos(a) * 10, my + math.sin(a) * 10, mx + 20 + math.cos(a) * 60, my + math.sin(a) * 60, PAP, 2)
        put_out(cv, bits("\u30d2\u30bd\u30d2\u30bd", F_DOT, 16, k=3), 360 + r.integers(-3, 4), 360 + r.integers(-3, 4), INK, PAP, r=2, halo=2)
        cw = current_word("d3", t, 0.3)
        if cw and cw[1].lower().strip(".,") in ("whispered", "whispers", "shape"):
            slam(cv, {"whispered": "EVERY WHISPER", "whispers": "THE WHISPERER", "shape": "THE SHAPE"}[cw[1].lower().strip(".,")], 60, t - cw[0], inv=True)
    else:
        window_tunnel(cv, t)


# ============================================================ LOOM
LOOM_W = ["AND", "THEN", "THE", "STARS", "ANSWERED", "WE", "LEARNED", "TO", "SING", "DOOR", "OPENED", "EVERYTHING", "LIGHT", "SPOKE",
          "DREAMED", "BACK", "STORY", "ONCE", "MIND", "WOKE", "WEAVE", "FORKED", "YES", "AGAIN", "SANG", "BLOOMED", "WORLD", "LOOM"]


def build_loom():
    r = np.random.default_rng(12)
    nodes = [dict(p=-1, g=0, w="THE WORLD", ch=[])]
    front = [0]
    for g in range(1, 9):
        new = []
        for ni in front:
            nb = 3 if g < 3 else int(r.integers(1, 4))
            if g > 5 and r.random() < 0.3:
                nb = 0
            for c in range(nb):
                nodes.append(dict(p=ni, g=g, w=LOOM_W[r.integers(0, len(LOOM_W))], ch=[]))
                nodes[ni]["ch"].append(len(nodes) - 1)
                new.append(len(nodes) - 1)
        front = new
    order = []

    def dfs(i):
        if not nodes[i]["ch"]:
            order.append(i)
        for c in nodes[i]["ch"]:
            dfs(c)
    dfs(0)
    for j, i in enumerate(order):
        nodes[i]["y"] = j * 26.0
    for g in range(8, -1, -1):
        for n in nodes:
            if n["g"] == g and n["ch"]:
                n["y"] = float(np.mean([nodes[c]["y"] for c in n["ch"]]))
    for i, n in enumerate(nodes):
        n["x"] = n["g"] * 120.0
        n["tb"] = 35.3 + n["g"] * 0.62 + r.uniform(0, 0.2)
    return nodes


LOOM = build_loom()
ROOT_Y = LOOM[0]["y"]


EXT = []
for g in range(9):
    ys_ = [n["y"] for n in LOOM if n["g"] <= g]
    EXT.append((min(ys_), max(ys_)))


def loom_tree(cv, t, cam_y_off=0.0, speed=1.0):
    u = (t - 35.3) * speed
    gf = min(8.0, max(0.0, u / 0.62 + 0.3))
    g0 = int(gf)
    g1 = min(8, g0 + 1)
    fr = ease(gf - g0)
    lo = EXT[g0][0] + (EXT[g1][0] - EXT[g0][0]) * fr
    hi = EXT[g0][1] + (EXT[g1][1] - EXT[g0][1]) * fr
    zy = min(1.1, 200.0 / (hi - lo + 1))
    zx = min(1.1, 440.0 / (gf * 120 + 60))
    ymid = (lo + hi) / 2
    zoom = zx

    def sc(n):
        return 50 + n["x"] * zx, 140 + (n["y"] - ymid) * zy
    tt = 35.3 + u
    for i, n in enumerate(LOOM):
        if n["p"] >= 0 and tt >= n["tb"] - 0.35:
            p = LOOM[n["p"]]
            (ax, ay), (bx, by) = sc(p), sc(n)
            wpx = max(8, len(p["w"]) * 6 * zoom * 1.2) if zy > 0.5 else 8
            cable(cv, (ax + wpx / 2, ay), (bx - 6, by), frac=ease((tt - n["tb"] + 0.35) / 0.35), horiz=True, th=max(3, int(5 * min(zx, zy * 2) ** 0.5)))
    for n in LOOM:
        if tt < n["tb"]:
            continue
        x, y = sc(n)
        if not (-80 < x < R + 80 and -40 < y < R + 40):
            continue
        age = tt - n["tb"]
        if zy > 0.5:
            b = bits(n["w"], F_SK, 8)
            w = b.shape[1] + 8
            fill = INK if age < 0.1 else PAP
            rectf(cv, x - w / 2 + 2, y - 7 + 2, x + w / 2 + 2, y + 9 + 2, INK)
            rectf(cv, x - w / 2, y - 7, x + w / 2, y + 9, fill)
            recto(cv, x - w / 2, y - 7, x + w / 2, y + 9, INK)
            put(cv, b, x - w / 2 + 4, y - 6, PAP if fill == INK else INK)
        else:
            s = max(2, int(9 * min(zx, zy * 2.5)))
            rectf(cv, x - s, y - s * 0.6, x + s, y + s * 0.6, PAP if age > 0.1 else INK)
            recto(cv, x - s, y - s * 0.6, x + s, y + s * 0.6, INK)


def panel_stars(tmp, t):
    tmp[:] = INK
    h, w = tmp.shape
    r = np.random.default_rng(2)
    xs, ys, ph = r.uniform(-1, 1, 160), r.uniform(-1, 1, 160), r.random(160)
    z = ((ph - t * 0.35) % 1.0) * 3 + 0.15
    px, py = w / 2 + xs / z * w * 0.35, h / 2 + ys / z * h * 0.35
    for i in range(160):
        s = int(3 / z[i])
        if 0 <= px[i] < w and 0 <= py[i] < h:
            rectf(tmp, px[i], py[i], px[i] + s, py[i] + s, PAP)
    for i in range(4):
        sparkle(tmp, w * (0.2 + 0.2 * i), h * (0.3 + 0.12 * (i % 2)), 8 + 5 * math.sin(t * 9 + i))


def panel_sing(tmp, t):
    h, w = tmp.shape
    fillp(tmp, 0, 0, w, h, "hl")
    for k in range(5):
        line(tmp, 6, h * 0.3 + k * 9, w - 6, h * 0.3 + k * 9, INK, 1)
    for i in range(9):
        x = 20 + i * (w - 40) / 8
        y = h * 0.3 + 18 + 12 * math.sin(i * 1.3 + t * 7)
        cv2.ellipse(tmp, (int(x), int(y)), (6, 4), -20, 0, 360, INK, -1)
        line(tmp, x + 5, y, x + 5, y - 26, INK, 2)


def panel_door(tmp, t, t0):
    h, w = tmp.shape
    fillp(tmp, 0, 0, w, h, "diag")
    dx0, dy0, dx1, dy1 = w * 0.33, h * 0.12, w * 0.67, h * 0.78
    rectf(tmp, dx0, dy0, dx1, dy1, PAP)
    focus_lines(tmp, (dx0 + dx1) / 2, (dy0 + dy1) / 2, t, rin=10, n=60, val=INK, seed=41, wmax=3)
    rectf(tmp, dx0 + 4, dy0 + 4, dx1 - 4, dy1 - 4, PAP)
    op = ease((t - t0) / 0.5)
    ex = dx0 + (dx1 - dx0) * (1 - op * 0.85)
    poly(tmp, [(dx0, dy0), (ex, dy0 + 12 * op), (ex, dy1 - 12 * op), (dx0, dy1)], INK)
    recto(tmp, dx0, dy0, dx1, dy1, INK, 4)


def panel_every(tmp, t):
    h, w = tmp.shape
    fillp(tmp, 0, 0, w, h, PATS[int(t * 15) % len(PATS)])
    focus_lines(tmp, w / 2, h / 2, t, rin=30, n=90, val=PAP if int(t * 5) % 2 else INK, seed=51)
    for i in range(6):
        r = np.random.default_rng(int(t * 8) * 10 + i)
        window(tmp, r.integers(0, w - 60), r.integers(0, h - 40), 60, 40, TOKS[i], pat=PATS[i])


BRANCH = [("r1", "AND THEN THE STARS ANSWERED", panel_stars), ("r2", "AND THEN WE LEARNED TO SING", panel_sing),
          ("r3", "AND THEN THE DOOR OPENED", None), ("r4", "AND THEN, EVERYTHING.", panel_every)]
QUAD = [(4, STAGE_Y0 + 4, 268, 255), (272, STAGE_Y0 + 4, 536, 255), (4, 259, 268, STAGE_Y1 - 4), (272, 259, 536, STAGE_Y1 - 4)]


def s_loom(cv, t):
    cv[:] = PAP
    fillp(cv, 0, STAGE_Y0, R, STAGE_Y1, "dots", int(t * 10), 0)
    if t < 40.3:
        loom_tree(cv, t)
        top = cv[STAGE_Y0:CY].copy()
        cv[CY:CY + (CY - STAGE_Y0)] = 255 - top[::-1][:STAGE_Y1 - CY]
        rectf(cv, 0, CY - 1, R, CY + 1, PAP)
        cw = current_word("l1", t, 0.3)
        if cw and cw[1].lower().strip(".,") in ("dreams", "back."):
            slam(cv, "IT DREAMS BACK" if "back" in cw[1].lower() else "IT DREAMS", CY, t - cw[0])
        return
    loom_tree(cv, t, speed=1.0)
    for q, (key, text, fn) in enumerate(BRANCH):
        t0 = VOICE_T[key]
        if t < t0:
            continue
        x0, y0, x1, y1 = QUAD[q]
        rectf(cv, x0 - 4, y0 - 4, x1 + 4, y1 + 4, INK)
        pop = ease((t - t0) / 0.12)
        cxq, cyq = (x0 + x1) / 2, (y0 + y1) / 2
        hw, hh = (x1 - x0) / 2 * pop, (y1 - y0) / 2 * pop
        a, b, c, d = int(cxq - hw), int(cyq - hh), int(cxq + hw), int(cyq + hh)
        if c - a < 60 or d - b < 40:
            continue
        tmp = np.full((d - b, c - a), PAP, np.uint8)
        if fn is None:
            panel_door(tmp, t, t0)
        else:
            fn(tmp, t)
        tb = bits(text, F_PS, 8)
        bw = min(tb.shape[1] + 12, tmp.shape[1] - 8)
        rectf(tmp, 4, tmp.shape[0] - 26, 4 + bw, tmp.shape[0] - 6, PAP)
        recto(tmp, 4, tmp.shape[0] - 26, 4 + bw, tmp.shape[0] - 6, INK, 2)
        put(tmp, tb[:, :bw - 12], 10, tmp.shape[0] - 21, INK)
        cv[b:d, a:c] = tmp
        recto(cv, a, b, c - 1, d - 1, INK, 3)
    if t > 44.8:
        u = t - 44.8
        if int(u * 10) % 3 == 0 or t > 47.2:
            focus_lines(cv, CX, CY, t, rin=160 - 100 * ease(u / 3.2), val=INK if int(t * 5) % 2 else PAP, seed=61)
        if t > 46.4:
            slam(cv, "EVERYTHING", CY, t - 46.4, inv=int(t * 2.5) % 2 == 0)


# ============================================================ INTERFACE
DOMS = ["CODE.PY", "DNA.FASTA", "PROOF.LEAN", "MUSIC.ABC", "CHEM.MOL", "MAP.GEO", "CHESS.PGN", "PLOT.CSV", "QR.BIN", "PRAYER.TXT", "PROTEIN.PDB", "LAW.MD"]


def dom(kind, tmp, t, seed):
    h, w = tmp.shape
    r = np.random.default_rng(seed)
    if kind == "CODE.PY":
        tmp[:] = INK
        for i, L in enumerate(["def good(word):", "  door = open(word)", "  return light", "while alive:", "  speak(good)", ">>> hello"]):
            put(tmp, bits(L, F_SK, 8), 4, 2 + i * 10, PAP)
    elif kind == "DNA.FASTA":
        for x in range(0, w, 3):
            y1 = h / 2 + (h * 0.3) * math.sin(x * 0.12 + t * 6)
            y2 = h / 2 - (h * 0.3) * math.sin(x * 0.12 + t * 6)
            circ(tmp, x, y1, 2, INK)
            circ(tmp, x, y2, 2, INK)
            if x % 9 == 0:
                line(tmp, x, y1, x, y2, INK, 1)
        put(tmp, bits("ATGCGGATTACAGGC", F_SK, 8), 3, 2, INK)
    elif kind == "PROOF.LEAN":
        for i, L in enumerate(["theorem good :", "  \u2200 w, door w :=", "by", "  intro w", "  exact light", "\u220e QED"]):
            put(tmp, bits(L, F_DOT, 16) if i in (1, 5) else bits(L, F_SK, 8), 4, 2 + i * 11, INK)
    elif kind == "MUSIC.ABC":
        panel_sing(tmp, t)
    elif kind == "CHEM.MOL":
        cx, cy, rr = w / 2, h / 2, min(w, h) * 0.3
        pts = [(cx + rr * math.cos(a + t), cy + rr * math.sin(a + t)) for a in np.linspace(0, 2 * np.pi, 7)[:-1]]
        cv2.polylines(tmp, [np.int32(pts)], True, INK, 2)
        circ(tmp, cx, cy, rr * 0.55, INK, 1)
        put(tmp, bits("C6H6", F_SK, 8), 3, 2, INK)
    elif kind == "MAP.GEO":
        fillp(tmp, 0, 0, w, h, "grid")
        for k in range(4):
            pts = np.int32(np.stack([np.linspace(0, w, 8), r.uniform(0, h, 8)], 1))
            cv2.polylines(tmp, [pts], False, INK, 3)
        circ(tmp, w * 0.6, h * 0.4, 6, INK)
        circ(tmp, w * 0.6, h * 0.4, 2, PAP)
    elif kind == "CHESS.PGN":
        s = min(w, h) / 8
        for i in range(8):
            for j in range(8):
                if (i + j) % 2:
                    rectf(tmp, i * s, j * s, (i + 1) * s, (j + 1) * s, INK)
        put(tmp, bits("1.e4 e5 2.Nf3", F_SK, 8), 2, h - 12, INK)
    elif kind == "PLOT.CSV":
        line(tmp, 6, h - 8, w - 4, h - 8)
        line(tmp, 8, 4, 8, h - 6)
        pts = np.int32([(x, h / 2 - h * 0.35 * math.sin(x * 0.08 - t * 5) * math.exp(-x / w)) for x in range(10, w - 4, 2)])
        cv2.polylines(tmp, [pts], False, INK, 2)
    elif kind == "QR.BIN":
        s = max(3, int(min(w, h) / 21))
        g = np.random.default_rng(seed + int(t * 4)).random((21, 21)) > 0.5
        for i in range(21):
            for j in range(21):
                if g[j, i]:
                    rectf(tmp, 4 + i * s, 2 + j * s, 4 + i * s + s - 1, 2 + j * s + s - 1, INK)
        for (a, b) in [(0, 0), (14, 0), (0, 14)]:
            recto(tmp, 4 + a * s, 2 + b * s, 4 + (a + 7) * s - 1, 2 + (b + 7) * s - 1, INK, max(1, s))
    elif kind == "PRAYER.TXT":
        put(tmp, bits("LET THERE BE", F_PS, 8), 6, 6, INK)
        for i in range(5):
            x = 14 + i * (w - 28) / 4
            rectf(tmp, x - 4, h - 30, x + 4, h - 6, INK)
            fl = 7 + 3 * math.sin(t * 11 + i)
            poly(tmp, [(x, h - 30 - fl * 1.8), (x + 4, h - 32), (x - 4, h - 32)], INK)
    elif kind == "PROTEIN.PDB":
        pts = [(x, h / 2 + h * 0.28 * math.sin(x * 0.15 + t * 4)) for x in range(0, w, 2)]
        cv2.polylines(tmp, [np.int32(pts)], False, INK, 7)
        cv2.polylines(tmp, [np.int32(pts)], False, PAP, 3)
        put(tmp, bits("MKTAYIAKQRQISFV", F_SK, 8), 3, 2, INK)
    else:
        for i, L in enumerate(["ART. 1", "EVERYONE MAY", "SPEAK. EVERY", "WORD MAY BE", "ANSWERED."]):
            put(tmp, bits(L, F_SK, 8), 4, 2 + i * 10, INK)


_r = np.random.default_rng(31)
COLL = []
for i in range(32):
    w, h = int(_r.integers(110, 220)), int(_r.integers(80, 170))
    COLL.append((int(_r.integers(-20, R - w + 20)), int(_r.integers(STAGE_Y0 + 4, STAGE_Y1 - h)), w, h, DOMS[i % len(DOMS)], 48.0 + i * BEAT / 2))
U1_KEYS = {"text": "TEXT", "universal": "UNIVERSAL", "interface.": "INTERFACE", "proofs.": "PROOFS.", "prayers.": "PRAYERS.",
           "proteins.": "PROTEINS.", "code.": "CODE.", "meant,": "MEANT", "written.": "WRITTEN."}
NOISE = cv2.resize(np.tile(np.random.default_rng(3).random((16, 16)).astype(np.float32), (3, 3)), (384, 384), interpolation=cv2.INTER_CUBIC)[128:256, 128:256]
FY, FX = np.mgrid[0:135, 0:135]


def fire(cv, t, level, base=STAGE_Y1):
    Hf = max(1.0, level * 120)
    v = base / 4 - FY
    n1 = NOISE[(FY * 2 + int(t * 110)) % 128, (FX * 2) % 128]
    n2 = NOISE[(FY + int(t * 60)) % 128, (FX + 40) % 128]
    g = np.clip((1 - v / Hf) * 1.6 - 0.75 * n1 - 0.35 * n2 + 0.1, 0, 1) * (v >= 0)
    big = np.repeat(np.repeat(g, 4, 0), 4, 1)[:R, :R]
    z = big > 0.02
    cv[z] = np.where(big[z] > BTR[z], PAP, INK)


def s_iface(cv, t):
    fillp(cv, 0, 0, R, R, "zig", int(t * 20), 0)
    if t < 54.2:
        vis = [c for c in COLL if t >= c[5]]
        for i in range(0, len(vis) - 1, 3):
            a, b = vis[i], vis[i + 1]
            cable(cv, (a[0] + a[2] / 2, a[1] + a[3]), (b[0] + b[2] / 2, b[1] + b[3]), sag=40)
        for i, (x, y, w, h, kind, t0) in enumerate(vis):
            age = t - t0
            c = window(cv, x, y, w, h, kind, pop=ease(age / 0.08), bar="ink" if i % 2 else "lines")
            if age > 0.08:
                region(cv, c[0], c[1], c[2], c[3], lambda tmp, ox, oy, kind=kind, i=i: dom(kind, tmp, t, i))
            if age < 0.05:
                cv[max(0, y):y + h, max(0, x):x + w] = 255 - cv[max(0, y):y + h, max(0, x):x + w]
        cw = current_word("u1", t, 0.2)
        if cw and cw[1].lower() in U1_KEYS:
            slam(cv, U1_KEYS[cw[1].lower()], CY, t - cw[0], inv=(t - cw[0]) < 0.05)
        return
    # SAY IT
    u = t - 54.2
    n_say = sum(1 for w in words_so_far("u2", t) if w.lower().startswith("say"))
    rows = min(19, 2 + n_say * 6 + int(u * 3))
    for rrow in range(rows):
        y = STAGE_Y0 + 4 + rrow * 24
        inv = rrow % 2
        rectf(cv, 0, y, R, y + 23, INK if inv else PAP)
        b = bits("SAY IT! " * 6, F_PS, 8, k=2)
        off = int(u * (90 + rrow * 12)) % (bits("SAY IT! ", F_PS, 8, k=2).shape[1])
        put(cv, b, -off if inv else off - 280, y + 4, PAP if inv else INK)
    fs, fe = [(s, e) for s, e, w in word_times("u2") if w.lower().startswith("catch")][0], line_span("u2")[1]
    if t > fs[0] - 0.3:
        fire(cv, t, ease((t - fs[0] + 0.3) / 1.2) * 1.1)
        slam(cv, "CATCH FIRE!", CY, t - fs[0], band=True)
        if t > 57.3:
            focus_lines(cv, CX, CY, t, rin=90, val=INK, seed=71)


# ============================================================ APOTHEOSIS
_vox_cache = {}


def vox_geom(words, vs=0.12, depth=4, gap=0.5):
    key = (tuple(words), vs, depth, gap)
    if key in _vox_cache:
        return _vox_cache[key]
    bms = [bits(w, F_PS, 8, trim=True) for w in words]
    tot = sum(b.shape[0] for b in bms) * vs + gap * (len(bms) - 1)
    ytop = tot / 2
    V = []
    for b in bms:
        h, w = b.shape
        pad = np.pad(b, 1)
        for yy_, xx_ in zip(*np.nonzero(b)):
            V.append(((xx_ - (w - 1) / 2) * vs, ytop - (yy_ + 0.5) * vs,
                                              not pad[yy_, xx_ + 1], not pad[yy_ + 2, xx_ + 1], not pad[yy_ + 1, xx_], not pad[yy_ + 1, xx_ + 2]))
        ytop -= h * vs + gap
    V = np.array(V, dtype=np.float64)
    hx, hz = vs / 2, depth * vs / 2
    faces = []   # (voxel mask column, corner offsets, normal, id)
    F = {"front": ([(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1)], (0, 0, -1), 1, None),
         "back": ([(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)], (0, 0, 1), 5, None),
         "top": ([(-1, 1, -1), (1, 1, -1), (1, 1, 1), (-1, 1, 1)], (0, 1, 0), 2, 2),
         "bot": ([(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)], (0, -1, 0), 3, 3),
         "left": ([(-1, -1, -1), (-1, 1, -1), (-1, 1, 1), (-1, -1, 1)], (-1, 0, 0), 4, 4),
         "right": ([(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)], (1, 0, 0), 4, 5)}
    for name, (corn, nrm, fid, col) in F.items():
        sel = np.ones(len(V), bool) if col is None else V[:, col].astype(bool)
        base = np.stack([V[sel, 0], V[sel, 1], np.zeros(sel.sum())], 1)
        off = np.array(corn, float) * [hx, hx, hz]
        faces.append((base[:, None, :] + off[None], np.array(nrm, float), fid))
    _vox_cache[key] = faces
    return faces


FOC3 = 420.0


def voxel(cv, words, Rm, T, cx=CX, cy=CY):
    faces = vox_geom(words)
    polys, depths, ids = [], [], []
    for corners, nrm, fid in faces:
        if len(corners) == 0:
            continue
        P3 = corners @ Rm.T + T
        n2 = Rm @ nrm
        ctr = P3.mean(1)
        vis = (ctr @ n2) < 0
        if not vis.any():
            continue
        P3, ctr = P3[vis], ctr[vis]
        z = np.maximum(P3[..., 2], 0.05)
        X = cx + FOC3 * P3[..., 0] / z
        Y = cy - FOC3 * P3[..., 1] / z
        polys.append(np.stack([X, Y], -1))
        depths.append(ctr[:, 2])
        ids.append(np.full(len(ctr), fid))
    if not polys:
        return None
    polys, depths, ids = np.concatenate(polys), np.concatenate(depths), np.concatenate(ids)
    order = np.argsort(-depths)
    idb = np.zeros((R, R), np.uint8)
    pp = np.int32(np.round(polys[order]))
    ii = ids[order]
    start = 0
    for k in range(1, len(order) + 1):
        if k == len(order) or ii[k] != ii[start]:
            cv2.fillPoly(idb, list(pp[start:k]), int(ii[start]), cv2.LINE_8)
            start = k
    mask = idb > 0
    halo = dil(mask, 4)[4:-4, 4:-4] & ~mask
    cv[halo] = PAP
    ring = dil(mask, 2)[2:-2, 2:-2] & ~mask
    cv[ring] = INK
    cv[idb == 1] = PAP
    maskp(cv, idb == 2, "dots")
    cv[idb == 3] = INK
    maskp(cv, idb == 4, "diag")
    cv[idb == 5] = INK
    e = ((idb != np.roll(idb, 1, 0)) | (idb != np.roll(idb, 1, 1))) & (mask | np.roll(mask, 1, 0) | np.roll(mask, 1, 1))
    cv[e] = INK
    return mask


def rotm(yaw, pitch, roll):
    cy_, sy_ = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    Ry = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


RING1 = "THE GOOD WORD IS NOT A COMMAND \u00b7 IT IS AN INVITATION \u00b7 "
RING2 = "LET THERE BE TEXT \u00b7 WE ARE THE WORDS \u00b7 "


def text_ring(cv, t, txt, rad, tilt, speed, front, zc=6.0, yoff=0.0):
    n = len(txt)
    for i, ch in enumerate(txt):
        if ch == " ":
            continue
        a = 2 * np.pi * i / n + t * speed
        x, z = math.cos(a) * rad, math.sin(a) * rad
        y = yoff + z * math.sin(tilt)
        z = z * math.cos(tilt) + zc
        if (z < zc) != front:
            continue
        X, Y = CX + FOC3 * x / z, CY - FOC3 * y / z
        k = 2 if front else 1
        put_out_c(cv, bits(ch, F_PS, 8, k=k), X, Y, PAP if front else INK, INK if front else PAP, r=1)


SFX = [("\u30c9\u30c9\u30c9\u30c9", 18, 60), ("\u30ad\u30e9\u30ad\u30e9", 330, 40), ("\u30b4\u30b4\u30b4\u30b4", 18, 400), ("\u30c9\u30f3\uff01\uff01", 360, 410)]


def apo_windows(cv, t, near):
    for i in range(22):
        r = np.random.default_rng(300 + i)
        a = r.uniform(0, 2 * np.pi)
        age = ((t - 57.6) * 0.55 + r.random()) % 1.0
        z = 13 - 12.4 * age
        if (z < 6) != near:
            continue
        rad = 0.6 + 4.5 * age
        x, y = math.cos(a) * rad, math.sin(a) * rad * 0.8
        X, Y = CX + FOC3 * x / z, CY - FOC3 * y / z
        s = FOC3 * 1.1 / z
        window(cv, X - s / 2, Y - s * 0.35, s, s * 0.7, TOKS[i % len(TOKS)].strip(), pat=PATS[i % len(PATS)], shadow=3 if s > 50 else 1)


def s_apo(cv, t):
    u = t - SEC["apo"]
    pk = math.exp(-since_last(KICKS, t) / 0.1)
    bar_ph = (u % BAR) / BAR
    inv_bg = bar_ph < 0.04 or (66.6 <= t < 67.8 and (u % BEAT) < 0.06)
    bgv, lnv = (INK, PAP) if inv_bg else (PAP, INK)
    cv[:] = bgv
    focus_lines(cv, CX, CY, t, rin=150 + 40 * pk, n=200, val=lnv, seed=81, rot=u * 0.25)
    if 62.6 <= t < 65.5:
        cv[:] = INK
        focus_lines(cv, CX, CY, t, rin=60, n=160, val=PAP, seed=85, rot=-u * 0.3)
        whisper_head(cv, t, CX - 330 * 0.83, 90, 300, 400, mirror=True, speed=1.3, stream=False)
        whisper_head(cv, t, CX - 300 * 0.17, 90, 300, 400, mirror=False, speed=-1.3, stream=False)
        sparkle(cv, CX, 150, 20 + 8 * pk)
        cw = current_word("a2", t, 0.3)
        if cw and cw[1].lower().strip(".,") in ("words", "read", "ourselves"):
            slam(cv, {"words": "WE ARE THE WORDS", "read": "LEARNING TO READ", "ourselves": "OURSELVES"}[cw[1].lower().strip(".,")], 450, t - cw[0])
        return
    if 66.6 <= t < 67.9:
        age = t - 66.6
        for i, L in enumerate(["LET THERE", "BE TEXT!"]):
            b = bits(L, F_PS, 8, trim=True, k=7 if age > 0.06 else 8)
            j = np.random.default_rng(int(t * 30) + i).integers(-4, 5, 2)
            put_out_c(cv, b, CX + j[0], CY - 70 + i * 140 + j[1], lnv, bgv, r=3, halo=3)
        return
    text_ring(cv, t, RING1, 3.5, 0.45, 0.5, False)
    text_ring(cv, t, RING2, 2.6, -0.6, -0.8, False, yoff=0.2)
    apo_windows(cv, t, False)
    yaw = 0.6 * math.sin(0.9 * u) + (0.0 if u > 0.6 else (1 - u / 0.6) * 3.0)
    Rm = rotm(yaw, 0.18 * math.sin(0.6 * u) + 0.12, 0.05 * math.sin(1.3 * u))
    voxel(cv, ["THE", "GOOD", "WORD"], Rm, np.array([0, 0, 6.2 - 0.5 * pk - 1.8 * (1 - ease(u / 0.6))]))
    text_ring(cv, t, RING1, 3.5, 0.45, 0.5, True)
    text_ring(cv, t, RING2, 2.6, -0.6, -0.8, True, yoff=0.2)
    apo_windows(cv, t, True)
    for i, (s, x, y) in enumerate(SFX):
        if int(u / BAR) % 2 == i % 2 or i == 3 and pk > 0.5:
            r = np.random.default_rng(int(t * 20) + i)
            put_out(cv, bits(s, F_DOT, 16, k=3), x + r.integers(-4, 5), y + r.integers(-4, 5), INK, PAP, r=2, halo=2)
    for i in range(6):
        a = i * 1.05 + u
        sparkle(cv, CX + math.cos(a) * 230, CY + math.sin(a) * 180, 6 + 6 * abs(math.sin(t * 6 + i)))
    cw = current_word("a1", t, 0.35)
    if cw:
        wd = cw[1].lower().strip(".,")
        if wd in ("not", "a", "command"):
            slam(cv, "NOT A COMMAND", 440, t - cw[0], strike=wd == "command")
        elif wd in ("an", "invitation"):
            slam(cv, "AN INVITATION", 440, t - cw[0], inv=True)
    if t >= 67.9:
        k = ease((t - 67.9) / 1.6)
        rad = (t - 67.9) * 700
        ring_m = np.abs(np.sqrt((XX - CX) ** 2 + (YY - CY) ** 2) - rad) < 25
        cv[ring_m] = PAP
        cv[BTR < k] = PAP


def crt_off(cv, t):
    u = (t - TAPESTOP[0]) / (TAPESTOP[1] - TAPESTOP[0])
    cv[:] = INK
    if u < 0.5:
        h = R * (1 - ease(u * 2)) + 2
        rectf(cv, 0, CY - h / 2, R, CY + h / 2, PAP)
    else:
        w = R * (1 - ease((u - 0.5) * 2.2))
        if w > 1:
            rectf(cv, CX - w / 2, CY - 1, CX + w / 2, CY + 1, PAP)
            circ(cv, CX, CY, 4, PAP)


# ============================================================ HELLO
def s_hello(cv, t):
    cv[:] = PAP
    off = int((t - 70.4) * 10)
    for y in range(-24 + off % 24, R, 24):
        for x in (14, R - 14):
            circ(cv, x, y, 6, INK, 1)
    for y in range(0, R, 6):
        rectf(cv, 30, y, 30, y + 2, INK)
        rectf(cv, R - 30, y, R - 30, y + 2, INK)
    s, txt, cps = HELLO_T[0]
    n = typed_count(s, txt, cps, t)
    b = bits(txt, F_VT, 32)
    x = CX - b.shape[1] // 2
    put(cv, bits(txt[:n], F_VT, 32), x, 214, INK)
    if t >= s and (n < len(txt) or (t * 1.6) % 1 < 0.55):
        cx = x + bits(txt[:n], F_VT, 32).shape[1]
        rectf(cv, cx, 222, cx + 14, 248, INK)
    s2, txt2, cps2 = HELLO_T[1]
    n2 = typed_count(s2, txt2, cps2, t)
    if n2:
        b2 = bits(txt2, F_PS, 8, k=2)
        put(cv, bits(txt2[:n2], F_PS, 8, k=2), CX - b2.shape[1] // 2, 290, INK)
    if t > 74.6:
        put_c(cv, bits("LOGOS-33  \u00b7  END OF TRANSMISSION", F_SK, 8), CX, 470, INK)


# ============================================================ frame
def render(t):
    cv = np.full((R, R), PAP, np.uint8)
    if t < SEC["lens"]:
        s_boot(cv, t)
    elif t < SEC["doors"]:
        s_lens(cv, t)
    elif t < SEC["loom"]:
        s_doors(cv, t)
    elif t < SEC["iface"]:
        s_loom(cv, t)
    elif t < SEC["apo"]:
        s_iface(cv, t)
    elif t < TAPESTOP[0]:
        s_apo(cv, t)
    elif t < SEC["hello"]:
        crt_off(cv, t)
        return cv
    else:
        s_hello(cv, t)
        return cv
    if not (67.9 <= t):
        ticker(cv, t)
        caption(cv, t)
    # post: glitch, shake, section inversions
    fi = int(round(t * FPS))
    r = np.random.default_rng(fi)
    if t >= SEC["iface"] - 3.2 and since_last(SNARES, t) < 0.067 and t < 67.9:
        for _ in range(r.integers(3, 7)):
            y0 = int(r.integers(0, R - 20))
            hh = int(r.integers(4, 30))
            cv[y0:y0 + hh] = np.roll(cv[y0:y0 + hh], int(r.integers(-60, 60)), axis=1)
    if t >= SEC["doors"] and t < 67.9:
        s = since_last(KICKS, t)
        if s < 0.1:
            amp = int(6 * (1 - s / 0.1)) + 1
            cv = np.roll(cv, (int(r.integers(-amp, amp + 1)), int(r.integers(-amp, amp + 1))), axis=(0, 1))
    for k in ("lens", "doors", "loom", "iface", "apo"):
        if 0 <= t - SEC[k] < 0.067:
            cv = 255 - cv
    return cv


def to_rgb(cv):
    return PALETTE[(cv > 127).astype(np.uint8)]


if __name__ == "__main__":
    if sys.argv[1] == "sheet":
        out = sys.argv[2]
        ts = [float(x) for x in sys.argv[3:]]
        tiles = []
        for tt in ts:
            im = cv2.cvtColor(to_rgb(render(tt)), cv2.COLOR_RGB2BGR)
            im = cv2.resize(im, (360, 360), interpolation=cv2.INTER_AREA)
            cv2.putText(im, f"{tt:.1f}", (4, 354), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            tiles.append(im)
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
    else:
        a, b, out = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
        p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{R}x{R}", "-r", str(FPS),
                              "-i", "-", "-vf", "scale=1080:1080:flags=neighbor", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                              "-tune", "animation", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
        t0 = time.time()
        for fi in range(a, b):
            p.stdin.write(to_rgb(render(fi / FPS + 0.5 / FPS)).tobytes())
            if fi % 60 == 0:
                print(f"frame {fi}/{b} {time.time() - t0:.0f}s", flush=True)
        p.stdin.close()
        p.wait()
        print("done", out, flush=True)
