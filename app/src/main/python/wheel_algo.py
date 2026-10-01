"""
Flywheel rotation counter - side-on view, one dark tape on the rim.

This is the algorithm from flywheel_rotation_counter.py, reorganised so it works on the
grayscale tracking-box frames the Android decoder delivers:

  1. measure_frame()      per frame: find the wheel (grey pixels) inside the tracking box ->
                          wheel box (x0, x1, top, bottom) + a 1-D row-brightness profile.
                          Only a few hundred floats are kept per frame, not the picture.
  2. solve()              over all frames: smooth the wheel box (sliding median), find the tape as
                          the longest run of dark rows, turn its vertical position into an angle
                          sin(theta) = (y_tape - y_centre) / R, count one pass of the tape as one more
                          turn, interpolate the hidden half of every turn and sum up the rotations.
  3. LiveEstimator        cheap causal estimate shown while the live preview is running.
"""
import math
import numpy as np
import cv2

# ---------------- tunables ----------------
GREY_V_MAX = 200         # wheel / tape pixels are darker than the white wall
SEARCH_ROWS = (0.06, 0.30)   # fraction of box height used to find the wheel's columns
EDGE_TRIM = 0.15         # ignore this fraction of the wheel width at each edge
MIN_WHEEL_H = 20         # smallest plausible wheel height (px)
MIN_TAPE_H = 4           # smallest tape band (px)
MAX_GAP_FRAMES = 3       # frames apart => the tape starts a new pass (= new turn)
WIN_X, WIN_Y = 5, 9      # median-filter windows for the wheel box
BAND_MARGIN = 2          # rows skipped at the wheel's top / bottom edge
MAX_TAIL_DEG = 180.0     # never extrapolate more than half a turn after the last sighting


# ---------------- per-frame measurement ----------------
def measure_frame(gray):
    """gray: uint8 image of the tracking box. Returns None (wheel not found) or
    {'box': (x0, x1, top, bot), 'prof': row means, 'ncols': columns averaged} in box coordinates."""
    H, W = gray.shape[:2]
    if H < 8 or W < 8:
        return None
    grey = gray < GREY_V_MAX
    r0 = int(SEARCH_ROWS[0] * H)
    r1 = max(int(SEARCH_ROWS[1] * H), r0 + 1)
    cols = grey[r0:r1].sum(0)
    if cols.max() == 0:
        return None
    xs = np.where(cols > cols.max() * 0.6)[0]
    runs = np.split(xs, np.where(np.diff(xs) > 3)[0] + 1)
    run = max(runs, key=len)
    x0, x1 = int(run[0]), int(run[-1])
    w = x1 - x0
    xi0, xi1 = x0 + int(EDGE_TRIM * w), x1 - int(EDGE_TRIM * w)
    if xi1 - xi0 < 3:
        return None
    ok = grey[:, xi0:xi1].mean(1) > 0.7
    if not ok.any():
        return None
    top = int(np.argmax(ok))
    y, gap = top, 0
    while y < H - 1 and gap < 8:
        y += 1
        gap = 0 if ok[y] else gap + 1
    bot = y - gap
    if bot - top < MIN_WHEEL_H:
        return None
    prof = gray[:, xi0:xi1].mean(1).astype(np.float32)
    return {'box': (x0, x1, top, bot), 'prof': prof, 'ncols': xi1 - xi0}


def _longest_run(mask):
    """(start, end) of the longest run of True in a 1-D bool array, or None."""
    if not mask.any():
        return None
    m = np.concatenate(([False], mask, [False])).astype(np.int8)
    d = np.diff(m)
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0] - 1
    k = int(np.argmax(ends - starts))
    return int(starts[k]), int(ends[k])


def tape_in_profile(prof, top, bot, ncols, p):
    """Tape band (y0, y1, area) in box coordinates, or None when it is on the hidden side."""
    a, b = int(top) + BAND_MARGIN, int(bot) - BAND_MARGIN
    if b - a < 8:
        return None
    seg = prof[a:b]
    run = _longest_run(seg < p['black_thresh'])
    if run is None:
        return None
    lo, hi = run
    h = hi - lo + 1
    area = h * ncols
    if h < MIN_TAPE_H or area < p['min_area'] or (p['max_area'] > 0 and area > p['max_area']):
        return None
    return lo + a, hi + a, area


def state_of(yc, top, bot):
    ny = (yc - top) / float(max(1, bot - top))
    return 'top' if ny < 0.33 else ('mid' if ny < 0.66 else 'bottom')


def frame_result(rec, ox, oy, p):
    """Raw (unsmoothed) single-frame result in FULL-FRAME coordinates - used for the overlay,
    the slider and the live preview. Same keys the UI already knows."""
    res = {'found': False, 'mask': None, 'y': None, 'state': 'hidden', 'area': 0,
           'rect': None, 'wheel': None}
    if rec is None:
        return res
    x0, x1, top, bot = rec['box']
    res['wheel'] = (ox + x0, oy + top, ox + x1, oy + bot)
    t = tape_in_profile(rec['prof'], top, bot, rec['ncols'], p)
    if t is None:
        return res
    y0, y1, area = t
    yc = (y0 + y1) / 2.0
    res.update(found=True, y=oy + yc, area=area, state=state_of(yc, top, bot),
               rect=(ox + x0, oy + y0, x1 - x0, y1 - y0 + 1))
    return res


# ---------------- whole-video solution ----------------
def _median_filter(a, k):
    h = k // 2
    pad = np.pad(a, (h, h), mode='edge')
    return np.median(np.stack([pad[i:i + len(a)] for i in range(k)]), axis=0)


def _smooth_boxes(recs):
    n = len(recs)
    B = np.full((n, 4), np.nan)
    for i, r in enumerate(recs):
        if r is not None:
            B[i] = r['box']
    valid = ~np.isnan(B[:, 0])
    if not valid.any():
        raise RuntimeError("The wheel was not found inside the tracking box. "
                           "Widen the box so it surrounds the whole wheel, or raise the grey limit.")
    first = int(np.argmax(valid))
    idx = np.where(valid, np.arange(n), -1)
    idx = np.maximum.accumulate(idx)
    idx[idx < 0] = first
    B = B[idx]
    return (_median_filter(B[:, 0], WIN_X), _median_filter(B[:, 1], WIN_X),
            _median_filter(B[:, 2], WIN_Y), _median_filter(B[:, 3], WIN_Y))


def solve(recs, p):
    """recs: list (one per frame, in order) of measure_frame() results (None allowed).
    Returns dict with per-frame arrays and the total number of rotations."""
    n = len(recs)
    x0, x1, top, bot = _smooth_boxes(recs)
    tapes = []
    for i, r in enumerate(recs):
        tapes.append(None if r is None else tape_in_profile(r['prof'], top[i], bot[i], r['ncols'], p))
    seen = [i for i, t in enumerate(tapes) if t is not None]

    # group sightings into passes (one pass = the tape crossing the visible half of the rim)
    groups = []
    for i in seen:
        if groups and i - groups[-1][-1] <= MAX_GAP_FRAMES:
            groups[-1].append(i)
        else:
            groups.append([i])
    sign = 1.0 if p['direction'] == 'down' else -1.0
    yc_of = lambda i: (tapes[i][0] + tapes[i][1]) / 2.0
    kept = []
    for g in groups:
        if len(g) < max(1, p['debounce']):
            continue                                   # too short: noise / glare
        if p['strict'] and len(g) >= 2 and sign * (yc_of(g[-1]) - yc_of(g[0])) <= 0:
            continue                                   # tape moved the wrong way
        kept.append(g)

    out = {'n': n, 'tapes': tapes, 'x0': x0, 'x1': x1, 'top': top, 'bot': bot,
           'passes': [(g[0], g[-1]) for g in kept], 'n_groups': len(groups)}
    if not kept:
        out.update(theta=np.zeros(n), rot=np.zeros(n), total=0.0, known=[])
        return out

    known = {}
    for k, g in enumerate(kept):
        for i in g:
            cy, R = (top[i] + bot[i]) / 2.0, max(1.0, (bot[i] - top[i]) / 2.0)
            s = float(np.clip(sign * (yc_of(i) - cy) / R, -1.0, 1.0))
            known[i] = 360.0 * k + math.degrees(math.asin(s))
    ks = sorted(known)
    theta = np.interp(np.arange(n), ks, [known[i] for i in ks])

    # after the last sighting: continue at the last measured rate, but never more than half a turn
    last = ks[-1]
    if last < n - 1 and len(ks) >= 3:
        rate = max(0.0, (known[last] - known[ks[-3]]) / float(last - ks[-3]))
        tail = known[last] + np.minimum(rate * np.arange(1, n - last), MAX_TAIL_DEG)
        theta[last + 1:] = tail
    theta = np.maximum.accumulate(theta)               # the wheel does not run backwards
    rot = (theta - theta[0]) / 360.0
    out.update(theta=theta, rot=rot, total=float(rot[-1]), known=ks)
    return out


# ---------------- live estimate (causal, for the preview only) ----------------
class LiveEstimator:
    """Running rotation count while frames arrive: counts passes of the tape and holds the last
    known angle while the tape is on the hidden side. The exact result comes from solve()."""

    def __init__(self, p):
        self.p, self.k, self.last_i, self.th0, self.cur = p, -1, None, None, 0.0
        self.sign = 1.0 if p['direction'] == 'down' else -1.0

    def step(self, i, res):
        if res['found'] and res['wheel'] is not None:
            if self.last_i is None or i - self.last_i > MAX_GAP_FRAMES:
                self.k += 1
            self.last_i = i
            top, bot = res['wheel'][1], res['wheel'][3]
            cy, R = (top + bot) / 2.0, max(1.0, (bot - top) / 2.0)
            s = float(np.clip(self.sign * (res['y'] - cy) / R, -1.0, 1.0))
            th = 360.0 * self.k + math.degrees(math.asin(s))
            if self.th0 is None:
                self.th0 = th
            self.cur = max(self.cur, (th - self.th0) / 360.0)
        return self.cur
