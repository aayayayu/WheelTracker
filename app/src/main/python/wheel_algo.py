"""
Flywheel rotation counter - side view, one black tape on the rim.

Port of flywheel_rotations.py to a streaming form, so a long video never has to be held in memory:
only a box and a 200-point brightness profile are kept per frame.

  1. detect_flywheel()   per frame: gray vertical block crossing the blue bracket -> (x, y, w, h).
                         With a TrackCfg (user calibration) the box SIZE never changes: BoxTracker only
                         lets the fixed box drift a few pixels, and when the detected area is off by
                         more than the tolerance it re-locates the box by template matching instead.
                         Without a TrackCfg: the last known box is reused when detection fails.
  2. profile_of()        brightness profile of the box, divided by its own running baseline.
  3. observe_tape()      divide every profile by the median profile over time -> static parts (rod,
                         shaft, edges) cancel, only the moving black tape is left as a dark band.
  4. phase_track()       tape height y -> angle asin(2y-1); y jumping back up = new lap; unwrap,
                         interpolate while the tape is hidden, extrapolate after the last sighting.
  5. rotations = (final phase - first phase) / 360

FrameExtractor does steps 1-2 while frames arrive (and can be run in several threads, one per part
of the video); merge_extractors() joins the parts; solve() does steps 3-5 on the merged result.
LiveTracker is a cheap causal version of 3-5 used only for the live preview.
"""
import math
import numpy as np
import cv2

N_PROFILE = 200      # vertical resolution of the brightness profile
DARK_THR = 0.70      # tape = profile below 70% of its usual brightness
NEW_LAP_JUMP = 0.30  # tape y jumping back up by more than this = new lap
DETECT_EVERY = 4     # calibrated runs: full-frame flywheel detection only on every 4th frame (box may only drift a few px anyway)
MAX_PENDING = 240    # frames kept (grayscale) while waiting for the very first flywheel detection


# --------------------------------------------------------------------------- #
# 1. Flywheel detection
# --------------------------------------------------------------------------- #
def _k(v, scale):
    """A pixel size tuned for a full-resolution frame, scaled to the frame actually analysed."""
    return max(1, int(round(v * scale)))


def detect_flywheel(frame, scale=1.0):
    """frame: BGR image (the original frame sub-sampled by 1/scale). Returns (x, y, w, h) in the
    pixels of that image, or None. scale = 1 gives exactly the thresholds of flywheel_rotations.py."""
    H, W = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    wall_v = np.percentile(hsv[::4, ::4, 2], 80)        # every 4th pixel is plenty for a percentile

    # Blue bracket -> gives the vertical level the wheel must cross
    blue = ((hsv[..., 0] > 100) & (hsv[..., 0] < 135) &
            (hsv[..., 1] > 120) & (hsv[..., 2] > 40)).astype(np.uint8)
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((_k(5, scale), _k(5, scale)), np.uint8))
    n, _, stats, cent = cv2.connectedComponentsWithStats(blue)
    if n < 2:
        return None
    j = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    if stats[j, cv2.CC_STAT_AREA] < 500 * scale * scale:
        return None
    cy = int(cent[j][1])

    # Gray (low saturation, darker than the wall) pixels
    gray = ((hsv[..., 1] < 70) & (hsv[..., 2] < wall_v - 22) &
            (hsv[..., 2] > 8)).astype(np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,
                            np.ones((_k(15, scale), _k(3, scale)), np.uint8)).astype(bool)

    # For every column: the vertical run of gray pixels that contains row cy (if any)
    min_run = 0.28 * H * 0.7
    up = gray[:cy + 1][::-1]                             # rows cy .. 0
    dn = gray[cy:]                                       # rows cy .. H-1
    s = np.where(up.all(0), 0, cy - np.argmin(up, 0) + 1)
    e = np.where(dn.all(0), H, cy + np.argmin(dn, 0))
    ok = gray[cy] & ((e - s) > min_run)
    cols = np.nonzero(ok)[0]
    if len(cols) < _k(10, scale):
        return None
    tops, bots = s[cols], e[cols]

    groups = np.split(cols, np.where(np.diff(cols) > max(1, 4 * scale))[0] + 1)
    g = max(groups, key=len)
    if len(g) < _k(20, scale) or len(g) > 0.5 * W:
        return None

    sel = np.isin(cols, g)
    t = int(np.median(tops[sel]))
    b = int(np.median(bots[sel]))
    return (int(g[0]), t, int(g[-1] - g[0] + 1), b - t)


# --------------------------------------------------------------------------- #
# 2. Brightness profile
# --------------------------------------------------------------------------- #
def _median_filter_nearest(a, k):
    """1-D median filter, odd window k, edges extended with the nearest value (= scipy 'nearest')."""
    h = k // 2
    pad = np.pad(a, (h, h), mode='edge')
    return np.median(np.stack([pad[i:i + len(a)] for i in range(k)]), axis=0)


def profile_of(gray, box):
    """gray: uint8 image, box: (x, y, w, h). 200-point vertical profile / its own baseline."""
    x, y, w, h = box
    roi = gray[y:y + h, x + int(.2 * w):x + int(.8 * w)]
    if roi.size == 0:
        return np.ones(N_PROFILE)
    p = roi.mean(1, dtype=np.float64)
    p = np.interp(np.linspace(0, len(p) - 1, N_PROFILE), np.arange(len(p)), p)
    base = _median_filter_nearest(p, 61)
    return p / np.maximum(base, 1)


class TrackCfg:
    """Calibration of the flywheel box: fixed size, small allowed drift, appearance template.
    box = (x, y, w, h) and frame = BGR image, both in the pixels that are analysed."""

    def __init__(self, box, frame, area_tol=0.10, drift=0.06):
        H, W = frame.shape[:2]
        x, y, w, h = [int(round(float(v))) for v in box]
        w, h = max(8, min(w, W)), max(8, min(h, H))
        x, y = min(max(0, x), W - w), min(max(0, y), H - h)
        self.ref = (x, y, w, h)
        self.area_tol = float(area_tol)                       # allowed |area / ref area - 1|
        self.shift = max(0, int(round(float(drift) * w)))     # allowed drift from ref, px (x and y)
        pad = max(4, int(.15 * w))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        self.tmpl = np.ascontiguousarray(g[y0:y1, x0:x1])
        self.off = (x - x0, y - y0)                           # where the box sits inside the template


class BoxTracker:
    """Keeps the flywheel box at the calibrated size and lets it move only a few pixels.

    Per frame, in this order:
      detected  detect_flywheel() gave a box whose area is within area_tol of the calibrated one:
                its centre is used (its size is ignored).
      anchored  the area is off (wheel partly hidden): per axis, the edge of the detection that is
                still consistent with the calibrated position is kept and the other edge is ignored.
      matched   detection missing, or an axis has no consistent edge: the calibration picture is
                searched in a small window around the calibrated position (normalised correlation).
      held      nothing usable: previous position.
    The position is always clamped to calibrated position +/- shift."""
    MIN_SCORE = 0.5

    def __init__(self, cfg):
        self.cfg = cfg
        self.pos = (float(cfg.ref[0]), float(cfg.ref[1]))
        self.state = 'held'
        self.last_measured = False

    def _match(self, gray):
        c = self.cfg
        th, tw = c.tmpl.shape[:2]
        H, W = gray.shape[:2]
        tx0, ty0 = c.ref[0] - c.off[0], c.ref[1] - c.off[1]
        s = c.shift
        sx0, sy0 = max(0, tx0 - s), max(0, ty0 - s)
        sx1, sy1 = min(W, tx0 + tw + s), min(H, ty0 + th + s)
        if sx1 - sx0 < tw or sy1 - sy0 < th:
            return None
        reg = cv2.GaussianBlur(gray[sy0:sy1, sx0:sx1], (3, 3), 0)
        res = cv2.matchTemplate(reg, c.tmpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if score < self.MIN_SCORE:
            return None
        return (sx0 + loc[0] + c.off[0], sy0 + loc[1] + c.off[1])

    def update(self, gray, raw):
        """gray: uint8 frame, raw: detect_flywheel() result or None.
        Returns ((x, y, w, h), measured) - measured is False when the position is only a guess."""
        c = self.cfg
        H, W = gray.shape[:2]
        rx, ry, rw, rh = c.ref
        pos, measured, self.state = None, False, 'held'
        if raw is not None and abs(raw[2] * raw[3] / float(rw * rh) - 1.0) <= c.area_tol:
            pos, measured, self.state = (raw[0] + (raw[2] - rw) / 2.0, raw[1] + (raw[3] - rh) / 2.0), True, 'detected'
        s = c.shift
        if pos is None and raw is not None:
            # Area is off (hand / glare / shadow hides part of the wheel). Per axis, one edge of the
            # detection is usually still right: use the edge whose implied box position stays within
            # the drift limit and is nearest to the previous box.
            x, y, w, h = raw
            px, py = self.pos

            def pick(lo, hi, p, r):
                ok = [v for v in (lo, hi) if abs(v - r) <= s + 1]
                return min(ok, key=lambda v: abs(v - p)) if ok else None
            ax, ay = pick(x, x + w - rw, px, rx), pick(y, y + h - rh, py, ry)
            if ax is not None and ay is not None:
                pos, measured, self.state = (ax, ay), True, 'anchored'
            else:
                m = self._match(gray)                      # an axis has no trustworthy edge
                if m is not None:
                    pos = (ax if ax is not None else m[0], ay if ay is not None else m[1])
                    measured, self.state = True, 'matched'
        if pos is None:
            m = self._match(gray)                          # not detected at all
            if m is not None:
                pos, measured, self.state = m, True, 'matched'
        if pos is None:
            pos = self.pos                                 # nothing usable: hold
        x = min(max(pos[0], rx - s), rx + s)
        y = min(max(pos[1], ry - s), ry + s)
        x = min(max(x, 0), W - rw)
        y = min(max(y, 0), H - rh)
        self.pos = (x, y)
        self.last_measured = measured
        return (int(round(x)), int(round(y)), rw, rh), measured

    def hold(self):
        """Frame without a detection pass (speed): keep the previous box and its measured flag."""
        rw, rh = self.cfg.ref[2], self.cfg.ref[3]
        return (int(round(self.pos[0])), int(round(self.pos[1])), rw, rh), self.last_measured


class FrameExtractor:
    """Steps 1-2 for a run of frames, fed in order. Keeps per frame: index, raw detection (or None),
    box in use and the profile. Frames that arrive before the first detection wait (as grayscale)
    until a box is known."""

    def __init__(self, scale=1.0, cfg=None, detect_every=DETECT_EVERY):
        self.scale = scale
        self.detect_every = max(1, int(detect_every))
        self.tracker = BoxTracker(cfg) if cfg is not None else None
        self.idx, self.raw, self.box, self.prof = [], [], [], []
        self.pending = {}                 # position -> gray frame, waiting for a box
        self.last = None

    def feed(self, i, bgr):
        """Returns (raw_box_or_None, box_in_use_or_None, profile_or_None) for this frame."""
        if self.tracker is not None:                      # calibrated: fixed-size box, small drift
            if self.detect_every > 1 and self.idx and int(i) % self.detect_every:
                # speed: no full-frame detection on this frame; only the box area is converted
                box, ok = self.tracker.hold()
                x, y, w, h = box
                sub = cv2.cvtColor(bgr[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
                pr = profile_of(sub, (0, 0, w, h))
            else:
                raw = detect_flywheel(bgr, self.scale)
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                box, ok = self.tracker.update(gray, raw)
                pr = profile_of(gray, box)
            raw = box if ok else None                     # raw = box was measured (else held/guessed)
            self.idx.append(int(i)); self.raw.append(raw); self.box.append(box); self.prof.append(pr)
            self.last = box
            return raw, box, pr
        raw = detect_flywheel(bgr, self.scale)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if raw is not None:
            self.last = raw
        pos = len(self.idx)
        self.idx.append(int(i)); self.raw.append(raw); self.box.append(self.last)
        if self.last is None:
            if len(self.pending) >= MAX_PENDING:
                raise RuntimeError("The flywheel was not detected in the first %d frames. "
                                   "It must be a gray block crossing the blue bracket." % MAX_PENDING)
            self.pending[pos] = gray
            self.prof.append(None)
            return raw, None, None
        pr = profile_of(gray, self.last)
        self.prof.append(pr)
        return raw, self.last, pr


def merge_extractors(exts):
    """Join extractors of consecutive parts of the video (in order) and apply the hold-last-known
    rule across part boundaries. Returns (idx, raw, box, prof) lists, all complete."""
    idx, raw, box, prof, pend = [], [], [], [], {}
    for e in exts:
        off = len(idx)
        idx += e.idx; raw += e.raw; box += e.box; prof += e.prof
        for k, g in e.pending.items():
            pend[off + k] = g
    last = None
    for k in range(len(idx)):
        if raw[k] is not None:
            last = raw[k]
        elif box[k] is None:
            box[k] = last
    first = next((b for b in raw if b is not None), None)
    if first is None:
        raise RuntimeError("Flywheel was never detected. It must be a gray block crossing the "
                           "blue bracket.")
    for k in sorted(pend):
        if box[k] is None:
            box[k] = first
        prof[k] = profile_of(pend[k], box[k])
    pend.clear()
    return idx, raw, box, prof


# --------------------------------------------------------------------------- #
# 3. Tape observation
# --------------------------------------------------------------------------- #
def observe_tape(P, thr=DARK_THR):
    """Per frame: (found, y_norm, min_ratio, width_px)."""
    med = np.maximum(np.median(P, axis=0), 1e-6)        # static structure (rod, shaft, edges)
    D = P / med
    obs = []
    for d in D:
        m = d < thr
        m[:6] = False
        m[-6:] = False
        if m.sum() >= 2:
            ys = np.nonzero(m)[0]
            wts = thr - d[m]
            obs.append((1, float((ys * wts).sum() / wts.sum() / N_PROFILE), float(d.min()), int(m.sum())))
        else:
            obs.append((0, float('nan'), float(d.min()), 0))
    return obs


# --------------------------------------------------------------------------- #
# 4. Phase tracking and rotation count
# --------------------------------------------------------------------------- #
def phase_track(obs, new_lap_jump=NEW_LAP_JUMP):
    """Unwrapped phase in degrees for every frame, the frames where laps start and the frames
    where the tape was seen."""
    n = len(obs)
    idx, ph, entries = [], [], []
    lap, prev = 0, None
    for f, o in enumerate(obs):
        if not o[0]:
            continue
        y = o[1]
        theta = math.degrees(math.asin(min(1.0, max(-1.0, 2 * y - 1))))  # -90 top ... +90 bottom
        if prev is not None and y < prev - new_lap_jump:
            lap += 1
            entries.append(f)
        prev = y
        idx.append(f)
        ph.append(360 * lap + theta)
    if not idx:
        raise RuntimeError("The tape was never detected. Try a higher dark threshold.")

    idx = np.array(idx)
    ph = np.maximum.accumulate(np.array(ph))            # rotation only goes one way
    full = np.interp(np.arange(n), idx, ph)

    # After the last sighting: extrapolate using the latest lap period
    last = idx[-1]
    if last < n - 1:
        if len(entries) >= 2:
            w = 360.0 / np.mean(np.diff(entries[-3:]))  # deg / frame
        elif entries:
            w = (ph[-1] - ph[np.searchsorted(idx, entries[-1])]) / max(1, last - entries[-1])
        else:
            w = 0.0
        full[last + 1:] = ph[-1] + w * np.arange(1, n - last)
    return full, entries, idx


def solve(idx, raw, box, prof, p):
    """idx/raw/box/prof from merge_extractors(); p = {'dark_thr', 'new_lap_jump'}."""
    P = np.asarray(prof, dtype=float)
    obs = observe_tape(P, p['dark_thr'])
    phase, entries, seen = phase_track(obs, p['new_lap_jump'])
    return {
        'n': len(idx),
        'idx': np.asarray(idx, dtype=np.int64),
        'raw_ok': np.array([r is not None for r in raw], dtype=bool),
        'box': np.array(box, dtype=float),
        'found': np.array([o[0] for o in obs], dtype=bool),
        'y': np.array([o[1] for o in obs], dtype=float),
        'phase': phase,
        'entries': entries,                 # positions (0..n-1) where a new lap starts
        'seen': seen,                       # positions where the tape was seen
        'total': float((phase[-1] - phase[0]) / 360.0),
        'seen_only': float((phase[seen[-1]] - phase[seen[0]]) / 360.0),
        'missed': int(sum(r is None for r in raw)),
    }


# --------------------------------------------------------------------------- #
# Live preview (causal, approximate): the exact numbers come from solve()
# --------------------------------------------------------------------------- #
class LiveTracker:
    """Runs steps 3-4 while frames arrive. The 'usual profile' is the median of everything seen so
    far (recomputed now and then; the first time it is computed the stored frames are replayed, so
    the count starts at the first frame). The phase is held while the tape is behind the wheel."""

    def __init__(self, p):
        self.p = p
        self.P, self.med = [], None
        self.lap, self.prev = 0, None
        self.ph_max = self.ph0 = None
        self.laps_seen = 0
        self.found, self.y = False, None

    def _observe(self, prof):
        thr = self.p['dark_thr']
        d = prof / self.med
        mk = d < thr
        mk[:6] = False
        mk[-6:] = False
        self.found, self.y = False, None
        if mk.sum() < 2:
            return
        ys = np.nonzero(mk)[0]
        wts = thr - d[mk]
        y = float((ys * wts).sum() / wts.sum() / N_PROFILE)
        if self.prev is not None and y < self.prev - self.p['new_lap_jump']:
            self.lap += 1
            self.laps_seen += 1
        self.prev = y
        theta = math.degrees(math.asin(min(1.0, max(-1.0, 2 * y - 1))))
        ph = 360.0 * self.lap + theta
        self.ph_max = ph if self.ph_max is None else max(self.ph_max, ph)
        if self.ph0 is None:
            self.ph0 = ph
        self.found, self.y = True, y

    def step(self, prof):
        """prof: profile of this frame (or None). Returns dict(found, y, phase, rot, laps)."""
        if prof is not None:
            self.P.append(prof)
            m = len(self.P)
            if m >= 15 and (self.med is None or m % (5 if m < 60 else max(25, m // 20)) == 0):
                first = self.med is None
                self.med = np.maximum(np.median(np.array(self.P), axis=0), 1e-6)
                if first:
                    for pr in self.P:               # replay what was seen before the median existed
                        self._observe(pr)
                else:
                    self._observe(prof)
            elif self.med is not None:
                self._observe(prof)
        out = {'found': self.found, 'y': self.y, 'phase': self.ph_max, 'rot': 0.0,
               'laps': self.laps_seen}
        if self.ph_max is not None:
            out['rot'] = (self.ph_max - self.ph0) / 360.0
        return out
