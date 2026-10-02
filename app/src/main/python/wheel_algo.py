"""
Flywheel rotation counter - side view, one black tape on the rim.

Port of flywheel_rotations.py to a streaming form, so a long video never has to be held in memory:
only a box and a 200-point brightness profile are kept per frame.

  1. detect_flywheel()   per frame: gray vertical block crossing the blue bracket -> (x, y, w, h).
                         If detection fails the last known box is reused (the first known box for
                         leading frames).
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


class FrameExtractor:
    """Steps 1-2 for a run of frames, fed in order. Keeps per frame: index, raw detection (or None),
    box in use and the profile. Frames that arrive before the first detection wait (as grayscale)
    until a box is known."""

    def __init__(self, scale=1.0):
        self.scale = scale
        self.idx, self.raw, self.box, self.prof = [], [], [], []
        self.pending = {}                 # position -> gray frame, waiting for a box
        self.last = None

    def feed(self, i, bgr):
        """Returns (raw_box_or_None, box_in_use_or_None, profile_or_None) for this frame."""
        raw = detect_flywheel(bgr, self.scale)
        if raw is not None:
            self.last = raw
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
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
            if m >= 15 and (self.med is None or m % (5 if m < 60 else 25) == 0):
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
