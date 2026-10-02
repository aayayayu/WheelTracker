"""
EP LAB - Flywheel — Flask Web Edition
Side-view flywheel with one black tape on the rim. Algorithm: wheel_algo.py (port of flywheel_rotations.py).
"""
import os, io, re, math, base64, uuid, shutil, collections, json as _json, tempfile, threading, time
from urllib.parse import unquote
import numpy as np, cv2, matplotlib
import wheel_algo as wa
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, Response

app = Flask(__name__)
app.secret_key = 'rw-tracker-secret-key'
SESSIONS, DISPLAY_MAX_W = {}, 900
PREVIEW_MS = 300            # uncapped live preview: at most one picture per this many ms
DETECT_W = 960              # frames are analysed sub-sampled to about this width (px)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class UserError(Exception):
    """A problem with the video / settings (not a decoder failure): shown to the user as is."""


# ---------- Drawing ----------
def fit_display(frame, max_w=DISPLAY_MAX_W):
    """Return (picture no wider than max_w, scale vs. the original). Always a NEW array,
    because decoded frames are shared with the frame cache and must never be drawn on."""
    h, w = frame.shape[:2]
    if w > max_w:
        sc = max_w / float(w)
        return cv2.resize(frame, (max_w, max(1, int(h * sc))), interpolation=cv2.INTER_AREA), sc
    return frame.copy(), 1.0


def draw_overlay(out, s, box, detected, tape_y=None, tape_state=None, phase=None, rot=None,
                 laps=None, t=None):
    """Annotate `out` (the frame scaled by s) like the annotated video of flywheel_rotations.py:
    flywheel box (green = detected, orange = held), red line at the tape, a dial with the tape
    phase, and the rotation counter. box = (x, y, w, h) in ORIGINAL pixels or None;
    tape_y in [0, 1] of the box height; tape_state 'found' | 'hidden' | None."""
    H, W = out.shape[:2]
    if box is not None:
        x, y, w, h = [int(round(float(v) * s)) for v in box]
        col = (0, 200, 0) if detected else (0, 165, 255)
        cv2.rectangle(out, (x, y), (x + w, y + h), col, 2)
        label = "flywheel: " + ("detected" if detected else "held (last known)")
    else:
        col, label = (0, 0, 255), "flywheel: not found"
    cv2.putText(out, label, (8, 22), FONT, 0.55, col, 2)
    if tape_state == 'found' and box is not None and tape_y is not None:
        ty = int(y + tape_y * h)
        cv2.line(out, (x, ty), (x + w, ty), (0, 0, 255), 3)
        cv2.putText(out, "tape: found", (8, 44), FONT, 0.55, (0, 0, 255), 2)
    elif tape_state == 'hidden':
        cv2.putText(out, "tape: behind wheel (phase estimated)", (8, 44), FONT, 0.55, (255, 160, 0), 2)
    if phase is not None:                                  # small dial: tape position around the wheel
        cx, cy, r = W - 60, H - 70, 40
        cv2.circle(out, (cx, cy), r, (255, 255, 255), -1)
        cv2.circle(out, (cx, cy), r, (60, 60, 60), 2)
        a = math.radians(phase)
        cv2.circle(out, (cx + int(r * .8 * math.cos(a)), cy + int(r * .8 * math.sin(a))), 6,
                   (0, 0, 255) if tape_state == 'found' else (255, 160, 0), -1)
    if rot is not None:
        cv2.rectangle(out, (0, H - 62), (250, H), (0, 0, 0), -1)
        cv2.putText(out, "Rotations: %.2f" % rot, (8, H - 36), FONT, 0.75, (255, 255, 255), 2)
        cv2.putText(out, "full laps seen: %d  t=%.2fs" % (laps or 0, t or 0.0),
                    (8, H - 10), FONT, 0.5, (200, 200, 200), 1)
    return out


def encode_jpeg(bgr, max_w=DISPLAY_MAX_W, quality=80):
    h, w = bgr.shape[:2]
    if w > max_w:
        bgr = cv2.resize(bgr, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    return 'data:image/jpeg;base64,' + base64.b64encode(
        cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]).decode()


def extract_params(d):
    def num(key, default, lo, hi):
        try: v = float(d.get(key, default))
        except (TypeError, ValueError): v = default
        return min(hi, max(lo, v))
    return {'dark_thr': num('dark_thr', wa.DARK_THR, 0.30, 0.95),
            'new_lap_jump': num('new_lap_jump', wa.NEW_LAP_JUMP, 0.10, 0.90)}


# ---------- Video storage / frame access ----------
# Every uploaded video lives in ONE folder inside the app cache. Only the video that is currently
# loaded is kept; everything else is deleted on upload, on app start and when the app is closed.
VIDEO_DIR = os.path.join(tempfile.gettempdir(), 'wheeltracker_videos')
_LEGACY_NAME = re.compile(r'^[0-9a-f]{32}_')   # old versions wrote <sid>_<name> straight into the temp dir


def _rm(path):
    try: os.remove(path)
    except OSError: pass


def startup_cleanup():
    """Delete every video left behind by earlier runs (this layout and the old one)."""
    os.makedirs(VIDEO_DIR, exist_ok=True)
    for d in (VIDEO_DIR, tempfile.gettempdir()):
        try: names = os.listdir(d)
        except OSError: continue
        for n in names:
            if d == VIDEO_DIR or _LEGACY_NAME.match(n): _rm(os.path.join(d, n))


def drop_session(sid):
    s = SESSIONS.pop(sid, None)
    if s:
        try: s['src'].close()
        except Exception: pass
        _rm(s['video_path'])


def drop_all():
    for sid in list(SESSIONS): drop_session(sid)
    startup_cleanup()


class FrameSource:
    """One persistent decoder per video (opening a decoder for every request is very slow on
    Android) plus a tiny LRU of decoded frames, so re-rendering the same frame costs no decoding at
    all. Cached frames are shared: never modify them."""

    def __init__(self, path, keep=2):
        self.path, self.keep = path, keep
        self.lock = threading.Lock()
        self.cap, self.cache = None, collections.OrderedDict()
        self.thumbs = collections.OrderedDict()      # frame index -> encoded thumbnail (data URL)

    def _open(self):
        if self.cap is None: self.cap = cv2.VideoCapture(self.path)
        return self.cap

    def info(self):
        """(fps, frame_count) or None if the video cannot be opened."""
        with self.lock:
            c = self._open()
            if not c.isOpened(): return None
            return c.get(cv2.CAP_PROP_FPS), int(c.get(cv2.CAP_PROP_FRAME_COUNT))

    def get(self, idx, cache=True):
        with self.lock:
            fr = self.cache.get(idx)
            if fr is not None:
                self.cache.move_to_end(idx)
                return fr
            c = self._open()
            c.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, fr = c.read()
            if not ok: return None
            if cache:
                self.cache[idx] = fr
                while len(self.cache) > self.keep: self.cache.popitem(last=False)
            return fr

    def thumb(self, idx, max_w=130):
        """Small JPEG data URL of one frame. Uses the decoder's own down-scaling when it has one
        (Android: no full-size bitmap copy per thumbnail) and remembers finished thumbnails."""
        with self.lock:
            t = self.thumbs.get(idx)
            if t is not None:
                self.thumbs.move_to_end(idx)
                return t
            c = self._open()
            small = None
            fn = getattr(c, 'read_thumb', None)
            if fn is not None:
                c.set(cv2.CAP_PROP_POS_FRAMES, idx)
                try: small = fn(max_w)
                except Exception: small = None
            if small is None:
                fr = self.cache.get(idx)
                if fr is None:
                    c.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ok, fr = c.read()
                    if not ok: return None
                h, w = fr.shape[:2]
                small = cv2.resize(fr, (max_w, max(1, int(h * max_w / w))), interpolation=cv2.INTER_AREA)
            t = encode_jpeg(small, max_w=max_w, quality=70)
            self.thumbs[idx] = t
            while len(self.thumbs) > 300: self.thumbs.popitem(last=False)
            return t

    def close(self):
        with self.lock:
            self.thumbs.clear()
            self.cache.clear()
            if self.cap is not None:
                try: self.cap.release()
                except Exception: pass
                self.cap = None


# ---------- Final result ----------
def finish_event(sol, fps, p, sf, ef, extra, sess):
    """Build the final 'done' event from wheel_algo.solve() output and keep what the frame viewer
    needs (box / tape / phase per frame) in the session."""
    n, idx, phase = sol['n'], sol['idx'], sol['phase']
    ph0 = float(phase[0])
    rot = (phase - ph0) / 360.0
    entries, seen = sol['entries'], sol['seen']
    lap_frames = [int(idx[e]) for e in entries]
    lap_periods = [float(idx[b] - idx[a]) / fps for a, b in zip(entries, entries[1:])] if fps > 0 else []

    L = [f"frames: {n}  fps: {fps:.2f}  flywheel missed: {sol['missed']}",
         f"lap start frames: {lap_frames}",
         f"lap periods (s): {[round(x, 2) for x in lap_periods]}",
         f"rotations (from tape sightings): {sol['seen_only']:.2f}",
         f"rotations (incl. estimate to end): {sol['total']:.2f}"]
    L.append(f"full rotations: {int(math.floor(sol['total'] + 1e-9))}")
    if fps > 0 and sol['seen_only'] > 0.05:
        per = (float(idx[seen[-1]] - idx[seen[0]]) / fps) / sol['seen_only']
        w = 2 * math.pi / per
        L += [f"average period per rotation: {per:.4f} s",
              f"average angular velocity: {w:.4f} rad/s ({w * 60 / (2 * math.pi):.2f} RPM)"]
    L += list(extra)
    summary = "\n".join(L)

    hist = []
    for k in range(n):
        hist.append({'frame': int(idx[k]), 'time_s': round(float(idx[k]) / fps if fps > 0 else 0.0, 5),
                     'flywheel': 'detected' if sol['raw_ok'][k] else 'held',
                     'tape': 'found' if sol['found'][k] else 'hidden',
                     'tape_y': round(float(sol['y'][k]), 4) if sol['found'][k] else None,
                     'phase_deg': round(float(phase[k] - ph0), 3),
                     'rotation_count': round(float(rot[k]), 3)})
    laps = []
    for k, e in enumerate(entries):
        laps.append({'lap': k + 1, 'frame': int(idx[e]), 'time_s': round(float(idx[e]) / fps if fps > 0 else 0.0, 3),
                     'period_s': round(lap_periods[k], 4) if k < len(lap_periods) else None})

    times = np.array([h['time_s'] for h in hist])
    fig, (a0, a1, a2) = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True)
    ys = np.where(sol['found'], sol['y'], np.nan)
    a0.plot(times, ys, '.', ms=3, color='crimson')
    for e in entries: a0.axvline(times[e], color='gray', lw=0.6, ls='--')
    a0.invert_yaxis()
    a0.set(ylabel="tape height (0 = top)", title="Tape height on the wheel (dashed = new lap)"); a0.grid(alpha=0.3)
    a1.plot(times, phase - ph0, '-', lw=1, color='lightsteelblue')
    a1.plot(times[seen], (phase - ph0)[seen], '.', ms=3, color='steelblue')
    a1.set(ylabel="cumulative angle (deg)",
           title="Tape angle vs time (dots = tape visible, line = interpolated / estimated)"); a1.grid(alpha=0.3)
    a2.plot(times, rot, color='darkorange')
    a2.set(xlabel="time (s)", ylabel="rotations", title="Cumulative rotations vs time"); a2.grid(alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=90, bbox_inches='tight')
    plt.close(fig); buf.seek(0)
    plot = 'data:image/png;base64,' + base64.b64encode(buf.read()).decode()

    sess['result'] = {'idx': idx, 'raw_ok': sol['raw_ok'], 'box': sol['box'], 'found': sol['found'],
                      'y': sol['y'], 'phase': phase, 'entries': np.array(entries, dtype=np.int64)}
    return {'done': True, 'summary': summary, 'rotation_count': round(sol['total'], 2),
            'rotations_seen': round(sol['seen_only'], 2), 'laps': laps,
            'history': hist, 'plot': plot, 'start_frame': sf, 'end_frame': ef}


def _frame_range(sess, t0, t1):
    fps, total = sess['fps'], sess['total_frames']
    sf = max(0, int(t0 * fps)) if t0 is not None else 0
    ef = min(total, int(t1 * fps)) if t1 is not None else total
    return fps, total, sf, ef


def draw_calib(out, s, c):
    """Calibration box (x, y, w, h in ORIGINAL pixels) as a cyan rectangle."""
    x, y, w, h = [int(round(float(c[k]) * s)) for k in 'xywh']
    cv2.rectangle(out, (x, y), (x + w, y + h), (255, 200, 0), 1)
    cv2.putText(out, "calibration", (x + 3, max(14, y - 5)), FONT, 0.45, (255, 200, 0), 1)


def _pct(d, key, default, lo, hi):
    try: v = float(d.get(key, default))
    except (TypeError, ValueError): v = default
    return min(hi, max(lo, v)) / 100.0


def make_cfg(sess, d, sf=0):
    """Fixed-size flywheel box for the whole run. The user's calibration box (original pixels) if
    given, otherwise the median detection over the first frames of the range."""
    st, sc = sess['step'], sess['scale']
    area_tol, drift = _pct(d, 'area_tol_pct', 10, 1, 50), _pct(d, 'max_drift_pct', 6, 0, 30)
    c = d.get('calib')
    if c:
        try:
            box = [float(c[k]) / st for k in 'xywh']
            fi = max(0, min(int(c.get('frame', sf)), sess['total_frames'] - 1))
        except (TypeError, ValueError, KeyError):
            raise UserError('The calibration box is not valid.')
        if box[2] < 8 or box[3] < 8: raise UserError('The calibration box is too small.')
        fr = sess['src'].get(fi)
        if fr is None: raise UserError('Cannot read the calibration frame.')
        return wa.TrackCfg(box, fr[::st, ::st], area_tol, drift)
    found, ref = [], None
    for fi in range(sf, min(sess['total_frames'], sf + 120), 15):
        fr = sess['src'].get(fi)
        if fr is None: continue
        small = fr[::st, ::st]
        b = wa.detect_flywheel(small, sc)
        if b is not None:
            found.append(b)
            if ref is None: ref = small
    if not found:
        raise UserError('The flywheel was not found automatically. Calibrate it by hand '
                        '(Flywheel Calibration panel).')
    return wa.TrackCfg(np.median(np.array(found, dtype=float), axis=0), ref, area_tol, drift)


def _feed(ext, i, bgr):
    try: return ext.feed(i, bgr)
    except RuntimeError as e: raise UserError(str(e))


def _solve(exts, p):
    try:
        idx, raw, box, prof = wa.merge_extractors(exts)
        return wa.solve(idx, raw, box, prof, p)
    except RuntimeError as e:
        raise UserError(str(e))


# ---------- Analysis stream ----------
def run_analysis_stream(sess, p, live=False, every=3, t0=None, t1=None, fps_cap=0, cfg=None):
    """Sequential analysis with an optional live preview (annotated frames with a running estimate;
    the exact result comes from the final solve). The hardware decoder delivers full colour frames
    (RANGE_READER); the preview is made from the very same frame, so it costs no extra decoding.
    fps_cap > 0 (live mode only) paces the analysis to at most that many frames per second and shows
    EVERY processed frame, each stamped with its playback time ('at'), so the page can play the
    preview back smoothly. fps_cap = 0: full speed with a sparse preview."""
    fps, total, sf, ef = _frame_range(sess, t0, t1)
    n_total = ef - sf
    if n_total <= 0:
        yield {'error': 'Nothing to analyse (empty time range).'}; return
    ext, ctr = wa.FrameExtractor(sess['scale'], cfg), wa.LiveTracker(p)
    cap = float(fps_cap) if (live and fps_cap and fps_cap > 0) else 0.0
    pe = (1 if cap else max(1, int(every))) if live else 0
    st = {'proc': 0, 'next': sf, 'last_pv': 0.0}
    clk = {'origin': None, 'begin': None}
    notes = []
    t_start = time.time()

    def consume(reader):
        for i, bgr in reader:
            raw, box, pr = _feed(ext, i, bgr)
            lv = ctr.step(pr)
            st['proc'] += 1; st['next'] = i + 1
            at = None
            if cap:
                now = time.perf_counter()
                if clk['origin'] is None: clk['origin'] = clk['begin'] = now
                due = clk['begin'] + (st['proc'] - 1) / cap
                if due > now: time.sleep(due - now)
                elif now - due > 0.25: clk['begin'] += now - due     # fell behind: no burst catch-up
                at = clk['begin'] + (st['proc'] - 1) / cap - clk['origin']
            want = False
            if live:
                if cap: want = True
                else:
                    now = time.time()
                    if (st['proc'] - 1) % pe == 0 and now - st['last_pv'] >= PREVIEW_MS / 1000.0:
                        want, st['last_pv'] = True, now
            if want:
                tape = 'found' if lv['found'] else ('hidden' if ctr.med is not None and box is not None else None)
                small, sc = fit_display(bgr, 640)
                draw_overlay(small, sc, box, raw is not None, lv['y'], tape, lv['phase'],
                             lv['rot'] if box is not None else None, lv['laps'], i / fps if fps > 0 else 0.0)
                yield {'preview': {'frame': i, 'time_s': round(i / fps if fps > 0 else 0.0, 5),
                                   'image': encode_jpeg(small, quality=70), 'at': at,
                                   'wheel': 'none' if box is None else ('detected' if raw is not None else 'held'),
                                   'tape': tape, 'y': lv['y'], 'rotation_count': round(lv['rot'], 2),
                                   'laps': lv['laps'], 'total_frames': n_total, 'start_frame': sf}}

    try:
        yield from consume(RANGE_READER(sess, sf, ef))
    except UserError:
        raise
    except Exception as e:
        if RANGE_READER is cv_range_reader: raise
        notes.append(f"Fast decoder failed ({e}); finished with the slow fallback")
        yield from consume(cv_range_reader(sess, st['next'], ef))
    proc = st['proc']
    if not proc:
        yield {'error': 'No frames could be decoded.'}; return
    dt = max(1e-6, time.time() - t_start)
    extra = ["parallel parts: 1", f"processing time: {dt:.2f} s ({proc / dt:.0f} frames/s)"]
    if proc < n_total: extra.append(f"warning: decoded {proc} of {n_total} frames")
    extra += notes
    print(f"ANALYSIS: {proc} frames in {dt:.1f}s ({proc / dt:.0f} fps)")
    sol = _solve([ext], p)
    yield finish_event(sol, fps, p, sf, ef, extra, sess)


# ---- Fast / parallel analysis: split the video into parts, measure each part on its own thread ----
def cv_range_reader(sess, sf, ef):
    """Generic reader: yields (frame_index, BGR frame sub-sampled by sess['step']) for frames
    [sf, ef). Android replaces RANGE_READER with a hardware-decoder version (main.py)."""
    st = sess['step']
    cap = cv2.VideoCapture(sess['video_path'])
    try:
        if sf: cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
        for i in range(sf, ef):
            ok, fr = cap.read()
            if not ok: break
            yield i, (np.ascontiguousarray(fr[::st, ::st]) if st > 1 else fr)
    finally:
        cap.release()


RANGE_READER = cv_range_reader


def run_parallel_stream(sess, p, parts=2, t0=None, t1=None, cfg=None):
    """Split [start, end) into `parts` chunks, find the flywheel and take the brightness profile of
    every frame of a chunk on its own thread (per-frame measurement is independent), then join the
    chunks (the 'last known box' rule carries over chunk borders) and track the tape ONCE, in order.
    Result is identical to a sequential run."""
    fps, total, sf, ef = _frame_range(sess, t0, t1)
    n_frames = ef - sf
    if n_frames <= 0:
        yield {'error': 'Nothing to analyse (empty time range).'}; return
    n = max(1, min(int(parts), 8, n_frames))
    edges = [sf + n_frames * k // n for k in range(n + 1)]
    exts = [wa.FrameExtractor(sess['scale'], cfg) for _ in range(n)]
    counts, errs, notes = [0] * n, [None] * n, []
    stop = threading.Event()

    def consume(k, reader):
        ext = exts[k]
        for i, bgr in reader:
            if stop.is_set(): return
            _feed(ext, i, bgr)
            counts[k] = len(ext.idx)

    def worker(k):
        a, b = edges[k], edges[k + 1]
        try:
            consume(k, RANGE_READER(sess, a, b))
        except UserError as e:
            errs[k] = str(e); stop.set()
        except Exception as e:
            if RANGE_READER is cv_range_reader:
                errs[k] = str(e); return
            notes.append(f"part {k + 1}: fast decoder failed ({e}); used slow fallback")
            exts[k] = wa.FrameExtractor(sess['scale'], cfg); counts[k] = 0
            try:
                consume(k, cv_range_reader(sess, a, b))
            except Exception as e2:
                errs[k] = str(e2)

    t_start = time.time()
    ths = [threading.Thread(target=worker, args=(k,), daemon=True) for k in range(n)]
    try:
        for t in ths: t.start()
        while any(t.is_alive() for t in ths):
            time.sleep(0.25)
            d = sum(counts)
            yield {'progress': min(0.99, d / n_frames), 'done_frames': d,
                   'total': n_frames, 'parts': n}
        for t in ths: t.join()
    finally:
        stop.set()
    if any(errs):
        yield {'error': next(e for e in errs if e)}; return
    done = sum(len(e.idx) for e in exts)
    if not done:
        yield {'error': 'No frames could be decoded.'}; return
    wall = max(1e-6, time.time() - t_start)
    sol = _solve(exts, p)
    extra = [f"parallel parts: {n}", f"processing time: {wall:.2f} s ({done / wall:.0f} frames/s)"]
    if done < n_frames: extra.append(f"warning: decoded {done} of {n_frames} frames")
    extra += notes
    print(f"PARALLEL ANALYSIS: {done} frames, {n} parts, {wall:.1f}s ({done / wall:.0f} fps)")
    yield finish_event(sol, fps, p, sf, ef, extra, sess)


# ---------- Routes ----------
@app.route('/')
def index(): return TEMPLATE


@app.route('/upload', methods=['POST'])
def upload():
    """The page posts the raw file body (X-Filename header) which is streamed straight to disk:
    no multipart parsing and no second temporary copy of the video."""
    os.makedirs(VIDEO_DIR, exist_ok=True)
    sid = uuid.uuid4().hex
    if request.mimetype == 'multipart/form-data':          # legacy form upload
        f = request.files.get('video')
        if not f or not f.filename: return jsonify({'error': 'no file'}), 400
        name = os.path.basename(f.filename)
        tmp = os.path.join(VIDEO_DIR, f"{sid}_{name}")
        f.save(tmp)
    else:
        name = os.path.basename(unquote(request.headers.get('X-Filename', 'video.mp4'))) or 'video.mp4'
        safe = re.sub(r'[^A-Za-z0-9._-]', '_', name)[-80:]
        tmp = os.path.join(VIDEO_DIR, f"{sid}_{safe}")
        with open(tmp, 'wb') as out:
            shutil.copyfileobj(request.stream, out, 1 << 20)
    src = FrameSource(tmp)
    inf = src.info()
    if inf is None:
        src.close(); _rm(tmp)
        return jsonify({'error': 'cannot open video'}), 400
    fps, total = inf
    if not fps or fps <= 0: fps = 30.0
    frame = src.get(0, cache=False)
    if frame is None:
        src.close(); _rm(tmp)
        return jsonify({'error': 'cannot read first frame'}), 400
    h, w = frame.shape[:2]
    step = max(1, int(math.ceil(w / float(DETECT_W))))
    for old in list(SESSIONS): drop_session(old)        # keep only the video that is loaded now
    SESSIONS[sid] = {'video_path': tmp, 'fps': fps, 'total_frames': total, 'src': src,
                     'orig_w': w, 'orig_h': h, 'filename': name, 'result': None,
                     'step': step, 'scale': 1.0 / step}
    return jsonify({'sid': sid, 'fps': fps, 'total_frames': total,
                    'orig_w': w, 'orig_h': h, 'filename': name,
                    'duration': total / fps if fps else 0})


@app.route('/render', methods=['POST'])
def render():
    """One annotated frame for the slider. After an analysis (use_result) the box, tape, dial and
    rotation counter come from that analysis; before it only the flywheel is detected in this frame."""
    t0 = time.time()
    d = request.get_json(force=True)
    sess = SESSIONS.get(d.get('sid'))
    if not sess: return jsonify({'error': 'invalid session'}), 400
    idx = max(0, min(int(d.get('frame_idx', 0)), sess['total_frames'] - 1))
    frame = sess['src'].get(idx)
    if frame is None: return jsonify({'error': 'cannot read frame'}), 400
    t1 = time.time()
    res = sess.get('result') if d.get('use_result') else None
    k = None
    if res is not None:
        k = int(np.searchsorted(res['idx'], idx))
        if k >= len(res['idx']) or res['idx'][k] != idx: k = None
    disp, sc = fit_display(frame)
    if k is not None:
        box, detected, found = res['box'][k] * sess['step'], bool(res['raw_ok'][k]), bool(res['found'][k])
        ty = float(res['y'][k]) if found else None
        ph = float(res['phase'][k])
        rot = (ph - float(res['phase'][0])) / 360.0
        laps = int(np.searchsorted(res['entries'], k, side='right'))
        draw_overlay(disp, sc, box, detected, ty, 'found' if found else 'hidden', ph, rot, laps,
                     idx / sess['fps'])
        out = {'wheel': 'detected' if detected else 'held', 'tape': 'found' if found else 'hidden',
               'y': ty, 'rotation_count': round(rot, 2), 'laps': laps}
    else:
        st = sess['step']
        box = wa.detect_flywheel(frame[::st, ::st], sess['scale'])
        if box is not None: box = tuple(v * st for v in box)
        draw_overlay(disp, sc, box, box is not None)
        out = {'wheel': 'detected' if box is not None else 'none', 'tape': None, 'y': None,
               'rotation_count': None, 'laps': None}
    c = d.get('calib')
    if c:
        try: draw_calib(disp, sc, c)
        except (TypeError, ValueError, KeyError): pass
    out['image'] = encode_jpeg(disp)
    print(f"RENDER frame {idx}: decode {1000 * (t1 - t0):.0f} ms, rest {1000 * (time.time() - t1):.0f} ms")
    return jsonify(out)


@app.route('/crop_strip', methods=['POST'])
def crop_strip():
    d = request.get_json(force=True)
    sess = SESSIONS.get(d.get('sid'))
    if not sess: return jsonify({'error': 'invalid session'}), 400
    n = max(2, min(int(d.get('n', 20)), 40))
    before, after = float(d.get('before', 3.0)), float(d.get('after', 3.0))
    fps, total = sess['fps'], sess['total_frames']
    dur = total / fps if fps else 0.0
    c = max(0.0, min(float(d.get('center_time', 0.0)), dur))
    ws, we = c - before, c + after
    if ws < 0: we = min(dur, we - ws); ws = 0.0
    if we > dur: ws = max(0.0, ws - (we - dur)); we = dur
    s_idx, e_idx = int(ws * fps), int(we * fps)
    idxs = [max(0, min(int(s_idx + (e_idx - s_idx) * i / (n - 1)), total - 1))
            for i in range(n)]
    # The page asks for the strip in small batches (offset/count) so thumbnails appear as they finish.
    off = max(0, int(d.get('offset', 0)))
    cnt = max(1, int(d.get('count', n)))
    slots = [{'slot': k, 'frame': i, 'time_s': round(i / fps if fps else 0, 3)} for k, i in enumerate(idxs)]
    thumbs = []
    for sl in slots[off:off + cnt]:
        img = sess['src'].thumb(sl['frame'], 130)
        if img is not None: thumbs.append(dict(sl, image=img))
    return jsonify({'thumbs': thumbs, 'slots': slots, 'thumb_h': max(1, round(130 * sess['orig_h'] / sess['orig_w'])),
                    'win_start': round(ws, 3), 'win_end': round(we, 3), 'center_time': round(c, 3),
                    'duration': round(dur, 3), 'fps': fps})


@app.route('/detect', methods=['POST'])
def detect():
    """Auto-detected flywheel box (original pixels) of one frame: start point for calibration."""
    d = request.get_json(force=True)
    sess = SESSIONS.get(d.get('sid'))
    if not sess: return jsonify({'error': 'invalid session'}), 400
    idx = max(0, min(int(d.get('frame_idx', 0)), sess['total_frames'] - 1))
    frame = sess['src'].get(idx)
    if frame is None: return jsonify({'error': 'cannot read frame'}), 400
    st = sess['step']
    b = wa.detect_flywheel(frame[::st, ::st], sess['scale'])
    if b is None: return jsonify({'error': 'Flywheel not detected in this frame - move the slider or enter the box by hand.'})
    return jsonify({'box': [int(v * st) for v in b], 'frame': idx})


def _time_range(d):
    t0 = d.get('time_start'); t1 = d.get('time_end')
    t0 = max(0.0, float(t0)) if t0 is not None else None
    t1 = max(0.0, float(t1)) if t1 is not None else None
    return t0, t1


@app.route('/analyze', methods=['POST'])
def analyze():
    d = request.get_json(force=True)
    sess = SESSIONS.get(d.get('sid'))
    if not sess: return jsonify({'error': 'invalid session'}), 400
    p = extract_params(d)
    live, every = bool(d.get('live_preview')), max(1, int(d.get('preview_every', 3)))
    try: fps_cap = max(0.0, min(120.0, float(d.get('fps_cap') or 0)))
    except (TypeError, ValueError): fps_cap = 0.0
    t0, t1 = _time_range(d)
    if t0 is not None and t1 is not None and t1 <= t0:
        return jsonify({'error': 'End time must be greater than start time.'}), 400

    def gen():
        try:
            # Without live preview nothing has to be drawn: use the plain one-part reader path.
            cfg = make_cfg(sess, d, _frame_range(sess, t0, t1)[2])
            it = run_analysis_stream(sess, p, live, every, t0, t1, fps_cap, cfg) if live \
                 else run_parallel_stream(sess, p, 1, t0, t1, cfg)
            for ev in it:
                yield 'data: ' + _json.dumps(ev) + '\n\n'
        except Exception as e:
            yield 'data: ' + _json.dumps({'error': str(e)}) + '\n\n'
    return Response(gen(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/fast_analyze', methods=['POST'])
def fast_analyze():
    d = request.get_json(force=True)
    sess = SESSIONS.get(d.get('sid'))
    if not sess: return jsonify({'error': 'invalid session'}), 400
    p = extract_params(d)
    parts = max(1, min(int(d.get('parts', 2) or 2), 8))
    t0, t1 = _time_range(d)
    if t0 is not None and t1 is not None and t1 <= t0:
        return jsonify({'error': 'End time must be greater than start time.'}), 400

    def gen():
        try:
            cfg = make_cfg(sess, d, _frame_range(sess, t0, t1)[2])
            for ev in run_parallel_stream(sess, p, parts, t0, t1, cfg):
                yield 'data: ' + _json.dumps(ev) + '\n\n'
        except Exception as e:
            yield 'data: ' + _json.dumps({'error': str(e)}) + '\n\n'
    return Response(gen(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


# ---------- Template ----------
TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>EP LAB - Flywheel</title>
<style>
*{box-sizing:border-box}
body{margin:0;font:13px -apple-system,"Segoe UI",Roboto,sans-serif;background:#181818;color:#eee}
header{background:#0f0f0f;padding:12px 22px;border-bottom:1px solid #2b2b2b}
header h1{margin:0;font-size:17px;letter-spacing:.5px}
.tabbar{display:flex;background:#0f0f0f;padding:0 22px;border-bottom:1px solid #2b2b2b}
.tabbar button{background:transparent;color:#9aa;border:none;padding:10px 18px;cursor:pointer;font-size:13px;border-bottom:2px solid transparent;border-radius:0}
.tabbar button:hover{color:#ddd;background:#1a1a1a}
.tabbar button.active{color:#63b3ed;border-bottom-color:#63b3ed;background:#181818}
.page{display:none}.page.active{display:block}
.container{display:grid;grid-template-columns:1fr 350px;gap:16px;padding:16px;max-width:1500px;margin:0 auto}
.panel{background:#222;border:1px solid #333;border-radius:6px;padding:12px;margin-bottom:14px}
.panel h3{margin:0 0 10px;font-size:12px;text-transform:uppercase;color:#9aa;letter-spacing:1.2px}
.video-wrap{background:#000;border-radius:4px;overflow:hidden;line-height:0}
#frameImg{display:block;width:100%;cursor:crosshair;user-select:none}
.row{display:flex;gap:8px;align-items:center;margin-bottom:8px;flex-wrap:wrap;font-size:12px}
.row label{color:#aaa}
input[type=number],select{background:#111;border:1px solid #444;color:#eee;padding:5px 7px;border-radius:4px;font-size:12px}
input[type=range]{width:100%;accent-color:#4299e1}
button{background:#2b6cb0;color:#fff;border:none;padding:8px 12px;border-radius:4px;cursor:pointer;font-size:13px}
button:hover:not(:disabled){background:#3182ce}button:disabled{background:#444;cursor:not-allowed}
button.secondary{background:#444}button.secondary:hover{background:#555}
button.pick-active{background:#d69e2e}
.status{font-size:12px;color:#8a8a8a;font-style:italic;margin:6px 0}
.stats{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:8px;font-size:11px}
.stats>div{background:#1a1a1a;padding:7px 9px;border-radius:4px;color:#888}
.stats .big{font-size:20px;font-weight:600;color:#63b3ed;display:block;margin-top:2px}
.roi-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:8px}
.roi-grid button{padding:7px 8px;font-size:12px}
.results{padding:0 16px 16px;max-width:1500px;margin:0 auto}
pre.summary{background:#111;padding:12px;border-radius:4px;font-size:12px;white-space:pre-wrap;margin:0;border:1px solid #2a2a2a}
img.plot{width:100%;background:#fff;border-radius:4px;margin-top:10px;display:block}
.moi-grid{display:grid;grid-template-columns:auto 1fr auto 1fr;gap:8px 10px;align-items:center;font-size:12px;margin-bottom:8px}
.moi-grid label{color:#aaa}
.maths-page{max-width:1100px;margin:0 auto;padding:16px}
.formula{font-size:20px;text-align:center;padding:14px 8px;background:#111;border-radius:6px;margin-bottom:12px;font-family:"Times New Roman",serif;overflow-x:auto;white-space:nowrap}
.formula .frac{display:inline-block;vertical-align:middle;text-align:center;margin:0 4px}
.formula .frac>span{display:block;padding:0 6px}
.formula .frac>span:first-child{border-bottom:1px solid #ccc}
.mtable-wrap{overflow-x:auto}
table.mtable{border-collapse:collapse;width:100%;font-size:12px;min-width:560px}
.mtable th,.mtable td{border:1px solid #3a3a3a;padding:5px 6px;text-align:center}
.mtable th{background:#1b1b1b;color:#9aa;font-weight:600}
.mtable input{width:100%;min-width:64px;box-sizing:border-box;text-align:center}
.mtable td.res{color:#63b3ed;font-weight:600}
.moi-result{font-size:15px;font-weight:600;color:#63b3ed;margin-top:10px}
progress{width:100%;height:6px;appearance:none;border:none;border-radius:3px;background:#333}
progress::-webkit-progress-bar{background:#333;border-radius:3px}
progress::-webkit-progress-value{background:#4299e1;border-radius:3px}
progress::-moz-progress-bar{background:#4299e1}
.badge{display:inline-block;padding:2px 8px;border-radius:3px;color:#fff;font-size:10px;font-weight:600;letter-spacing:1px;margin-left:6px;opacity:0;transition:opacity .2s}
.badge.active{opacity:1}
.live-badge{background:#2f855a}.crop-badge{background:#b7791f}
.crop-page{max-width:1200px;margin:0 auto;padding:16px}
.crop-section{background:#222;border:1px solid #333;border-radius:6px;padding:14px;margin-bottom:16px}
.crop-section h3{margin:0 0 10px;font-size:12px;text-transform:uppercase;color:#9aa;letter-spacing:1.2px}
.crop-section .hint{font-size:12px;color:#888;margin-bottom:8px}
.thumb-strip{display:flex;gap:4px;overflow-x:auto;padding:8px 4px;background:#161616;border:1px solid #2a2a2a;border-radius:4px}
.thumb-strip .thumb{flex:0 0 auto;text-align:center;cursor:pointer;padding:2px;border-radius:4px;border:2px solid transparent;transition:border .1s}
.thumb-strip .thumb:hover{border-color:#4299e1}
.thumb-strip .thumb.selected{border-color:#ecc94b;background:#2a2410}
.thumb-strip img{display:block;border-radius:3px}
.thumb-strip .tlabel{font-size:9px;color:#aaa;margin-top:1px}
.thumb-strip .thumb.selected .tlabel{color:#ecc94b;font-weight:600}
.crop-controls{display:grid;grid-template-columns:1fr 1fr auto;gap:10px;align-items:center;margin-top:10px}
.crop-controls .field{display:flex;flex-direction:column;gap:3px}
.crop-controls label{font-size:11px;color:#888}
.crop-controls input[type=number]{width:100%}
.window-info{font-size:12px;color:#888;margin-top:6px}
.selection-info{font-size:13px;color:#ecc94b;margin-top:8px;font-weight:600}
.summary-bar{position:sticky;bottom:0;background:#1a1a1a;border-top:1px solid #333;padding:12px 20px;display:flex;gap:20px;align-items:center;justify-content:center;font-size:13px}
.summary-bar .val{color:#ecc94b;font-weight:600}

.stats.three{grid-template-columns:1fr 1fr 1fr}
.stats .mid{font-size:15px}
.hint{font-size:12px;color:#888}
.ok{color:#68d391 !important}.warn{color:#f6ad55 !important}.bad{color:#fc8181 !important}
</style>
</head>
<body>
<header><h1>EP LAB - Flywheel</h1></header>
<div class="tabbar">
  <button id="tabAnalysis" class="active" onclick="showTab('analysis')">Analysis</button>
  <button id="tabCrop" onclick="showTab('crop')">Time Crop</button>
  <button id="tabFast" onclick="showTab('fast')">Parallel</button>
  <button id="tabMaths" onclick="showTab('maths')">Maths</button>
</div>

<!-- ============ ANALYSIS PAGE ============ -->
<div id="pageAnalysis" class="page active">
<div class="container">
  <div>
    <div class="panel"><div class="row">
      <input type="file" id="videoFile" accept="video/*">
      <button onclick="uploadVideo()">Load Video</button>
      <span id="fileLabel" class="status">No video loaded</span>
    </div></div>

    <div class="panel">
      <div class="video-wrap"><img id="frameImg" alt="Video frame"></div>
      <input type="range" id="frameSlider" min="0" max="0" value="0" disabled>
      <div id="frameInfo" class="status">Frame: 0 / 0 &nbsp; Time: 0.00s</div>
      <div id="statusMsg" class="status">Load a side-view video of the flywheel: a gray block crossing the blue bracket, one black tape on the rim. Nothing has to be marked by hand &ndash; the flywheel is found automatically in every frame. Drag the slider to check it (green box = detected, orange = last known position kept).</div>
      <div class="stats three">
        <div>Flywheel<span id="lblWheel" class="big mid">-</span></div>
        <div>Tape<span id="lblTape" class="big mid">-</span></div>
        <div>Tape height<span id="lblY" class="big mid">-</span></div>
        <div>Laps seen<span id="lblLaps" class="big mid">-</span></div>
        <div>Elapsed<span id="lblTime" class="big mid">0.00s</span></div>
        <div>Rotations<span id="lblCount" class="big">0</span></div>
      </div>
    </div>
  </div>

  <div>
    <div class="panel">
      <h3>Flywheel Calibration <span id="calBadge" class="badge crop-badge">CALIBRATED</span></h3>
      <div class="hint" style="margin-bottom:8px">Pick a frame where the whole flywheel is visible, press <b>Auto-fill</b> (or type the box), then nudge it onto the flywheel. The box keeps this exact size for the whole video and may only drift a few pixels; if the detected area suddenly changes by more than the tolerance the box is re-located by matching the calibrated picture instead of being resized.</div>
      <div class="row"><button onclick="calAuto()">Auto-fill from this frame</button>
        <button class="secondary" onclick="calClear()">Clear</button></div>
      <div class="row"><label style="width:34px">X</label><input type="number" id="calX" style="width:78px" onchange="calEdit()">
        <button class="secondary" onclick="calNudge(-2,0,0,0)">&#9664;</button><button class="secondary" onclick="calNudge(2,0,0,0)">&#9654;</button>
        <label style="width:34px;margin-left:8px">Y</label><input type="number" id="calY" style="width:78px" onchange="calEdit()">
        <button class="secondary" onclick="calNudge(0,-2,0,0)">&#9650;</button><button class="secondary" onclick="calNudge(0,2,0,0)">&#9660;</button></div>
      <div class="row"><label style="width:34px">W</label><input type="number" id="calW" style="width:78px" onchange="calEdit()">
        <button class="secondary" onclick="calNudge(0,0,-2,0)">&minus;</button><button class="secondary" onclick="calNudge(0,0,2,0)">+</button>
        <label style="width:34px;margin-left:8px">H</label><input type="number" id="calH" style="width:78px" onchange="calEdit()">
        <button class="secondary" onclick="calNudge(0,0,0,-2)">&minus;</button><button class="secondary" onclick="calNudge(0,0,0,2)">+</button></div>
      <div class="row"><label style="flex:1">Max drift (% of box width)</label>
        <input type="number" id="driftPct" value="6" min="0" max="30" step="1" style="width:70px"></div>
      <div class="row"><label style="flex:1">Area change tolerance (%)</label>
        <input type="number" id="areaTol" value="10" min="1" max="50" step="1" style="width:70px"></div>
      <div id="calInfo" class="status">Not calibrated: the box size is taken from the first detections of the video.</div>
    </div>

    <div class="panel">
      <h3>Tape Detection</h3>
      <div class="row"><label style="flex:1">Dark threshold</label>
        <input type="number" id="darkNum" value="0.70" min="0.30" max="0.95" step="0.01" style="width:80px"></div>
      <input type="range" id="darkRange" min="0.30" max="0.95" step="0.01" value="0.70">
      <div class="hint" style="margin:4px 0 10px">The tape is where the brightness of a wheel row falls below this fraction of its usual value (the median over the whole video, so the rod, shaft and edges cancel out). Raise it if the tape is not found, lower it if glare is picked up.</div>
      <div class="row"><label style="flex:1">New-lap jump</label>
        <input type="number" id="lapNum" value="0.30" min="0.10" max="0.90" step="0.01" style="width:80px"></div>
      <input type="range" id="lapRange" min="0.10" max="0.90" step="0.01" value="0.30">
      <div class="hint" style="margin-top:4px">A new lap starts when the tape jumps back up the wheel by more than this fraction of its height.</div>
    </div>

    <div class="panel">
      <h3>Time Crop <span id="cropBadge" class="badge crop-badge">CROPPED</span></h3>
      <div class="row"><input type="checkbox" id="enableCrop">
        <label for="enableCrop">Use crop range from Time Crop page</label></div>
      <div id="cropInfo" class="status">Crop: full video</div>
      <div class="row"><button class="secondary" onclick="showTab('crop')">Open Time Crop page</button></div>
    </div>

    <div class="panel">
      <h3>Run Analysis <span id="liveBadge" class="badge live-badge">LIVE</span></h3>
      <div class="row"><input type="checkbox" id="livePreview" checked>
        <label for="livePreview">Show live frame preview</label></div>
      <div class="row"><label style="flex:1">FPS cap (smooth playback)</label>
        <select id="fpsCap" style="width:130px">
          <option value="0" selected>Unlimited (fastest)</option>
          <option value="60">60 fps</option>
          <option value="30">30 fps</option>
          <option value="15">15 fps</option>
          <option value="10">10 fps</option>
          <option value="5">5 fps</option>
        </select></div>
      <div class="row"><label style="flex:1">Preview every N frames <span class="status">(unlimited only)</span></label>
        <input type="number" id="previewEvery" value="3" min="1" max="60" style="width:70px"></div>
      <div class="status">The live preview shows a running estimate; the exact count (the tape height is compared with the median of the whole video) appears when the run finishes. With a cap, every frame is analysed and shown at that speed so you can count along. Choose Unlimited or untick the preview for the fastest run.</div>
      <div class="row">
        <button id="btnRun" onclick="runAnalysis()">Run Full Analysis</button>
        <button class="secondary" onclick="resetAnalysis()">Reset</button>
      </div>
      <progress id="progress" value="0" max="100"></progress>
    </div>
  </div>
</div>

<div class="results">
  <div class="panel">
    <h3>Results</h3>
    <pre class="summary" id="summary">Run an analysis to see results here.</pre>
    <div class="mtable-wrap" id="lapsWrap" style="display:none;margin-top:10px">
      <table class="mtable" style="min-width:320px">
        <thead><tr><th>Lap</th><th>Starts at frame</th><th>Time (s)</th><th>Period (s)</th></tr></thead>
        <tbody id="lapsBody"></tbody>
      </table>
    </div>
    <div class="row" style="margin-top:10px"><button onclick="exportCSV()">Export CSV</button></div>
    <img id="plot" class="plot" style="display:none">
  </div>
</div>
</div>

<!-- ============ TIME CROP PAGE ============ -->
<div id="pageCrop" class="page">
<div class="crop-page">
  <div class="panel" style="margin-bottom:16px">
    <div class="row" style="margin-bottom:0">
      <strong>Video:</strong>
      <span id="cropFileName" class="status">No video loaded</span>
      <span id="cropDuration" class="status"></span>
    </div>
  </div>

  <div class="crop-section">
    <h3>Start of crop — 20 thumbnails centered on selection</h3>
    <div class="hint">Click a thumbnail to set the crop <b>start</b> time, or drag the slider. Use <b>Load</b> to re-center.</div>
    <div id="startStrip" class="thumb-strip"><div class="hint" style="padding:20px">Load a video first.</div></div>
    <div class="crop-controls">
      <div class="field"><label>Center (s)</label><input type="number" id="startWinCenter" value="0" step="0.1" min="0"></div>
      <div class="field"><label>Selected start (s)</label><input type="number" id="startSelNum" value="0" step="0.01" min="0"></div>
      <div class="field"><label>&nbsp;</label><button onclick="loadStartWindow()">Load</button></div>
    </div>
    <input type="range" id="startSlider" min="0" max="0" value="0" step="0.01">
    <div id="startWinInfo" class="window-info">Window: —</div>
    <div id="startSelInfo" class="selection-info">Start selected: 0.00 s</div>
  </div>

  <div class="crop-section">
    <h3>End of crop — 20 thumbnails centered on selection</h3>
    <div class="hint">Click a thumbnail to set the crop <b>end</b> time, or drag the slider. Use <b>Load</b> to re-center.</div>
    <div id="endStrip" class="thumb-strip"><div class="hint" style="padding:20px">Load a video first.</div></div>
    <div class="crop-controls">
      <div class="field"><label>Center (s)</label><input type="number" id="endWinCenter" value="0" step="0.1" min="0"></div>
      <div class="field"><label>Selected end (s)</label><input type="number" id="endSelNum" value="0" step="0.01" min="0"></div>
      <div class="field"><label>&nbsp;</label><button onclick="loadEndWindow()">Load</button></div>
    </div>
    <input type="range" id="endSlider" min="0" max="0" value="0" step="0.01">
    <div id="endWinInfo" class="window-info">Window: —</div>
    <div id="endSelInfo" class="selection-info">End selected: 0.00 s</div>
  </div>

  <div class="crop-section">
    <h3>Actions</h3>
    <div class="row">
      <button class="secondary" onclick="resetCropToFull()">Reset to full video</button>
      <button onclick="applyCropToAnalysis()">Apply &amp; go to Analysis</button>
      <button onclick="applyCropToFast()">Apply &amp; go to Parallel</button>
    </div>
  </div>
</div>
</div>

<!-- ============ PARALLEL PAGE ============ -->
<div id="pageFast" class="page">
<div class="crop-page">
  <div class="panel">
    <h3>Parallel Analysis <span class="badge live-badge" id="fastBadge">RUNNING</span></h3>
    <div class="hint" style="margin-bottom:8px">
      Uses the video and the tape-detection settings from the Analysis tab. The video is split into
      parts; the flywheel is found and measured in all parts at the same time. The tape is then tracked
      once, in order, so the result is the same as a normal run.</div>
    <div id="fastRoiInfo" class="status">No video loaded (use the Analysis tab).</div>
    <div class="row"><label style="flex:1">Parallel parts (1-8)</label>
      <input type="number" id="fastParts" value="2" min="1" max="8" style="width:70px"></div>
    <div class="row"><input type="checkbox" id="fastCrop">
      <label for="fastCrop">Use crop range from Time Crop page</label></div>
    <div id="fastCropInfo" class="status">Crop: full video</div>
    <div class="row"><button class="secondary" onclick="showTab('crop')">Open Time Crop page</button></div>
    <div class="row">
      <button id="btnFast" onclick="runFast()">Run Parallel Analysis</button>
      <button class="secondary" onclick="resetAnalysis()">Reset</button>
    </div>
    <progress id="fastProgress" value="0" max="100"></progress>
    <div id="fastStatus" class="status">Load a video on the Analysis tab first.</div>
  </div>
  <div class="panel">
    <h3>Results</h3>
    <pre class="summary" id="fastSummary">Run an analysis to see results here.</pre>
    <div class="row" style="margin-top:10px"><button onclick="exportCSV()">Export CSV</button></div>
    <img id="fastPlot" class="plot" style="display:none">
  </div>
</div>
</div>

<!-- ============ MATHS PAGE ============ -->
<div id="pageMaths" class="page">
<div class="maths-page">
  <div class="panel">
    <h3>Moment of Inertia (falling load method)</h3>
    <div class="formula">
      I &nbsp;=&nbsp;
      <span class="frac"><span>r &middot; g &middot; t&sup2; &middot; m</span>
        <span>4&pi; &middot; n<sub>2</sub> &middot; (1 + n<sub>2</sub>/n<sub>1</sub>)</span></span>
    </div>
    <div class="row">
      <label>Radius r (cm)</label><input type="number" id="mR" step="any" placeholder="e.g. 1.25">
      <label>g (cm/s&sup2;)</label><input type="number" id="mG" value="981" step="any">
    </div>
    <div class="status">m in g, t in s, r in cm, g in cm/s&sup2; &rarr; I in g&middot;cm&sup2;. n<sub>1</sub> = revolutions before the load detaches, n<sub>2</sub> = revolutions after it detaches until rest, t = time for n<sub>2</sub> revolutions.</div>
  </div>
  <div class="panel">
    <h3>Observations</h3>
    <div class="mtable-wrap">
    <table class="mtable">
      <thead><tr>
        <th>Mass of hanging load m (g)</th><th>Revolutions before detached n<sub>1</sub></th>
        <th>Revolutions to rest after detached n<sub>2</sub></th><th>Time for n<sub>2</sub> revolutions t (s)</th>
        <th>Calculated I (g&middot;cm&sup2;)</th><th></th>
      </tr></thead>
      <tbody id="mBody"></tbody>
    </table>
    </div>
    <div class="row" style="margin-top:10px">
      <button onclick="mathsAddRow()">Add row</button>
      <button class="secondary" onclick="mathsClear()">Clear all</button>
    </div>
  </div>
  <div class="panel">
    <h3>Averages</h3>
    <div class="mtable-wrap">
    <table class="mtable">
      <thead><tr><th>Mass m (g)</th><th>Readings</th><th>Average n<sub>2</sub></th>
        <th>Average t (s)</th><th>Average I (g&middot;cm&sup2;)</th></tr></thead>
      <tbody id="mAvgBody"></tbody>
    </table>
    </div>
    <div id="mOverall" class="moi-result">Average I = -</div>
  </div>
</div>
</div>

<div class="summary-bar">
  <div>Crop start: <span class="val" id="barStart">0.00 s</span></div>
  <div>Crop end: <span class="val" id="barEnd">0.00 s</span></div>
  <div>Length: <span class="val" id="barLen">0.00 s</span></div>
</div>

<script>
const S = {sid:null, fps:30, totalFrames:0, origW:0, origH:0, duration:0,
  currentFrame:0, history:[], laps:[], analyzing:false, cropStart:0, cropEnd:0,
  startSel:0, endSel:0, calib:null};
const STRIP_BEFORE = 3.0, STRIP_AFTER = 3.0, STRIP_N = 20;

const $ = id => document.getElementById(id);
const postJSON = async (url, body) => (await fetch(url, {
  method: 'POST', headers: {'Content-Type': 'application/json'},
  body: JSON.stringify(body)})).json();

function showTab(name) {
  [['analysis','Analysis'], ['crop','Crop'], ['fast','Fast'], ['maths','Maths']].forEach(([n, cap]) => {
    $('page' + cap).classList.toggle('active', name === n);
    $('tab' + cap).classList.toggle('active', name === n);
  });
  if (name === 'crop' && S.sid && !$('startStrip').dataset.loaded) {
    loadStartWindow(); loadEndWindow();
  }
  if (name === 'fast') updateFastInfo();
}

function updateFastInfo() {
  $('fastRoiInfo').textContent = !S.sid ? 'No video loaded (use the Analysis tab).'
    : `Video: ${S.origW} x ${S.origH}, ${S.totalFrames} frames. Dark threshold ${(+$('darkNum').value).toFixed(2)}, ` +
      `new-lap jump ${(+$('lapNum').value).toFixed(2)}.`;
}

function getParams() {
  return {sid: S.sid, frame_idx: S.currentFrame,
    dark_thr: +$('darkNum').value, new_lap_jump: +$('lapNum').value,
    use_result: S.history.length > 0,
    calib: S.calib || null, max_drift_pct: +$('driftPct').value, area_tol_pct: +$('areaTol').value};
}

// ---- flywheel calibration (box in ORIGINAL pixels, taken on frame S.calib.frame) ----
function calShow() {
  const c = S.calib;
  ['X','Y','W','H'].forEach(k => { $('cal' + k).value = c ? c[k.toLowerCase()] : ''; });
  $('calBadge').classList.toggle('active', !!c);
  $('calInfo').textContent = c
    ? `Calibrated on frame ${c.frame}: x ${c.x}, y ${c.y}, ${c.w} x ${c.h} px. Size is locked for the whole analysis.`
    : 'Not calibrated: the box size is taken from the first detections of the video.';
}
async function calAuto() {
  if (!S.sid) return alert('Load a video first.');
  let d;
  try { d = await postJSON('/detect', {sid: S.sid, frame_idx: S.currentFrame}); }
  catch (e) { return alert('Detection failed: ' + e.message); }
  if (d.error) return alert(d.error);
  S.calib = {x: d.box[0], y: d.box[1], w: d.box[2], h: d.box[3], frame: d.frame};
  calShow(); renderFrame();
}
function calEdit() {
  if (!S.sid) return;
  const v = k => Math.round(+$('cal' + k).value);
  const c = {x: v('X'), y: v('Y'), w: v('W'), h: v('H'), frame: S.currentFrame};
  if (!(c.w >= 8 && c.h >= 8)) return alert('Width and height must be at least 8 px.');
  S.calib = c; calShow(); renderFrame();
}
function calNudge(dx, dy, dw, dh) {
  if (!S.calib) return alert('Press Auto-fill (or type a box) first.');
  const c = S.calib;
  S.calib = {x: c.x + dx, y: c.y + dy, w: Math.max(8, c.w + dw), h: Math.max(8, c.h + dh), frame: S.currentFrame};
  calShow(); renderFrame();
}
function calClear() { S.calib = null; calShow(); renderFrame(); }

// ---- status labels (flywheel / tape / height / laps / rotations) ----
function setLabels(d) {
  const w = d.wheel, t = d.tape;
  const lw = $('lblWheel'), lt = $('lblTape');
  lw.textContent = w === 'detected' ? 'DETECTED' : w === 'held' ? 'HELD' : w === 'none' ? 'NOT FOUND' : '-';
  lw.className = 'big mid ' + (w === 'detected' ? 'ok' : w === 'held' ? 'warn' : w === 'none' ? 'bad' : '');
  lt.textContent = t === 'found' ? 'FOUND' : t === 'hidden' ? 'BEHIND WHEEL' : '-';
  lt.className = 'big mid ' + (t === 'found' ? 'ok' : t === 'hidden' ? 'warn' : '');
  $('lblY').textContent = (t === 'found' && d.y != null) ? (100 * d.y).toFixed(0) + ' %' : '-';
  $('lblLaps').textContent = d.laps != null ? d.laps : '-';
  if (d.rotation_count != null) $('lblCount').textContent = d.rotation_count;
}

// ---- Upload ----
async function uploadVideo() {
  const file = $('videoFile').files[0];
  if (!file) return alert('Select a video file first.');
  $('fileLabel').textContent = 'Loading...';
  let data;
  try {
    data = await (await fetch('/upload', {method:'POST', body:file, headers:{
      'Content-Type': 'application/octet-stream',
      'X-Filename': encodeURIComponent(file.name)}})).json();
  } catch (e) { $('fileLabel').textContent = 'Upload failed'; return alert('Upload failed: ' + e.message); }
  if (data.error) { $('fileLabel').textContent = 'No video loaded'; return alert(data.error); }
  Object.assign(S, {sid: data.sid, fps: data.fps,
    totalFrames: data.total_frames, origW: data.orig_w, origH: data.orig_h,
    duration: data.duration || data.total_frames / data.fps,
    currentFrame: 0, history: [], laps: [], calib: null, cropStart: 0, cropEnd: data.duration,
    startSel: 0, endSel: data.duration});
  $('fileLabel').textContent = data.filename;
  $('cropFileName').textContent = data.filename;
  $('cropDuration').textContent = `Duration: ${S.duration.toFixed(2)} s @ ${S.fps.toFixed(2)} fps`;
  const sl = $('frameSlider');
  sl.max = Math.max(0, data.total_frames - 1); sl.value = 0; sl.disabled = false;
  sl.oninput = () => {
    if (S.analyzing) return;
    S.currentFrame = +sl.value;
    $('frameInfo').textContent =
      `Frame: ${S.currentFrame} / ${S.totalFrames}   Time: ${(S.currentFrame / S.fps).toFixed(2)}s`;
    renderFrame();
  };
  resetAnalysis(); calShow();
  updateCropBadge(); updateCropBar(); updateCropInfo(); renderFrame(); updateFastInfo();
  $('startStrip').dataset.loaded = ''; $('endStrip').dataset.loaded = '';
  if ($('pageCrop').classList.contains('active')) { loadStartWindow(); loadEndWindow(); }
}

// ---- Render ----
// Only ONE /render request is in flight at a time. While the slider is dragged, further calls just
// mark the request as dirty and the newest position is rendered when the current one returns, so
// the server never builds a queue of frames nobody will look at.
// After an analysis the frame is drawn from its results (box, tape, dial, rotation counter);
// before it, only the flywheel is detected in the shown frame.
let _rBusy = false, _rDirty = false;
async function renderFrame() {
  if (!S.sid || S.analyzing) return;
  if (_rBusy) { _rDirty = true; return; }
  _rBusy = true;
  try {
    do {
      _rDirty = false;
      const p = getParams();
      let data;
      try { data = await postJSON('/render', p); }
      catch (e) { console.warn('render failed', e); continue; }
      if (S.analyzing) break;
      if (data.error) { console.warn(data.error); continue; }
      $('frameImg').src = data.image;
      $('frameInfo').textContent =
        `Frame: ${p.frame_idx} / ${S.totalFrames}   Time: ${(p.frame_idx / S.fps).toFixed(2)}s`;
      $('lblTime').textContent = (p.frame_idx / S.fps).toFixed(2) + 's';
      setLabels(data);
    } while (_rDirty);
  } finally { _rBusy = false; }
}

// ---- Tape-detection settings ----
function bindPair(numId, rangeId, lo, hi) {
  const num = $(numId), rng = $(rangeId);
  const changed = () => {
    updateFastInfo();
    if (S.history.length) $('statusMsg').textContent = 'Settings changed - run the analysis again to update the result.';
  };
  rng.oninput = () => { num.value = (+rng.value).toFixed(2); changed(); };
  num.onchange = () => {
    const v = Math.max(lo, Math.min(hi, +num.value || lo));
    num.value = v.toFixed(2); rng.value = v; changed();
  };
}
bindPair('darkNum', 'darkRange', 0.30, 0.95);
bindPair('lapNum', 'lapRange', 0.10, 0.90);

// ---- Crop helpers ----
function updateCropBadge() {
  const full = S.cropStart <= 0.001 && Math.abs(S.cropEnd - S.duration) <= 0.001;
  $('cropBadge').classList.toggle('active', $('enableCrop').checked && !full);
}
$('enableCrop').onchange = updateCropBadge;

function updateCropInfo() {
  if (!S.sid) return $('cropInfo').textContent = 'Crop: full video';
  const full = S.cropStart <= 0.001 && Math.abs(S.cropEnd - S.duration) <= 0.001;
  $('cropInfo').textContent = full ? 'Crop: full video'
    : `Crop: ${S.cropStart.toFixed(2)}s → ${S.cropEnd.toFixed(2)}s (${(S.cropEnd - S.cropStart).toFixed(2)}s)`;
}

const _updateCropInfo = updateCropInfo;
updateCropInfo = function () { _updateCropInfo(); $('fastCropInfo').textContent = $('cropInfo').textContent; };

function updateCropBar() {
  $('barStart').textContent = S.cropStart.toFixed(2) + ' s';
  $('barEnd').textContent = S.cropEnd.toFixed(2) + ' s';
  $('barLen').textContent = Math.max(0, S.cropEnd - S.cropStart).toFixed(2) + ' s';
}

// ---- Strips ----
const loadStartWindow = () => S.sid ? fetchStrip('start', +$('startWinCenter').value || 0) : alert('Load a video first.');
const loadEndWindow = () => S.sid ? fetchStrip('end', +$('endWinCenter').value || 0) : alert('Load a video first.');

const stripTok = {start: 0, end: 0};
async function fetchStrip(which, center) {
  const tok = ++stripTok[which], el = $(which + 'Strip'), BATCH = 4;
  el.innerHTML = '<div class="hint" style="padding:20px">Loading…</div>';
  const ask = off => postJSON('/crop_strip', {sid: S.sid, center_time: center,
    before: STRIP_BEFORE, after: STRIP_AFTER, n: STRIP_N, offset: off, count: BATCH});
  const data = await ask(0);
  if (tok !== stripTok[which]) return;                       // a newer Load replaced this one
  if (data.error) return el.innerHTML = `<div class="hint" style="padding:20px">Error: ${data.error}</div>`;
  $(which + 'WinCenter').value = data.center_time.toFixed(2);
  $(which + 'WinInfo').textContent =
    `Window: ${data.win_start.toFixed(2)}s → ${data.win_end.toFixed(2)}s ` +
    `(${(data.win_end - data.win_start).toFixed(2)}s, ${STRIP_N} thumbs)`;
  const sl = $(which + 'Slider');
  sl.min = data.win_start; sl.max = data.win_end; sl.step = 0.01;
  el.innerHTML = ''; el.dataset.loaded = '1';
  // all slots appear at once (with their time labels); the pictures fill in batch by batch
  const cells = data.slots.map(t => {
    const d = document.createElement('div');
    d.className = 'thumb'; d.dataset.time = t.time_s;
    d.innerHTML = `<img style="width:130px;height:${data.thumb_h}px;background:#111">` +
                  `<div class="tlabel">${t.time_s.toFixed(2)}s</div>`;
    d.onclick = () => selectTime(which, t.time_s);
    el.appendChild(d);
    return d;
  });
  const fill = r => r.thumbs.forEach(t => { cells[t.slot].querySelector('img').src = t.image; });
  fill(data);
  const cur = which === 'start' ? S.startSel : S.endSel;
  selectTime(which, (cur < data.win_start || cur > data.win_end) ? data.center_time : cur, false);
  for (let off = BATCH; off < data.slots.length; off += BATCH) {
    const r = await ask(off);
    if (tok !== stripTok[which]) return;
    if (!r.error) fill(r);
  }
}

function selectTime(which, t, skipSlider) {
  t = Math.max(0, Math.min(t, S.duration));
  if (which === 'start') {
    if (t > S.endSel - 0.01) t = Math.max(0, S.endSel - 0.01);
    S.startSel = S.cropStart = t;
    $('startSelNum').value = t.toFixed(2);
    $('startSelInfo').textContent = `Start selected: ${t.toFixed(2)} s`;
    if (!skipSlider) { const s = $('startSlider'); if (t >= +s.min && t <= +s.max) s.value = t; }
  } else {
    if (t < S.startSel + 0.01) t = Math.min(S.duration, S.startSel + 0.01);
    S.endSel = S.cropEnd = t;
    $('endSelNum').value = t.toFixed(2);
    $('endSelInfo').textContent = `End selected: ${t.toFixed(2)} s`;
    if (!skipSlider) { const s = $('endSlider'); if (t >= +s.min && t <= +s.max) s.value = t; }
  }
  const strip = $(which + 'Strip');
  let best = null, bd = Infinity;
  strip.querySelectorAll('.thumb').forEach(el => {
    const d = Math.abs(+el.dataset.time - t);
    if (d < bd) { bd = d; best = el; }
  });
  strip.querySelectorAll('.thumb').forEach(el => el.classList.toggle('selected', el === best));
  updateCropBar(); updateCropInfo(); updateCropBadge();
}

$('startSlider').oninput = e => selectTime('start', +e.target.value, true);
$('endSlider').oninput = e => selectTime('end', +e.target.value, true);
$('startSelNum').onchange = e => selectTime('start', +e.target.value || 0);
$('endSelNum').onchange = e => selectTime('end', +e.target.value || 0);
$('startWinCenter').onchange = loadStartWindow;
$('endWinCenter').onchange = loadEndWindow;

function resetCropToFull() {
  S.startSel = S.cropStart = 0;
  S.endSel = S.cropEnd = S.duration;
  $('startSelNum').value = '0.00'; $('endSelNum').value = S.duration.toFixed(2);
  $('startSelInfo').textContent = 'Start selected: 0.00 s';
  $('endSelInfo').textContent = `End selected: ${S.duration.toFixed(2)} s`;
  updateCropBar(); updateCropInfo(); updateCropBadge();
}

function applyCrop(target) {
  if (!S.sid) return alert('Load a video first.');
  const full = S.cropStart <= 0.001 && Math.abs(S.cropEnd - S.duration) <= 0.001;
  $('enableCrop').checked = !full;
  $('fastCrop').checked = !full;
  updateCropBadge();
  showTab(target);
}
const applyCropToAnalysis = () => applyCrop('analysis');
const applyCropToFast = () => applyCrop('fast');

// ---- Live preview playback ----
// Capped runs stamp every preview with its playback time ('at', seconds since the first frame). The page
// keeps a small jitter buffer and shows each frame when its time comes, so playback stays even even if
// the phone delivers the frames in bursts. Uncapped runs (sparse previews) are shown immediately.
const Live = {q: [], base: null, raf: 0};
function liveShow(q) {
  $('frameImg').src = q.image;
  setLabels({wheel: q.wheel, tape: q.tape, y: q.y, laps: q.laps, rotation_count: q.rotation_count});
  $('lblTime').textContent = q.time_s.toFixed(2) + 's';
  $('frameInfo').textContent =
    `Frame: ${q.frame} / ${q.total_frames + (q.start_frame || 0)}   Time: ${q.time_s.toFixed(2)}s`;
  $('frameSlider').value = q.frame; S.currentFrame = q.frame;
  const frac = q.total_frames > 0 ? (q.frame - (q.start_frame || 0)) / q.total_frames : 0;
  $('progress').value = Math.min(95, frac * 95);
}
function livePush(q) {
  if (q.at == null) return liveShow(q);
  if (Live.base === null) Live.base = performance.now() + 250;      // 250 ms jitter buffer
  Live.q.push(q);
  if (!Live.raf) Live.raf = requestAnimationFrame(liveTick);
}
function liveTick() {
  Live.raf = 0;
  const t = performance.now() - Live.base;
  let last = null;
  while (Live.q.length && Live.q[0].at * 1000 <= t) last = Live.q.shift();
  if (last) liveShow(last);
  if (S.analyzing && Live.base !== null) Live.raf = requestAnimationFrame(liveTick);
}
function liveReset() {
  Live.q = []; Live.base = null;
  if (Live.raf) { cancelAnimationFrame(Live.raf); Live.raf = 0; }
}
function syncLiveControls() {
  const live = $('livePreview').checked, cap = +$('fpsCap').value;
  $('fpsCap').disabled = !live;
  $('previewEvery').disabled = !live || cap > 0;
}
$('livePreview').onchange = $('fpsCap').onchange = syncLiveControls;
syncLiveControls();

// ---- Results ----
function showResult(ev, prefix) {
  S.history = ev.history || [];
  S.laps = ev.laps || [];
  $('summary').textContent = ev.summary;
  $('lblCount').textContent = ev.rotation_count;
  $('lblLaps').textContent = S.laps.length;
  const body = $('lapsBody');
  body.innerHTML = S.laps.map(l =>
    `<tr><td>${l.lap}</td><td>${l.frame}</td><td>${l.time_s.toFixed(3)}</td>` +
    `<td>${l.period_s == null ? '-' : l.period_s.toFixed(3)}</td></tr>`).join('');
  $('lapsWrap').style.display = S.laps.length ? 'block' : 'none';
  if (ev.plot) { $('plot').src = ev.plot; $('plot').style.display = 'block'; }
  if (prefix === 'fast') {
    $('fastSummary').textContent = ev.summary;
    if (ev.plot) { $('fastPlot').src = ev.plot; $('fastPlot').style.display = 'block'; }
  }
  $('statusMsg').textContent = `Done: ${ev.rotation_count} rotations (${ev.rotations_seen} from tape sightings). Drag the slider to review any frame.`;
}

async function readStream(resp, onEvent) {
  const reader = resp.body.getReader(), dec = new TextDecoder();
  let buf = '';
  while (true) {
    const {value, done} = await reader.read();
    if (done) break;
    buf += dec.decode(value, {stream: true});
    let i;
    while ((i = buf.indexOf('\n\n')) !== -1) {
      const raw = buf.slice(0, i).trim(); buf = buf.slice(i + 2);
      if (!raw.startsWith('data:')) continue;
      let ev; try { ev = JSON.parse(raw.slice(5).trim()); } catch { continue; }
      if (onEvent(ev) === false) return false;
    }
  }
  return true;
}

async function runAnalysis() {
  if (!S.sid) return alert('Load a video first.');
  const cropOn = $('enableCrop').checked;
  if (cropOn && S.cropEnd <= S.cropStart) return alert('End time must be greater than start time.');
  const btn = $('btnRun'), badge = $('liveBadge'), live = $('livePreview').checked;
  S.analyzing = true; btn.disabled = true; btn.textContent = 'Analyzing...';
  $('progress').value = 2;
  if (live) badge.classList.add('active');
  S.history = []; S.laps = [];
  $('lblCount').textContent = '0';

  const p = getParams();
  p.live_preview = live; p.preview_every = +$('previewEvery').value || 3;
  p.fps_cap = live ? +$('fpsCap').value : 0;
  liveReset();
  if (cropOn) { p.time_start = S.cropStart; p.time_end = S.cropEnd; }

  try {
    const resp = await fetch('/analyze', {method: 'POST',
      headers: {'Content-Type': 'application/json'}, body: JSON.stringify(p)});
    if (!resp.ok || !resp.body) {
      let m = 'Analysis failed';
      try { m = (await resp.json()).error || m; } catch {}
      return alert(m);
    }
    await readStream(resp, ev => {
      if (ev.error) { alert(ev.error); return false; }
      if (ev.progress !== undefined) $('progress').value = Math.max(2, ev.progress * 95);
      if (ev.preview) livePush(ev.preview);
      if (ev.done) { liveReset(); showResult(ev, 'main'); $('progress').value = 100; }
    });
  } catch (e) { alert('Analysis error: ' + e.message); }
  finally {
    S.analyzing = false; btn.disabled = false; btn.textContent = 'Run Full Analysis';
    liveReset();
    badge.classList.remove('active');
    setTimeout(() => $('progress').value = 0, 1200);
    renderFrame();
  }
}

async function runFast() {
  if (!S.sid) return alert('Load a video on the Analysis tab first.');
  const cropOn = $('fastCrop').checked;
  if (cropOn && S.cropEnd <= S.cropStart) return alert('End time must be greater than start time.');
  const btn = $('btnFast'), badge = $('fastBadge');
  S.analyzing = true; btn.disabled = true; btn.textContent = 'Analyzing...';
  badge.classList.add('active'); $('fastProgress').value = 1;
  $('fastStatus').textContent = 'Starting...';
  S.history = []; S.laps = [];
  const p = getParams();
  p.parts = Math.max(1, Math.min(8, +$('fastParts').value || 2));
  if (cropOn) { p.time_start = S.cropStart; p.time_end = S.cropEnd; }
  const t0 = performance.now();
  try {
    const resp = await fetch('/fast_analyze', {method: 'POST',
      headers: {'Content-Type': 'application/json'}, body: JSON.stringify(p)});
    if (!resp.ok || !resp.body) {
      let m = 'Analysis failed';
      try { m = (await resp.json()).error || m; } catch {}
      return alert(m);
    }
    await readStream(resp, ev => {
      if (ev.error) { alert(ev.error); return false; }
      if (ev.progress !== undefined) {
        $('fastProgress').value = ev.progress * 100;
        $('fastStatus').textContent =
          `Processed ${ev.done_frames} / ${ev.total} frames (${ev.parts} parts)`;
      }
      if (ev.done) {
        showResult(ev, 'fast');
        $('fastProgress').value = 100;
        $('fastStatus').textContent =
          `Finished in ${((performance.now() - t0) / 1000).toFixed(1)} s`;
      }
    });
  } catch (e) { alert('Analysis error: ' + e.message); }
  finally {
    S.analyzing = false; btn.disabled = false; btn.textContent = 'Run Parallel Analysis';
    badge.classList.remove('active');
    renderFrame();
  }
}

function resetAnalysis() {
  S.history = []; S.laps = [];
  $('summary').textContent = 'Run an analysis to see results here.';
  $('lblCount').textContent = '0'; $('lblWheel').textContent = '-'; $('lblWheel').className = 'big mid';
  $('lblTape').textContent = '-'; $('lblTape').className = 'big mid';
  $('lblY').textContent = '-'; $('lblLaps').textContent = '-';
  $('lblTime').textContent = '0.00s'; $('plot').style.display = 'none';
  $('lapsWrap').style.display = 'none'; $('lapsBody').innerHTML = '';
  $('progress').value = 0;
  $('fastSummary').textContent = 'Run an analysis to see results here.';
  $('fastPlot').style.display = 'none'; $('fastProgress').value = 0;
  $('fastStatus').textContent = '';
  if (S.sid) renderFrame();
}

// ---- CSV ----
function exportCSV() {
  if (!S.history.length) return alert('Run an analysis first.');
  const cols = ['frame','time_s','flywheel','tape','tape_y','phase_deg','rotation_count'];
  const lines = [cols.join(','), ...S.history.map(h => cols.map(c => h[c] ?? '').join(','))];
  const url = URL.createObjectURL(new Blob([lines.join('\n')], {type: 'text/csv'}));
  const a = document.createElement('a'); a.href = url; a.download = 'rotation_data.csv';
  a.click(); URL.revokeObjectURL(url);
}

// ---- Maths page ----
function mathsMomentOfInertia(m, n1, n2, t, r, g) {
  const v = r * g * t * t * m / (4 * Math.PI * n2 * (1 + n2 / n1));
  return Number.isFinite(v) ? v : null;
}
const mFmt = v => v === null ? '-' : (Math.abs(v) >= 1e6 || (v !== 0 && Math.abs(v) < 1e-2))
  ? v.toExponential(4) : v.toFixed(2);
const mNum = el => el.value === '' ? NaN : +el.value;

function mathsAddRow() {
  const tr = document.createElement('tr');
  tr.innerHTML = '<td><input type="number" step="any"></td>'.repeat(4) +
    '<td class="res">-</td><td><button class="secondary" title="Remove row">&times;</button></td>';
  tr.querySelectorAll('input').forEach(i => i.oninput = mathsRecalc);
  tr.querySelector('button').onclick = () => { tr.remove(); mathsRecalc(); };
  $('mBody').appendChild(tr);
}

function mathsClear() {
  $('mBody').innerHTML = '';
  for (let i = 0; i < 3; i++) mathsAddRow();
  mathsRecalc();
}

function mathsRecalc() {
  const r = mNum($('mR')), g = mNum($('mG'));
  const groups = new Map(), all = [];
  $('mBody').querySelectorAll('tr').forEach(tr => {
    const [m, n1, n2, t] = [...tr.querySelectorAll('input')].map(mNum);
    const ok = [m, n1, n2, t, r, g].every(v => Number.isFinite(v)) && n1 !== 0 && n2 !== 0;
    const I = ok ? mathsMomentOfInertia(m, n1, n2, t, r, g) : null;
    tr.querySelector('.res').textContent = mFmt(I);
    if (I === null) return;
    all.push(I);
    if (!groups.has(m)) groups.set(m, {n2: [], t: [], I: []});
    const grp = groups.get(m); grp.n2.push(n2); grp.t.push(t); grp.I.push(I);
  });
  const avg = a => a.reduce((x, y) => x + y, 0) / a.length;
  const rows = [...groups.entries()].sort((a, b) => a[0] - b[0]).map(([m, grp]) =>
    `<tr><td>${m}</td><td>${grp.I.length}</td><td>${avg(grp.n2).toFixed(2)}</td>` +
    `<td>${avg(grp.t).toFixed(3)}</td><td class="res">${mFmt(avg(grp.I))}</td></tr>`);
  $('mAvgBody').innerHTML = rows.join('') || '<tr><td colspan="5">-</td></tr>';
  $('mOverall').textContent = all.length
    ? `Average I = ${mFmt(avg(all))} g\u00b7cm\u00b2  (${all.length} reading${all.length > 1 ? 's' : ''})`
    : 'Average I = -';
}
$('mR').oninput = $('mG').oninput = mathsRecalc;
mathsClear();

updateCropBar();

updateCropBar();
</script>
</body>
</html>
"""

if __name__ == '__main__':
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
