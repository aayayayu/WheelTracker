"""
EP LAB - Flywheel — Flask Web Edition
"""
import os, io, re, csv, math, base64, uuid, shutil, collections, json as _json, tempfile, threading, time
from urllib.parse import unquote
import numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, Response

app = Flask(__name__)
app.secret_key = 'rw-tracker-secret-key'
SESSIONS, DISPLAY_MAX_W = {}, 900


# ---------- CV ----------
def process_frame(frame, p):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, p['black_thresh'], 255, cv2.THRESH_BINARY_INV)
    yt, yb, xl, xr = p['y_top'], p['y_bottom'], p['x_left'], p['x_right']
    res = {'found': False, 'mask': mask, 'y': None, 'state': 'hidden', 'area': 0, 'rect': None}
    if None in (yt, yb, xl, xr): return res
    y0, y1, x0, x1 = min(yt, yb), max(yt, yb), min(xl, xr), max(xl, xr)
    roi = np.zeros_like(mask)
    roi[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    mask = roi
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    res['mask'] = mask
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < p['min_area'] or (p['max_area'] > 0 and a > p['max_area']): continue
        x, y, w, h = cv2.boundingRect(c)
        cands.append({'area': a, 'cy': y + h / 2.0, 'rect': (x, y, w, h)})
    if not cands: return res
    ch = max(cands, key=lambda c: c['area'])
    span = (yb - yt) or 1
    ny = (ch['cy'] - yt) / span
    state = 'top' if ny < 0.33 else ('mid' if ny < 0.66 else 'bottom')
    res.update(found=True, y=ch['cy'], state=state, area=ch['area'], rect=ch['rect'])
    return res


def angle_from_y(y, p):
    yt, yb = p['y_top'], p['y_bottom']
    yc, r = (yt + yb) / 2.0, abs(yb - yt) / 2.0
    if r == 0: return 0.0
    v = (yc - y) / r if p['direction'] == 'down' else (y - yc) / r
    return math.degrees(math.acos(max(-1.0, min(1.0, v))))


def fit_display(frame, max_w=DISPLAY_MAX_W):
    """Return (picture no wider than max_w, scale vs. the original). Always a NEW array,
    because decoded frames are shared with the frame cache and must never be drawn on."""
    h, w = frame.shape[:2]
    if w > max_w:
        sc = max_w / float(w)
        return cv2.resize(frame, (max_w, max(1, int(h * sc))), interpolation=cv2.INTER_AREA), sc
    return frame.copy(), 1.0


def draw_overlay(out, res, p, s):
    """Draw boundary lines / detection box / state text IN PLACE on `out`, which is `s` times
    the size of the original frame (boundaries and detections are in original coordinates).
    Drawing after down-scaling is ~5x cheaper than drawing on the full frame."""
    H, W = out.shape[:2]
    th = max(1, int(round(2 * s)))
    sc = lambda v: int(round(v * s))
    for y in (p.get('y_top'), p.get('y_bottom')):
        if y is not None: cv2.line(out, (0, sc(y)), (W, sc(y)), (255, 0, 0), th)
    for x in (p.get('x_left'), p.get('x_right')):
        if x is not None: cv2.line(out, (sc(x), 0), (sc(x), H), (0, 255, 0), th)
    if res['found']:
        x, y, w, h = res['rect']
        cv2.rectangle(out, (sc(x), sc(y)), (sc(x + w), sc(y + h)), (0, 0, 255), th)
        txt, col = f"STATE: {res['state'].upper()} | Y: {int(res['y'])}", (0, 255, 255)
    else:
        txt, col = "Tape HIDDEN (back side)", (0, 0, 255)
    fs = max(0.5, 0.9 * s)
    cv2.putText(out, txt, (8, int(10 + 22 * fs)), cv2.FONT_HERSHEY_SIMPLEX, fs, col, max(1, th))
    return out


def encode_jpeg(bgr, max_w=DISPLAY_MAX_W, quality=80):
    h, w = bgr.shape[:2]
    if w > max_w:
        bgr = cv2.resize(bgr, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    return 'data:image/jpeg;base64,' + base64.b64encode(
        cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]).decode()


def extract_params(d):
    return {k: d.get(k) for k in ('y_top', 'y_bottom', 'x_left', 'x_right')} | {
        'black_thresh': int(d.get('black_thresh', 60)),
        'min_area': int(d.get('min_area', 100)),
        'max_area': int(d.get('max_area', 0)),
        'debounce': int(d.get('debounce', 2)),
        'direction': d.get('direction', 'down'),
        'strict': bool(d.get('strict', False)),
    }


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
    Android) plus a tiny LRU of decoded frames, so re-rendering the same frame after a threshold /
    boundary change costs no decoding at all. Cached frames are shared: never modify them."""

    def __init__(self, path, keep=2):
        self.path, self.keep = path, keep
        self.lock = threading.Lock()
        self.cap, self.cache = None, collections.OrderedDict()

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

    def close(self):
        with self.lock:
            self.cache.clear()
            if self.cap is not None:
                try: self.cap.release()
                except Exception: pass
                self.cap = None


_K5 = np.ones((5, 5), np.uint8)
_PAD = 6   # >= reach of the open(1x)+close(2x) 5x5 morphology


def _empty_res():
    return {'found': False, 'mask': None, 'y': None, 'state': 'hidden', 'area': 0, 'rect': None}


def roi_rect(p, W, H):
    """Tracking box (x0, y0, x1, y1) clamped to the frame, or None if not usable."""
    yt, yb, xl, xr = p['y_top'], p['y_bottom'], p['x_left'], p['x_right']
    if None in (yt, yb, xl, xr): return None
    y0, y1 = max(0, min(yt, yb)), min(H, max(yt, yb))
    x0, x1 = max(0, min(xl, xr)), min(W, max(xl, xr))
    if y1 <= y0 or x1 <= x0: return None
    return x0, y0, x1, y1


def detect_gray(gray, x0, y0, p):
    """Detect the tape in an already-cropped grayscale ROI whose top-left corner is
    (x0, y0) in the full frame. Returned y / rect are in full-frame coordinates."""
    res = _empty_res()
    yt, yb = p['y_top'], p['y_bottom']
    _, m = cv2.threshold(gray, p['black_thresh'], 255, cv2.THRESH_BINARY_INV)
    m = cv2.copyMakeBorder(m, _PAD, _PAD, _PAD, _PAD, cv2.BORDER_CONSTANT, value=0)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _K5, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _K5, iterations=2)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ox, oy = x0 - _PAD, y0 - _PAD
    cands = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < p['min_area'] or (p['max_area'] > 0 and a > p['max_area']): continue
        x, y, w, h = cv2.boundingRect(c)
        cands.append({'area': a, 'cy': y + h / 2.0 + oy, 'rect': (x + ox, y + oy, w, h)})
    if not cands: return res
    ch = max(cands, key=lambda c: c['area'])
    span = (yb - yt) or 1
    ny = (ch['cy'] - yt) / span
    state = 'top' if ny < 0.33 else ('mid' if ny < 0.66 else 'bottom')
    res.update(found=True, y=ch['cy'], state=state, area=ch['area'], rect=ch['rect'])
    return res


# ---------- Analysis stream ----------
class RotationCounter:
    """Sequential state machine (debounce + top/mid/bottom/hidden sequence).
    Cheap, so it always runs once, in order, over per-frame detections."""

    def __init__(self, p, fps):
        self.p, self.fps = p, fps
        self.seq = ['top', 'mid', 'bottom', 'hidden'] if p['direction'] == 'down' \
                   else ['bottom', 'mid', 'top', 'hidden']
        self.exp, self.cand, self.ccnt, self.stable = 0, None, 0, None
        self.rot, self.rtimes, self.cum = 0, [], 0.0
        self.prev, self.max_jump, self.hist = None, 0.0, []

    def step(self, fi, res):
        p, seq = self.p, self.seq
        t = fi / self.fps if self.fps > 0 else 0.0
        st = res['state']
        if st == self.cand: self.ccnt += 1
        else: self.cand, self.ccnt = st, 1
        if self.ccnt >= p['debounce'] and st != self.stable:
            first = self.stable is None
            self.stable = st
            if first:
                if st in seq: self.exp = (seq.index(st) + 1) % 4
            else:
                if st == seq[self.exp]:
                    self.exp = (self.exp + 1) % 4
                    if self.exp == 0: self.rot += 1; self.rtimes.append(t)
                elif not p['strict'] and st in seq:
                    self.exp = (seq.index(st) + 1) % 4
                    if self.exp == 0: self.rot += 1; self.rtimes.append(t)
        th = None
        if res['found']:
            th = angle_from_y(res['y'], p)
            self.cum = self.rot * 360.0 + th
            if self.prev is not None: self.max_jump = max(self.max_jump, abs(self.cum - self.prev))
            self.prev = self.cum
        self.hist.append({'frame': fi, 'time_s': round(t, 5),
                          'y_position': res['y'] if res['found'] else None,
                          'unwrapped_angle_deg': self.cum if res['found'] else None,
                          'state': st, 'stable_state': self.stable, 'rotation_count': self.rot})
        return th


def finish_event(ctr, proc, sf, ef, total, fps, extra=()):
    rot, rtimes, hist = ctr.rot, ctr.rtimes, ctr.hist
    total_time = proc / fps if fps > 0 else 0.0
    L = [f"Frames processed: {proc}", f"Video duration processed: {total_time:.2f} s"]
    L += list(extra)
    L.append(f"Total full rotations counted: {rot}")
    if rot and rtimes:
        off = sf / fps if fps > 0 else 0.0
        per = (rtimes[-1] - off) / rot
        w = 2 * math.pi / per
        L += [f"Average period per rotation: {per:.4f} s",
              f"Average angular velocity: {w:.4f} rad/s ({w * 60 / (2 * math.pi):.2f} RPM)"]
    else:
        L.append("No complete rotations were counted.")
    summary = "\n".join(L)

    times = [h['time_s'] for h in hist if h['unwrapped_angle_deg'] is not None]
    angs = [h['unwrapped_angle_deg'] for h in hist if h['unwrapped_angle_deg'] is not None]
    all_t = [h['time_s'] for h in hist]
    cnts = [h['rotation_count'] for h in hist]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6))
    if times: a1.plot(times, angs, '.', ms=2, color='steelblue')
    a1.set(xlabel="time (s)", ylabel="cumulative angle (deg)",
           title="Unwrapped marker angle vs time"); a1.grid(alpha=0.3)
    if all_t: a2.step(all_t, cnts, where='post', color='darkorange')
    a2.set(xlabel="time (s)", ylabel="rotations",
           title="Cumulative rotation count vs time"); a2.grid(alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=90, bbox_inches='tight')
    plt.close(fig); buf.seek(0)
    plot = 'data:image/png;base64,' + base64.b64encode(buf.read()).decode()

    return {'done': True, 'summary': summary, 'rotation_count': rot,
            'history': hist, 'plot': plot, 'start_frame': sf, 'end_frame': ef}


def _frame_range(sess, t0, t1):
    fps, total = sess['fps'], sess['total_frames']
    sf = max(0, int(t0 * fps)) if t0 is not None else 0
    ef = min(total, int(t1 * fps)) if t1 is not None else total
    return fps, total, sf, ef


def run_analysis_stream(sess, p, live=False, every=3, t0=None, t1=None):
    """Sequential analysis with an optional live preview. It uses the SAME fast reader as the
    parallel mode (RANGE_READER: only the tracking box is decoded); the preview picture comes
    from the reader itself, so it costs no extra decoding."""
    fps, total, sf, ef = _frame_range(sess, t0, t1)
    rect = roi_rect(p, sess['orig_w'], sess['orig_h'])
    if rect is None:
        yield {'error': 'Tracking box is empty or outside the frame.'}; return
    n_total = ef - sf
    if n_total <= 0:
        yield {'error': 'Nothing to analyse (empty time range).'}; return
    ctr = RotationCounter(p, fps)
    pe = max(1, int(every)) if live else 0
    st = {'proc': 0, 'next': sf}
    notes = []
    t_start = time.time()

    def consume(reader):
        for i, g, pv in reader:
            res = detect_gray(g, rect[0], rect[1], p)
            th = ctr.step(i, res)
            st['proc'] += 1; st['next'] = i + 1
            if pv is not None:
                img, sc = pv
                out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
                draw_overlay(out, res, p, sc)
                yield {'preview': {'frame': i, 'time_s': round(i / fps if fps > 0 else 0.0, 5),
                                   'image': encode_jpeg(out, quality=75), 'state': res['state'],
                                   'found': res['found'], 'y': res['y'] if res['found'] else None, 'rotation_count': ctr.rot,
                                   'total_frames': n_total, 'start_frame': sf}}

    try:
        yield from consume(RANGE_READER(sess, sf, ef, rect, pe))
    except Exception as e:
        if RANGE_READER is cv_range_reader: raise
        notes.append(f"Fast decoder failed ({e}); finished with the slow fallback")
        yield from consume(cv_range_reader(sess, st['next'], ef, rect, pe))
    proc = st['proc']
    if not proc:
        yield {'error': 'No frames could be decoded.'}; return
    dt = max(1e-6, time.time() - t_start)
    extra = ["Parallel parts: 1", f"Processing time: {dt:.2f} s ({proc / dt:.0f} frames/s)"]
    if proc < n_total: extra.append(f"Warning: decoded {proc} of {n_total} frames")
    extra += notes
    print(f"ANALYSIS: {proc} frames in {dt:.1f}s ({proc / dt:.0f} fps)")
    yield finish_event(ctr, proc, sf, ef, total, fps, extra)


# ---- Fast / parallel analysis: read only the ROI, split the video into parts ----
def cv_range_reader(sess, sf, ef, rect, preview_every=0):
    """Generic reader: yields (frame_index, gray ROI, preview) where preview is None or
    (picture, scale). Android replaces RANGE_READER with a hardware-decoder version (main.py)."""
    x0, y0, x1, y1 = rect
    cap = cv2.VideoCapture(sess['video_path'])
    try:
        if sf: cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
        last_pv = 0.0
        for i in range(sf, ef):
            ok, fr = cap.read()
            if not ok: break
            pv = None
            if preview_every and (i - sf) % preview_every == 0 and time.time() - last_pv >= 0.3:
                last_pv = time.time()
                pv = fit_display(fr, 640)
            yield i, cv2.cvtColor(fr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), pv
    finally:
        cap.release()


RANGE_READER = cv_range_reader


def run_parallel_stream(sess, p, parts=2, t0=None, t1=None):
    """Split [start, end) into `parts` chunks, detect the tape in each chunk on its own
    thread (per-frame detection is independent), then run the rotation state machine ONCE,
    in order, over the merged detections. Result is identical to a sequential run, and
    there is nothing to reconcile at chunk boundaries."""
    fps, total, sf, ef = _frame_range(sess, t0, t1)
    rect = roi_rect(p, sess['orig_w'], sess['orig_h'])
    if rect is None:
        yield {'error': 'Tracking box is empty or outside the frame.'}; return
    n_frames = ef - sf
    if n_frames <= 0:
        yield {'error': 'Nothing to analyse (empty time range).'}; return
    n = max(1, min(int(parts), 8, n_frames))
    edges = [sf + n_frames * k // n for k in range(n + 1)]
    results = [[] for _ in range(n)]
    counts, errs, notes = [0] * n, [None] * n, []
    stop = threading.Event()

    def consume(k, reader):
        out = results[k]
        for i, g, _pv in reader:
            if stop.is_set(): return
            r = detect_gray(g, rect[0], rect[1], p)
            out.append((i, r['found'], r['y'], r['state']))
            counts[k] = len(out)

    def worker(k):
        a, b = edges[k], edges[k + 1]
        try:
            consume(k, RANGE_READER(sess, a, b, rect))
        except Exception as e:
            if RANGE_READER is cv_range_reader:
                errs[k] = str(e); return
            notes.append(f"Part {k + 1}: fast decoder failed ({e}); used slow fallback")
            results[k].clear(); counts[k] = 0
            try:
                consume(k, cv_range_reader(sess, a, b, rect))
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
    dets = [x for r in results for x in r]
    if not dets:
        yield {'error': 'No frames could be decoded.'}; return
    wall = max(1e-6, time.time() - t_start)
    ctr = RotationCounter(p, fps)
    for i, found, y, state in dets:
        ctr.step(i, {'found': found, 'y': y, 'state': state})
    extra = [f"Parallel parts: {n}",
             f"Processing time: {wall:.2f} s ({len(dets) / wall:.0f} frames/s)"]
    if len(dets) < n_frames:
        extra.append(f"Warning: decoded {len(dets)} of {n_frames} frames")
    extra += notes
    print(f"PARALLEL ANALYSIS: {len(dets)} frames, {n} parts, {wall:.1f}s ({len(dets) / wall:.0f} fps)")
    yield finish_event(ctr, len(dets), sf, ef, total, fps, extra)


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
    for old in list(SESSIONS): drop_session(old)        # keep only the video that is loaded now
    SESSIONS[sid] = {'video_path': tmp, 'fps': fps, 'total_frames': total, 'src': src,
                     'orig_w': w, 'orig_h': h, 'filename': name}
    return jsonify({'sid': sid, 'fps': fps, 'total_frames': total,
                    'orig_w': w, 'orig_h': h, 'filename': name,
                    'duration': total / fps if fps else 0})


@app.route('/render', methods=['POST'])
def render():
    t0 = time.time()
    d = request.get_json(force=True)
    sess = SESSIONS.get(d.get('sid'))
    if not sess: return jsonify({'error': 'invalid session'}), 400
    idx = max(0, min(int(d.get('frame_idx', 0)), sess['total_frames'] - 1))
    frame = sess['src'].get(idx)
    if frame is None: return jsonify({'error': 'cannot read frame'}), 400
    t1 = time.time()
    p = extract_params(d)
    if d.get('show_mask'):
        res = process_frame(frame, p)                       # full-frame mask, only when asked for
        disp = cv2.cvtColor(res['mask'], cv2.COLOR_GRAY2BGR)
        disp, sc = fit_display(disp)
    else:
        rect = roi_rect(p, frame.shape[1], frame.shape[0])
        if rect is None: res = _empty_res()
        else:
            x0, y0, x1, y1 = rect                            # detect inside the tracking box only
            res = detect_gray(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), x0, y0, p)
        disp, sc = fit_display(frame)
        draw_overlay(disp, res, p, sc)
    out = {'image': encode_jpeg(disp), 'state': res['state'], 'found': res['found'],
           'y': res['y'] if res['found'] else None}
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
    thumbs = []
    for i in idxs:
        fr = sess['src'].get(i, cache=False)
        if fr is None: continue
        h, w = fr.shape[:2]
        th = cv2.resize(fr, (130, int(h * 130 / w)), interpolation=cv2.INTER_AREA)
        thumbs.append({'time_s': round(i / fps if fps else 0, 3), 'frame': i,
                       'image': encode_jpeg(th, max_w=130)})
    return jsonify({'thumbs': thumbs, 'win_start': round(ws, 3),
                    'win_end': round(we, 3), 'center_time': round(c, 3),
                    'duration': round(dur, 3), 'fps': fps})


@app.route('/analyze', methods=['POST'])
def analyze():
    d = request.get_json(force=True)
    sess = SESSIONS.get(d.get('sid'))
    if not sess: return jsonify({'error': 'invalid session'}), 400
    p = extract_params(d)
    if None in (p['y_top'], p['y_bottom'], p['x_left'], p['x_right']):
        return jsonify({'error': 'Set TOP, BOTTOM, LEFT, RIGHT boundaries first.'}), 400
    live, every = bool(d.get('live_preview')), max(1, int(d.get('preview_every', 3)))
    t0 = d.get('time_start'); t1 = d.get('time_end')
    t0 = max(0.0, float(t0)) if t0 is not None else None
    t1 = max(0.0, float(t1)) if t1 is not None else None
    if t0 is not None and t1 is not None and t1 <= t0:
        return jsonify({'error': 'End time must be greater than start time.'}), 400

    def gen():
        try:
            # Without live preview no colour frames are needed: use the fast ROI-only path.
            it = run_analysis_stream(sess, p, live, every, t0, t1) if live \
                 else run_parallel_stream(sess, p, 1, t0, t1)
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
    if None in (p['y_top'], p['y_bottom'], p['x_left'], p['x_right']):
        return jsonify({'error': 'Set TOP, BOTTOM, LEFT, RIGHT boundaries first.'}), 400
    parts = max(1, min(int(d.get('parts', 2) or 2), 8))
    t0 = d.get('time_start'); t1 = d.get('time_end')
    t0 = max(0.0, float(t0)) if t0 is not None else None
    t1 = max(0.0, float(t1)) if t1 is not None else None
    if t0 is not None and t1 is not None and t1 <= t0:
        return jsonify({'error': 'End time must be greater than start time.'}), 400

    def gen():
        try:
            for ev in run_parallel_stream(sess, p, parts, t0, t1):
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
      <div class="video-wrap"><img id="frameImg" alt="Click to set ROI"></div>
      <input type="range" id="frameSlider" min="0" max="0" value="0" disabled>
      <div id="frameInfo" class="status">Frame: 0 / 0 &nbsp; Time: 0.00s</div>
      <div class="roi-grid">
        <button id="btnTop" onclick="setPickMode('top')">1) Set Wheel TOP</button>
        <button id="btnBottom" onclick="setPickMode('bottom')">2) Set Wheel BOTTOM</button>
        <button id="btnLeft" onclick="setPickMode('left')">3) Set Wheel LEFT edge</button>
        <button id="btnRight" onclick="setPickMode('right')">4) Set Wheel RIGHT edge</button>
      </div>
      <div id="statusMsg" class="status">Set the Top, Bottom, Left, and Right boundaries of the wheel to create a tracking box.</div>
      <div class="stats">
        <div>State<span id="lblState" class="big">-</span></div>
        <div>Y<span id="lblY" class="big">-</span></div>
        <div>Expecting<span id="lblExpect" class="big">top</span></div>
        <div>Elapsed<span id="lblTime" class="big">0.00s</span></div>
        <div style="grid-column:span 2">Rotations<span id="lblCount" class="big">0</span></div>
      </div>
    </div>
  </div>

  <div>
    <div class="panel">
      <h3>Grayscale Threshold</h3>
      <div class="row"><label>Black Limit</label>
        <input type="number" id="threshNum" value="60" min="0" max="255" style="width:70px"></div>
      <input type="range" id="threshRange" min="0" max="255" value="60">
      <div class="row" style="margin-top:8px"><input type="checkbox" id="showMask">
        <label for="showMask">Show binary mask</label></div>
    </div>

    <div class="panel">
      <h3>Detection Parameters</h3>
      <div class="row"><label style="flex:1">Min blob area (px)</label>
        <input type="number" id="minArea" value="100" style="width:90px"></div>
      <div class="row"><label style="flex:1">Max blob area (0 = unlimited)</label>
        <input type="number" id="maxArea" value="0" style="width:90px"></div>
      <div class="row"><label style="flex:1">Debounce frames</label>
        <input type="number" id="debounce" value="2" min="1" max="30" style="width:70px"></div>
      <div class="row"><label style="flex:1">Tape motion</label>
        <select id="direction" style="flex:1">
          <option value="down">Top &rarr; Mid &rarr; Bottom &rarr; Hidden</option>
          <option value="up">Bottom &rarr; Mid &rarr; Top &rarr; Hidden</option>
        </select></div>
      <div class="row"><input type="checkbox" id="strict">
        <label for="strict">Strict sequence mode</label></div>
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
      <div class="row"><label style="flex:1">Preview every N frames</label>
        <input type="number" id="previewEvery" value="3" min="1" max="60" style="width:70px"></div>
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
    <div class="hint" style="font-size:12px;color:#888;margin-bottom:8px">
      Uses the video, tracking box and detection settings from the Analysis tab. The video is
      split into parts that are decoded and analysed at the same time; rotations are then counted
      once, in order, so the result is the same as a normal run.</div>
    <div id="fastRoiInfo" class="status">Tracking box: not set</div>
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
  currentFrame:0, pickMode:null, y_top:null, y_bottom:null, x_left:null,
  x_right:null, history:[], analyzing:false, cropStart:0, cropEnd:0,
  startSel:0, endSel:0};
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
  if (name === 'fast') updateFastRoi();
}

function updateFastRoi() {
  const ok = [S.y_top, S.y_bottom, S.x_left, S.x_right].every(v => v !== null);
  $('fastRoiInfo').textContent = !S.sid ? 'No video loaded (use the Analysis tab).'
    : ok ? `Tracking box: x ${Math.min(S.x_left, S.x_right)}-${Math.max(S.x_left, S.x_right)}, ` +
           `y ${Math.min(S.y_top, S.y_bottom)}-${Math.max(S.y_top, S.y_bottom)}`
         : 'Tracking box: not set (set TOP, BOTTOM, LEFT, RIGHT on the Analysis tab).';
}

function getParams() {
  return {sid: S.sid, frame_idx: S.currentFrame, y_top: S.y_top,
    y_bottom: S.y_bottom, x_left: S.x_left, x_right: S.x_right,
    black_thresh: +$('threshNum').value, min_area: +$('minArea').value,
    max_area: +$('maxArea').value, debounce: +$('debounce').value,
    direction: $('direction').value, strict: $('strict').checked,
    show_mask: $('showMask').checked};
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
    currentFrame: 0, y_top: null, y_bottom: null, x_left: null, x_right: null,
    history: [], cropStart: 0, cropEnd: data.duration,
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
  $('lblCount').textContent = '0';
  $('summary').textContent = 'Run an analysis to see results here.';
  $('plot').style.display = 'none';
  $('fastSummary').textContent = 'Run an analysis to see results here.';
  $('fastPlot').style.display = 'none'; $('fastStatus').textContent = '';
  updateCropBadge(); updateCropBar(); updateCropInfo(); renderFrame();
  $('startStrip').dataset.loaded = ''; $('endStrip').dataset.loaded = '';
  if ($('pageCrop').classList.contains('active')) { loadStartWindow(); loadEndWindow(); }
}

// ---- Render ----
// Only ONE /render request is in flight at a time. While the slider is dragged, further calls just
// mark the request as dirty and the newest position is rendered when the current one returns, so
// the server never builds a queue of frames nobody will look at.
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
      $('lblState').textContent = data.found ? data.state.toUpperCase() : 'HIDDEN';
      $('lblY').textContent = (data.found && data.y != null) ? Math.round(data.y) + ' px' : '-';
    } while (_rDirty);
  } finally { _rBusy = false; }
}

// ---- ROI ----
function setPickMode(mode) {
  if (!S.sid) return alert('Load a video first.');
  if (S.analyzing) return;
  S.pickMode = mode;
  ['top','bottom','left','right'].forEach(m => {
    $('btn' + m[0].toUpperCase() + m.slice(1))
      .classList.toggle('pick-active', m === mode);
  });
  $('statusMsg').textContent =
    `Click on the video frame to set the ${mode.toUpperCase()} boundary.`;
}
$('frameImg').addEventListener('click', e => {
  if (!S.pickMode || !S.sid || S.analyzing) return;
  const r = e.target.getBoundingClientRect();
  const x = Math.round((e.clientX - r.left) * (S.origW || e.target.naturalWidth) / r.width);
  const y = Math.round((e.clientY - r.top) * (S.origH || e.target.naturalHeight) / r.height);
  if (S.pickMode === 'top') S.y_top = y;
  else if (S.pickMode === 'bottom') S.y_bottom = y;
  else if (S.pickMode === 'left') S.x_left = x;
  else S.x_right = x;
  $('statusMsg').textContent = `Boundary ${S.pickMode.toUpperCase()} set.`;
  S.pickMode = null;
  ['top','bottom','left','right'].forEach(m =>
    $('btn' + m[0].toUpperCase() + m.slice(1)).classList.remove('pick-active'));
  updateFastRoi(); renderFrame();
});

// ---- Threshold ----
$('threshRange').oninput = e => { $('threshNum').value = e.target.value; renderFrame(); };
$('threshNum').onchange = e => {
  const v = Math.max(0, Math.min(255, +e.target.value || 0));
  e.target.value = v; $('threshRange').value = v; renderFrame();
};
$('showMask').onchange = renderFrame;
['minArea','maxArea'].forEach(id => $(id).onchange = renderFrame);

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

async function fetchStrip(which, center) {
  const el = $(which + 'Strip');
  el.innerHTML = '<div class="hint" style="padding:20px">Loading…</div>';
  const data = await postJSON('/crop_strip', {sid: S.sid, center_time: center,
    before: STRIP_BEFORE, after: STRIP_AFTER, n: STRIP_N});
  if (data.error) return el.innerHTML = `<div class="hint" style="padding:20px">Error: ${data.error}</div>`;
  $(which + 'WinCenter').value = data.center_time.toFixed(2);
  $(which + 'WinInfo').textContent =
    `Window: ${data.win_start.toFixed(2)}s → ${data.win_end.toFixed(2)}s ` +
    `(${(data.win_end - data.win_start).toFixed(2)}s, ${STRIP_N} thumbs)`;
  const sl = $(which + 'Slider');
  sl.min = data.win_start; sl.max = data.win_end; sl.step = 0.01;
  el.innerHTML = ''; el.dataset.loaded = '1';
  data.thumbs.forEach(t => {
    const d = document.createElement('div');
    d.className = 'thumb'; d.dataset.time = t.time_s;
    d.innerHTML = `<img src="${t.image}"><div class="tlabel">${t.time_s.toFixed(2)}s</div>`;
    d.onclick = () => selectTime(which, t.time_s);
    el.appendChild(d);
  });
  const cur = which === 'start' ? S.startSel : S.endSel;
  selectTime(which, (cur < data.win_start || cur > data.win_end) ? data.center_time : cur, false);
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

// ---- Analysis ----
async function runAnalysis() {
  if (!S.sid) return alert('Load a video first.');
  if (S.y_top === null || S.y_bottom === null || S.x_left === null || S.x_right === null)
    return alert('Set TOP, BOTTOM, LEFT, RIGHT boundaries first.');
  const cropOn = $('enableCrop').checked;
  if (cropOn && S.cropEnd <= S.cropStart) return alert('End time must be greater than start time.');
  const btn = $('btnRun'), badge = $('liveBadge'), live = $('livePreview').checked;
  S.analyzing = true; btn.disabled = true; btn.textContent = 'Analyzing...';
  $('progress').value = 2;
  if (live) badge.classList.add('active');

  const p = getParams();
  p.live_preview = live; p.preview_every = +$('previewEvery').value || 3;
  if (cropOn) { p.time_start = S.cropStart; p.time_end = S.cropEnd; }

  try {
    const resp = await fetch('/analyze', {method: 'POST',
      headers: {'Content-Type': 'application/json'}, body: JSON.stringify(p)});
    if (!resp.ok || !resp.body) {
      let m = 'Analysis failed';
      try { m = (await resp.json()).error || m; } catch {}
      return alert(m);
    }
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
        if (ev.error) return alert(ev.error);
        if (ev.progress !== undefined) $('progress').value = Math.max(2, ev.progress * 95);
        if (ev.preview) {
          const q = ev.preview;
          $('frameImg').src = q.image;
          $('lblState').textContent = (q.state || 'hidden').toUpperCase();
          $('lblY').textContent = q.y != null ? Math.round(q.y) + ' px' : '-';
          $('lblCount').textContent = q.rotation_count;
          $('lblTime').textContent = q.time_s.toFixed(2) + 's';
          $('frameInfo').textContent =
            `Frame: ${q.frame} / ${q.total_frames + (q.start_frame || 0)}   Time: ${q.time_s.toFixed(2)}s`;
          $('frameSlider').value = q.frame; S.currentFrame = q.frame;
          const frac = q.total_frames > 0 ? (q.frame - (q.start_frame || 0)) / q.total_frames : 0;
          $('progress').value = Math.min(95, frac * 95);
        }
        if (ev.done) {
          S.history = ev.history || [];
          $('summary').textContent = ev.summary;
          $('lblCount').textContent = ev.rotation_count;
          if (ev.plot) { $('plot').src = ev.plot; $('plot').style.display = 'block'; }
          $('progress').value = 100;
        }
      }
    }
  } catch (e) { alert('Analysis error: ' + e.message); }
  finally {
    S.analyzing = false; btn.disabled = false; btn.textContent = 'Run Full Analysis';
    badge.classList.remove('active');
    setTimeout(() => $('progress').value = 0, 1200);
  }
}

async function runFast() {
  if (!S.sid) return alert('Load a video on the Analysis tab first.');
  if (S.y_top === null || S.y_bottom === null || S.x_left === null || S.x_right === null)
    return alert('Set TOP, BOTTOM, LEFT, RIGHT boundaries on the Analysis tab first.');
  const cropOn = $('fastCrop').checked;
  if (cropOn && S.cropEnd <= S.cropStart) return alert('End time must be greater than start time.');
  const btn = $('btnFast'), badge = $('fastBadge');
  S.analyzing = true; btn.disabled = true; btn.textContent = 'Analyzing...';
  badge.classList.add('active'); $('fastProgress').value = 1;
  $('fastStatus').textContent = 'Starting...';
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
        if (ev.error) return alert(ev.error);
        if (ev.progress !== undefined) {
          $('fastProgress').value = ev.progress * 100;
          $('fastStatus').textContent =
            `Processed ${ev.done_frames} / ${ev.total} frames (${ev.parts} parts)`;
        }
        if (ev.done) {
          S.history = ev.history || [];
          $('fastSummary').textContent = ev.summary; $('summary').textContent = ev.summary;
          $('lblCount').textContent = ev.rotation_count;
          if (ev.plot) {
            $('fastPlot').src = ev.plot; $('fastPlot').style.display = 'block';
            $('plot').src = ev.plot; $('plot').style.display = 'block';
          }
          $('fastProgress').value = 100;
          $('fastStatus').textContent =
            `Finished in ${((performance.now() - t0) / 1000).toFixed(1)} s`;
        }
      }
    }
  } catch (e) { alert('Analysis error: ' + e.message); }
  finally {
    S.analyzing = false; btn.disabled = false; btn.textContent = 'Run Parallel Analysis';
    badge.classList.remove('active');
  }
}

function resetAnalysis() {
  S.history = [];
  $('summary').textContent = 'Run an analysis to see results here.';
  $('lblCount').textContent = '0'; $('lblState').textContent = '-';
  $('lblY').textContent = '-'; $('lblExpect').textContent = 'top';
  $('lblTime').textContent = '0.00s'; $('plot').style.display = 'none';
  $('progress').value = 0;
  $('fastSummary').textContent = 'Run an analysis to see results here.';
  $('fastPlot').style.display = 'none'; $('fastProgress').value = 0;
  $('fastStatus').textContent = '';
}

// ---- CSV ----
function exportCSV() {
  if (!S.history.length) return alert('Run an analysis first.');
  const cols = ['frame','time_s','y_position','unwrapped_angle_deg','state','stable_state','rotation_count'];
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
</script>
</body>
</html>
"""

if __name__ == '__main__':
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
