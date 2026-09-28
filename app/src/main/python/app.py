"""
Edge-On Wheel Rotation Counter — Flask Web Edition
"""
import os, io, csv, math, base64, uuid, json as _json, tempfile
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


def annotate(frame, res, p, s):
    out = frame.copy()
    W, H = s['orig_w'], s['orig_h']
    for y, col in ((p.get('y_top'), (255, 0, 0)), (p.get('y_bottom'), (255, 0, 0))):
        if y is not None: cv2.line(out, (0, y), (W, y), col, 2)
    for x, col in ((p.get('x_left'), (0, 255, 0)), (p.get('x_right'), (0, 255, 0))):
        if x is not None: cv2.line(out, (x, 0), (x, H), col, 2)
    if res['found']:
        x, y, w, h = res['rect']
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
        txt, col = f"STATE: {res['state'].upper()} | Y: {int(res['y'])}", (0, 255, 255)
    else:
        txt, col = "Tape HIDDEN (back side)", (0, 0, 255)
    cv2.putText(out, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
    return out


def encode_jpeg(bgr, max_w=DISPLAY_MAX_W):
    h, w = bgr.shape[:2]
    if w > max_w:
        bgr = cv2.resize(bgr, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    return 'data:image/jpeg;base64,' + base64.b64encode(
        cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode()


def extract_params(d):
    return {k: d.get(k) for k in ('y_top', 'y_bottom', 'x_left', 'x_right')} | {
        'black_thresh': int(d.get('black_thresh', 60)),
        'min_area': int(d.get('min_area', 100)),
        'max_area': int(d.get('max_area', 0)),
        'debounce': int(d.get('debounce', 2)),
        'direction': d.get('direction', 'down'),
        'strict': bool(d.get('strict', False)),
    }


def grab_frames(path, indices):
    """Open video, seek to each index, return list of (idx, frame)."""
    cap = cv2.VideoCapture(path)
    out = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, f = cap.read()
        if ret: out.append((i, f))
    cap.release()
    return out


# ---------- Analysis stream ----------
def run_analysis_stream(sess, p, live=False, every=3, t0=None, t1=None):
    fps, total = sess['fps'], sess['total_frames']
    sf = max(0, int(t0 * fps)) if t0 is not None else 0
    ef = min(total, int(t1 * fps)) if t1 is not None else total
    seq = ['top', 'mid', 'bottom', 'hidden'] if p['direction'] == 'down' \
          else ['bottom', 'mid', 'top', 'hidden']

    cap = cv2.VideoCapture(sess['video_path'])
    if sf: cap.set(cv2.CAP_PROP_POS_FRAMES, sf)

    exp, cand, ccnt, stable = 0, None, 0, None
    rot, rtimes, cum = 0, [], 0.0
    prev, max_jump, hist = None, 0.0, []

    fi, proc = sf, 0
    while fi < ef:
        ret, frame = cap.read()
        if not ret: break
        t = fi / fps if fps > 0 else 0.0
        res = process_frame(frame, p)
        st = res['state']

        if st == cand: ccnt += 1
        else: cand, ccnt = st, 1
        if ccnt >= p['debounce'] and st != stable:
            first = stable is None
            stable = st
            if first:
                if st in seq: exp = (seq.index(st) + 1) % 4
            else:
                if st == seq[exp]:
                    exp = (exp + 1) % 4
                    if exp == 0: rot += 1; rtimes.append(t)
                elif not p['strict'] and st in seq:
                    exp = (seq.index(st) + 1) % 4
                    if exp == 0: rot += 1; rtimes.append(t)

        th = None
        if res['found']:
            th = angle_from_y(res['y'], p)
            cum = rot * 360.0 + th
            if prev is not None: max_jump = max(max_jump, abs(cum - prev))
            prev = cum

        hist.append({'frame': fi, 'time_s': round(t, 5),
                     'y_position': res['y'] if res['found'] else None,
                     'unwrapped_angle_deg': cum if res['found'] else None,
                     'state': st, 'stable_state': stable, 'rotation_count': rot})

        if live and proc % every == 0:
            yield {'preview': {'frame': fi, 'time_s': round(t, 5),
                               'image': encode_jpeg(annotate(frame, res, p, sess)),
                               'state': st, 'found': res['found'], 'angle': th,
                               'rotation_count': rot, 'total_frames': ef - sf,
                               'start_frame': sf}}
        fi += 1; proc += 1
    cap.release()

    total_time = proc / fps if fps > 0 else 0.0
    L = ["Analysis completed.", f"Frames processed: {proc}",
         f"Video duration processed: {total_time:.2f} s"]
    if sf or ef < total:
        L.append(f"Time crop: {sf / fps:.2f}s to {ef / fps:.2f}s")
    L.append(f"Total full rotations counted: {rot}")
    if rot and rtimes:
        off = sf / fps if fps > 0 else 0.0
        per = (rtimes[-1] - off) / rot
        w = 2 * math.pi / per
        L += [f"Average period per rotation: {per:.4f} s",
              f"Average angular velocity: {w:.4f} rad/s ({w * 60 / (2 * math.pi):.2f} RPM)"]
    else:
        L.append("No complete rotations were counted.")
    if max_jump: L.append(f"Max single-frame angular jump: {max_jump:.2f} deg")
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

    yield {'done': True, 'summary': summary, 'rotation_count': rot,
           'history': hist, 'plot': plot, 'start_frame': sf, 'end_frame': ef}


# ---------- Routes ----------
@app.route('/')
def index(): return TEMPLATE


@app.route('/upload', methods=['POST'])
def upload():
    f = request.files.get('video')
    if not f or not f.filename: return jsonify({'error': 'no file'}), 400
    sid = uuid.uuid4().hex
    name = os.path.basename(f.filename)
    tmp = os.path.join(tempfile.gettempdir(), f"{sid}_{name}")
    f.save(tmp)
    cap = cv2.VideoCapture(tmp)
    if not cap.isOpened(): return jsonify({'error': 'cannot open video'}), 400
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0: fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = cap.read(); cap.release()
    if not ret: return jsonify({'error': 'cannot read first frame'}), 400
    h, w = frame.shape[:2]
    SESSIONS[sid] = {'video_path': tmp, 'fps': fps, 'total_frames': total,
                     'orig_w': w, 'orig_h': h, 'filename': name}
    return jsonify({'sid': sid, 'fps': fps, 'total_frames': total,
                    'orig_w': w, 'orig_h': h, 'filename': name,
                    'duration': total / fps if fps else 0})


@app.route('/render', methods=['POST'])
def render():
    d = request.get_json(force=True)
    sess = SESSIONS.get(d.get('sid'))
    if not sess: return jsonify({'error': 'invalid session'}), 400
    idx = max(0, min(int(d.get('frame_idx', 0)), sess['total_frames'] - 1))
    cap = cv2.VideoCapture(sess['video_path'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read(); cap.release()
    if not ret: return jsonify({'error': 'cannot read frame'}), 400
    p = extract_params(d)
    res = process_frame(frame, p)
    disp = cv2.cvtColor(res['mask'], cv2.COLOR_GRAY2BGR) if d.get('show_mask') \
           else annotate(frame, res, p, sess)
    th = angle_from_y(res['y'], p) if res['found'] else None
    return jsonify({'image': encode_jpeg(disp), 'state': res['state'],
                    'found': res['found'], 'angle': th})


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
    for i, fr in grab_frames(sess['video_path'], idxs):
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
            for ev in run_analysis_stream(sess, p, live, every, t0, t1):
                yield 'data: ' + _json.dumps(ev) + '\n\n'
        except Exception as e:
            yield 'data: ' + _json.dumps({'error': str(e)}) + '\n\n'
    return Response(gen(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/fit_alpha', methods=['POST'])
def fit_alpha():
    d = request.get_json(force=True)
    h = d.get('history', [])
    t0, t1 = float(d.get('t0', 0.0)), float(d.get('t1', 1e9))
    pts = [(x['time_s'], x['unwrapped_angle_deg']) for x in h
           if x['unwrapped_angle_deg'] is not None and t0 <= x['time_s'] <= t1]
    if len(pts) < 5:
        return jsonify({'error': 'Not enough data in the specified range.'}), 400
    c = np.polyfit([p[0] for p in pts], np.radians([p[1] for p in pts]), 2)
    return jsonify({'alpha': float(2.0 * c[0])})


# ---------- Template ----------
TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Edge-On Reaction Wheel Tracker</title>
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
<header><h1>Edge-On Reaction Wheel Tracker</h1></header>
<div class="tabbar">
  <button id="tabAnalysis" class="active" onclick="showTab('analysis')">Analysis</button>
  <button id="tabCrop" onclick="showTab('crop')">Time Crop</button>
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
        <div>Angle<span id="lblAngle" class="big">-</span></div>
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
        <label for="livePreview">Show live frame-by-frame preview</label></div>
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
  <div class="panel">
    <h3>Moment of Inertia Calculator</h3>
    <div class="moi-grid">
      <label>Fit start (s)</label><input type="number" id="fitStart" value="0" step="0.01">
      <label>Fit end (s)</label><input type="number" id="fitEnd" value="1000000" step="0.01">
    </div>
    <div class="row"><button onclick="fitAlpha()">Fit angular acceleration (&alpha;)</button></div>
    <div class="moi-grid">
      <label>&alpha; (rad/s&sup2;)</label><input type="number" id="alpha" value="0" step="0.000001">
      <label>Method</label>
      <select id="moiMethod">
        <option value="falling">Falling mass: I = m r&sup2;(g/a - 1)</option>
        <option value="torque">Known torque: I = &tau; / &alpha;</option>
      </select>
    </div>
    <div class="moi-grid">
      <label>Mass m (kg)</label><input type="number" id="mass" value="0.05" step="0.001">
      <label>Radius r (m)</label><input type="number" id="radius" value="0.01" step="0.00001">
      <label>g (m/s&sup2;)</label><input type="number" id="gVal" value="9.81" step="0.01">
      <label>Torque (N&middot;m)</label><input type="number" id="torque" value="0" step="0.001" disabled>
    </div>
    <div class="row"><button onclick="computeMOI()">Compute Moment of Inertia</button></div>
    <div id="moiResult" class="moi-result">I = -</div>
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
    </div>
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
  $('pageAnalysis').classList.toggle('active', name === 'analysis');
  $('pageCrop').classList.toggle('active', name === 'crop');
  $('tabAnalysis').classList.toggle('active', name === 'analysis');
  $('tabCrop').classList.toggle('active', name === 'crop');
  if (name === 'crop' && S.sid && !$('startStrip').dataset.loaded) {
    loadStartWindow(); loadEndWindow();
  }
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
  const fd = new FormData(); fd.append('video', file);
  $('fileLabel').textContent = 'Uploading...';
  const data = await (await fetch('/upload', {method:'POST', body:fd})).json();
  if (data.error) return alert(data.error);
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
  sl.oninput = () => { if (!S.analyzing) { S.currentFrame = +sl.value; renderFrame(); } };
  $('lblCount').textContent = '0';
  $('summary').textContent = 'Run an analysis to see results here.';
  $('plot').style.display = 'none';
  updateCropBadge(); updateCropBar(); renderFrame();
  $('startStrip').dataset.loaded = ''; $('endStrip').dataset.loaded = '';
  if ($('pageCrop').classList.contains('active')) { loadStartWindow(); loadEndWindow(); }
}

// ---- Render ----
async function renderFrame() {
  if (!S.sid || S.analyzing) return;
  const data = await postJSON('/render', getParams());
  if (data.error) return console.warn(data.error);
  $('frameImg').src = data.image;
  $('frameInfo').textContent =
    `Frame: ${S.currentFrame} / ${S.totalFrames}   Time: ${(S.currentFrame / S.fps).toFixed(2)}s`;
  $('lblState').textContent = data.found ? data.state.toUpperCase() : 'HIDDEN';
  $('lblAngle').textContent = (data.found && data.angle != null)
    ? data.angle.toFixed(1) + ' deg' : '-';
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
  const x = Math.round((e.clientX - r.left) * e.target.naturalWidth / r.width);
  const y = Math.round((e.clientY - r.top) * e.target.naturalHeight / r.height);
  if (S.pickMode === 'top') S.y_top = y;
  else if (S.pickMode === 'bottom') S.y_bottom = y;
  else if (S.pickMode === 'left') S.x_left = x;
  else S.x_right = x;
  $('statusMsg').textContent = `Boundary ${S.pickMode.toUpperCase()} set.`;
  S.pickMode = null;
  ['top','bottom','left','right'].forEach(m =>
    $('btn' + m[0].toUpperCase() + m.slice(1)).classList.remove('pick-active'));
  renderFrame();
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

function applyCropToAnalysis() {
  if (!S.sid) return alert('Load a video first.');
  const full = S.cropStart <= 0.001 && Math.abs(S.cropEnd - S.duration) <= 0.001;
  $('enableCrop').checked = !full;
  updateCropBadge();
  showTab('analysis');
}

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
        if (ev.preview) {
          const q = ev.preview;
          $('frameImg').src = q.image;
          $('lblState').textContent = (q.state || 'hidden').toUpperCase();
          $('lblAngle').textContent = q.angle != null ? q.angle.toFixed(1) + ' deg' : '-';
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

function resetAnalysis() {
  S.history = [];
  $('summary').textContent = 'Run an analysis to see results here.';
  $('lblCount').textContent = '0'; $('lblState').textContent = '-';
  $('lblAngle').textContent = '-'; $('lblExpect').textContent = 'top';
  $('lblTime').textContent = '0.00s'; $('plot').style.display = 'none';
  $('progress').value = 0; $('moiResult').textContent = 'I = -';
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

// ---- MOI ----
async function fitAlpha() {
  if (!S.history.length) return alert('Run an analysis first.');
  const data = await postJSON('/fit_alpha', {history: S.history,
    t0: +$('fitStart').value || 0, t1: +$('fitEnd').value || 1e9});
  if (data.error) return alert(data.error);
  $('alpha').value = data.alpha;
}

$('moiMethod').onchange = e => {
  const tq = e.target.value === 'torque';
  $('torque').disabled = !tq; $('mass').disabled = tq; $('gVal').disabled = tq;
};

function computeMOI() {
  const alpha = +$('alpha').value;
  if (!alpha) return alert('alpha is zero or missing.');
  const r = +$('radius').value, method = $('moiMethod').value;
  let out;
  if (method === 'falling') {
    const m = +$('mass').value, g = +$('gVal').value, a = r * Math.abs(alpha);
    if (a >= g) return alert('a >= g, physically impossible.');
    out = `I = ${(m * r * r * (g / a - 1)).toExponential(6)} kg·m²   (falling-mass; a = ${a.toFixed(5)} m/s²)`;
  } else {
    out = `I = ${((+$('torque').value) / Math.abs(alpha)).toExponential(6)} kg·m²   (known-torque)`;
  }
  $('moiResult').textContent = out;
}

updateCropBar();
</script>
</body>
</html>
"""

if __name__ == '__main__':
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
