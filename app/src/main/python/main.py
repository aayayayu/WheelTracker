"""
Android entry point. Keeps app.py (the Flask app) almost untouched and patches it at runtime:
 1. cv2.VideoCapture is replaced by a look-alike backed by Android's MediaMetadataRetriever.
    No conversion / transcoding step: loading a video is instant. Used for single frames only
    (slider, thumbnails); app.py keeps ONE retriever open per video instead of one per request.
 2. ALL analysis (normal, live preview and parallel) uses the phone's hardware video decoder
    (FastDecoder.kt). The flywheel detector needs colour, so every frame arrives whole, sub-sampled
    to about DETECT_W pixels wide and converted to BGR. The live preview is drawn on that same frame,
    so it costs no extra decoding.
 3. Mobile-friendly CSS / viewport, and CSV export through the native bridge.
"""
import os, tempfile, time
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import numpy as np
import cv2

_orig_vc = cv2.VideoCapture


class JavaCap:
    """Minimal cv2.VideoCapture look-alike backed by android.media.MediaMetadataRetriever.
    Fine for single frames (slider, thumbnails); analysis uses FastDecoder instead."""

    def __init__(self, path):
        self.ok, self.pos, self.total, self.fps, self.w, self.h = False, 0, 0, 30.0, 0, 0
        self.rot = 0
        try:
            from java import jclass
            self._MMR = jclass("android.media.MediaMetadataRetriever")
            self._BB = jclass("java.nio.ByteBuffer")
            self._BAOS = jclass("java.io.ByteArrayOutputStream")
            self._CF = jclass("android.graphics.Bitmap$CompressFormat")
            self.r = self._MMR()
            self.r.setDataSource(path)
            g = lambda k: self.r.extractMetadata(k)
            dur_ms = float(g(9) or 0)            # METADATA_KEY_DURATION
            n = g(32)                            # METADATA_KEY_VIDEO_FRAME_COUNT
            self.w, self.h = int(g(18) or 0), int(g(19) or 0)
            self.rot = int(g(24) or 0)           # METADATA_KEY_VIDEO_ROTATION
            self.total = int(n) if n else 0
            if self.total and dur_ms > 0:
                self.fps = self.total / (dur_ms / 1000.0)
            elif dur_ms > 0:
                cf = g(25)                       # METADATA_KEY_CAPTURE_FRAMERATE
                try:
                    self.fps = float(cf) if cf else 30.0
                except ValueError:
                    self.fps = 30.0
                self.total = max(1, int(round(dur_ms / 1000.0 * self.fps)))
            self.ok = self.total > 0
        except Exception as e:
            print("JavaCap open failed:", e)

    def isOpened(self):
        return self.ok

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: self.fps,
                cv2.CAP_PROP_FRAME_COUNT: float(self.total),
                cv2.CAP_PROP_FRAME_WIDTH: float(self.w),
                cv2.CAP_PROP_FRAME_HEIGHT: float(self.h)}.get(prop, 0.0)

    def set(self, prop, val):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.pos = int(val)
            return True
        return False

    def _to_bgr(self, bmp):
        """Bitmap -> BGR numpy array. Copies the raw pixels (no JPEG encode + decode round trip,
        which used to cost more than the frame grab itself); JPEG is only a fallback."""
        try:
            if str(bmp.getConfig()) == "ARGB_8888":
                from java import jarray, jbyte
                w, h, rb = bmp.getWidth(), bmp.getHeight(), bmp.getRowBytes()
                arr = jarray(jbyte)(rb * h)
                bmp.copyPixelsToBuffer(self._BB.wrap(arr))
                try:
                    a = np.frombuffer(arr, np.uint8)
                except Exception:
                    a = np.frombuffer(bytes(arr), np.uint8)
                return cv2.cvtColor(a.reshape(h, rb)[:, :w * 4].reshape(h, w, 4), cv2.COLOR_RGBA2BGR)
        except Exception as e:
            print("JavaCap raw copy failed, using JPEG:", e)
        baos = self._BAOS()
        bmp.compress(self._CF.JPEG, 92, baos)
        buf = np.frombuffer(bytes(baos.toByteArray()), np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    def read_thumb(self, max_w):
        """Small BGR picture of frame `pos`, scaled by the decoder itself (no full-size bitmap copy).
        Returns None if that is not possible; the caller then falls back to read()."""
        if not self.ok or not (0 <= self.pos < self.total):
            return None
        dw, dh = (self.h, self.w) if self.rot in (90, 270) else (self.w, self.h)   # displayed size
        if dw <= 0 or dh <= 0:
            return None
        tw = min(int(max_w), dw)
        th = max(1, int(round(dh * tw / dw)))
        us = int(self.pos * 1_000_000 / (self.fps or 30.0))
        bmp = self.r.getScaledFrameAtTime(us, 3, tw, th)          # 3 = OPTION_CLOSEST
        if bmp is None:
            return None
        try:
            return self._to_bgr(bmp)
        finally:
            bmp.recycle()

    def read(self):
        if not self.ok or self.pos >= self.total:
            return False, None
        try:
            try:
                bmp = self.r.getFrameAtIndex(self.pos)
            except Exception:
                us = int(self.pos * 1_000_000 / (self.fps or 30.0))
                bmp = self.r.getFrameAtTime(us, 3)   # OPTION_CLOSEST
            self.pos += 1
            if bmp is None:
                return False, None
            try:
                frame = self._to_bgr(bmp)
            finally:
                bmp.recycle()
            return (frame is not None), frame
        except Exception as e:
            print("JavaCap read failed:", e)
            return False, None

    def release(self):
        try:
            self.r.release()
        except Exception:
            pass


def _video_capture(path, *a):
    c = JavaCap(path)
    if c.isOpened():
        return c
    print("MediaMetadataRetriever could not open the video; trying OpenCV")
    return _orig_vc(path, *a)


cv2.VideoCapture = _video_capture


# ---------------- hardware-decoder range reader ----------------
def _to_np(jarr):
    """Java byte[] -> uint8 numpy array."""
    try:
        return np.frombuffer(jarr, dtype=np.uint8)          # buffer protocol (fast)
    except Exception:
        pass
    try:
        return np.frombuffer(bytes(jarr), dtype=np.uint8)   # one memcpy: still fast
    except Exception:
        return np.array(jarr, dtype=np.int8).view(np.uint8)  # very slow, last resort


def _to_bgr(f):
    """FdFrame (raw orientation, BGR bytes) -> displayed-orientation BGR array."""
    a = _to_np(f.data).reshape(f.h, f.w, 3)
    k = (4 - int(f.rotation) // 90) % 4                    # raw frame -> displayed orientation
    if k:
        a = np.ascontiguousarray(np.rot90(a, k))
    return a


def hw_range_reader(sess, sf, ef, light=None):
    """Yield (frame_index, frame) for frames [sf, ef) using MediaCodec. frame is a BGR image
    sub-sampled by sess['step'] or, when `light` = {'roi': (x0, y0, x1, y1), 'every': N} is given, a
    wheel_algo.LightFrame (grayscale of the roi only) for every frame that is not a multiple of N."""
    from java import jclass
    import wheel_algo as wa
    FD = jclass("com.example.wheeltracker.FastDecoder")
    every = int(light['every']) if light else 1
    s = FD.open(sess['video_path'], int(sf), int(ef), float(sess['fps']), int(sess['step']), every)
    n, roi_set = 0, False
    try:
        while True:
            f = s.pollWait(20)        # blocks in Java (GIL released) until a frame is ready
            if f is None:
                if s.isDone():
                    break
                continue
            n += 1
            if f.light:
                k = (4 - int(f.rotation) // 90) % 4
                a = _to_np(f.data).reshape(f.h, f.w)
                if k:
                    a = np.rot90(a, k)
                yield int(f.idx), wa.LightFrame(a, light['roi'][0], light['roi'][1])
                continue
            bgr = _to_bgr(f)
            if light and not roi_set:
                # the decoder works in the raw orientation: tell it where the roi is there
                x0, y0, x1, y1 = light['roi']
                dh, dw = bgr.shape[:2]
                x0, y0, x1, y1 = max(0, x0), max(0, y0), min(dw, x1), min(dh, y1)
                k = (4 - int(f.rotation) // 90) % 4
                m = np.zeros((dh, dw), bool)
                m[y0:y1, x0:x1] = True
                ys, xs = np.nonzero(np.rot90(m, -k))
                s.setRoi(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
                light = dict(light, roi=(x0, y0, x1, y1))
                roi_set = True
            yield int(f.idx), bgr
        err = s.errorMessage()
        if err:
            raise RuntimeError(err)
        if n == 0:
            raise RuntimeError("hardware decoder returned no frames")
    finally:
        s.close()


MOBILE_CSS = """
button{min-height:40px}
@media(max-width:900px){
 .container{grid-template-columns:1fr}
 .crop-controls{grid-template-columns:1fr 1fr}
 .moi-grid{grid-template-columns:auto 1fr}
 .summary-bar{flex-wrap:wrap;gap:8px}
 .tabbar{padding:0 8px}
}
"""


def cleanup():
    """Called from MainActivity when the app is closed: delete the cached video copies."""
    try:
        import app as wt
        wt.drop_all()
    except Exception as e:
        print("cleanup failed:", e)


def start():
    import app as wt
    wt.startup_cleanup()            # also removes videos left by older versions / crashed runs
    wt.RANGE_READER = hw_range_reader
    t = wt.TEMPLATE
    t = t.replace('<meta charset="utf-8">',
                  '<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">', 1)
    t = t.replace('</style>', MOBILE_CSS + '</style>', 1)
    t = t.replace('const url = URL.createObjectURL(',
                  'if (window.Android) { Android.saveCsv(lines.join("\\n")); return; }\n  const url = URL.createObjectURL(', 1)
    wt.TEMPLATE = t
    wt.app.run(host='127.0.0.1', port=5000, debug=False, threaded=True, use_reloader=False)
