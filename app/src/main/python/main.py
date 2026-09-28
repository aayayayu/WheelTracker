"""
Android entry point. Keeps app.py (the Flask app) almost untouched and patches it at runtime:
 1. cv2.VideoCapture is replaced by a look-alike backed by Android's MediaMetadataRetriever.
    No conversion / transcoding step: loading a video is instant.
 2. Fast analysis uses the phone's hardware video decoder (FastDecoder.kt) and reads only the
    tracking box (luma plane), instead of decoding whole colour frames.
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
        try:
            from java import jclass
            self._MMR = jclass("android.media.MediaMetadataRetriever")
            self._BAOS = jclass("java.io.ByteArrayOutputStream")
            self._CF = jclass("android.graphics.Bitmap$CompressFormat")
            self.r = self._MMR()
            self.r.setDataSource(path)
            g = lambda k: self.r.extractMetadata(k)
            dur_ms = float(g(9) or 0)            # METADATA_KEY_DURATION
            n = g(32)                            # METADATA_KEY_VIDEO_FRAME_COUNT
            self.w, self.h = int(g(18) or 0), int(g(19) or 0)
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
            baos = self._BAOS()
            bmp.compress(self._CF.JPEG, 92, baos)
            bmp.recycle()
            buf = np.frombuffer(bytes(baos.toByteArray()), np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
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
_LIMITED_LUT = np.clip((np.arange(256) - 16) * 255.0 / 219.0, 0, 255).astype(np.uint8)


def _to_np(jarr):
    """Java byte[] -> uint8 numpy array."""
    try:
        return np.frombuffer(jarr, dtype=np.uint8)          # buffer protocol (fast)
    except Exception:
        return np.array(jarr, dtype=np.int8).view(np.uint8)  # slow but always works


def _to_gray(f):
    a = _to_np(f.data).reshape(f.h, f.w)
    if not f.full:                      # limited-range luma (16..235) -> full 0..255
        a = cv2.LUT(a, _LIMITED_LUT)
    k = (4 - int(f.rotation) // 90) % 4  # raw frame -> displayed orientation
    if k:
        a = np.ascontiguousarray(np.rot90(a, k))
    return a


def hw_range_reader(sess, sf, ef, rect):
    """Yield (frame_index, gray ROI) for frames [sf, ef) using MediaCodec."""
    from java import jclass
    FD = jclass("com.example.wheeltracker.FastDecoder")
    x0, y0, x1, y1 = rect
    s = FD.open(sess['video_path'], int(sf), int(ef), float(sess['fps']),
                int(x0), int(y0), int(x1), int(y1))
    n = 0
    try:
        while True:
            f = s.poll()
            if f is None:
                if s.isDone():
                    break
                time.sleep(0.001)
                continue
            n += 1
            yield int(f.idx), _to_gray(f)
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


def start():
    import app as wt
    wt.RANGE_READER = hw_range_reader
    t = wt.TEMPLATE
    t = t.replace('<meta charset="utf-8">',
                  '<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">', 1)
    t = t.replace('</style>', MOBILE_CSS + '</style>', 1)
    t = t.replace('const url = URL.createObjectURL(',
                  'if (window.Android) { Android.saveCsv(lines.join("\\n")); return; }\n  const url = URL.createObjectURL(', 1)
    wt.TEMPLATE = t
    wt.app.run(host='127.0.0.1', port=5000, debug=False, threaded=True, use_reloader=False)
