"""
Android entry point. Keeps your original app.py untouched and patches it at runtime:
 1. cv2.VideoCapture falls back to FFmpeg (ffmpeg-kit) transcoding to MJPEG AVI, then to
    Android's MediaMetadataRetriever, if OpenCV's build cannot decode the video.
 2. Mobile-friendly CSS / viewport.
 3. CSV export goes through the native bridge (WebView can't download blob: URLs).
"""
import os, tempfile
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import numpy as np
import cv2

_orig_vc = cv2.VideoCapture


class JavaCap:
    """Minimal cv2.VideoCapture look-alike backed by android.media.MediaMetadataRetriever."""

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
            bmp = self.r.getFrameAtIndex(self.pos)
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


def _transcode(path):
    """Use FFmpeg (ffmpeg-kit) to make an MJPEG .avi that OpenCV can always read and seek fast."""
    out = path + ".mjpg.avi"
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out
    try:
        from java import jclass, jarray
        FK = jclass("com.arthenica.ffmpegkit.FFmpegKit")
        RC = jclass("com.arthenica.ffmpegkit.ReturnCode")
        args = ["-y", "-i", path, "-an", "-vsync", "0",
                "-vf", "scale='min(1280,iw)':-2", "-pix_fmt", "yuvj420p",
                "-c:v", "mjpeg", "-q:v", "3", out]
        sess = FK.executeWithArguments(jarray(jclass("java.lang.String"))(args))
        if RC.isSuccess(sess.getReturnCode()):
            return out
        print("ffmpeg failed:", sess.getOutput())
    except Exception as e:
        print("ffmpeg unavailable:", e)
    try:
        os.remove(out)
    except OSError:
        pass
    return None


def _try_open(path):
    c = _orig_vc(path)
    try:
        if c.isOpened():
            ok, _ = c.read()
            if ok:
                c.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return c
    except Exception:
        pass
    try:
        c.release()
    except Exception:
        pass
    return None


def _video_capture(path, *a):
    # 1) OpenCV directly  2) FFmpeg -> MJPEG AVI -> OpenCV  3) MediaMetadataRetriever
    c = _try_open(path)
    if c is not None:
        return c
    print("OpenCV cannot decode this video; transcoding with FFmpeg")
    t = _transcode(path)
    if t:
        c = _try_open(t)
        if c is not None:
            return c
    print("FFmpeg path failed; using Android MediaMetadataRetriever")
    return JavaCap(path)


cv2.VideoCapture = _video_capture

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
    t = wt.TEMPLATE
    t = t.replace('<meta charset="utf-8">',
                  '<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">', 1)
    t = t.replace('</style>', MOBILE_CSS + '</style>', 1)
    t = t.replace('const url = URL.createObjectURL(',
                  'if (window.Android) { Android.saveCsv(lines.join("\\n")); return; }\n  const url = URL.createObjectURL(', 1)
    wt.TEMPLATE = t
    wt.app.run(host='127.0.0.1', port=5000, debug=False, threaded=True, use_reloader=False)
