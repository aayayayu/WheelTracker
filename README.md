# Flywheel Rotation Counter (Android, Chaquopy)

Runs your Flask + OpenCV app fully on-device. Nothing is uploaded anywhere.

1. Install Android Studio (Koala or newer) with the default SDK.
2. File > Open > select this folder. Let Gradle sync (first sync downloads Python wheels; needs internet).
3. Enable USB debugging on your phone, plug in, press Run (green triangle).
   Or Build > Build APK(s) and install app/build/outputs/apk/debug/app-debug.apk.

## Performance notes
- All analysis (normal, live preview, parallel) uses the hardware decoder and reads only the tracking box.
  The live preview is a small grayscale picture taken from that same pass, so it costs almost nothing.
- The frame slider keeps one decoder open per video, caches the last frames, and only sends one
  request at a time. Logcat (tag `python`) prints `RENDER frame N: decode X ms` so you can see the raw decode cost.
- Only the currently loaded video is kept in the app cache. Older copies are deleted on upload,
  on app start and when the app is closed.

## Build
The GitHub Action builds the debug APK and publishes it as `Flywheel.apk` (artifact "Flywheel").

## Algorithm (wheel_algo.py)

The tracker was replaced by the side-on tape tracker from `flywheel_rotation_counter.py`.
The UI, the hardware decoder (FastDecoder.kt), time crop, parallel mode and CSV export are unchanged.

1. The **tracking box** (Top / Bottom / Left / Right) only has to surround the wheel. Inside it the
   wheel is found automatically in every frame (grey pixels darker than the wall), giving a wheel box.
2. The **tape** is the longest run of dark rows inside the wheel (`Black Limit`).
3. Tape height on the rim gives its angle: `sin(theta) = (y_tape - y_centre) / R`.
4. Every separate appearance of the tape is one more turn; the hidden half of each turn is
   interpolated, so the result is a fractional number of rotations (e.g. 6.13).
5. Only the grayscale tracking box is used, and only ~one row profile per frame is kept in memory.

How the existing controls are used now:

| Control | Meaning |
|---|---|
| Black Limit | row brightness below this = tape |
| Min / Max blob area | allowed tape band area (px); 0 = unlimited |
| Debounce frames | a tape pass must be seen in at least this many frames (rejects glare/noise) |
| Tape motion | direction the tape moves across the visible rim |
| Strict sequence mode | ignore passes where the tape moved the wrong way |

Assumptions: side-on camera, one dark tape on the rim, wheel does not reverse, wall brighter than the
wheel, and the wheel turns less than half a turn between frames.
