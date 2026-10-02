# Flywheel Rotation Counter (Android, Chaquopy)

Runs your Flask + OpenCV app fully on-device. Nothing is uploaded anywhere.

1. Install Android Studio (Koala or newer) with the default SDK.
2. File > Open > select this folder. Let Gradle sync (first sync downloads Python wheels; needs internet).
3. Enable USB debugging on your phone, plug in, press Run (green triangle).
   Or Build > Build APK(s) and install app/build/outputs/apk/debug/app-debug.apk.

## Performance notes
- All analysis (normal, live preview, parallel) uses the hardware decoder (FastDecoder.kt). The flywheel
  detector needs colour, so every frame is delivered whole, sub-sampled to about 960 px wide and converted
  to BGR. The live preview is drawn on that same frame, so it costs no extra decoding.
- The frame slider keeps one decoder open per video, caches the last frames, and only sends one
  request at a time. Logcat (tag `python`) prints `RENDER frame N: decode X ms` so you can see the raw decode cost.
- Only the currently loaded video is kept in the app cache. Older copies are deleted on upload,
  on app start and when the app is closed.

## Build
The GitHub Action builds the debug APK and publishes it as `Flywheel.apk` (artifact "Flywheel").

## Algorithm (wheel_algo.py)

`wheel_algo.py` is a port of `flywheel_rotations.py` (side view, one black tape on the rim). Nothing has to
be marked on the video any more: there is no tracking box, no black limit and no blob-area setting.

1. **Flywheel box, every frame.** The wheel is the gray vertical block (low saturation, darker than the
   wall) that crosses the blue bracket. If it is not found in a frame, the last known box is reused (the
   first known box for the frames before the first detection).
2. **Brightness profile.** Inside the box a 200-point vertical brightness profile is taken (middle 60 % of
   the width) and divided by its own smoothed baseline.
3. **Tape = what moves.** Every profile is divided by the *median profile over the whole video*, so the
   static parts (rod, shaft, edges) cancel and only the moving black tape is left, as a dark band
   (profile < `Dark threshold`, default 0.70).
4. **Phase.** Tape height y in [0, 1] gives the angle `asin(2y - 1)`. A new lap starts when y jumps back up
   by more than `New-lap jump` (default 0.30). The angle is unwrapped (never goes backwards), interpolated
   while the tape is behind the wheel, and extrapolated after the last sighting with the latest lap period.
5. **rotations = (final phase - first phase) / 360.**

Because step 3 needs the whole video, the exact result is computed when the run has finished. While the live
preview runs, a causal estimate (median of the frames seen so far) is shown instead.

### Controls

| Control | Meaning |
|---|---|
| Dark threshold | tape = profile below this fraction of its usual brightness |
| New-lap jump | tape moving back up by more than this fraction of the wheel height = new lap |
| Time Crop / Parallel / FPS cap / live preview | unchanged |

The result shows what `flywheel_rotations.py` prints (frames, fps, flywheel missed, lap start frames, lap
periods, rotations from tape sightings, rotations including the estimate to the end), a lap table, a plot, and
the per-frame CSV (frame, time, flywheel detected/held, tape found/hidden, tape height, phase, rotations).
After a run the frame slider shows the same annotation as the annotated video of the Python script: flywheel
box (green = detected, orange = held), red line at the tape, dial, rotation counter.

Assumptions: side-on camera, one dark tape on the rim, a blue bracket crossing the wheel, wheel does not
reverse, and the wheel turns less than half a turn between frames. Detection runs on frames sub-sampled to
about 960 px wide (pixel thresholds of the script are scaled accordingly).
