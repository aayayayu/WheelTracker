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
