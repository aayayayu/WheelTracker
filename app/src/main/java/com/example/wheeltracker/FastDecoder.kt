package com.example.wheeltracker

import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.TimeUnit

/** One decoded frame: the luma (grayscale) values of the tracking box, in the video's raw orientation. */
class FdFrame(
    @JvmField val idx: Int,
    @JvmField val w: Int,
    @JvmField val h: Int,
    @JvmField val full: Boolean,
    @JvmField val rotation: Int,
    @JvmField val data: ByteArray,
)

/**
 * Decodes frames [startFrame, endFrame) with the phone's hardware decoder on a background thread
 * and queues only the luma of the tracking box (given in *displayed* coordinates).
 * Python polls it with poll() until isDone().
 */
class FdSession(
    private val path: String,
    private val startFrame: Int,
    private val endFrame: Int,
    private val fps: Double,
    private val dx0: Int, private val dy0: Int, private val dx1: Int, private val dy1: Int,
) {
    private val queue = ArrayBlockingQueue<FdFrame>(32)
    @Volatile private var finished = false
    @Volatile private var stopFlag = false
    @Volatile private var err: String? = null

    private val worker = Thread { run() }.apply { isDaemon = true; start() }

    fun poll(): FdFrame? = queue.poll()
    fun isDone(): Boolean = finished && queue.isEmpty()
    fun errorMessage(): String? = err
    fun close() { stopFlag = true }

    private fun offer(f: FdFrame): Boolean {
        while (!stopFlag) {
            if (queue.offer(f, 100, TimeUnit.MILLISECONDS)) return true
        }
        return false
    }

    /** Displayed-orientation box -> raw (unrotated) box, clamped. Returns [x0, y0, x1, y1]. */
    private fun rawRect(rw: Int, rh: Int, rotation: Int): IntArray {
        val dispW = if (rotation == 90 || rotation == 270) rh else rw
        val dispH = if (rotation == 90 || rotation == 270) rw else rh
        val ax0 = dx0.coerceIn(0, dispW); val ax1 = dx1.coerceIn(0, dispW)
        val ay0 = dy0.coerceIn(0, dispH); val ay1 = dy1.coerceIn(0, dispH)
        return when (rotation) {
            90 -> intArrayOf(ay0, rh - ax1, ay1, rh - ax0)
            180 -> intArrayOf(rw - ax1, rh - ay1, rw - ax0, rh - ay0)
            270 -> intArrayOf(rw - ay1, ax0, rw - ay0, ax1)
            else -> intArrayOf(ax0, ay0, ax1, ay1)
        }
    }

    private fun run() {
        val ex = MediaExtractor()
        var codec: MediaCodec? = null
        try {
            ex.setDataSource(path)
            var track = -1
            var fmt: MediaFormat? = null
            for (i in 0 until ex.trackCount) {
                val f = ex.getTrackFormat(i)
                if ((f.getString(MediaFormat.KEY_MIME) ?: "").startsWith("video/")) {
                    track = i; fmt = f; break
                }
            }
            if (track < 0 || fmt == null) throw IllegalStateException("no video track")
            ex.selectTrack(track)
            val mime = fmt.getString(MediaFormat.KEY_MIME)!!
            val rotation = if (fmt.containsKey(MediaFormat.KEY_ROTATION))
                ((fmt.getInteger(MediaFormat.KEY_ROTATION) % 360) + 360) % 360 else 0
            var fullRange = fmt.containsKey(MediaFormat.KEY_COLOR_RANGE) &&
                fmt.getInteger(MediaFormat.KEY_COLOR_RANGE) == MediaFormat.COLOR_RANGE_FULL

            // presentation time of the first frame (usually 0)
            ex.seekTo(0, MediaExtractor.SEEK_TO_PREVIOUS_SYNC)
            val baseUs = maxOf(0L, ex.sampleTime)

            fmt.setInteger(
                MediaFormat.KEY_COLOR_FORMAT,
                MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible
            )
            val c = MediaCodec.createDecoderByType(mime)
            codec = c
            c.configure(fmt, null, null, 0)
            c.start()

            val frameUs = 1_000_000.0 / fps
            val seekUs = maxOf(0L, baseUs + ((startFrame - 0.5) * frameUs).toLong())
            ex.seekTo(seekUs, MediaExtractor.SEEK_TO_PREVIOUS_SYNC)

            val info = MediaCodec.BufferInfo()
            var inputDone = false
            var outputDone = false
            var rect: IntArray? = null
            var rectW = -1
            var rectH = -1

            while (!outputDone && !stopFlag) {
                if (!inputDone) {
                    val ii = c.dequeueInputBuffer(5_000)
                    if (ii >= 0) {
                        val buf = c.getInputBuffer(ii)!!
                        val n = ex.readSampleData(buf, 0)
                        if (n < 0) {
                            c.queueInputBuffer(ii, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            c.queueInputBuffer(ii, 0, n, ex.sampleTime, 0)
                            ex.advance()
                        }
                    }
                }

                val oi = c.dequeueOutputBuffer(info, 5_000)
                if (oi >= 0) {
                    val eos = (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0
                    val idx = Math.round((info.presentationTimeUs - baseUs) / frameUs).toInt()
                    if (idx >= endFrame) {
                        outputDone = true
                    } else if (idx >= startFrame && info.size > 0) {
                        val frame = extract(c, oi, info, idx, rotation, fullRange,
                            rect, rectW, rectH)
                        if (frame != null) {
                            rect = frame.second; rectW = frame.third; rectH = frame.fourth
                            if (!offer(frame.first)) outputDone = true
                        }
                    }
                    c.releaseOutputBuffer(oi, false)
                    if (eos) outputDone = true
                } else if (oi == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    val of = c.outputFormat
                    if (of.containsKey(MediaFormat.KEY_COLOR_RANGE)) {
                        fullRange = of.getInteger(MediaFormat.KEY_COLOR_RANGE) ==
                            MediaFormat.COLOR_RANGE_FULL
                    }
                    rect = null
                }
            }
        } catch (t: Throwable) {
            err = "${t.javaClass.simpleName}: ${t.message}"
        } finally {
            try { codec?.stop() } catch (_: Throwable) {}
            try { codec?.release() } catch (_: Throwable) {}
            try { ex.release() } catch (_: Throwable) {}
            finished = true
        }
    }

    private class Quad<A, B, C, D>(val first: A, val second: B, val third: C, val fourth: D)

    /** Copies the luma of the tracking box out of one decoder output buffer. */
    private fun extract(
        c: MediaCodec, oi: Int, info: MediaCodec.BufferInfo, idx: Int, rotation: Int,
        fullRange: Boolean, cachedRect: IntArray?, cachedW: Int, cachedH: Int,
    ): Quad<FdFrame, IntArray, Int, Int>? {
        val img = try { c.getOutputImage(oi) } catch (_: Throwable) { null }
        if (img != null) {
            try {
                val crop = img.cropRect
                val rw = crop.width()
                val rh = crop.height()
                val r = if (cachedRect != null && cachedW == rw && cachedH == rh) cachedRect
                        else rawRect(rw, rh, rotation)
                val cw = r[2] - r[0]
                val ch = r[3] - r[1]
                if (cw <= 0 || ch <= 0) return null
                val plane = img.planes[0]
                val bb = plane.buffer
                val rs = plane.rowStride
                val ps = plane.pixelStride
                val out = ByteArray(cw * ch)
                for (row in 0 until ch) {
                    val base = (crop.top + r[1] + row) * rs + (crop.left + r[0]) * ps
                    if (ps == 1) {
                        bb.position(base)
                        bb.get(out, row * cw, cw)
                    } else {
                        // e.g. 16-bit (10-bit HDR) luma: keep the high byte
                        for (col in 0 until cw) out[row * cw + col] = bb.get(base + col * ps + (ps - 1))
                    }
                }
                return Quad(FdFrame(idx, cw, ch, fullRange, rotation, out), r, rw, rh)
            } finally {
                img.close()
            }
        }

        // Fallback: raw byte buffer (Y plane first, then chroma)
        val of = c.outputFormat
        val w0 = of.getInteger(MediaFormat.KEY_WIDTH)
        val h0 = of.getInteger(MediaFormat.KEY_HEIGHT)
        val cl = if (of.containsKey("crop-left")) of.getInteger("crop-left") else 0
        val ct = if (of.containsKey("crop-top")) of.getInteger("crop-top") else 0
        val cr = if (of.containsKey("crop-right")) of.getInteger("crop-right") else w0 - 1
        val cb = if (of.containsKey("crop-bottom")) of.getInteger("crop-bottom") else h0 - 1
        val rw = cr - cl + 1
        val rh = cb - ct + 1
        val rs = if (of.containsKey("stride")) of.getInteger("stride") else w0
        val ob = c.getOutputBuffer(oi) ?: return null
        val r = if (cachedRect != null && cachedW == rw && cachedH == rh) cachedRect
                else rawRect(rw, rh, rotation)
        val cw = r[2] - r[0]
        val ch = r[3] - r[1]
        if (cw <= 0 || ch <= 0) return null
        val out = ByteArray(cw * ch)
        for (row in 0 until ch) {
            ob.position(info.offset + (ct + r[1] + row) * rs + cl + r[0])
            ob.get(out, row * cw, cw)
        }
        return Quad(FdFrame(idx, cw, ch, fullRange, rotation, out), r, rw, rh)
    }
}

object FastDecoder {
    /** Starts decoding [startFrame, endFrame); the box is x0,y0,x1,y1 in displayed pixels. */
    @JvmStatic
    fun open(
        path: String, startFrame: Int, endFrame: Int, fps: Double,
        x0: Int, y0: Int, x1: Int, y1: Int,
    ): FdSession = FdSession(path, startFrame, endFrame, fps, x0, y0, x1, y1)
}
