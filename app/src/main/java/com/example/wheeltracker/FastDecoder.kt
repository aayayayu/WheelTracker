package com.example.wheeltracker

import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.TimeUnit

/**
 * One decoded frame: the WHOLE frame, sub-sampled by `step` in both directions (nearest pixel) and
 * converted to interleaved BGR (w * h * 3 bytes, row-major), in the video's raw orientation.
 * Python rotates it to the displayed orientation with [rotation].
 *
 * The side-view flywheel detector needs colour (blue bracket, gray wheel, wall brightness), which is
 * why this is no longer just the luma of a tracking box.
 */
class FdFrame(
    @JvmField val idx: Int,
    @JvmField val w: Int,
    @JvmField val h: Int,
    @JvmField val rotation: Int,
    @JvmField val data: ByteArray,
    @JvmField val light: Boolean = false,   // true: data = grayscale of the region set with setRoi (w x h)
)

/**
 * Decodes frames [startFrame, endFrame) with the phone's hardware decoder on a background thread
 * and queues them as sub-sampled BGR pictures. Python polls it with poll() until isDone().
 */
class FdSession(
    private val path: String,
    private val startFrame: Int,
    private val endFrame: Int,
    private val fps: Double,
    private val step: Int,           // keep every step-th pixel in x and y (1 = full resolution)
    private val every: Int,          // >1: only every `every`-th frame is delivered whole, the others as a gray region
) {
    private val queue = ArrayBlockingQueue<FdFrame>(64)
    @Volatile private var finished = false
    @Volatile private var stopFlag = false
    @Volatile private var err: String? = null
    @Volatile private var roi: IntArray? = null     // x0, y0, x1, y1 on the sub-sampled grid, raw orientation

    private val worker = Thread { run() }.apply { isDaemon = true; start() }

    fun poll(): FdFrame? = queue.poll()
    fun pollWait(ms: Long): FdFrame? = queue.poll(ms, TimeUnit.MILLISECONDS)
    fun isDone(): Boolean = finished && queue.isEmpty()
    fun errorMessage(): String? = err
    fun close() { stopFlag = true }
    fun setRoi(x0: Int, y0: Int, x1: Int, y1: Int) { roi = intArrayOf(x0, y0, x1, y1) }

    private fun offer(f: FdFrame): Boolean {
        while (!stopFlag) {
            if (queue.offer(f, 100, TimeUnit.MILLISECONDS)) return true
        }
        return false
    }

    /** Colour conversion constants (x256), chosen from the stream's range / colour standard. */
    private class Yuv(val off: Int, val cy: Int, val crv: Int, val cgu: Int, val cgv: Int, val cbu: Int)

    private fun yuvFor(fullRange: Boolean, bt709: Boolean): Yuv = when {
        !fullRange && bt709 -> Yuv(16, 298, 459, 55, 136, 541)
        !fullRange -> Yuv(16, 298, 409, 100, 208, 516)
        bt709 -> Yuv(0, 256, 403, 48, 120, 475)
        else -> Yuv(0, 256, 359, 88, 183, 454)
    }

    private fun isBt709(fmt: MediaFormat, rawLongSide: Int): Boolean {
        if (fmt.containsKey(MediaFormat.KEY_COLOR_STANDARD)) {
            return when (fmt.getInteger(MediaFormat.KEY_COLOR_STANDARD)) {
                MediaFormat.COLOR_STANDARD_BT601_PAL, MediaFormat.COLOR_STANDARD_BT601_NTSC -> false
                else -> true
            }
        }
        return rawLongSide >= 1280          // no information: HD and larger is BT.709
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
            val longSide = maxOf(
                if (fmt.containsKey(MediaFormat.KEY_WIDTH)) fmt.getInteger(MediaFormat.KEY_WIDTH) else 0,
                if (fmt.containsKey(MediaFormat.KEY_HEIGHT)) fmt.getInteger(MediaFormat.KEY_HEIGHT) else 0
            )
            var bt709 = isBt709(fmt, longSide)

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
            var delivered = 0

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
                        val k = yuvFor(fullRange, bt709)
                        val r = roi
                        val frame = if (every > 1 && r != null && delivered > 0 && idx % every != 0)
                            extractLight(c, oi, idx, rotation, k, r)
                        else extract(c, oi, idx, rotation, k)
                        delivered++
                        if (!offer(frame)) outputDone = true
                    }
                    c.releaseOutputBuffer(oi, false)
                    if (eos) outputDone = true
                } else if (oi == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    val of = c.outputFormat
                    if (of.containsKey(MediaFormat.KEY_COLOR_RANGE)) {
                        fullRange = of.getInteger(MediaFormat.KEY_COLOR_RANGE) ==
                            MediaFormat.COLOR_RANGE_FULL
                    }
                    if (of.containsKey(MediaFormat.KEY_COLOR_STANDARD)) {
                        bt709 = isBt709(of, longSide)
                    }
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

    private fun clamp(v: Int): Int = if (v < 0) 0 else if (v > 255) 255 else v

    /** Fast path: only the luma of region r (sub-sampled grid), scaled like the BGR conversion does. */
    private fun extractLight(c: MediaCodec, oi: Int, idx: Int, rotation: Int, k: Yuv, r: IntArray): FdFrame {
        val img = c.getOutputImage(oi)
            ?: throw IllegalStateException("decoder gives no Image output")
        try {
            val crop = img.cropRect
            val ow = crop.width() / step
            val oh = crop.height() / step
            val x0 = maxOf(0, r[0]); val y0 = maxOf(0, r[1])
            val x1 = minOf(ow, r[2]); val y1 = minOf(oh, r[3])
            val w = x1 - x0
            val h = y1 - y0
            if (w <= 0 || h <= 0) throw IllegalStateException("empty region")
            val yP = img.planes[0]
            val yb = yP.buffer
            val yrs = yP.rowStride
            val yps = yP.pixelStride
            val hi = if (yps > 1) 1 else 0
            val yCol = IntArray(w) { it * step * yps + hi }
            val rowBuf = ByteArray(yCol[w - 1] + 1)
            val out = ByteArray(w * h)
            for (row in 0 until h) {
                val sy = crop.top + (y0 + row) * step
                yb.position(sy * yrs + (crop.left + x0 * step) * yps)
                yb.get(rowBuf, 0, minOf(rowBuf.size, yb.remaining()))
                val o = row * w
                for (col in 0 until w) {
                    val y = rowBuf[yCol[col]].toInt() and 0xFF
                    out[o + col] = clamp(((y - k.off) * k.cy) shr 8).toByte()
                }
            }
            return FdFrame(idx, w, h, rotation, out, true)
        } finally {
            img.close()
        }
    }

    /** Reads one decoder output picture, sub-samples it by `step` and converts it to BGR. */
    private fun extract(c: MediaCodec, oi: Int, idx: Int, rotation: Int, k: Yuv): FdFrame {
        val img = c.getOutputImage(oi)
            ?: throw IllegalStateException("decoder gives no Image output")
        try {
            val crop = img.cropRect
            val rw = crop.width()
            val rh = crop.height()
            val ow = rw / step
            val oh = rh / step
            if (ow <= 0 || oh <= 0) throw IllegalStateException("empty frame")

            val yP = img.planes[0]
            val uP = img.planes[1]
            val vP = img.planes[2]
            val yb = yP.buffer
            val ub = uP.buffer
            val vb = vP.buffer
            val yrs = yP.rowStride
            val yps = yP.pixelStride
            val urs = uP.rowStride
            val ups = uP.pixelStride
            val vrs = vP.rowStride
            val vps = vP.pixelStride
            // 16-bit (10-bit HDR) samples: keep the high byte
            val hi = if (yps > 1) 1 else 0

            val out = ByteArray(ow * oh * 3)
            // Column offsets are the same for every row: compute once. Rows are then copied out of
            // the (slow, bounds-checked) ByteBuffers with ONE bulk get per row instead of 3 per pixel.
            val yCol = IntArray(ow) { it * step * yps + hi }
            val cCol0 = crop.left shr 1
            val uCol = IntArray(ow) { (((crop.left + it * step) shr 1) - cCol0) * ups + hi }
            val vCol = IntArray(ow) { (((crop.left + it * step) shr 1) - cCol0) * vps + hi }
            val yRowBuf = ByteArray(yCol[ow - 1] + 1)
            val uRowBuf = ByteArray(uCol[ow - 1] + 1)
            val vRowBuf = ByteArray(vCol[ow - 1] + 1)
            var lastC = -1
            var o = 0
            for (row in 0 until oh) {
                val sy = crop.top + row * step
                yb.position(sy * yrs + crop.left * yps)
                yb.get(yRowBuf, 0, minOf(yRowBuf.size, yb.remaining()))
                val cy = sy shr 1
                if (cy != lastC) {
                    ub.position(cy * urs + cCol0 * ups)
                    ub.get(uRowBuf, 0, minOf(uRowBuf.size, ub.remaining()))
                    vb.position(cy * vrs + cCol0 * vps)
                    vb.get(vRowBuf, 0, minOf(vRowBuf.size, vb.remaining()))
                    lastC = cy
                }
                for (col in 0 until ow) {
                    val y = yRowBuf[yCol[col]].toInt() and 0xFF
                    val u = (uRowBuf[uCol[col]].toInt() and 0xFF) - 128
                    val v = (vRowBuf[vCol[col]].toInt() and 0xFF) - 128
                    val yy = (y - k.off) * k.cy
                    out[o] = clamp((yy + k.cbu * u) shr 8).toByte()                  // B
                    out[o + 1] = clamp((yy - k.cgu * u - k.cgv * v) shr 8).toByte()  // G
                    out[o + 2] = clamp((yy + k.crv * v) shr 8).toByte()              // R
                    o += 3
                }
            }
            return FdFrame(idx, ow, oh, rotation, out)
        } finally {
            img.close()
        }
    }
}

object FastDecoder {
    /**
     * Starts decoding [startFrame, endFrame). Every frame is delivered whole, sub-sampled by `step`
     * (1 = full resolution) and converted to BGR.
     */
    @JvmStatic
    fun open(path: String, startFrame: Int, endFrame: Int, fps: Double, step: Int, every: Int): FdSession =
        FdSession(path, startFrame, endFrame, fps, maxOf(1, step), maxOf(1, every))
}
