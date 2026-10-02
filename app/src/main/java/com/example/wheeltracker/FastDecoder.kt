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
) {
    private val queue = ArrayBlockingQueue<FdFrame>(16)
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
                        val frame = extract(c, oi, idx, rotation, yuvFor(fullRange, bt709))
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
            var o = 0
            for (row in 0 until oh) {
                val sy = crop.top + row * step
                val yRow = sy * yrs
                val cRowU = (sy shr 1) * urs
                val cRowV = (sy shr 1) * vrs
                for (col in 0 until ow) {
                    val sx = crop.left + col * step
                    val y = yb.get(yRow + sx * yps + hi).toInt() and 0xFF
                    val u = (ub.get(cRowU + (sx shr 1) * ups + hi).toInt() and 0xFF) - 128
                    val v = (vb.get(cRowV + (sx shr 1) * vps + hi).toInt() and 0xFF) - 128
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
    fun open(path: String, startFrame: Int, endFrame: Int, fps: Double, step: Int): FdSession =
        FdSession(path, startFrame, endFrame, fps, maxOf(1, step))
}
