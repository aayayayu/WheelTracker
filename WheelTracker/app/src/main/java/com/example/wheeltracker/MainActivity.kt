package com.example.wheeltracker

import android.content.ContentValues
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.webkit.*
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import java.net.Socket

class MainActivity : ComponentActivity() {

    private lateinit var web: WebView
    private var chooserCb: ValueCallback<Array<Uri>>? = null

    private val picker = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { res ->
        chooserCb?.onReceiveValue(
            WebChromeClient.FileChooserParams.parseResult(res.resultCode, res.data)
        )
        chooserCb = null
    }

    inner class Bridge {
        @JavascriptInterface
        fun saveCsv(text: String) {
            val v = ContentValues().apply {
                put(MediaStore.Downloads.DISPLAY_NAME, "rotation_data.csv")
                put(MediaStore.Downloads.MIME_TYPE, "text/csv")
                put(MediaStore.Downloads.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS)
            }
            val uri = contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, v)
            if (uri != null) {
                contentResolver.openOutputStream(uri)?.use { it.write(text.toByteArray()) }
                runOnUiThread {
                    Toast.makeText(this@MainActivity, "Saved to Downloads/rotation_data.csv",
                        Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        web = WebView(this)
        setContentView(web)
        web.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            allowContentAccess = true
            mediaPlaybackRequiresUserGesture = false
        }
        web.addJavascriptInterface(Bridge(), "Android")
        web.webViewClient = WebViewClient()
        web.webChromeClient = object : WebChromeClient() {
            override fun onShowFileChooser(
                view: WebView, cb: ValueCallback<Array<Uri>>,
                params: FileChooserParams
            ): Boolean {
                chooserCb?.onReceiveValue(null)
                chooserCb = cb
                return try {
                    picker.launch(params.createIntent()); true
                } catch (e: Exception) {
                    chooserCb = null; false
                }
            }
        }
        web.loadData("<body style='background:#181818;color:#eee;font:16px sans-serif;" +
            "padding:30px'>Starting engine… first launch can take ~20 s.</body>",
            "text/html", "utf-8")

        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (web.canGoBack()) web.goBack() else finish()
            }
        })

        // Start Python + Flask on a background thread
        if (!Python.isStarted()) Python.start(AndroidPlatform(this))
        Thread {
            try {
                Python.getInstance().getModule("main").callAttr("start")
            } catch (e: Exception) {
                runOnUiThread {
                    Toast.makeText(this, "Python error: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }.start()

        // Wait until Flask accepts connections, then load it
        Thread {
            repeat(120) {
                try {
                    Socket("127.0.0.1", 5000).close()
                    runOnUiThread { web.loadUrl("http://127.0.0.1:5000/") }
                    return@Thread
                } catch (_: Exception) {
                    Thread.sleep(500)
                }
            }
            runOnUiThread {
                Toast.makeText(this, "Server did not start (see Logcat)", Toast.LENGTH_LONG).show()
            }
        }.start()
    }

    override fun onDestroy() {
        val closing = isFinishing
        web.destroy()
        super.onDestroy()
        if (closing) {
            // App is being closed: delete the cached copy of the video so the cache does not pile up.
            // (On a background thread; it must never block the UI while Python is busy.)
            Thread {
                try { Python.getInstance().getModule("main").callAttr("cleanup") } catch (_: Exception) {}
            }.start()
        }
    }
}
