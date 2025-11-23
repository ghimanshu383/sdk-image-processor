package com.os.imageprocessor

import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.MediaMetadataRetriever.BitmapParams
import android.os.Bundle
import android.util.Log
import android.widget.ImageView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.camera2.internal.annotation.CameraExecutor
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.databinding.DataBindingUtil
import com.os.image.processing.sdk.R
import com.os.image.processing.sdk.databinding.ActivityCameraFrameCaptureBinding
import java.nio.ByteBuffer
import java.security.Permission
import java.security.Permissions
import java.util.concurrent.Executor
import java.util.concurrent.Executors

class CameraFrameCapture : AppCompatActivity() {
    private lateinit var previewView: PreviewView
    private val cameraExecutor = Executors.newSingleThreadExecutor()
    lateinit var binding: ActivityCameraFrameCaptureBinding
    lateinit var imageView: ImageView
    var viewBitMap: Bitmap? = null

    private var nv21Buffer: ByteArray? = null
    private var outputBitmap: Bitmap? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = DataBindingUtil.setContentView(this, R.layout.activity_camera_frame_capture)
        previewView = binding.preview
        imageView = binding.ImageView

        if (ContextCompat.checkSelfPermission(
                applicationContext,
                android.Manifest.permission.CAMERA
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), 101)
        } else {
            startCamera();
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        outputBitmap?.recycle()
        outputBitmap = null
    }

    private fun startCamera() {
        val cameraFutureProvider = ProcessCameraProvider.getInstance(this)
        cameraFutureProvider.addListener({
            val cameraProvider = cameraFutureProvider.get()

//            val preview = Preview.Builder().build().also {
//                it.setSurfaceProvider(previewView.surfaceProvider)
//            }
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageRotationEnabled(true)
                .build();

            analysis.setAnalyzer(Executors.newSingleThreadExecutor()) { imageProxy ->
                handleFrames(imageProxy)
                imageProxy.close()
            }

            val selector = CameraSelector.DEFAULT_BACK_CAMERA
            cameraProvider.unbindAll()

            cameraProvider.bindToLifecycle(this, selector, analysis)

        }, ContextCompat.getMainExecutor(this))
    }

    private fun handleFrames(image: ImageProxy) {
        val w = image.width
        val h = image.height

        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer

        val yStride = yPlane.rowStride
        val uRowStride = uPlane.rowStride
        val vRowStride = vPlane.rowStride
        val uPixelStride = uPlane.pixelStride
        val vPixelStride = vPlane.pixelStride

        val yArray = ByteArray(yBuffer.remaining()).also { yBuffer.get(it) }
        val uArray = ByteArray(uBuffer.remaining()).also { uBuffer.get(it) }
        val vArray = ByteArray(vBuffer.remaining()).also { vBuffer.get(it) }

        if (viewBitMap == null || viewBitMap?.width != w || viewBitMap?.height != h) {
            viewBitMap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        }


        JniBridge.convert_yuv_rgba(
            yArray,
            vArray,
            uArray,
            viewBitMap,
            w,
            h,
            yStride,
            w * 4,
            uRowStride,
            vRowStride,
            uPixelStride,
            vPixelStride,
            false
        );

        runOnUiThread {
            imageView.setImageBitmap(viewBitMap)
        }

    }
}