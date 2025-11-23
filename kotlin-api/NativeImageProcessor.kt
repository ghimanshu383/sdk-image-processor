package com.os.imageprocessor

import android.content.Context
import android.graphics.Bitmap
import android.widget.ImageView
import android.widget.Toast
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class NativeImageProcessor {
    companion object {
        enum class PROCESS_TYPE {
            GRAY,
            NEGATIVE,
            Blur,
            SHARPEN,
            EMBOSS,
            SOBEL_EDGE
        }

        suspend fun processImage(
            context: Context,
            bitmap: Bitmap,
            process: PROCESS_TYPE,
            optimizeNeon: Boolean,
            radius: Int = 3,
            sigma: Int = 5
        ): Bitmap = withContext(Dispatchers.Default) {
            val mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true)
            val result = when (process) {
                PROCESS_TYPE.GRAY -> JniBridge.GrayScaleImage(mutable, optimizeNeon)
                PROCESS_TYPE.NEGATIVE -> JniBridge.CreateNegative(mutable, optimizeNeon)
                PROCESS_TYPE.Blur -> JniBridge.BlurImage(
                    mutable,
                    radius,
                    sigma,
                    optimizeNeon
                )

                PROCESS_TYPE.SHARPEN -> JniBridge.Sharpen(mutable, optimizeNeon)
                PROCESS_TYPE.EMBOSS -> JniBridge.Embross(mutable, optimizeNeon)
                PROCESS_TYPE.SOBEL_EDGE -> JniBridge.EdgeDetection(mutable, optimizeNeon)
            }
            if (!result) Toast.makeText(context, "Image processing Failed", Toast.LENGTH_LONG)
                .show()
            return@withContext mutable
        }
    }
}