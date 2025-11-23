#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <jni.h>
#include <android/bitmap.h>

namespace ip {
    class ImageProcessor {
    public:
        static bool GrayScale(JNIEnv *env, jobject bitmap);

        static bool NegativeImage(JNIEnv *env, jobject bitmap);

        static bool BlurImage(JNIEnv *env, jobject bitmap, int radius, float sigma);

        static bool SharpenImage(JNIEnv *env, jobject bitmap);

        static bool EmbrossImage(JNIEnv *env, jobject bitmap);

        static bool EdgeDetection(JNIEnv *env, jobject bitmap);

        static void
        convert_yuv_rgba_scalar(const uint8_t *yPtr, const uint8_t *vPtr, const uint8_t *uPtr,
                                uint8_t *outrgba,
                                size_t width, size_t height,
                                size_t yStride, size_t yDstStride, size_t uRowStride,
                                size_t vRowStride,
                                size_t uPixelStride, size_t vPixelStride, int xStart = 0);

    private:
        static void gray_scale_scalar(void *pixelData, AndroidBitmapInfo &bitmapInfo);

        static void negative_scalar(void *pixelData, AndroidBitmapInfo &bitmapInfo);

        static void sharpen_scalar(void *pixels, AndroidBitmapInfo &bitmapInfo);

        static void emboss_scalar(void *pixels, AndroidBitmapInfo &bitmapInfo);

        static void
        gaussian_blur_scalar(void *pixels, AndroidBitmapInfo &bitmapInfo, int radius, float sigma);


        static uint8_t clamp255(int v) {
            if (v < 0) return 0;
            if (v > 255) return 255;
            return (uint8_t) v;
        }
    };

}
#endif