
#include <android/log.h>


#include "ImageProcessor.h"
#include "ImageProcessorSIMD.h"
#include "Utility.h"


#define LOG_TAG "core_native_image"
#define LOG_INFO(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOG_ERROR(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace ip {
    bool ImageProcessor::GrayScale(JNIEnv *env, jobject bitmap) {
        AndroidBitmapInfo bitmapInfo;
        void *pixels = nullptr;

        if (AndroidBitmap_getInfo(env, bitmap, &bitmapInfo) < 0) {
            LOG_ERROR("Failed to get the android bit map info");
            return false;
        }
        if (bitmapInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
            LOG_ERROR("Invalid Non Supported Bit format %d map should be RGBA_8888",
                      bitmapInfo.format);
            return false;

        }
        // get the pixel access
        if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) {
            LOG_ERROR("Locking the android bit map pixels failed");
            return false;
        }
        if (ImageProcessorSIMD::device_support_neon()) {
            LOG_INFO("Device Support NEON");
            uint8_t *outData = new uint8_t[bitmapInfo.height * bitmapInfo.width * 4];
            ImageProcessorSIMD::gray_scale_neon_simd(reinterpret_cast<uint8_t *>(pixels),
                                                     outData,
                                                     bitmapInfo.width * bitmapInfo.height);
            memcpy(pixels, outData, bitmapInfo.height * bitmapInfo.width * 4);
            delete[] outData;
        } else {
            gray_scale_scalar(pixels, bitmapInfo);
        }
        AndroidBitmap_unlockPixels(env, bitmap);
        return true;
    }

    void ImageProcessor::gray_scale_scalar(void *pixels, AndroidBitmapInfo &bitmapInfo) {
        const int width = bitmapInfo.width;
        const int height = bitmapInfo.height;
        const int stride = bitmapInfo.stride;

        uint8_t *base = reinterpret_cast<uint8_t *>(pixels);
        for (int y = 0; y < height; y++) {
            uint32_t *row = reinterpret_cast<uint32_t *>(base + (size_t) y * (size_t) stride);
            for (int x = 0; x < width; x++) {
                uint32_t color = row[x];
                // The bit map is RGBA_8888 so each entry is 32 bits and can be extracted for 8 bits each
                uint8_t a = (color >> 24) & 0xFF;
                uint8_t b = (color >> 16) & 0xFF;
                uint8_t g = (color >> 8) & 0xFf;
                uint8_t r = color & 0xFF;

                int gray = (77 * r + 150 * g + 29 * b + 128) >> 8;
                if (gray < 0) gray = 0;
                if (gray > 255) gray = 255;

                row[x] = (static_cast<uint32_t>(a) << 24) |
                         (static_cast<uint32_t>(gray) << 16) |
                         (static_cast<uint32_t>(gray) << 8) |
                         (static_cast<uint32_t>(gray));
            }
        }
    }

    bool ImageProcessor::NegativeImage(JNIEnv *env, jobject bitmap) {
        AndroidBitmapInfo bitmapInfo;
        void *pixels = nullptr;
        if (AndroidBitmap_getInfo(env, bitmap, &bitmapInfo) < 0) {
            LOG_ERROR("Failed to get the bit map info");
            return false;
        }
        if (bitmapInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
            LOG_ERROR("Un-Supported Bit map format");
            return false;
        }
        if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) {
            LOG_ERROR("Failed to lock the bitmap pixels");
            return false;
        }

        if (ImageProcessorSIMD::device_support_neon()) {
            LOG_INFO("Device Support NEON");
            uint8_t *outData = new uint8_t[bitmapInfo.width * bitmapInfo.height * 4];
            ImageProcessorSIMD::negative_neon_simd(reinterpret_cast<uint8_t *>(pixels), outData,
                                                   bitmapInfo.height * bitmapInfo.width);
            memcpy(pixels, outData, bitmapInfo.width * bitmapInfo.height * 4);
            delete[] outData;
        } else {
            negative_scalar(pixels, bitmapInfo);
        }
        AndroidBitmap_unlockPixels(env, bitmap);
        return true;
    }

    void ImageProcessor::negative_scalar(void *pixels, AndroidBitmapInfo &bitmapInfo) {
        int width = bitmapInfo.width;
        int height = bitmapInfo.height;
        int stride = bitmapInfo.stride;

        uint8_t *base = reinterpret_cast<uint8_t *>(pixels);
        for (int y = 0; y < height; y++) {
            uint32_t *row = reinterpret_cast<uint32_t *>(base + (size_t) y * (size_t) stride);
            for (int x = 0; x < width; x++) {
                uint32_t color = row[x];
                uint8_t a = (color >> 24) & 0xFF;
                uint8_t b = (color >> 16) & 0xFF;
                uint8_t g = (color >> 8) & 0xFf;
                uint8_t r = color & 0xFF;

                row[x] = (static_cast<uint32_t>(a) << 24) |
                         (static_cast<uint32_t>(255 - r) << 16) |
                         (static_cast<uint32_t>(255 - g) << 8) |
                         (static_cast<uint32_t>(255 - b));
            }
        }
    }

    bool ImageProcessor::BlurImage(JNIEnv *env, jobject bitmap, int radius, float sigma) {
        AndroidBitmapInfo bitmapInfo;
        void *pixels = nullptr;
        if (AndroidBitmap_getInfo(env, bitmap, &bitmapInfo) < 0) {
            LOG_ERROR("Failed to get the bit map info");
            return false;
        }
        if (bitmapInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
            LOG_ERROR("Invalid Bit map format");
            return false;
        }
        if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) {
            LOG_ERROR("Failed to lock the bitmap info");
            return false;
        }

        if (ImageProcessorSIMD::device_support_neon()) {
            uint8_t *dst = new uint8_t[bitmapInfo.width * bitmapInfo.height * 4];
            uint8_t *safeSrc = new uint8_t[bitmapInfo.width * bitmapInfo.height * 4];
            memcpy(safeSrc, pixels, bitmapInfo.width * bitmapInfo.height * 4);
            ImageProcessorSIMD::blur_neon_simd_float(safeSrc, dst,
                                                     bitmapInfo.width, bitmapInfo.height,
                                                     bitmapInfo.stride, radius, sigma);
            memcpy(pixels, dst, bitmapInfo.height * bitmapInfo.width * 4);
            delete[] dst;
            delete[] safeSrc;
        } else {
            gaussian_blur_scalar(pixels, bitmapInfo, radius, sigma);
        }
        AndroidBitmap_unlockPixels(env, bitmap);

        return true;
    }

    void
    ImageProcessor::gaussian_blur_scalar(void *pixels, AndroidBitmapInfo &bitmapInfo, int radius,
                                         float sigma) {
        uint32_t *src = reinterpret_cast<uint32_t *>(pixels);
        uint32_t width = bitmapInfo.width;
        uint32_t height = bitmapInfo.height;
        uint32_t stride = bitmapInfo.stride / 4;


        std::vector<std::vector<float>> kernel = Utility::generate_gaussian_kernel(radius, sigma);
        std::vector<uint32_t> output(width * height);

        for (int y = radius; y < height - radius; y++) {
            for (int x = radius; x < width - radius; x++) {
                float r = 0, g = 0, b = 0.0f;
                for (int ky = -radius; ky <= radius; ky++) {
                    for (int kx = -radius; kx <= radius; kx++) {
                        float weight = kernel[ky + radius][kx + radius];
                        uint32_t color = src[(y + ky) * stride + (x + kx)];
                        b += (float) ((color >> 16) & 0xFF) * weight;
                        g += (float) ((color >> 8) & 0xFF) * weight;
                        r += (float) (color & 0xFF) * weight;
                    }
                }
                uint8_t a = (src[y * stride + x] >> 24) & 0xFF;
                uint32_t newColor = a << 24 |
                                    ((uint8_t) b << 16) | ((uint8_t) g << 8) | ((uint8_t) r);
                output[y * stride + x] = newColor;
            }
        }

        memcpy(src, output.data(), height * width * sizeof(uint32_t));
    }

    bool ImageProcessor::SharpenImage(JNIEnv *env, jobject bitmap) {
        AndroidBitmapInfo bitmapInfo;
        void *pixels = nullptr;
        if (AndroidBitmap_getInfo(env, bitmap, &bitmapInfo) < 0) {
            LOG_ERROR("Failed to get the android bitmap info");
            return false;
        }
        if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) {
            LOG_ERROR("Failed to lock the bitmap pixels");
            return false;
        }
        if (ImageProcessorSIMD::device_support_neon()) {
            uint8_t *outData = new uint8_t[bitmapInfo.height * bitmapInfo.width * 4];
            ImageProcessorSIMD::sharp_neon_simd(reinterpret_cast<uint8_t *>(pixels), outData,
                                                bitmapInfo.width, bitmapInfo.height,
                                                bitmapInfo.stride);
            memcpy(pixels, outData, bitmapInfo.width * bitmapInfo.height * 4);
            delete[] outData;
        } else {
            sharpen_scalar(pixels, bitmapInfo);
        }
        AndroidBitmap_unlockPixels(env, bitmap);
        return true;
    }


    void ImageProcessor::sharpen_scalar(void *pixels, AndroidBitmapInfo &bitmapInfo) {
        uint32_t *src = reinterpret_cast<uint32_t *>(pixels);
        std::vector<std::vector<float>> kernel = {
                {-1, -1, -1},
                {-1, 9,  -1},
                {-1, -1, -1}
        };
        uint32_t width = bitmapInfo.width;
        uint32_t height = bitmapInfo.height;
        uint32_t stride = bitmapInfo.stride / 4;
        std::vector<std::uint32_t> output(height * width);

        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                float r = 0, g = 0, b = 0;
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        float weight = kernel[ky + 1][kx + 1];
                        uint32_t color = src[(y + ky) * stride + kx + x];
                        b += (float) ((color >> 16) & 0xFF) * weight;
                        g += (float) ((color >> 8) & 0xFF) * weight;
                        r += (float) (color & 0xFF) * weight;
                    }
                }
                r = std::min(255.0f, std::max(0.0f, r));
                g = std::min(255.0f, std::max(0.0f, g));
                b = std::min(255.0f, std::max(0.0f, b));

                uint32_t a = (src[y * stride + x] >> 24) & 0xFF;
                output[y * stride + x] =
                        a << 24 | (uint32_t) b << 16 | (uint32_t) g << 8 | (uint32_t) r;
            }
        }
        memcpy(src, output.data(), sizeof(uint32_t) * height * width);
    }

    bool ImageProcessor::EmbrossImage(JNIEnv *env, jobject bitmap) {
        AndroidBitmapInfo info;
        void *pixelData = nullptr;
        if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) {
            LOG_INFO("Failed to get the bitmap info");
            return false;
        }
        if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
            LOG_INFO("Invalid bit map format ");
            return false;
        }
        if (AndroidBitmap_lockPixels(env, bitmap, &pixelData) < 0) {
            LOG_INFO("Failed to lock the pixel information");
            return false;
        }
        if (ImageProcessorSIMD::device_support_neon()) {
            uint8_t *dst = new uint8_t[info.width * info.height * 4];
            ImageProcessorSIMD::emboss_neon_simd_float(reinterpret_cast<uint8_t *>(pixelData), dst,
                                                       info.width, info.height, info.stride);
            memcpy(pixelData, dst, info.width * info.height * 4);
            delete[] dst;
        } else {
            emboss_scalar(pixelData, info);
        }
        AndroidBitmap_unlockPixels(env, bitmap);
        return true;
    }


    void ImageProcessor::emboss_scalar(void *pixelData, AndroidBitmapInfo &info) {
        std::vector<std::vector<int32_t>> kernel = {
                {-2, -1, 0},
                {-1, 1,  1},
                {0,  1,  2}
        };
        uint32_t width = info.width;
        uint32_t height = info.height;
        uint32_t stride = info.stride / 4;
        std::vector<std::uint32_t> output(height * width);

        uint32_t *src = reinterpret_cast<uint32_t *>(pixelData);
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                float r = 0, g = 0, b = 0;

                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        uint32_t color = src[(y + ky) * stride + (x + kx)];
                        uint8_t blue = (color >> 16) & 0xFF;
                        uint8_t green = (color >> 8) & 0xFF;
                        uint8_t red = color & 0xFF;
                        uint8_t gray = static_cast<uint8_t>(0.3f * red + 0.59f * green +
                                                            0.11f * blue);
                        int32_t weight = kernel[ky + 1][kx + 1];
                        r += gray * weight;
                        g += gray * weight;
                        b += gray * weight;
                    }
                }

                r = std::clamp(r + 128.0f, 0.0f, 255.0f);
                g = std::clamp(g + 128.0f, 0.0f, 255.0f);
                b = std::clamp(b + 128.0f, 0.0f, 255.0f);

                uint32_t a = (src[y * stride + x] >> 24) & 0xFF;
                output[y * width + x] = (a << 24) |
                                        ((uint32_t) b << 16) |
                                        ((uint32_t) g << 8) |
                                        ((uint32_t) r);
            }
        }

        // Copy back respecting stride
        uint8_t *dstBytes = reinterpret_cast<uint8_t *>(pixelData);
        for (uint32_t y = 0; y < height; y++) {
            memcpy(dstBytes + y * info.stride,
                   output.data() + y * width,
                   width * sizeof(uint32_t));
        }
    }

    void ImageProcessor::convert_yuv_rgba_scalar(const uint8_t *yPtr, const uint8_t *vPtr,
                                                 const uint8_t *uPtr, uint8_t *outrgba,
                                                 size_t width, size_t height,
                                                 size_t yStride, size_t yDstStride,
                                                 size_t uRowStride,
                                                 size_t vRowStride, size_t uPixelStride,
                                                 size_t vPixelStride, int xStart) {
        for (int y = 0; y < height; y++) {
            const uint8_t *yRow = yPtr + y * yStride;
            int chromaY = y >> 1;
            const uint8_t *uRow = uPtr + chromaY * uRowStride;
            const uint8_t *vRow = vPtr + chromaY * vRowStride;

            uint8_t *outRow = outrgba + y * yDstStride;

            for (int x = xStart; x < width; x++) {
                int yPix = yRow[x];
                int chromaX = x >> 1;
                int vPix = vRow[chromaX * vPixelStride];
                int uPix = uRow[chromaX * uPixelStride];

                int C = yPix - 16;
                int D = uPix - 128;
                int E = vPix - 128;

                int R = (298 * C + 409 * E + 128) >> 8;
                int G = (298 * C - 100 * D - 208 * E + 128) >> 8;
                int B = (298 * C + 516 * D + 128) >> 8;

                outRow[4 * x + 0] = clamp255(R);
                outRow[4 * x + 1] = clamp255(G);
                outRow[4 * x + 2] = clamp255(B);
                outRow[4 * x + 3] = clamp255(255);
            }
        }

    }

    bool ImageProcessor::EdgeDetection(JNIEnv *env, jobject bitmap) {
        AndroidBitmapInfo info;
        void *pixelData = nullptr;
        if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) {
            LOG_INFO("Failed to get the bitmap info");
            return false;
        }
        if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
            LOG_INFO("Invalid bit map format ");
            return false;
        }
        if (AndroidBitmap_lockPixels(env, bitmap, &pixelData) < 0) {
            LOG_INFO("Failed to lock the pixel information");
            return false;
        }
        if (ImageProcessorSIMD::device_support_neon()) {
            uint8_t *outData = new uint8_t[info.height * info.width * 4];
            ImageProcessorSIMD::edge_detection_simd_float(reinterpret_cast<uint8_t *>(pixelData),
                                                          outData, info.width, info.height,
                                                          info.stride);
            memcpy(pixelData, outData, 4 * info.width * info.height);
            delete[] outData;
            return true;
        }
        AndroidBitmap_unlockPixels(env, bitmap);
        return true;
    }

}


extern "C" {
JNIEXPORT jboolean JNICALL
Java_com_os_imageprocessor_JniBridge_GrayScaleImage(JNIEnv *env, jclass clazz, jobject bitmap,
                                                    jboolean optimizeNeon) {
    bool imageProcessed = ip::ImageProcessor::GrayScale(env, bitmap);
    return imageProcessed ? JNI_TRUE : JNI_FALSE;
}
JNIEXPORT jboolean JNICALL
Java_com_os_imageprocessor_JniBridge_CreateNegative(JNIEnv *env, jclass clazz, jobject bitmap,
                                                    jboolean optimizeNeon) {
    bool imageProcessed = ip::ImageProcessor::NegativeImage(env, bitmap);
    return imageProcessed ? JNI_TRUE : JNI_FALSE;
}
JNIEXPORT jboolean JNICALL
Java_com_os_imageprocessor_JniBridge_BlurImage(JNIEnv *env, jclass clazz, jobject bitmap,
                                               int radius, int sigma,
                                               jboolean optimizeNeon) {
    bool imageProcessed = ip::ImageProcessor::BlurImage(env, bitmap, radius, sigma);
    return imageProcessed ? JNI_TRUE : JNI_FALSE;
}
JNIEXPORT jboolean JNICALL
Java_com_os_imageprocessor_JniBridge_Embross(JNIEnv *env, jclass clazz, jobject bitmap,
                                             jboolean optimizeNeon) {
    bool imageProcessed = ip::ImageProcessor::EmbrossImage(env, bitmap);
    return imageProcessed ? JNI_TRUE : JNI_FALSE;
}
JNIEXPORT jboolean JNICALL
Java_com_os_imageprocessor_JniBridge_Sharpen(JNIEnv *env, jclass clazz, jobject bitmap,
                                             jboolean optimize_neon) {
    bool imageProcessed = ip::ImageProcessor::SharpenImage(env, bitmap);
    return imageProcessed ? JNI_TRUE : JNI_FALSE;

}
JNIEXPORT jboolean JNICALL
Java_com_os_imageprocessor_JniBridge_EdgeDetection(JNIEnv *env, jclass clazz, jobject bitmap,
                                                   jboolean optimize_neon) {
    bool imageProcessed = ip::ImageProcessor::EdgeDetection(env, bitmap);
    return imageProcessed ? JNI_TRUE : JNI_FALSE;

}
JNIEXPORT void JNICALL
Java_com_os_imageprocessor_JniBridge_convert_1yuv_1rgba(JNIEnv *env, jclass clazz,
                                                        jbyteArray y_pixels, jbyteArray v_pixels,
                                                        jbyteArray u_pixels,
                                                        jobject outBitmap,
                                                        jint width, jint height, jint y_stride,
                                                        jint dst_stride, jint u_row_stride,
                                                        jint v_row_stride, jint u_pixel_stride,
                                                        jint v_pixel_stride,
                                                        jboolean optimizeNeon) {
    jbyte *yPtr = env->GetByteArrayElements(y_pixels, nullptr);
    jbyte *uPixels = env->GetByteArrayElements(u_pixels, nullptr);
    jbyte *vPixels = env->GetByteArrayElements(v_pixels, nullptr);
    void *lockPixels = nullptr;
    if (AndroidBitmap_lockPixels(env, outBitmap, &lockPixels) < 0) {
        LOG_ERROR("Failed to lock the pixels");
        return;
    }
    if (ip::ImageProcessorSIMD::device_support_neon() && optimizeNeon) {
        ip::ImageProcessorSIMD::convert_yuv_rgba_neon(reinterpret_cast<uint8_t *>(yPtr),
                                                      reinterpret_cast<uint8_t *>(uPixels),
                                                      reinterpret_cast<uint8_t *>(vPixels),
                                                      reinterpret_cast<uint8_t *>(lockPixels),
                                                      height,
                                                      width, y_stride, dst_stride, u_row_stride,
                                                      v_row_stride, v_pixel_stride, v_pixel_stride);
    } else {
        if (optimizeNeon) {
            LOG_ERROR("Device Does not support neon.. falling back to scalar");
        }
        ip::ImageProcessor::convert_yuv_rgba_scalar(reinterpret_cast<uint8_t *>(yPtr),
                                                    reinterpret_cast<uint8_t *>(vPixels),
                                                    reinterpret_cast<uint8_t *>(uPixels),
                                                    reinterpret_cast<uint8_t *>(lockPixels),
                                                    width, height, y_stride, dst_stride,
                                                    u_row_stride,
                                                    v_row_stride, u_pixel_stride, v_pixel_stride);
    }
    AndroidBitmap_unlockPixels(env, outBitmap);
    env->ReleaseByteArrayElements(y_pixels, yPtr, 0);
    env->ReleaseByteArrayElements(u_pixels, uPixels, 0);
    env->ReleaseByteArrayElements(v_pixels, vPixels, 0);
}
}
