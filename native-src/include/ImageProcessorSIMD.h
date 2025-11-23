//
// Created by ghima on 09-11-2025.
//

#ifndef OSFEATURENDKDEMO_IMAGEPROCESSORSIMD_H
#define OSFEATURENDKDEMO_IMAGEPROCESSORSIMD_H

#include <cstdint>
#include <arm_neon.h>

namespace ip {
    class ImageProcessorSIMD {
    public:
        static bool device_support_neon();

        static void gray_scale_neon_simd(uint8_t *src, uint8_t *dst, size_t pixels);

        static void negative_neon_simd(uint8_t *src, uint8_t *dst, size_t pixels);

        static void
        sharp_neon_simd(uint8_t *src, uint8_t *dst, size_t width, size_t height,
                        size_t stride);

        static void
        blur_neon_simd(uint8_t *src, uint8_t *dst, size_t width, size_t height, size_t stride,
                       int radius, float sigma);

        static void
        blur_neon_simd_float(uint8_t *src, uint8_t *dst, size_t width, size_t height, size_t stride,
                             int radius, float sigma);

        static void
        edge_detection_simd_float(uint8_t *src, uint8_t *dst, size_t width, size_t height,
                                  size_t stride);

        static void
        emboss_neon_simd_float(uint8_t *src, uint8_t *dst, size_t width, size_t height,
                               size_t stride);

        static void
        convert_yuv_rgba_neon(uint8_t *yPixel, uint8_t *uPix,
                              uint8_t *vPix, uint8_t *dstRGBA,
                              size_t height, size_t width,
                              size_t yStride, size_t yDstStride,
                              size_t uRowStride,
                              size_t vRowStride, size_t vPixelStride,
                              size_t uPixelStride);

        static uint8x8_t get_real_uv_pattern_for_stride_two(const uint8_t *src);
    };
}
#endif //OSFEATURENDKDEMO_IMAGEPROCESSORSIMD_H
