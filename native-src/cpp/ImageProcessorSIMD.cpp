//
// Created by ghima on 09-11-2025.
//
#include <cstddef>
#include <vector>
#include <android/log.h>
#include "ImageProcessorSIMD.h"
#include "ThreadPool.h"
#include "Utility.h"
#include "ImageProcessor.h"

#define LOG_TAG "core_native_image"
#define LOG_INFO(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOG_ERROR(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
namespace ip {
    void ImageProcessorSIMD::gray_scale_neon_simd(uint8_t *src, uint8_t *dst, size_t pixels) {
        const size_t vec_px = 16;
        const size_t bytes_per_pixel = 4;
        const size_t groups = pixels / vec_px;

        uint32_t threads = std::thread::hardware_concurrency();
        if (threads < 2) threads = 4;
        ThreadPool pool{threads};

        int per = groups / threads;
        int rem = groups % threads;
        int gStart = 0;
        for (int t = 0; t < threads; t++) {
            int gCount = per + (t < rem ? 1 : 0);
            int yStart = gStart;
            int yEnd = gStart + gCount;
            gStart = yEnd;
            pool.enqueue_task([&, yStart, yEnd]() -> void {
                for (int gr = yStart; gr < yEnd; gr++) {
                    // this will load the 64 bytes see this as a array of 128 bit registers
                    size_t i = gr * vec_px;
                    uint8x16x4_t ch = vld4q_u8(src + i * 4);

                    // there is no 16 x16 thing as it will overflow so have to take either high or low
                    uint16x8_t r_u16 = vmovl_u8(vget_low_u8(ch.val[2]));
                    uint16x8_t g_u16 = vmovl_u8(vget_low_u8(ch.val[1]));
                    uint16x8_t b_u16 = vmovl_u8(vget_low_u8(ch.val[0]));

                    uint16x8_t y_low = vmulq_n_u16(r_u16, 77);
                    y_low = vmlaq_n_u16(y_low, g_u16, 150);
                    y_low = vmlaq_n_u16(y_low, b_u16, 29);
                    y_low = vshrq_n_u16(y_low, 8);

                    uint8x8_t gray_low = vmovn_u16(y_low);

                    // doing it for the high lanes
                    uint16x8_t r_u16_high = vmovl_u8(vget_high_u8(ch.val[2]));
                    uint16x8_t g_u16_high = vmovl_u8(vget_high_u8(ch.val[1]));
                    uint16x8_t b_u16_high = vmovl_u8(vget_high_u8(ch.val[0]));

                    uint16x8_t y_high = vmulq_n_u16(r_u16_high, 77);
                    y_high = vmlaq_n_u16(y_high, g_u16_high, 150);
                    y_high = vmlaq_n_u16(y_high, b_u16_high, 29);
                    y_high = vshrq_n_u16(y_high, 8);
                    uint8x8_t gray_high = vmovn_u16(y_high);

                    uint8x16x4_t out;
                    out.val[0] = vcombine_u8(gray_low, gray_high);
                    out.val[1] = vcombine_u8(gray_low, gray_high);
                    out.val[2] = vcombine_u8(gray_low, gray_high);
                    out.val[3] = ch.val[3];

                    vst4q_u8(dst + i * 4, out);
                }
            });
        }
    }

    void ImageProcessorSIMD::negative_neon_simd(uint8_t *src, uint8_t *dst, size_t pixels) {
        const size_t vec_px = 16;
        const size_t bytes_per_pixel = 4;
        const size_t groups = pixels / vec_px;

        uint32_t threads = std::thread::hardware_concurrency();
        if (threads < 2) threads = 4;
        ThreadPool pool{threads};
        size_t per = groups / threads;
        size_t rem = groups % threads;

        int gStart = 0;
        for (int t = 0; t < threads; t++) {
            int gCount = per + (t < rem ? 1 : 0);
            int yStart = gStart;
            int yEnd = yStart + gCount;
            gStart = yEnd;
            pool.enqueue_task([&, yStart, yEnd]() -> void {
                for (size_t gr = yStart; gr < yEnd; gr++) {
                    // loading the 64 bytes data for 16 pixels;
                    size_t i = gr * vec_px;
                    uint8x16x4_t ch = vld4q_u8(src + i * bytes_per_pixel);
                    uint8x16_t subVal = vdupq_n_u8(255);
                    uint8x16_t b = vsubq_u8(subVal, ch.val[2]);
                    uint8x16_t g = vsubq_u8(subVal, ch.val[1]);
                    uint8x16_t r = vsubq_u8(subVal, ch.val[0]);

                    uint8x16x4_t out;
                    out.val[0] = r;
                    out.val[1] = g;
                    out.val[2] = b;
                    out.val[3] = ch.val[3];

                    vst4q_u8(dst + i * bytes_per_pixel, out);
                }
            });
        }
    }

    void
    ImageProcessorSIMD::sharp_neon_simd(uint8_t *src, uint8_t *dst, size_t width,
                                        size_t height,
                                        size_t stride) {
        std::vector<std::vector<int8_t>> kernel = {
                {-1, -1, -1},
                {-1, 9,  -1},
                {-1, -1, -1}
        };

        uint32_t threads = std::thread::hardware_concurrency();
        if (threads < 2) threads = 4;
        ThreadPool pool{threads};
        size_t slice = (height - 2) / threads;
        for (int t = 0; t < threads; t++) {
            int yStart = 1 + t * slice;
            int yEnd = (t == threads - 1) ? height - 1 : yStart + slice;
            pool.enqueue_task([&, yStart, yEnd]() -> void {
                for (int y = yStart; y < yEnd; y++) {
                    for (int x = 1; x < width - 1; x += 16) {
                        int16x8_t b_lo = vdupq_n_s16(0);
                        int16x8_t b_hi = vdupq_n_s16(0);
                        int16x8_t g_lo = b_lo, g_hi = b_hi;
                        int16x8_t r_lo = b_lo, r_hi = b_hi;

                        for (int ky = -1; ky <= 1; ky++) {
                            for (int kx = -1; kx <= 1; kx++) {
                                int8_t weight = kernel[ky + 1][kx + 1];
                                if (weight == 0) continue;
                                uint8_t *point = src + (y + ky) * stride + (x + kx) * 4;

                                uint8x16x4_t pixels = vld4q_u8(point);
                                int16x8_t b_l = vreinterpretq_s16_u16(
                                        vmovl_u8(vget_low_u8(pixels.val[2])));
                                int16x8_t b_h = vreinterpretq_s16_u16(
                                        vmovl_u8(vget_high_u8(pixels.val[2])));
                                int16x8_t g_l = vreinterpretq_s16_u16(
                                        vmovl_u8(vget_low_u8(pixels.val[1])));
                                int16x8_t g_h = vreinterpretq_s16_u16(
                                        vmovl_u8(vget_high_u8(pixels.val[1])));
                                int16x8_t r_l = vreinterpretq_s16_u16(
                                        vmovl_u8(vget_low_u8(pixels.val[0])));
                                int16x8_t r_h = vreinterpretq_s16_u16(
                                        vmovl_u8(vget_high_u8(pixels.val[0])));

                                int16x8_t w = vdupq_n_s16(weight);
                                b_lo = vmlaq_s16(b_lo, b_l, w);
                                b_hi = vmlaq_s16(b_hi, b_h, w);
                                g_lo = vmlaq_s16(g_lo, g_l, w);
                                g_hi = vmlaq_s16(g_hi, g_h, w);
                                r_lo = vmlaq_s16(r_lo, r_l, w);
                                r_hi = vmlaq_s16(r_hi, r_h, w);

                            }
                        }
                        int16x8_t zero = vdupq_n_s16(0);
                        int16x8_t maxv = vdupq_n_s16(255);

                        b_lo = vmaxq_s16(zero, vminq_s16(b_lo, maxv));
                        b_hi = vmaxq_s16(zero, vminq_s16(b_hi, maxv));
                        g_lo = vmaxq_s16(zero, vminq_s16(g_lo, maxv));
                        g_hi = vmaxq_s16(zero, vminq_s16(g_hi, maxv));
                        r_lo = vmaxq_s16(zero, vminq_s16(r_lo, maxv));
                        r_hi = vmaxq_s16(zero, vminq_s16(r_hi, maxv));

                        uint8x16_t b_8 = vcombine_u8(vqmovun_s16(b_lo), vqmovun_s16(b_hi));
                        uint8x16_t g_8 = vcombine_u8(vqmovun_s16(g_lo), vqmovun_s16(g_hi));
                        uint8x16_t r_8 = vcombine_u8(vqmovun_s16(r_lo), vqmovun_s16(r_hi));
                        uint8x16x4_t src_pix = vld4q_u8(src + y * stride + x * 4);
                        uint8x16_t a_8 = src_pix.val[3];

                        uint8x16x4_t out;
                        out.val[0] = r_8;
                        out.val[1] = g_8;
                        out.val[2] = b_8;
                        out.val[3] = a_8;

                        vst4q_u8(dst + (y * stride) + x * 4, out);
                    }
                }
            });
        }
    }

    void ImageProcessorSIMD::blur_neon_simd(uint8_t *src, uint8_t *dst, size_t width, size_t height,
                                            size_t stride, int radius, float sigma) {
        std::vector<std::vector<float>> kernel = Utility::generate_gaussian_kernel(radius, sigma);
        for (int y = radius; y < height - radius; y++) {
            for (int x = radius; x < width - radius; x += 16) {
                int16x8_t b_lo = vdupq_n_s16(0);
                int16x8_t b_hi = vdupq_n_s16(0);
                int16x8_t g_lo = vdupq_n_s16(0);
                int16x8_t g_hi = vdupq_n_s16(0);
                int16x8_t r_lo = vdupq_n_s16(0);
                int16x8_t r_hi = vdupq_n_s16(0);
                for (int ky = -radius; ky <= radius; ky++) {
                    for (int kx = -radius; kx <= radius; kx++) {
                        float weight = kernel[ky + radius][kx + radius];
                        if (weight == 0) continue;

                        uint8_t *p = src + (y + ky) * stride + (x + kx) * 4;
                        uint8x16x4_t ch = vld4q_u8(p);
                        int16x8_t r_l = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(ch.val[0])));
                        int16x8_t r_h = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(ch.val[0])));
                        int16x8_t g_l = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(ch.val[1])));
                        int16x8_t g_h = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(ch.val[1])));
                        int16x8_t b_l = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(ch.val[2])));
                        int16x8_t b_h = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(ch.val[2])));

                        float scale = 256;
                        int16x8_t w = vdupq_n_s16((int16_t) (weight * scale));
                        b_lo = vmlaq_s16(b_lo, b_l, w);
                        b_hi = vmlaq_s16(b_hi, b_h, w);
                        g_lo = vmlaq_s16(g_lo, g_l, w);
                        g_hi = vmlaq_s16(g_hi, g_h, w);
                        r_lo = vmlaq_s16(r_lo, r_l, w);
                        r_hi = vmlaq_s16(r_hi, r_h, w);
                    }
                }

                b_lo = vshrq_n_s16(b_lo, 8);
                b_hi = vshrq_n_s16(b_hi, 8);
                g_lo = vshrq_n_s16(g_lo, 8);
                g_hi = vshrq_n_s16(g_hi, 8);
                r_lo = vshrq_n_s16(r_lo, 8);
                r_hi = vshrq_n_s16(r_hi, 8);
                // clamping the output
                int16x8_t min = vdupq_n_s16(0);
                int16x8_t max = vdupq_n_s16(255);

                b_lo = vmaxq_s16(min, vminq_s16(b_lo, max));
                b_hi = vmaxq_s16(min, vminq_s16(b_hi, max));
                g_lo = vmaxq_s16(min, vminq_s16(g_lo, max));
                g_hi = vmaxq_s16(min, vminq_s16(g_hi, max));
                r_lo = vmaxq_s16(min, vminq_s16(r_lo, max));
                r_hi = vmaxq_s16(min, vminq_s16(r_hi, max));

                uint8x16x4_t out;
                out.val[0] = vcombine_u8(vqmovun_s16(r_lo), vqmovun_s16(r_hi));
                out.val[1] = vcombine_u8(vqmovun_s16(g_lo), vqmovun_s16(g_hi));
                out.val[2] = vcombine_u8(vqmovun_s16(b_lo), vqmovun_s16(b_hi));
                uint8_t *currentPix = src + y * stride + x * 4;
                out.val[3] = vld4q_u8(currentPix).val[3];
                vst4q_u8(dst + y * stride + x * 4, out);
            }

        }

    }

    void ImageProcessorSIMD::blur_neon_simd_float(uint8_t *src, uint8_t *dst, size_t width,
                                                  size_t height, size_t stride, int radius,
                                                  float sigma) {
        auto threads = std::thread::hardware_concurrency();
        if (threads <= 1) threads = 4;
        ThreadPool pool{threads};
        int slice = (height - radius) / threads;
        std::vector<std::vector<float>> kernel = Utility::generate_gaussian_kernel(radius, sigma);

        for (int t = 0; t < threads; t++) {
            int yStart = radius + t * slice;
            int yEnd = yStart + slice;
            yStart -= radius;
            yEnd += radius;
            if (yStart < radius) yStart = radius;
            if (yEnd > height - radius) yEnd = height - radius;
            pool.enqueue_task([&, yStart, yEnd]() -> void {
                for (int y = yStart; y < yEnd; y++) {
                    for (int x = radius; x < width - radius; x += 16) {
                        float32x4_t b_f_l_1 = vdupq_n_f32(0.0);
                        float32x4_t b_f_l_2 = vdupq_n_f32(0.0);
                        float32x4_t b_f_h_1 = vdupq_n_f32(0.0);
                        float32x4_t b_f_h_2 = vdupq_n_f32(0.0);
                        float32x4_t g_f_l_1 = vdupq_n_f32(0.0);
                        float32x4_t g_f_l_2 = vdupq_n_f32(0.0);
                        float32x4_t g_f_h_1 = vdupq_n_f32(0.0);
                        float32x4_t g_f_h_2 = vdupq_n_f32(0.0);
                        float32x4_t r_f_l_1 = vdupq_n_f32(0.0);
                        float32x4_t r_f_l_2 = vdupq_n_f32(0.0);
                        float32x4_t r_f_h_1 = vdupq_n_f32(0.0);
                        float32x4_t r_f_h_2 = vdupq_n_f32(0.0);

                        for (int ky = -radius; ky <= radius; ky++) {
                            for (int kx = -radius; kx <= radius; kx++) {
                                uint8_t *p = src + (y + ky) * stride + (kx + x) * 4;
                                uint8x16x4_t ch = vld4q_u8(p);
                                float weight = kernel[ky + radius][kx + radius];
                                float32x4_t w = vdupq_n_f32(weight);

                                float32x4_t r_l_1 = vcvtq_f32_u32(
                                        vmovl_u16(vget_low_u16(vmovl_u8(
                                                vget_low_u8(ch.val[0])))));
                                float32x4_t r_l_2 = vcvtq_f32_u32(
                                        vmovl_u16(vget_high_u16(vmovl_u8(
                                                vget_low_u8(ch.val[0])))));
                                float32x4_t r_h_1 = vcvtq_f32_u32(
                                        vmovl_u16(vget_low_u16(vmovl_u8(
                                                vget_high_u8(ch.val[0])))));
                                float32x4_t r_h_2 = vcvtq_f32_u32(
                                        vmovl_u16(vget_high_u16(vmovl_u8(
                                                vget_high_u8(ch.val[0])))));
                                float32x4_t g_l_1 = vcvtq_f32_u32(
                                        vmovl_u16(vget_low_u16(vmovl_u8(
                                                vget_low_u8(ch.val[1])))));
                                float32x4_t g_l_2 = vcvtq_f32_u32(
                                        vmovl_u16(vget_high_u16(vmovl_u8(
                                                vget_low_u8(ch.val[1])))));
                                float32x4_t g_h_1 = vcvtq_f32_u32(
                                        vmovl_u16(vget_low_u16(vmovl_u8(
                                                vget_high_u8(ch.val[1])))));
                                float32x4_t g_h_2 = vcvtq_f32_u32(
                                        vmovl_u16(vget_high_u16(vmovl_u8(
                                                vget_high_u8(ch.val[1])))));
                                float32x4_t b_l_1 = vcvtq_f32_u32(
                                        vmovl_u16(vget_low_u16(vmovl_u8(
                                                vget_low_u8(ch.val[2])))));
                                float32x4_t b_l_2 = vcvtq_f32_u32(
                                        vmovl_u16(vget_high_u16(vmovl_u8(
                                                vget_low_u8(ch.val[2])))));
                                float32x4_t b_h_1 = vcvtq_f32_u32(
                                        vmovl_u16(vget_low_u16(vmovl_u8(
                                                vget_high_u8(ch.val[2])))));
                                float32x4_t b_h_2 = vcvtq_f32_u32(
                                        vmovl_u16(vget_high_u16(vmovl_u8(
                                                vget_high_u8(ch.val[2])))));


                                b_f_l_1 = vmlaq_f32(b_f_l_1, b_l_1, w);
                                b_f_l_2 = vmlaq_f32(b_f_l_2, b_l_2, w);
                                b_f_h_1 = vmlaq_f32(b_f_h_1, b_h_1, w);
                                b_f_h_2 = vmlaq_f32(b_f_h_2, b_h_2, w);
                                g_f_l_1 = vmlaq_f32(g_f_l_1, g_l_1, w);
                                g_f_l_2 = vmlaq_f32(g_f_l_2, g_l_2, w);
                                g_f_h_1 = vmlaq_f32(g_f_h_1, g_h_1, w);
                                g_f_h_2 = vmlaq_f32(g_f_h_2, g_h_2, w);
                                r_f_l_1 = vmlaq_f32(r_f_l_1, r_l_1, w);
                                r_f_l_2 = vmlaq_f32(r_f_l_2, r_l_2, w);
                                r_f_h_1 = vmlaq_f32(r_f_h_1, r_h_1, w);
                                r_f_h_2 = vmlaq_f32(r_f_h_2, r_h_2, w);
                            }
                        }
                        // clamping the data between 0 and 255

                        float32x4_t min = vdupq_n_f32(0);
                        float32x4_t max = vdupq_n_f32(255);

                        b_f_l_1 = vmaxq_f32(min, vminq_f32(b_f_l_1, max));
                        b_f_l_2 = vmaxq_f32(min, vminq_f32(b_f_l_2, max));
                        b_f_h_1 = vmaxq_f32(min, vminq_f32(b_f_h_1, max));
                        b_f_h_2 = vmaxq_f32(min, vminq_f32(b_f_h_2, max));
                        g_f_l_1 = vmaxq_f32(min, vminq_f32(g_f_l_1, max));
                        g_f_l_2 = vmaxq_f32(min, vminq_f32(g_f_l_2, max));
                        g_f_h_1 = vmaxq_f32(min, vminq_f32(g_f_h_1, max));
                        g_f_h_2 = vmaxq_f32(min, vminq_f32(g_f_h_2, max));
                        r_f_l_1 = vmaxq_f32(min, vminq_f32(r_f_l_1, max));
                        r_f_l_2 = vmaxq_f32(min, vminq_f32(r_f_l_2, max));
                        r_f_h_1 = vmaxq_f32(min, vminq_f32(r_f_h_1, max));
                        r_f_h_2 = vmaxq_f32(min, vminq_f32(r_f_h_2, max));

                        uint16x4_t b_low_16_1 = vqmovun_s32(vcvtq_s32_f32(b_f_l_1));
                        uint16x4_t b_low_16_2 = vqmovun_s32(vcvtq_s32_f32(b_f_l_2));
                        uint8x8_t b_low_8 = vqmovn_u16(vcombine_u16(b_low_16_1, b_low_16_2));
                        uint16x4_t b_high_16_1 = vqmovun_s32(vcvtq_s32_f32(b_f_h_1));
                        uint16x4_t b_high_16_2 = vqmovun_s32(vcvtq_s32_f32(b_f_h_2));
                        uint8x8_t b_high_8 = vqmovn_u16(vcombine_u16(b_high_16_1, b_high_16_2));
                        uint8x16_t b_out = vcombine_u8(b_low_8, b_high_8);

                        uint16x4_t g_low_16_1 = vqmovun_s32(vcvtq_s32_f32(g_f_l_1));
                        uint16x4_t g_low_16_2 = vqmovun_s32(vcvtq_s32_f32(g_f_l_2));
                        uint8x8_t g_low_8 = vqmovn_u16(vcombine_u16(g_low_16_1, g_low_16_2));
                        uint16x4_t g_high_16_1 = vqmovun_s32(vcvtq_s32_f32(g_f_h_1));
                        uint16x4_t g_high_16_2 = vqmovun_s32(vcvtq_s32_f32(g_f_h_2));
                        uint8x8_t g_high_8 = vqmovn_u16(vcombine_u16(g_high_16_1, g_high_16_2));
                        uint8x16_t g_out = vcombine_u8(g_low_8, g_high_8);

                        uint16x4_t r_low_16_1 = vqmovun_s32(vcvtq_s32_f32(r_f_l_1));
                        uint16x4_t r_low_16_2 = vqmovun_s32(vcvtq_s32_f32(r_f_l_2));
                        uint8x8_t r_low_8 = vqmovn_u16(vcombine_u16(r_low_16_1, r_low_16_2));
                        uint16x4_t r_high_16_1 = vqmovun_s32(vcvtq_s32_f32(r_f_h_1));
                        uint16x4_t r_high_16_2 = vqmovun_s32(vcvtq_s32_f32(r_f_h_2));
                        uint8x8_t r_high_8 = vqmovn_u16(vcombine_u16(r_high_16_1, r_high_16_2));
                        uint8x16_t r_out = vcombine_u8(r_low_8, r_high_8);

                        uint8x16x4_t out;
                        out.val[0] = r_out;
                        out.val[1] = g_out;
                        out.val[2] = b_out;
                        uint8_t *currentPix = src + y * stride + x * 4;
                        out.val[3] = vld4q_u8(currentPix).val[3];
                        vst4q_u8(dst + y * stride + x * 4, out);
                    }
                }
            });
        }
        pool.joinAll();
    }

    void ImageProcessorSIMD::edge_detection_simd_float(uint8_t *src, uint8_t *dst, size_t width,
                                                       size_t height, size_t stride) {
        float Gx[3][3] = {
                {-1, 0, 1},
                {-2, 0, 2},
                {-1, 0, 1}
        };

        float Gy[3][3] = {
                {-1, -2, -1},
                {0,  0,  0},
                {1,  2,  1}
        };
        uint threads = std::thread::hardware_concurrency();
        if (threads <= 1) threads = 4;
        size_t slice = (height - 2) / threads;

        ThreadPool pool{threads};
        for (int t = 0; t < threads; t++) {
            int yStart = 1 + t * slice;
            int yEnd = (t == threads - 1) ? height - 1 : yStart + slice;
            pool.enqueue_task([&, yStart, yEnd]() -> void {
                for (int y = yStart; y < yEnd; y++) {
                    for (int x = 1; x < width - 1; x += 16) {
                        float32x4_t x_b_l_1 = vdupq_n_f32(0.0);
                        float32x4_t x_b_l_2 = vdupq_n_f32(0.0);
                        float32x4_t x_b_h_1 = vdupq_n_f32(0.0);
                        float32x4_t x_b_h_2 = vdupq_n_f32(0.0);
                        float32x4_t x_g_l_1 = vdupq_n_f32(0.0);
                        float32x4_t x_g_l_2 = vdupq_n_f32(0.0);
                        float32x4_t x_g_h_1 = vdupq_n_f32(0.0);
                        float32x4_t x_g_h_2 = vdupq_n_f32(0.0);
                        float32x4_t x_r_l_1 = vdupq_n_f32(0.0);
                        float32x4_t x_r_l_2 = vdupq_n_f32(0.0);
                        float32x4_t x_r_h_1 = vdupq_n_f32(0.0);
                        float32x4_t x_r_h_2 = vdupq_n_f32(0.0);

                        float32x4_t y_b_l_1 = vdupq_n_f32(0.0);
                        float32x4_t y_b_l_2 = vdupq_n_f32(0.0);
                        float32x4_t y_b_h_1 = vdupq_n_f32(0.0);
                        float32x4_t y_b_h_2 = vdupq_n_f32(0.0);
                        float32x4_t y_g_l_1 = vdupq_n_f32(0.0);
                        float32x4_t y_g_l_2 = vdupq_n_f32(0.0);
                        float32x4_t y_g_h_1 = vdupq_n_f32(0.0);
                        float32x4_t y_g_h_2 = vdupq_n_f32(0.0);
                        float32x4_t y_r_l_1 = vdupq_n_f32(0.0);
                        float32x4_t y_r_l_2 = vdupq_n_f32(0.0);
                        float32x4_t y_r_h_1 = vdupq_n_f32(0.0);
                        float32x4_t y_r_h_2 = vdupq_n_f32(0.0);

                        for (int ky = -1; ky <= 1; ky++) {
                            for (int kx = -1; kx <= 1; kx++) {
                                uint8_t *ptr = src + (y + ky) * stride + (x + kx) * 4;
                                uint8x16x4_t ch = vld4q_u8(ptr);


                                float32x4_t r_c_l_1 = vcvtq_f32_u32(vmovl_u16(
                                        vget_low_u16(vmovl_u8(vget_low_u8(ch.val[0])))));
                                float32x4_t r_c_l_2 = vcvtq_f32_u32(vmovl_u16(
                                        vget_high_u16(vmovl_u8(vget_low_u8(ch.val[0])))));
                                float32x4_t r_c_h_1 = vcvtq_f32_u32(vmovl_u16(
                                        vget_low_u16(vmovl_u8(vget_high_u8(ch.val[0])))));
                                float32x4_t r_c_h_2 = vcvtq_f32_u32(vmovl_u16(
                                        vget_high_u16(vmovl_u8(vget_high_u8(ch.val[0])))));

                                float32x4_t g_c_l_1 = vcvtq_f32_u32(vmovl_u16(
                                        vget_low_u16(vmovl_u8(vget_low_u8(ch.val[1])))));
                                float32x4_t g_c_l_2 = vcvtq_f32_u32(vmovl_u16(
                                        vget_high_u16(vmovl_u8(vget_low_u8(ch.val[1])))));
                                float32x4_t g_c_h_1 = vcvtq_f32_u32(vmovl_u16(
                                        vget_low_u16(vmovl_u8(vget_high_u8(ch.val[1])))));
                                float32x4_t g_c_h_2 = vcvtq_f32_u32(vmovl_u16(
                                        vget_high_u16(vmovl_u8(vget_high_u8(ch.val[1])))));

                                float32x4_t b_c_l_1 = vcvtq_f32_u32(vmovl_u16(
                                        vget_low_u16(vmovl_u8(vget_low_u8(ch.val[2])))));
                                float32x4_t b_c_l_2 = vcvtq_f32_u32(vmovl_u16(
                                        vget_high_u16(vmovl_u8(vget_low_u8(ch.val[2])))));
                                float32x4_t b_c_h_1 = vcvtq_f32_u32(vmovl_u16(
                                        vget_low_u16(vmovl_u8(vget_high_u8(ch.val[2])))));
                                float32x4_t b_c_h_2 = vcvtq_f32_u32(vmovl_u16(
                                        vget_high_u16(vmovl_u8(vget_high_u8(ch.val[2])))));


                                float32x4_t x_weight = vdupq_n_f32(Gx[ky + 1][kx + 1]);
                                float32x4_t y_weight = vdupq_n_f32(Gy[ky + 1][kx + 1]);

                                x_b_l_1 = vmlaq_f32(x_b_l_1, b_c_l_1, x_weight);
                                x_b_l_2 = vmlaq_f32(x_b_l_2, b_c_l_2, x_weight);
                                x_b_h_1 = vmlaq_f32(x_b_h_1, b_c_h_1, x_weight);
                                x_b_h_2 = vmlaq_f32(x_b_h_2, b_c_h_2, x_weight);
                                x_g_l_1 = vmlaq_f32(x_g_l_1, g_c_l_1, x_weight);
                                x_g_l_2 = vmlaq_f32(x_g_l_2, g_c_l_2, x_weight);
                                x_g_h_1 = vmlaq_f32(x_g_h_1, g_c_h_1, x_weight);
                                x_g_h_2 = vmlaq_f32(x_g_h_2, g_c_h_2, x_weight);
                                x_r_l_1 = vmlaq_f32(x_r_l_1, r_c_l_1, x_weight);
                                x_r_l_2 = vmlaq_f32(x_r_l_2, r_c_l_2, x_weight);
                                x_r_h_1 = vmlaq_f32(x_r_h_1, r_c_h_1, x_weight);
                                x_r_h_2 = vmlaq_f32(x_r_h_2, r_c_h_2, x_weight);

                                y_b_l_1 = vmlaq_f32(y_b_l_1, b_c_l_1, y_weight);
                                y_b_l_2 = vmlaq_f32(y_b_l_2, b_c_l_2, y_weight);
                                y_b_h_1 = vmlaq_f32(y_b_h_1, b_c_h_1, y_weight);
                                y_b_h_2 = vmlaq_f32(y_b_h_2, b_c_h_2, y_weight);
                                y_g_l_1 = vmlaq_f32(y_g_l_1, g_c_l_1, y_weight);
                                y_g_l_2 = vmlaq_f32(y_g_l_2, g_c_l_2, y_weight);
                                y_g_h_1 = vmlaq_f32(y_g_h_1, g_c_h_1, y_weight);
                                y_g_h_2 = vmlaq_f32(y_g_h_2, g_c_h_2, y_weight);
                                y_r_l_1 = vmlaq_f32(y_r_l_1, r_c_l_1, y_weight);
                                y_r_l_2 = vmlaq_f32(y_r_l_2, r_c_l_2, y_weight);
                                y_r_h_1 = vmlaq_f32(y_r_h_1, r_c_h_1, y_weight);
                                y_r_h_2 = vmlaq_f32(y_r_h_2, r_c_h_2, y_weight);
                            }
                        }
                        x_b_l_1 = vsqrtq_f32(
                                vaddq_f32(vmulq_f32(x_b_l_1, x_b_l_1),
                                          vmulq_f32(y_b_l_1, y_b_l_1)));
                        x_b_l_2 = vsqrtq_f32(
                                vaddq_f32(vmulq_f32(x_b_l_2, x_b_l_2),
                                          vmulq_f32(y_b_l_2, y_b_l_2)));
                        x_b_h_1 = vsqrtq_f32(
                                vaddq_f32(vmulq_f32(x_b_h_1, x_b_h_1),
                                          vmulq_f32(y_b_h_1, y_b_h_1)));
                        x_b_h_2 = vsqrtq_f32(
                                vaddq_f32(vmulq_f32(x_b_h_2, x_b_h_2),
                                          vmulq_f32(y_b_h_2, y_b_h_2)));

                        x_g_l_1 = vsqrtq_f32(
                                vaddq_f32(vmulq_f32(x_g_l_1, x_g_l_1),
                                          vmulq_f32(y_g_l_1, y_g_l_1)));
                        x_g_l_2 = vsqrtq_f32(
                                vaddq_f32(vmulq_f32(x_g_l_2, x_g_l_2),
                                          vmulq_f32(y_g_l_2, y_g_l_2)));
                        x_g_h_1 = vsqrtq_f32(
                                vaddq_f32(vmulq_f32(x_g_h_1, x_g_h_1),
                                          vmulq_f32(y_g_h_1, y_g_h_1)));
                        x_g_h_2 = vsqrtq_f32(
                                vaddq_f32(vmulq_f32(x_g_h_2, x_g_h_2),
                                          vmulq_f32(y_g_h_2, y_g_h_2)));

                        x_r_l_1 = vsqrtq_f32(
                                vaddq_f32(vmulq_f32(x_r_l_1, x_r_l_1),
                                          vmulq_f32(y_r_l_1, y_r_l_1)));
                        x_r_l_2 = vsqrtq_f32(
                                vaddq_f32(vmulq_f32(x_r_l_2, x_r_l_2),
                                          vmulq_f32(y_r_l_2, y_r_l_2)));
                        x_r_h_1 = vsqrtq_f32(
                                vaddq_f32(vmulq_f32(x_r_h_1, x_r_h_1),
                                          vmulq_f32(y_r_h_1, y_r_h_1)));
                        x_r_h_2 = vsqrtq_f32(
                                vaddq_f32(vmulq_f32(x_r_h_2, x_r_h_2),
                                          vmulq_f32(y_r_h_2, y_r_h_2)));

                        uint16x8_t b_low = vcombine_u16(vqmovun_s32(vcvtq_s32_f32(x_b_l_1)),
                                                        vqmovun_s32(vcvtq_s32_f32(x_b_l_2)));
                        uint16x8_t b_high = vcombine_u16(vqmovun_s32(vcvtq_s32_f32(x_b_h_1)),
                                                         vqmovun_s32(vcvtq_s32_f32(x_b_h_2)));
                        uint8x16_t b_out = vcombine_u8(vqmovn_u16(b_low), vqmovn_u16(b_high));

                        uint16x8_t g_low = vcombine_u16(vqmovun_s32(vcvtq_s32_f32(x_g_l_1)),
                                                        vqmovun_s32(vcvtq_s32_f32(x_g_l_2)));
                        uint16x8_t g_high = vcombine_u16(vqmovun_s32(vcvtq_s32_f32(x_g_h_1)),
                                                         vqmovun_s32(vcvtq_s32_f32(x_g_h_2)));
                        uint8x16_t g_out = vcombine_u8(vqmovn_u16(g_low), vqmovn_u16(g_high));

                        uint16x8_t r_low = vcombine_u16(vqmovun_s32(vcvtq_s32_f32(x_r_l_1)),
                                                        vqmovun_s32(vcvtq_s32_f32(x_r_l_2)));
                        uint16x8_t r_high = vcombine_u16(vqmovun_s32(vcvtq_s32_f32(x_r_h_1)),
                                                         vqmovun_s32(vcvtq_s32_f32(x_r_h_2)));
                        uint8x16_t r_out = vcombine_u8(vqmovn_u16(r_low), vqmovn_u16(r_high));

                        uint8x16x4_t ptr = vld4q_u8(src + y * stride + x * 4);
                        uint8x16_t a = ptr.val[3];

                        uint8x16x4_t out;
                        out.val[0] = r_out;
                        out.val[1] = g_out;
                        out.val[2] = b_out;
                        out.val[3] = a;

                        vst4q_u8(dst + y * stride + x * 4, out);
                    }
                }
            });
        }

    }

    void ImageProcessorSIMD::emboss_neon_simd_float(uint8_t *src, uint8_t *dst, size_t width,
                                                    size_t height, size_t stride) {
        std::vector<std::vector<float>> kernel = {
                {-2, -1, 0},
                {-1, 1,  1},
                {0,  1,  2}};
        uint32_t threads = std::thread::hardware_concurrency();
        if (threads < 2) threads = 4;
        int slice = (height - 1) / threads;
        ThreadPool pool{threads};
        for (int t = 0; t < threads; t++) {
            int yStart = 1 + t * slice;
            int yEnd = (t == threads - 1) ? height - 1 : yStart + slice;
            pool.enqueue_task([&, yStart, yEnd]() -> void {
                for (int y = yStart; y < yEnd; y++) {
                    for (int x = 1; x < width - 1; x += 16) {
                        // Getting the values for the pixels
                        float32x4_t r_l_1 = vdupq_n_f32(0.0);
                        float32x4_t r_l_2 = vdupq_n_f32(0.0);
                        float32x4_t r_h_1 = vdupq_n_f32(0.0);
                        float32x4_t r_h_2 = vdupq_n_f32(0.0);
                        float32x4_t g_l_1 = vdupq_n_f32(0.0);
                        float32x4_t g_l_2 = vdupq_n_f32(0.0);
                        float32x4_t g_h_1 = vdupq_n_f32(0.0);
                        float32x4_t g_h_2 = vdupq_n_f32(0.0);
                        float32x4_t b_l_1 = vdupq_n_f32(0.0);
                        float32x4_t b_l_2 = vdupq_n_f32(0.0);
                        float32x4_t b_h_1 = vdupq_n_f32(0.0);
                        float32x4_t b_h_2 = vdupq_n_f32(0.0);

                        float32x4_t r_o_l_1 = vdupq_n_f32(0.0);
                        float32x4_t r_o_l_2 = vdupq_n_f32(0.0);
                        float32x4_t r_o_h_1 = vdupq_n_f32(0.0);
                        float32x4_t r_o_h_2 = vdupq_n_f32(0.0);
                        float32x4_t g_o_l_1 = vdupq_n_f32(0.0);
                        float32x4_t g_o_l_2 = vdupq_n_f32(0.0);
                        float32x4_t g_o_h_1 = vdupq_n_f32(0.0);
                        float32x4_t g_o_h_2 = vdupq_n_f32(0.0);
                        float32x4_t b_o_l_1 = vdupq_n_f32(0.0);
                        float32x4_t b_o_l_2 = vdupq_n_f32(0.0);
                        float32x4_t b_o_h_1 = vdupq_n_f32(0.0);
                        float32x4_t b_o_h_2 = vdupq_n_f32(0.0);
                        for (int ky = -1; ky <= 1; ky++) {
                            for (int kx = -1; kx <= 1; kx++) {
                                float weight = kernel[ky + 1][kx + 1];
                                uint8x16x4_t ch = vld4q_u8(src + (y + ky) * stride + (x + kx) * 4);
                                r_l_1 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(
                                        vget_low_u8(ch.val[0])))));
                                r_l_2 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8(
                                        vget_low_u8(ch.val[0])))));
                                r_h_1 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(
                                        vget_high_u8(ch.val[0])))));
                                r_h_2 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8(
                                        vget_high_u8(ch.val[0])))));
                                g_l_1 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(
                                        vget_low_u8(ch.val[1])))));
                                g_l_2 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8(
                                        vget_low_u8(ch.val[1])))));
                                g_h_1 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(
                                        vget_high_u8(ch.val[1])))));
                                g_h_2 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8(
                                        vget_high_u8(ch.val[1])))));
                                b_l_1 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(
                                        vget_low_u8(ch.val[2])))));
                                b_l_2 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8(
                                        vget_low_u8(ch.val[2])))));
                                b_h_1 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(
                                        vget_high_u8(ch.val[2])))));
                                b_h_2 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8(
                                        vget_high_u8(ch.val[2])))));
                                // getting the gray for the values
                                float32x4_t gray_l_1 = vmulq_n_f32(r_l_1, .3);
                                gray_l_1 = vmlaq_n_f32(gray_l_1, g_l_1, .59);
                                gray_l_1 = vmlaq_n_f32(gray_l_1, b_l_1, .11);

                                r_o_l_1 = vmlaq_n_f32(r_o_l_1, gray_l_1, weight);
                                g_o_l_1 = vmlaq_n_f32(g_o_l_1, gray_l_1, weight);
                                b_o_l_1 = vmlaq_n_f32(b_o_l_1, gray_l_1, weight);

                                float32x4_t gray_l_2 = vmulq_n_f32(r_l_2, .3);
                                gray_l_2 = vmlaq_n_f32(gray_l_2, g_l_2, .59);
                                gray_l_2 = vmlaq_n_f32(gray_l_2, b_l_2, .11);

                                r_o_l_2 = vmlaq_n_f32(r_o_l_2, gray_l_2, weight);
                                g_o_l_2 = vmlaq_n_f32(g_o_l_2, gray_l_2, weight);
                                b_o_l_2 = vmlaq_n_f32(b_o_l_2, gray_l_2, weight);

                                float32x4_t gray_h_1 = vmulq_n_f32(r_h_1, .3);
                                gray_h_1 = vmlaq_n_f32(gray_h_1, g_h_1, .59);
                                gray_h_1 = vmlaq_n_f32(gray_h_1, b_h_1, .11);

                                r_o_h_1 = vmlaq_n_f32(r_o_h_1, gray_h_1, weight);
                                g_o_h_1 = vmlaq_n_f32(g_o_h_1, gray_h_1, weight);
                                b_o_h_1 = vmlaq_n_f32(b_o_h_1, gray_h_1, weight);

                                float32x4_t gray_h_2 = vmulq_n_f32(r_h_2, .3);
                                gray_h_2 = vmlaq_n_f32(gray_h_2, g_h_2, .59);
                                gray_h_2 = vmlaq_n_f32(gray_h_2, b_h_2, .11);

                                r_o_h_2 = vmlaq_n_f32(r_o_h_2, gray_h_2, weight);
                                g_o_h_2 = vmlaq_n_f32(g_o_h_2, gray_h_2, weight);
                                b_o_h_2 = vmlaq_n_f32(b_o_h_2, gray_h_2, weight);
                            }
                        }
                        float32x4_t min = vdupq_n_f32(0);
                        float32x4_t max = vdupq_n_f32(255);
                        float32x4_t add_val = vdupq_n_f32(128);
                        r_o_l_1 = vmaxq_f32(min, vminq_f32(vaddq_f32(r_o_l_1, add_val), max));
                        r_o_l_2 = vmaxq_f32(min, vminq_f32(vaddq_f32(r_o_l_2, add_val), max));
                        r_o_h_1 = vmaxq_f32(min, vminq_f32(vaddq_f32(r_o_h_1, add_val), max));
                        r_o_h_2 = vmaxq_f32(min, vminq_f32(vaddq_f32(r_o_h_2, add_val), max));

                        g_o_l_1 = vmaxq_f32(min, vminq_f32(vaddq_f32(g_o_l_1, add_val), max));
                        g_o_l_2 = vmaxq_f32(min, vminq_f32(vaddq_f32(g_o_l_2, add_val), max));
                        g_o_h_1 = vmaxq_f32(min, vminq_f32(vaddq_f32(g_o_h_1, add_val), max));
                        g_o_h_2 = vmaxq_f32(min, vminq_f32(vaddq_f32(g_o_h_2, add_val), max));

                        b_o_l_1 = vmaxq_f32(min, vminq_f32(vaddq_f32(b_o_l_1, add_val), max));
                        b_o_l_2 = vmaxq_f32(min, vminq_f32(vaddq_f32(b_o_l_2, add_val), max));
                        b_o_h_1 = vmaxq_f32(min, vminq_f32(vaddq_f32(b_o_h_1, add_val), max));
                        b_o_h_2 = vmaxq_f32(min, vminq_f32(vaddq_f32(b_o_h_2, add_val), max));
                        // setting back the values
                        uint16x8_t R_L = vcombine_u16(vqmovn_u32(vcvtq_s32_f32(r_o_l_1)),
                                                      vqmovn_u32(vcvtq_s32_f32(r_o_l_2)));
                        uint16x8_t R_H = vcombine_u16(vqmovn_u32(vcvtq_s32_f32(r_o_h_1)),
                                                      vqmovn_s32(vcvtq_s32_f32(r_o_h_2)));
                        uint8x16_t R = vcombine_u8(vmovn_u16(R_L), vmovn_u16(R_H));

                        uint16x8_t G_L = vcombine_u16(vqmovn_u32(vcvtq_s32_f32(g_o_l_1)),
                                                      vqmovn_u32(vcvtq_s32_f32(g_o_l_2)));
                        uint16x8_t G_H = vcombine_u16(vqmovn_u32(vcvtq_s32_f32(g_o_h_1)),
                                                      vqmovn_s32(vcvtq_s32_f32(g_o_h_2)));
                        uint8x16_t G = vcombine_u8(vmovn_u16(G_L), vmovn_u16(G_H));
                        uint16x8_t B_L = vcombine_u16(vqmovn_u32(vcvtq_s32_f32(b_o_l_1)),
                                                      vqmovn_u32(vcvtq_s32_f32(b_o_l_2)));
                        uint16x8_t B_H = vcombine_u16(vqmovn_u32(vcvtq_s32_f32(b_o_h_1)),
                                                      vqmovn_s32(vcvtq_s32_f32(b_o_h_2)));
                        uint8x16_t B = vcombine_u8(vmovn_u16(B_L), vmovn_u16(B_H));

                        uint8x16x4_t out;
                        out.val[0] = R;
                        out.val[1] = G;
                        out.val[2] = B;
                        out.val[3] = vld4q_u8(src + y * stride + x * 4).val[3];

                        vst4q_u8(dst + y * stride + x * 4, out);
                    }
                }
            });
        }
    }

    void ImageProcessorSIMD::convert_yuv_rgba_neon(uint8_t *yPixel, uint8_t *uPix,
                                                   uint8_t *vPix, uint8_t *dstRGBA,
                                                   size_t height, size_t width,
                                                   size_t yStride, size_t yDstStride,
                                                   size_t uRowStride,
                                                   size_t vRowStride, size_t vPixelStride,
                                                   size_t uPixelStride) {
        uint threads = std::thread::hardware_concurrency();
        if (threads < 1) threads = 4;
        if ((size_t) threads > height) threads = static_cast<unsigned>(height);
        int slice = height / threads;
        ThreadPool pool{threads};
        for (int t = 0; t < threads; t++) {
            int yStart = t * slice;
            int yEnd = t == threads - 1 ? height : yStart + slice;
            pool.enqueue_task([&, yStart, yEnd]() -> void {
                for (int y = yStart; y < yEnd; y++) {
                    const uint8_t *yRow = yPixel + y * yStride;
                    int chromaY = y >> 1;
                    const uint8_t *uRow = uPix + uRowStride * chromaY;
                    const uint8_t *vRow = vPix + vRowStride * chromaY;

                    uint8_t *dstRow = dstRGBA + y * yDstStride;

                    uint8x8_t ch_v_s;
                    uint8x8_t ch_u_s;

                    for (int x = 0; x < width; x += 16) {
                        if (width - x < 16) {
                            ImageProcessor::convert_yuv_rgba_scalar(yRow, vRow, uRow, dstRow,
                                                                    width, 1, yStride,
                                                                    uRowStride, vRowStride,
                                                                    uPixelStride, vPixelStride,
                                                                    x);
                            break;
                        }
                        uint8x16_t ch_y = vld1q_u8(yRow + x);
                        int chromaX = x >> 1;
                        //loading the values for u and v and handling the stride is not equal to 1 (ie. is not tightly packed)
                        if (uPixelStride == 1) {
                            ch_v_s = vld1_u8(vRow + chromaX);
                            ch_u_s = vld1_u8(uRow + chromaX);
                        } else if (uPixelStride == 2) {
                            if (chromaX <= (width >> 1) - 8) {
                                ch_v_s = get_real_uv_pattern_for_stride_two(
                                        vRow + vPixelStride * chromaX);
                                ch_u_s = get_real_uv_pattern_for_stride_two(
                                        uRow + uPixelStride * chromaX);
                            } else {
                                uint8_t temp_u[8], temp_v[8];
                                for (int i = 0; i < 8; i++) {
                                    temp_u[i] = uRow[(chromaX + i) * uPixelStride];
                                    temp_v[i] = vRow[(chromaX + i) * vPixelStride];
                                }
                                ch_v_s = vld1_u8(temp_v);
                                ch_u_s = vld1_u8(temp_u);
                            }

                        } else {
                            uint8_t temp_u[8], temp_v[8];
                            for (int i = 0; i < 8; i++) {
                                temp_u[i] = uRow[(chromaX + i) * uPixelStride];
                                temp_v[i] = vRow[(chromaX + i) * vPixelStride];
                            }
                            ch_v_s = vld1_u8(temp_v);
                            ch_u_s = vld1_u8(temp_u);
                        }


                        // duplicating the v and u for the correct functioning, two adjacent pixels need to be multiplied by the same value of u and v
                        uint8x8x2_t d_u = vzip_u8(ch_u_s, ch_u_s);
                        uint8x8x2_t d_v = vzip_u8(ch_v_s, ch_v_s);

                        uint8x16_t ch_u = vcombine_u8(d_u.val[0], d_u.val[1]);
                        uint8x16_t ch_v = vcombine_u8(d_v.val[0], d_v.val[1]);

                        int16x8_t ch_y_l = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(ch_y)));
                        int16x8_t ch_y_h = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(ch_y)));
                        int16x8_t ch_v_l = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(ch_v)));
                        int16x8_t ch_v_h = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(ch_v)));
                        int16x8_t ch_u_l = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(ch_u)));
                        int16x8_t ch_u_h = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(ch_u)));

                        ch_y_l = vsubq_s16(ch_y_l, vdupq_n_s16(16));
                        ch_y_h = vsubq_s16(ch_y_h, vdupq_n_s16(16));
                        ch_u_l = vsubq_s16(ch_u_l, vdupq_n_s16(128));
                        ch_u_h = vsubq_s16(ch_u_h, vdupq_n_s16(128));
                        ch_v_l = vsubq_s16(ch_v_l, vdupq_n_s16(128));
                        ch_v_h = vsubq_s16(ch_v_h, vdupq_n_s16(128));

                        int32x4_t r0 = vmull_n_s16(vget_low_s16(ch_y_l), 298);
                        r0 = vmlal_n_s16(r0, vget_low_s16(ch_v_l), 409);
                        r0 = vaddq_s32(r0, vdupq_n_s32(128));
                        r0 = vrshrq_n_s32(r0, 8);

                        int32x4_t r1 = vmull_n_s16(vget_high_s16(ch_y_l), 298);
                        r1 = vmlal_n_s16(r1, vget_high_s16(ch_v_l), 409);
                        r1 = vaddq_s32(r1, vdupq_n_s32(128));
                        r1 = vrshrq_n_s32(r1, 8);
                        int32x4_t r2 = vmull_n_s16(vget_low_s16(ch_y_h), 298);
                        r2 = vmlal_n_s16(r2, vget_low_s16(ch_v_h), 409);
                        r2 = vaddq_s32(r2, vdupq_n_s32(128));
                        r2 = vrshrq_n_s32(r2, 8);
                        int32x4_t r3 = vmull_n_s16(vget_high_s16(ch_y_h), 298);
                        r3 = vmlal_n_s16(r3, vget_high_s16(ch_v_h), 409);
                        r3 = vaddq_s32(r3, vdupq_n_s32(128));
                        r3 = vrshrq_n_s32(r3, 8);
                        int32x4_t g0 = vmull_n_s16(vget_low_s16(ch_y_l), 298);
                        g0 = vmlal_n_s16(g0, vget_low_s16(ch_u_l), -100);
                        g0 = vmlal_n_s16(g0, vget_low_s16(ch_v_l), -208);
                        g0 = vaddq_s32(g0, vdupq_n_s32(128));
                        g0 = vrshrq_n_s32(g0, 8);
                        int32x4_t g1 = vmull_n_s16(vget_high_s16(ch_y_l), 298);
                        g1 = vmlal_n_s16(g1, vget_high_s16(ch_u_l), -100);
                        g1 = vmlal_n_s16(g1, vget_high_s16(ch_v_l), -208);
                        g1 = vaddq_s32(g1, vdupq_n_s32(128));
                        g1 = vrshrq_n_s32(g1, 8);

                        int32x4_t g2 = vmull_n_s16(vget_low_s16(ch_y_h), 298);
                        g2 = vmlal_n_s16(g2, vget_low_s16(ch_u_h), -100);
                        g2 = vmlal_n_s16(g2, vget_low_s16(ch_v_h), -208);
                        g2 = vaddq_s32(g2, vdupq_n_s32(128));
                        g2 = vrshrq_n_s32(g2, 8);

                        int32x4_t g3 = vmull_n_s16(vget_high_s16(ch_y_h), 298);
                        g3 = vmlal_n_s16(g3, vget_high_s16(ch_u_h), -100);
                        g3 = vmlal_n_s16(g3, vget_high_s16(ch_v_h), -208);
                        g3 = vaddq_s32(g3, vdupq_n_s32(128));
                        g3 = vrshrq_n_s32(g3, 8);

                        int32x4_t b0 = vmull_n_s16(vget_low_s16(ch_y_l), 298);
                        b0 = vmlal_n_s16(b0, vget_low_s16(ch_u_l), 516);
                        b0 = vaddq_s32(b0, vdupq_n_s32(128));
                        b0 = vrshrq_n_s32(b0, 8);

                        int32x4_t b1 = vmull_n_s16(vget_high_s16(ch_y_l), 298);
                        b1 = vmlal_n_s16(b1, vget_high_s16(ch_u_l), 516);
                        b1 = vaddq_s32(b1, vdupq_n_s32(128));
                        b1 = vrshrq_n_s32(b1, 8);

                        int32x4_t b2 = vmull_n_s16(vget_low_s16(ch_y_h), 298);
                        b2 = vmlal_n_s16(b2, vget_low_s16(ch_u_h), 516);
                        b2 = vaddq_s32(b2, vdupq_n_s32(128));
                        b2 = vrshrq_n_s32(b2, 8);

                        int32x4_t b3 = vmull_n_s16(vget_high_s16(ch_y_h), 298);
                        b3 = vmlal_n_s16(b3, vget_high_s16(ch_u_h), 516);
                        b3 = vaddq_s32(b3, vdupq_n_s32(128));
                        b3 = vrshrq_n_s32(b3, 8);
                        // scaling back to 8x16
                        uint16x8_t r_low = vcombine_u16(vqmovun_s32(r0), vqmovun_s32(r1));
                        uint16x8_t r_high = vcombine_u16(vqmovun_s32(r2), vqmovun_s32(r3));
                        uint8x16_t r = vcombine_u8(vqmovn_u16(r_low), vqmovn_u16(r_high));

                        uint16x8_t g_low = vcombine_u16(vqmovun_s32(g0), vqmovun_s32(g1));
                        uint16x8_t g_high = vcombine_u16(vqmovun_s32(g2), vqmovun_s32(g3));
                        uint8x16_t g = vcombine_u8(vqmovn_u16(g_low), vqmovn_u16(g_high));

                        uint16x8_t b_low = vcombine_u16(vqmovun_s32(b0), vqmovun_s32(b1));
                        uint16x8_t b_high = vcombine_u16(vqmovun_s32(b2), vqmovun_s32(b3));
                        uint8x16_t b = vcombine_u8(vqmovn_u16(b_low), vqmovn_u16(b_high));

                        uint8x16x4_t out;
                        out.val[0] = r;
                        out.val[1] = g;
                        out.val[2] = b;
                        out.val[3] = vdupq_n_u8(255);

                        vst4q_u8(dstRow + x * 4, out);

                    }
                }
            });
        }
    }

    uint8x8_t ImageProcessorSIMD::get_real_uv_pattern_for_stride_two(const uint8_t *src) {
        uint8x16_t raw = vld1q_u8(src);

        uint8x8_t realIndex = {0, 2, 4, 6, 8, 10, 12, 14};
        return vqtbl1_u8(raw, realIndex);
    }

#ifndef __ANDROID__
    bool ImageProcessorSIMD::device_support_neon() {
        return false;
    }
#else

#include <cpu-features.h>

    bool ImageProcessorSIMD::device_support_neon() {
        AndroidCpuFamily family = android_getCpuFamily();
        uint64_t features = android_getCpuFeatures();
        return (family == ANDROID_CPU_FAMILY_ARM64) ||
               (family == ANDROID_CPU_FAMILY_ARM && (features & ANDROID_CPU_ARM_FEATURE_NEON));
    }


#endif
}