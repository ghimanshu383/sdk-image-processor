package com.os.imageprocessor;

import android.graphics.Bitmap;

public class JniBridge {
    static {
        System.loadLibrary("core_native_image_processor");
    }

    public static native boolean GrayScaleImage(Bitmap bitmap, boolean optimizeNeon);

    public static native boolean CreateNegative(Bitmap bitmap, boolean optimizeNeon);

    public static native boolean BlurImage(Bitmap bitmap, int radius, int sigma, boolean optimizeNeon);

    public static native boolean Embross(Bitmap bitmap, boolean optimizeNeon);

    public static native boolean Sharpen(Bitmap bitmap, boolean optimizeNeon);

    public static native boolean EdgeDetection(Bitmap bitmap, boolean optimizeNeon);

    public static native void convert_yuv_rgba(byte[] yPixels, byte[] vPixels, byte[] uPixels, Bitmap outBitmap,
                                               int width, int height, int yStride, int dstStride,
                                               int uRowStride, int vRowStride, int uPixelStride, int vPixelStride, boolean optimizeNeon);
}
