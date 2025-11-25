# Sdk-image-processor

This repository contains the native C++ code and Kotlin bindings for a lightweight, offline image-processing SDK for Android.
The goal of the SDK is to provide fast, dependency-free image effects using NDK, ARM NEON SIMD, and a small multi-threaded pipeline suitable for real-time CameraX usage.

**The SDK exposes both SIMD-optimized and scalar implementations for all filters, and includes prebuilt binaries for direct integration.**

## 1. Overview

The SDK receives images in YUV (CameraX) or RGBA8888 format and applies various filters.
A dedicated NEON pipeline is used for vectorized operations, and a small custom ThreadPool parallelizes the work across available CPU cores.

**Supported filters (SIMD + scalar versions):**

- Grayscale

- Negative

- Blur

- Sharpen

- Edge detection

- Emboss

- The YUV → RGBA conversion is also optimized using NEON, but a scalar fallback is available for unsupported paths.

Kotlin API and the JNI bridge are included for easy integration from Android apps.

## 2. Features
**SIMD Optimized Filters (ARM NEON)**

- Vectorized processing using 16-pixel batches

- vld4q / vst4q for interleaved loads/stores

- SIMD acceleration for convolution kernels

- Auto fallback to scalar code upon misalignment

**Multithreaded Pipeline**

- Simple C++ thread pool

- Each filter splits the image into row-ranges

- Parallel execution across available cores

- Suitable for real-time camera preview filters

**CameraX YUV Support**

- Fast YUV_420_888 → RGBA8888 path

- NEON-optimized conversion (optional)

- Designed for live frame processing

## 3. Native Pipeline

The general pipeline for CameraX frames is:

1. Receive YUV420 frame from Kotlin

2. Convert YUV → RGBA (NEON or scalar)

3. Divide image into row chunks

4. For each chunk:

    - Load RGBA pixels

    - Apply selected filter (SIMD or scalar)

    - Write output

    - Join threads

5. Return processed buffer back to Kotlin

**All SIMD paths operate on 128-bit vectors handling 16 pixels at a time.**

## 4. Kotlin API

A simple Kotlin wrapper is provided:

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
        ): Bitmap

        suspend fun convertYuvToRGBA(
            yPixels: ByteArray,
            vPixels: ByteArray,
            uPixels: ByteArray,
            outBitmap: Bitmap,
            width: Int,
            height: Int,
            yStride: Int,
            dstStride: Int,
            uRowStride: Int,
            vRowStride: Int,
            uPixelStride: Int,
            vPixelStride: Int,
            optimizeNeon: Boolean
        ) = withContext(Dispatchers.Default) {
            JniBridge.convert_yuv_rgba(
                yPixels,
                vPixels,
                uPixels,
                outBitmap,
                width,
                height,
                yStride,
                dstStride,
                uRowStride,
                vRowStride,
                uPixelStride,
                vPixelStride,
                optimizeNeon
            )
        }
Returned data is always RGBA8888.

The API can be used from:

- CameraX analyzer

- Image editors

- Custom pipelines

## 5. SIMD Implementation Details

The NEON code uses:

- uint8x16x4_t for loading 16 RGBA pixels

- vld4q_u8 / vst4q_u8 for interleaved memory access

- Saturated arithmetic for kernel outputs

- 128-bit parallel lanes

- Scalar fallback is automatically triggered when:

    - Width is not divisible by vector width

    - Device lacks NEON (rare on modern devices)

## 6. ThreadPool

The SDK includes a tiny C++17 ThreadPool:

- Thread count = hardware_concurrency

- Each worker receives a row range

- No futures/promises in this version (simple synchronous dispatch)

- Designed to avoid memory allocations inside hot loops

- This allows real-time filters even for 1080p frames.

## 7. Using this SDK

#### 1. Add the AAR

Place the AAR inside any folder in your project and add:

implementation(files("folderName/native-image-processor-sdk-1.0.0.aar"))

### 2. CameraX Dependencies

(Only needed if using the CameraX demo included in kotlin-api)

var camerax_version = "1.3.1"

api("androidx.camera:camera-core:$camerax_version")

api("androidx.camera:camera-camera2:$camerax_version")

api("androidx.camera:camera-lifecycle:$camerax_version")
 
api("androidx.camera:camera-video:$camerax_version")
 
api("androidx.camera:camera-view:$camerax_version")
 
api("androidx.camera:camera-extensions:$camerax_version")


### 3. Example Usage

CoroutineScope(Dispatchers.IO).launch {

    val result = NativeImageProcessor.processImage(
    
        applicationContext,
        
        sourceBitmap,
        
        NativeImageProcessor.PROCESS_TYPE.BLUR,
        
        optimizeNeon = true,
        
        radius = 3,
        
        sigma = 5
    )
}


CameraX → YUV integration example is provided inside the kotlin-api directory.

## 8. Notes & Limitations

- Filters are CPU-based (no GPU/Vulkan compute in this version)

- NEON SIMD requires ARM64 (arm64-v8a)

- Heavy filters like blur may cost more time on very large images

- YUV conversion performance depends on memory stride/alignment

## 9. Future Paths and updates

- Add Vulkan compute path for GPU-accelerated filters

- Add real-time beauty filters

- Add face detection + effect pipeline


## 📊 Performance Benchmarks (High-Resolution Image)

| Filter               | Scalar (ms) | NEON SIMD (ms) | Speedup        |
|---------------------|-------------|----------------|----------------|
| **Grayscale**       | 26.94       | 39.29          | 0.7× (Scalar faster) |
| **Negative**        | 12.44       | 26.00          | 0.48× (Scalar faster) |
| **Blur 3×3**        | 357.54      | 108.65         | **3.29× faster** |
| **Blur 5×5**        | 1540.39     | 325.19         | **4.73× faster** |
| **Blur 9×9**        | 2952.92     | 391.53         | **7.54× faster** |
| **Sharpen (3×3)**   | 193.18      | 100.97         | **1.91× faster** |
| **Emboss**          | 565.00      | 91.62          | **6.17× faster** |
| **Sobel Edge**      | 138.28      | 110.14         | **1.25× faster** |

### Notes
- Simple per-pixel operations (grayscale, negative) are **memory-bound**, so NEON does not provide a large benefit and may be slower due to interleaved vector load overhead (`vld4q_u8`).
- Convolution-based filters (blur, sharpen, emboss, sobel) are **compute-bound** and benefit heavily from NEON SIMD.
- Larger Gaussian kernels show exponential SIMD gains (up to 7–8×).
- All benchmarks were measured on a high-resolution camera image (~12MP), so absolute times are naturally higher than 1080p/720p results.





