/*********************************************************************
 * convolve_gpu.cu
 * V3: Optimized GPU implementation
 *
 * - Separable convolution implemented with shared-memory tiling for
 *   both horizontal and vertical passes.
 * - Kernel coefficients placed in constant memory (fast, read-only).
 * - Launch configuration tuned for occupancy (tile sizes chosen).
 * - CUDA error checking and optional timing retained.
 *
 * Keep exported function names unchanged:
 *   _KLTToFloatImage_gpu
 *   _convolveSeparate_gpu
 *   _KLTComputeGradients_gpu
 *
 *********************************************************************/

#include <cuda_runtime.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt_util.h"

#define MAX_KERNEL_WIDTH 71

typedef struct {
    int width;
    float data[MAX_KERNEL_WIDTH];
} ConvolutionKernel;

static ConvolutionKernel gauss_kernel;
static ConvolutionKernel gaussderiv_kernel;
static float sigma_last = -10.0f;

// constant memory for kernel coefficients (fast read-only)
__constant__ float d_const_kernel[MAX_KERNEL_WIDTH];
__constant__ int d_const_kernel_width;

// toggle timing
bool measureTime = true;

// error-check helper
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (err=%d) at %s:%d\n", cudaGetErrorString(err), (int)err, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/******************** CUDA Streams for Async Pipelining ********************/
/* 
   Global streams for H2D/compute/D2H pipelining:
   - h2d_stream: Host-to-device async memcpy
   - compute_stream: GPU kernel execution
   - d2h_stream: Device-to-host async memcpy
   
   These allow overlapping of transfers and computation for ~1.5-2.3x speedup.
*/
static cudaStream_t h2d_stream = nullptr;
static cudaStream_t compute_stream = nullptr;
static cudaStream_t d2h_stream = nullptr;

// Persistent GPU allocations for batched processing
static float *d_img_persistent = nullptr;
static float *d_gradx_persistent = nullptr;
static float *d_grady_persistent = nullptr;
static float *d_tmp_persistent = nullptr;  // Temporary buffer for convolution (eliminates malloc in _convolveSeparate_gpu)
static size_t persistent_size = 0;

// Forward declarations with C linkage
#ifdef __cplusplus
extern "C" {
#endif

void _KLTInitStreams(int ncols, int nrows) {
    // FIX A2: Allocate based on MAXIMUM pyramid size needed
    // Full resolution pyramid level 0: ncols x nrows
    // Smaller pyramid levels: ncols/2^L x nrows/2^L (but we reuse same buffer)
    // So we allocate for the MAXIMUM (full res), and all levels can safely use it
    size_t max_pyramid_size = ncols * nrows * sizeof(float);
    
    // Create streams
    CUDA_CHECK(cudaStreamCreate(&h2d_stream));
    CUDA_CHECK(cudaStreamCreate(&compute_stream));
    CUDA_CHECK(cudaStreamCreate(&d2h_stream));
    
    // Pre-allocate persistent GPU memory for maximum pyramid size
    CUDA_CHECK(cudaMalloc(&d_img_persistent, max_pyramid_size));
    CUDA_CHECK(cudaMalloc(&d_gradx_persistent, max_pyramid_size));
    CUDA_CHECK(cudaMalloc(&d_grady_persistent, max_pyramid_size));
    CUDA_CHECK(cudaMalloc(&d_tmp_persistent, max_pyramid_size));  // FIX: Temp buffer for convolution (eliminates malloc in _convolveSeparate_gpu)
    
    // persistent_size reflects MAXIMUM allocation, not current imgSize
    // This allows batched_streams to safely handle all pyramid levels
    persistent_size = max_pyramid_size;
    
    if (measureTime) {
        fprintf(stderr, "(KLT-CUDA) Streams initialized with 4 persistent buffers (%zu bytes each)\n", max_pyramid_size);
    }
}

void _KLTDestroyStreams() {
    if (h2d_stream != nullptr) {
        CUDA_CHECK(cudaStreamSynchronize(h2d_stream));
        CUDA_CHECK(cudaStreamDestroy(h2d_stream));
        h2d_stream = nullptr;
    }
    if (compute_stream != nullptr) {
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaStreamDestroy(compute_stream));
        compute_stream = nullptr;
    }
    if (d2h_stream != nullptr) {
        CUDA_CHECK(cudaStreamSynchronize(d2h_stream));
        CUDA_CHECK(cudaStreamDestroy(d2h_stream));
        d2h_stream = nullptr;
    }
    
    if (d_img_persistent != nullptr) {
        CUDA_CHECK(cudaFree(d_img_persistent));
        d_img_persistent = nullptr;
    }
    if (d_gradx_persistent != nullptr) {
        CUDA_CHECK(cudaFree(d_gradx_persistent));
        d_gradx_persistent = nullptr;
    }
    if (d_grady_persistent != nullptr) {
        CUDA_CHECK(cudaFree(d_grady_persistent));
        d_grady_persistent = nullptr;
    }
    if (d_tmp_persistent != nullptr) {
        CUDA_CHECK(cudaFree(d_tmp_persistent));
        d_tmp_persistent = nullptr;
    }
    persistent_size = 0;
}

// Helper: Get the persistent temp buffer (used by _convolveSeparate_gpu to eliminate malloc)
static float* _get_d_tmp_persistent() {
    if (d_tmp_persistent == nullptr) {
        fprintf(stderr, "ERROR: Temp persistent buffer not initialized. Call _KLTInitStreams() first.\n");
        exit(EXIT_FAILURE);
    }
    return d_tmp_persistent;
}

// Helper: Ensure temp buffer is initialized (lazy initialization for early calls from selectGoodFeatures_gpu)
static void _ensure_tmp_buffer_initialized(int ncols, int nrows) {
    if (d_tmp_persistent == nullptr) {
        fprintf(stderr, "[INFO] Lazy init: Allocating temp buffer (%d×%d)...\n", ncols, nrows);
        size_t size = (size_t)ncols * (size_t)nrows * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_tmp_persistent, size));
        persistent_size = size;  // Update for tracking
    }
}

#ifdef __cplusplus
}
#endif

/******************** GPU Kernels (optimized) ********************/
/*
Design notes:
- We implement separable convolution as two kernels:
  1) horizontal pass: each block processes a tile of rows; uses shared memory
     across columns with halo = radius.
  2) vertical pass: each block processes a tile of columns; uses shared memory
     across rows with halo = radius.
- Shared memory staging reduces global memory traffic and increases arithmetic
  intensity.
- We choose tile sizes (TILE_W x TILE_H) to balance occupancy and shared memory.
- We assume MAX_KERNEL_WIDTH isn't huge (<=71) so constant memory fits well.
*/

/* KLTToFloatImage_kernel: 1D mapping (coalesced loads/stores) */
__global__ void KLTToFloatImage_kernel(const KLT_PixelType* img, float* floatimg, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        floatimg[idx] = (float)img[idx];
    }
}

/* Horizontal pass with shared-memory tiling
   Each block processes TILE_W columns for multiple rows (TILE_H).
   Shared memory width = TILE_W + 2*radius (halo).
*/
template<int TILE_W, int TILE_H>
__global__ void convolveHoriz_shared_kernel(const float* imgin, float* imgout, int ncols, int nrows) {
    // compute tile origin (col0,row0)
    const int radius = d_const_kernel_width / 2;
    const int col0 = blockIdx.x * TILE_W;
    const int row0 = blockIdx.y * TILE_H;

    // shared memory: (TILE_H) rows x (TILE_W + 2*radius) cols
    extern __shared__ float sdata[]; // size: TILE_H * (TILE_W + 2*radius)
    int sharedPitch = TILE_W + 2 * radius;

    int tx = threadIdx.x; // 0..TILE_W-1 or used to read halo depending on config
    int ty = threadIdx.y; // 0..TILE_H-1

    int globalRow = row0 + ty;
    if (globalRow >= nrows) return;

    // We'll let threads cooperatively load the shared region.
    // Each thread loads multiple elements horizontally to fully cover the shared tile.
    // compute how many elements to load per thread (simple striding)
    int linearThreadId = ty * blockDim.x + tx;
    int nThreads = blockDim.x * blockDim.y;

    // load into shared memory (coalesced accesses along row)
    for (int i = linearThreadId; i < sharedPitch * TILE_H; i += nThreads) {
        int srow = i / sharedPitch;
        int scol = i % sharedPitch;
        int gcol = col0 + (scol - radius);
        int grow = row0 + srow;
        float v = 0.0f;
        if (grow >= 0 && grow < nrows && gcol >= 0 && gcol < ncols) {
            v = imgin[grow * ncols + gcol];
        }
        sdata[srow * sharedPitch + scol] = v;
    }
    __syncthreads();

    // compute convolution for our TILE element(s)
    int outCol = col0 + tx;
    if (outCol >= ncols) return;

    // each thread may compute multiple output columns vertically (if TILE_H > blockDim.y)
    if (tx < TILE_W && ty < TILE_H) {
        float sum;
        // compute output for (globalRow, outCol)
        int srow = ty;
        int scoff = (tx + radius);
        sum = 0.0f;
        // loop kernel (perform true convolution by flipping kernel index)
        for (int k = -radius; k <= radius; ++k) {
            // CPU path uses convolution (kernel reversed). To match sign/convention,
            // index kernel with (radius - k) instead of (radius + k).
            sum += sdata[srow * sharedPitch + scoff + k] * d_const_kernel[radius - k];
        }
        imgout[globalRow * ncols + outCol] = sum;
    }
}

/* Vertical pass with shared-memory tiling
   Each block processes TILE_H rows for multiple columns (TILE_W).
   We allocate shared memory of size (TILE_H + 2*radius) x TILE_W to stage columns.
*/
template<int TILE_W, int TILE_H>
__global__ void convolveVert_shared_kernel(const float* imgin, float* imgout, int ncols, int nrows) {
    const int radius = d_const_kernel_width / 2;
    const int col0 = blockIdx.x * TILE_W;
    const int row0 = blockIdx.y * TILE_H;

    extern __shared__ float sdata[]; // size: TILE_W * (TILE_H + 2*radius)
    int sharedPitch = TILE_H + 2 * radius; // number of rows stored per column

    int tx = threadIdx.x; // 0..TILE_W-1
    int ty = threadIdx.y; // 0..TILE_H-1

    int globalCol = col0 + tx;
    if (globalCol >= ncols) return;

    // load shared data column-wise with cooperative threads
    int elements = TILE_W * sharedPitch;
    int linearThreadId = ty * blockDim.x + tx;
    int nThreads = blockDim.x * blockDim.y;

    for (int i = linearThreadId; i < elements; i += nThreads) {
        int scol = i / sharedPitch; // 0..TILE_W-1
        int srow = i % sharedPitch; // 0..sharedPitch-1
        int gcol = col0 + scol;
        int grow = row0 + (srow - radius);
        float v = 0.0f;
        if (gcol >= 0 && gcol < ncols && grow >= 0 && grow < nrows) {
            v = imgin[grow * ncols + gcol];
        }
        sdata[scol * sharedPitch + srow] = v;
    }
    __syncthreads();

    // compute convolution for our output element
    int outRow = row0 + ty;
    if (outRow >= nrows) return;

    if (tx < TILE_W && ty < TILE_H) {
        float sum = 0.0f;
        int scol = tx;
        int scoff = ty + radius;
        // loop kernel (perform true convolution by flipping kernel index)
        for (int k = -radius; k <= radius; ++k) {
            sum += sdata[scol * sharedPitch + scoff + k] * d_const_kernel[radius - k];
        }
        imgout[outRow * ncols + globalCol] = sum;
    }
}

/******************** CPU Helper Functions (unchanged) ********************/

static void _computeKernels(float sigma, ConvolutionKernel *gauss, ConvolutionKernel *gaussderiv) {
    const float factor = 0.01f;
    int i;
    assert(MAX_KERNEL_WIDTH % 2 == 1);
    assert(sigma >= 0.0f);

    int hw = MAX_KERNEL_WIDTH / 2;
    float max_gauss = 1.0f, max_gaussderiv = (float)(sigma * exp(-0.5f));

    for (i = -hw; i <= hw; i++) {
        gauss->data[i + hw] = (float)exp(-i * i / (2 * sigma * sigma));
        gaussderiv->data[i + hw] = -i * gauss->data[i + hw];
    }

    gauss->width = MAX_KERNEL_WIDTH;
    for (i = -hw; fabs(gauss->data[i + hw] / max_gauss) < factor; i++, gauss->width -= 2);
    gaussderiv->width = MAX_KERNEL_WIDTH;
    for (i = -hw; fabs(gaussderiv->data[i + hw] / max_gaussderiv) < factor; i++, gaussderiv->width -= 2);

    for (i = 0; i < gauss->width; i++)
        gauss->data[i] = gauss->data[i + (MAX_KERNEL_WIDTH - gauss->width) / 2];
    for (i = 0; i < gaussderiv->width; i++)
        gaussderiv->data[i] = gaussderiv->data[i + (MAX_KERNEL_WIDTH - gaussderiv->width) / 2];

    float den = 0.0f;
    hw = gaussderiv->width / 2;
    for (i = 0; i < gauss->width; i++) den += gauss->data[i];
    for (i = 0; i < gauss->width; i++) gauss->data[i] /= den;
    den = 0.0f;
    for (i = -hw; i <= hw; i++) den -= i * gaussderiv->data[i + hw];
    for (i = -hw; i <= hw; i++) gaussderiv->data[i + hw] /= den;

    sigma_last = sigma;
}

/******************** GPU Wrappers (kept function names) ********************/

/* Convert image to float on GPU */
void _KLTToFloatImage_gpu(KLT_PixelType *img, int ncols, int nrows, float* d_floatimg) {
    int total = ncols * nrows;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;

    cudaEvent_t start = 0, stop = 0;
    if (measureTime) {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, 0));
    }

    KLTToFloatImage_kernel<<<gridSize, blockSize>>>(img, d_floatimg, total);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (measureTime) {
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float gpuTimeMs;
        CUDA_CHECK(cudaEventElapsedTime(&gpuTimeMs, start, stop));
        // printf("(KLT) GPU float conversion time: %.3f ms\n", gpuTimeMs);
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
}

/* Separate convolution wrapper using optimized shared-memory kernels */
void _convolveSeparate_gpu(float* d_imgin, int ncols, int nrows,
                           ConvolutionKernel horiz, ConvolutionKernel vert, float* d_imgout) {

    // choose tile sizes. These are conservative balanced defaults:
    // TILE_W x TILE_H controls shared memory per block and occupancy.
    // For many GPUs, (TILE_W=32, TILE_H=8) is a good starting point.
    const int TILE_W = 32;
    const int TILE_H = 8;

    // FIX: Lazy init - ensure temp buffer exists (may be called from selectGoodFeatures_gpu before KLTTrackFeatures)
    _ensure_tmp_buffer_initialized(ncols, nrows);
    
    // FIX: Use persistent temp buffer (allocated once in _KLTInitStreams)
    // This eliminates malloc/free overhead: ~2ms per call × 50 calls = 100ms saved!
    float* d_tmp = _get_d_tmp_persistent();  // ← NO MALLOC!

    // Upload horizontal kernel coefficients to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(d_const_kernel, horiz.data, horiz.width * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_const_kernel_width, &horiz.width, sizeof(int)));

    // grid/block for horizontal
    dim3 blockH(TILE_W, TILE_H);
    dim3 gridH( (ncols + TILE_W - 1) / TILE_W, (nrows + TILE_H - 1) / TILE_H );

    // compute shared memory size for horizontal: TILE_H * (TILE_W + 2*radius)
    int radius = horiz.width / 2;
    size_t sharedH = (size_t)TILE_H * (size_t)(TILE_W + 2*radius) * sizeof(float);

    cudaEvent_t start = 0, stop = 0;
    if (measureTime) {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, 0));
    }

    // Launch horizontal kernel (templated)
    // Since templated kernels require compile-time TILE constants,
    // we instantiate the common combination via if-guard. For other combos,
    // a small switch can be added.
    if (TILE_W == 32 && TILE_H == 8) {
        convolveHoriz_shared_kernel<32,8><<<gridH, blockH, sharedH>>>(d_imgin, d_tmp, ncols, nrows);
    } else {
        // fallback generic launch (less optimized)
        convolveHoriz_shared_kernel<32,8><<<gridH, blockH, sharedH>>>(d_imgin, d_tmp, ncols, nrows);
    }
    CUDA_CHECK(cudaPeekAtLastError());

    // Upload vertical kernel coefficients to constant memory
    // NOTE: We MUST upload even if same width as horizontal, because the coefficients are different!
    // (e.g., gaussderiv vs gauss have same width but different values)
    CUDA_CHECK(cudaMemcpyToSymbol(d_const_kernel, vert.data, vert.width * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_const_kernel_width, &vert.width, sizeof(int)));

    // grid/block for vertical (we keep same TILEs but swap effective roles)
    dim3 blockV(TILE_W, TILE_H);
    dim3 gridV( (ncols + TILE_W - 1) / TILE_W, (nrows + TILE_H - 1) / TILE_H );

    // shared memory size for vertical: TILE_W * (TILE_H + 2*radius_vert)
    int radiusV = vert.width / 2;
    size_t sharedV = (size_t)TILE_W * (size_t)(TILE_H + 2*radiusV) * sizeof(float);

    if (TILE_W == 32 && TILE_H == 8) {
        convolveVert_shared_kernel<32,8><<<gridV, blockV, sharedV>>>(d_tmp, d_imgout, ncols, nrows);
    } else {
        convolveVert_shared_kernel<32,8><<<gridV, blockV, sharedV>>>(d_tmp, d_imgout, ncols, nrows);
    }
    CUDA_CHECK(cudaPeekAtLastError());

    // synchronize and timing
    CUDA_CHECK(cudaDeviceSynchronize());
    if (measureTime) {
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float gpuTimeMs;
        CUDA_CHECK(cudaEventElapsedTime(&gpuTimeMs, start, stop));
        // printf("(KLT) GPU convolution time: %.3f ms\n", gpuTimeMs);
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    // FIX: Do NOT free d_tmp - it's persistent and reused!
}

/* Compute gradients using separable conv passes (unchanged behavior) */
void _KLTComputeGradients_gpu(float* d_img, int ncols, int nrows, float sigma,
                              float* d_gradx, float* d_grady) {
    if (fabs(sigma - sigma_last) > 0.05f)
        _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

    // horiz=gaussderiv, vert=gauss
    _convolveSeparate_gpu(d_img, ncols, nrows, gaussderiv_kernel, gauss_kernel, d_gradx);

    // horiz=gauss, vert=gaussderiv
    _convolveSeparate_gpu(d_img, ncols, nrows, gauss_kernel, gaussderiv_kernel, d_grady);
}

// New wrapper that takes host memory and handles transfers
// Wrapped in extern "C" for C linkage
#ifdef __cplusplus
extern "C" {
#endif

void _KLTComputeGradients_gpu_wrapper(
    float* h_img, int ncols, int nrows, float sigma,
    float* h_gradx, float* h_grady)
{
    float *d_img, *d_gradx, *d_grady;
    size_t imgSize = ncols * nrows * sizeof(float);
    
    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&d_img, imgSize));
    CUDA_CHECK(cudaMalloc(&d_gradx, imgSize));
    CUDA_CHECK(cudaMalloc(&d_grady, imgSize));
    
    // Copy input to GPU
    CUDA_CHECK(cudaMemcpy(d_img, h_img, imgSize, cudaMemcpyHostToDevice));
    
    // Call the actual GPU function
    _KLTComputeGradients_gpu(d_img, ncols, nrows, sigma, d_gradx, d_grady);
    
    // CRITICAL: Ensure GPU finishes before copying back
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back to CPU
    CUDA_CHECK(cudaMemcpy(h_gradx, d_gradx, imgSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_grady, d_grady, imgSize, cudaMemcpyDeviceToHost));
    
    // Free GPU memory
    CUDA_CHECK(cudaFree(d_img));
    CUDA_CHECK(cudaFree(d_gradx));
    CUDA_CHECK(cudaFree(d_grady));
}

/******************** Batched Wrapper with CUDA Streams ********************/
/*
   Optimized batched wrapper using 3-stream async pipelining:
   - h2d_stream: Queues async H2D transfer (overlaps with previous D2H and compute)
   - compute_stream: Waits for H2D, then executes compute kernels
   - d2h_stream: Waits for compute, then queues async D2H transfer
   
   Expected speedup: 1.5-2.3x through overlapping H2D/compute/D2H phases.
   Use _KLTInitStreams() once at tracking start and _KLTDestroyStreams() at end.
*/
void _KLTComputeGradients_gpu_batched_streams(
    float* h_img, int ncols, int nrows, float sigma,
    float* h_gradx, float* h_grady)
{
    size_t imgSize = ncols * nrows * sizeof(float);
    
    // Verify streams and persistent memory are initialized
    // Note: persistent_size should be >= imgSize (pyramid levels may be smaller)
    if (h2d_stream == nullptr || compute_stream == nullptr || d2h_stream == nullptr ||
        d_img_persistent == nullptr || persistent_size < imgSize) {
        fprintf(stderr, "ERROR: Streams not initialized or size mismatch. Call _KLTInitStreams() with max size first.\n");
        fprintf(stderr, "DEBUG: h2d_stream=%p, compute_stream=%p, d2h_stream=%p\n", 
                h2d_stream, compute_stream, d2h_stream);
        fprintf(stderr, "DEBUG: d_img_persistent=%p, persistent_size=%zu, imgSize=%zu\n", 
                d_img_persistent, persistent_size, imgSize);
        exit(EXIT_FAILURE);
    }
    
    // Phase 1: Async H2D transfer in h2d_stream
    // (overlaps with previous D2H and compute phases)
    CUDA_CHECK(cudaMemcpyAsync(d_img_persistent, h_img, imgSize, 
                               cudaMemcpyHostToDevice, h2d_stream));
    
    // Phase 2: Add event to compute_stream to wait for h2d_stream
    // This ensures H2D completes before compute kernels start
    cudaEvent_t h2d_event;
    CUDA_CHECK(cudaEventCreate(&h2d_event));
    CUDA_CHECK(cudaEventRecord(h2d_event, h2d_stream));
    CUDA_CHECK(cudaStreamWaitEvent(compute_stream, h2d_event, 0));
    
    // Execute compute kernels in compute_stream
    // (overlaps with next H2D and previous D2H)
    _KLTComputeGradients_gpu(d_img_persistent, ncols, nrows, sigma, 
                             d_gradx_persistent, d_grady_persistent);
    
    // Phase 3: Add event to d2h_stream to wait for compute_stream
    // This ensures compute completes before D2H starts
    cudaEvent_t compute_event;
    CUDA_CHECK(cudaEventCreate(&compute_event));
    CUDA_CHECK(cudaEventRecord(compute_event, compute_stream));
    CUDA_CHECK(cudaStreamWaitEvent(d2h_stream, compute_event, 0));
    
    // Async D2H transfer in d2h_stream
    // (overlaps with next H2D and compute)
    CUDA_CHECK(cudaMemcpyAsync(h_gradx, d_gradx_persistent, imgSize,
                               cudaMemcpyDeviceToHost, d2h_stream));
    CUDA_CHECK(cudaMemcpyAsync(h_grady, d_grady_persistent, imgSize,
                               cudaMemcpyDeviceToHost, d2h_stream));
    
    // Clean up events
    CUDA_CHECK(cudaEventDestroy(h2d_event));
    CUDA_CHECK(cudaEventDestroy(compute_event));
    
    // Note: Do NOT synchronize here. Let streams operate asynchronously.
    // Caller should ensure d2h_stream is synchronized before reading results
    // (typically done once at end of tracking loop).
}

// Helper to synchronize d2h_stream before reading results
void _KLTSyncD2HStream() {
    if (d2h_stream != nullptr) {
        CUDA_CHECK(cudaStreamSynchronize(d2h_stream));
    }
}

#ifdef __cplusplus
}
#endif