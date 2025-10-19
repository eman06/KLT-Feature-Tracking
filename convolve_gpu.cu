/*********************************************************************
 * convolve_gpu.cu
 * Improved naive GPU implementation with kernel timing
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

// Use constant memory for faster kernel access (simple optimization)
__constant__ float d_const_kernel[MAX_KERNEL_WIDTH];
__constant__ int d_const_kernel_width;

// Enable/disable GPU kernel timing
bool measureTime = true;

/******************** GPU Kernels ********************/

__global__ void KLTToFloatImage_kernel(const KLT_PixelType* img, float* floatimg, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        floatimg[idx] = (float)img[idx];
    }
}

__global__ void convolveHoriz_kernel(const float* imgin, float* imgout, int ncols, int nrows) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= ncols || row >= nrows) return;

    int radius = d_const_kernel_width / 2;
    float sum = 0.0f;
    for (int k = -radius; k <= radius; k++) {
        int c = col + k;
        if (c >= 0 && c < ncols) {
            sum += imgin[row * ncols + c] * d_const_kernel[radius + k];
        }
    }
    imgout[row * ncols + col] = sum;
}

__global__ void convolveVert_kernel(const float* imgin, float* imgout, int ncols, int nrows) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= ncols || row >= nrows) return;

    int radius = d_const_kernel_width / 2;
    float sum = 0.0f;
    for (int k = -radius; k <= radius; k++) {
        int r = row + k;
        if (r >= 0 && r < nrows) {
            sum += imgin[r * ncols + col] * d_const_kernel[radius + k];
        }
    }
    imgout[row * ncols + col] = sum;
}

/******************** CPU Helper Functions ********************/

static void _computeKernels(float sigma, ConvolutionKernel *gauss, ConvolutionKernel *gaussderiv) {
    const float factor = 0.01f;
    int i;
    assert(MAX_KERNEL_WIDTH % 2 == 1);
    assert(sigma >= 0.0);

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

/******************** GPU Wrappers ********************/

void _KLTToFloatImage_gpu(KLT_PixelType *img, int ncols, int nrows, float* d_floatimg) {
    int total = ncols * nrows;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    if (measureTime) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
    }

    KLTToFloatImage_kernel<<<gridSize, blockSize>>>(img, d_floatimg, total);
    cudaDeviceSynchronize();

    if (measureTime) {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float gpuTimeMs;
        cudaEventElapsedTime(&gpuTimeMs, start, stop);
        printf("(KLT) GPU float conversion time: %.3f ms\n", gpuTimeMs);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
}

void _convolveSeparate_gpu(float* d_imgin, int ncols, int nrows,
                           ConvolutionKernel horiz, ConvolutionKernel vert, float* d_imgout) {
    float* d_tmp;
    cudaMalloc(&d_tmp, ncols * nrows * sizeof(float));

    dim3 blockDim(16,16);
    dim3 gridDim( (ncols + blockDim.x - 1)/blockDim.x,
                  (nrows + blockDim.y - 1)/blockDim.y );

    cudaEvent_t start, stop;
    if (measureTime) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
    }

    // Horizontal pass
    cudaMemcpyToSymbol(d_const_kernel, horiz.data, horiz.width * sizeof(float));
    cudaMemcpyToSymbol(d_const_kernel_width, &horiz.width, sizeof(int));
    convolveHoriz_kernel<<<gridDim, blockDim>>>(d_imgin, d_tmp, ncols, nrows);

    // Vertical pass
    cudaMemcpyToSymbol(d_const_kernel, vert.data, vert.width * sizeof(float));
    cudaMemcpyToSymbol(d_const_kernel_width, &vert.width, sizeof(int));
    convolveVert_kernel<<<gridDim, blockDim>>>(d_tmp, d_imgout, ncols, nrows);

    cudaDeviceSynchronize();

    if (measureTime) {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float gpuTimeMs;
        cudaEventElapsedTime(&gpuTimeMs, start, stop);
        printf("(KLT) GPU convolution time: %.3f ms\n", gpuTimeMs);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    cudaFree(d_tmp);
}

void _KLTComputeGradients_gpu(float* d_img, int ncols, int nrows, float sigma,
                              float* d_gradx, float* d_grady) {
    if (fabs(sigma - sigma_last) > 0.05f)
        _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

    _convolveSeparate_gpu(d_img, ncols, nrows, gaussderiv_kernel, gauss_kernel, d_gradx);
    _convolveSeparate_gpu(d_img, ncols, nrows, gauss_kernel, gaussderiv_kernel, d_grady);
}