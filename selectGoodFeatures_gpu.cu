#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "klt.h"
#include "error.h"
#include "selectGoodFeatures_gpu.h"
#include "convolve.h"  // For Gaussian convolution

#ifdef __cplusplus
extern "C" {
#endif

int KLTInitCUDA() { return 0; }
void KLTShutdownCUDA() { }
int KLTSelectGoodFeatures_GPU(KLT_TrackingContext tc,
                              unsigned char *img,
                              int ncols, int nrows,
                              KLT_FeatureList fl);

#ifdef __cplusplus
}
#endif

// ------------------------------------------------------------
// Optional timing flag
// ------------------------------------------------------------
static bool measureTime = true;

// External GPU convolution functions from convolve_gpu.cu
extern void _KLTToFloatImage_gpu(KLT_PixelType *img, int ncols, int nrows, float* d_floatimg);
extern void _KLTComputeGradients_gpu(float* d_img, int ncols, int nrows, float sigma,
                                     float* d_gradx, float* d_grady);

// ------------------------------------------------------------
// Device helper function
// ------------------------------------------------------------
__device__ float _minEigenvalue_GPU(float gxx, float gxy, float gyy) {
    float trace = gxx + gyy;
    float det = gxx*gyy - gxy*gxy;
    float diskrim = trace*trace - 4*det;
    if (diskrim < 0.0f) diskrim = 0.0f;
    return 0.5f * (trace - sqrtf(diskrim));
}

// ------------------------------------------------------------
// Compute minimum eigenvalue over window - OPTIMIZED VERSION
// Uses shared memory tiling, __restrict__ pointers, and occupancy hints
// ------------------------------------------------------------
__global__ void __launch_bounds__(256, 4)  // Occupancy: 256 threads/block, min 4 blocks/SM
computeMinEigenvalue_kernel(const float * __restrict__ gradx, 
                            const float * __restrict__ grady, 
                            float * __restrict__ minEigenvalue,
                            int width, int height, int window_hw, int window_hh, int nSkippedPixels){
    
    // Shared memory tile for gradient reuse (including halo region)
    // TILE_SIZE = 16, so with halo we need (16 + 2*window_hw) x (16 + 2*window_hh)
    // For typical 7x7 window (hw=3, hh=3): need (16+6)x(16+6) = 22x22 = 484 floats per gradient
    // Total: 2 * 484 * 4 bytes = ~3.9 KB per block (well within 48KB limit)
    __shared__ float s_gradx[22][22];
    __shared__ float s_grady[22][22];
    
    // Output pixel coordinates (accounting for stride)
    int x_out = (blockIdx.x*blockDim.x + threadIdx.x)*(nSkippedPixels+1);
    int y_out = (blockIdx.y*blockDim.y + threadIdx.y)*(nSkippedPixels+1);
    
    // Shared memory coordinates (local to tile)
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Top-left corner of the tile in global memory (includes halo)
    int tile_x_start = blockIdx.x * blockDim.x * (nSkippedPixels+1) - window_hw;
    int tile_y_start = blockIdx.y * blockDim.y * (nSkippedPixels+1) - window_hh;
    
    int tile_width = blockDim.x + 2*window_hw;
    int tile_height = blockDim.y + 2*window_hh;
    
    // Cooperatively load tile into shared memory (each thread loads multiple elements)
    for(int i = ty; i < tile_height; i += blockDim.y){
        for(int j = tx; j < tile_width; j += blockDim.x){
            int global_x = tile_x_start + j;
            int global_y = tile_y_start + i;
            
            // Clamp to image boundaries
            global_x = max(0, min(global_x, width-1));
            global_y = max(0, min(global_y, height-1));
            
            int global_idx = global_y * width + global_x;
            s_gradx[i][j] = gradx[global_idx];
            s_grady[i][j] = grady[global_idx];
        }
    }
    
    __syncthreads();  // Wait for all threads to finish loading
    
    // Now compute eigenvalue using shared memory
    if(x_out >= window_hw && x_out < width-window_hw && 
       y_out >= window_hh && y_out < height-window_hh){
        
        float gxx = 0.0f, gxy = 0.0f, gyy = 0.0f;
        
        // Access from shared memory (much faster!)
        #pragma unroll 4
        for(int dy = 0; dy <= 2*window_hh; dy++){
            #pragma unroll 4
            for(int dx = 0; dx <= 2*window_hw; dx++){
                int s_x = tx + dx;
                int s_y = ty + dy;
                
                float gx = s_gradx[s_y][s_x];
                float gy = s_grady[s_y][s_x];
                
                gxx += gx * gx;
                gxy += gx * gy;
                gyy += gy * gy;
            }
        }
        
        minEigenvalue[y_out * width + x_out] = _minEigenvalue_GPU(gxx, gxy, gyy);
    }
}

// ------------------------------------------------------------
// Main GPU feature selection function with timing
// ------------------------------------------------------------
int KLTSelectGoodFeatures_GPU(KLT_TrackingContext tc, unsigned char *img, int ncols, int nrows, KLT_FeatureList fl){
    int maxFeatures = MAX_FEATURES;
    unsigned char *d_img; 
    float *d_floatimg, *d_gradx, *d_grady, *d_minEigenvalues;

    cudaMalloc(&d_img, ncols*nrows*sizeof(unsigned char));
    cudaMalloc(&d_floatimg, ncols*nrows*sizeof(float));
    cudaMalloc(&d_gradx, ncols*nrows*sizeof(float));
    cudaMalloc(&d_grady, ncols*nrows*sizeof(float));
    cudaMalloc(&d_minEigenvalues, ncols*nrows*sizeof(float));

    cudaMemcpy(d_img, img, ncols*nrows*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(d_minEigenvalues,0,ncols*nrows*sizeof(float));

    // OPTIMIZATION: Better launch configuration
    // Use 16x16 blocks (256 threads) for good occupancy
    // Adjust grid to account for stride (nSkippedPixels)
    dim3 blockDim(16, 16);
    int grid_x = (ncols / (tc->nSkippedPixels + 1) + blockDim.x - 1) / blockDim.x;
    int grid_y = (nrows / (tc->nSkippedPixels + 1) + blockDim.y - 1) / blockDim.y;
    dim3 gridDim(grid_x, grid_y);

    // ------------------------------------------------------------
    // GPU timing
    // ------------------------------------------------------------
    cudaEvent_t start, stop;
    if(measureTime){
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
    }

    // Convert to float image and compute Gaussian gradients (like CPU version)
    _KLTToFloatImage_gpu(d_img, ncols, nrows, d_floatimg);
    _KLTComputeGradients_gpu(d_floatimg, ncols, nrows, tc->grad_sigma, d_gradx, d_grady);
    
    // Compute minimum eigenvalue map
    computeMinEigenvalue_kernel<<<gridDim,blockDim>>>(d_gradx,d_grady,d_minEigenvalues,ncols,nrows,
                                                     tc->window_width/2,tc->window_height/2,tc->nSkippedPixels);

    cudaDeviceSynchronize();

    if(measureTime){
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float gpuTimeMs;
        cudaEventElapsedTime(&gpuTimeMs, start, stop);
        printf("(KLT-CUDA) GPU kernel time: %.3f ms\n", gpuTimeMs);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    // ------------------------------------------------------------

    // OPTIMIZATION: Use pinned memory for faster D2H transfer
    float *h_minEigenvalues;
    cudaMallocHost(&h_minEigenvalues, ncols*nrows*sizeof(float));  // Pinned memory
    cudaMemcpy(h_minEigenvalues,d_minEigenvalues,ncols*nrows*sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(d_img); cudaFree(d_floatimg); cudaFree(d_gradx); cudaFree(d_grady); cudaFree(d_minEigenvalues);

    int *pointlist = (int*)malloc(ncols*nrows*3*sizeof(int));
    int npoints = 0;

    for(int y=tc->bordery; y<nrows-tc->bordery; y+=tc->nSkippedPixels+1)
        for(int x=tc->borderx; x<ncols-tc->borderx; x+=tc->nSkippedPixels+1){
            float val = h_minEigenvalues[y*ncols + x];
            if(val>0){
                pointlist[3*npoints] = x; pointlist[3*npoints+1] = y; pointlist[3*npoints+2] = (int)val; 
                npoints++;
            }
        }

    cudaFreeHost(h_minEigenvalues);  // Free pinned memory

    // Sort descending by minEigenvalue
    for(int i=0;i<npoints-1;i++){
        int max_idx=i;
        for(int j=i+1;j<npoints;j++)
            if(pointlist[3*j+2] > pointlist[3*max_idx+2]) max_idx=j;
        if(max_idx!=i)
            for(int k=0;k<3;k++){
                int tmp=pointlist[3*i+k];
                pointlist[3*i+k]=pointlist[3*max_idx+k];
                pointlist[3*max_idx+k]=tmp;
            }
    }

    // Fill feature list
    fl->nFeatures=0;
    unsigned char *featuremap = (unsigned char*)calloc(ncols*nrows,sizeof(unsigned char));
    int mindist = (tc->mindist<0?0:tc->mindist)-1;

    for(int i=0; i<npoints && fl->nFeatures<maxFeatures; i++){

        int x=pointlist[3*i], y=pointlist[3*i+1], val=pointlist[3*i+2];
        if(!featuremap[y*ncols+x] && val>=tc->min_eigenvalue){
            fl->feature[fl->nFeatures]->x=x;
            fl->feature[fl->nFeatures]->y=y;
            fl->feature[fl->nFeatures]->val=val;
            fl->nFeatures++;
            for(int iy=y-mindist; iy<=y+mindist; iy++)
                for(int ix=x-mindist; ix<=x+mindist; ix++)
                    if(ix>=0 && ix<ncols && iy>=0 && iy<nrows) featuremap[iy*ncols+ix]=1;
        }
    }

    free(featuremap);
    free(pointlist);

    if(KLT_verbose>=1)
        fprintf(stderr,"(KLT-CUDA) %d features found.\n", fl->nFeatures);

    return 0;
}