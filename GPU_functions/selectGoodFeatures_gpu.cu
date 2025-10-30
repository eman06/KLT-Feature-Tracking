#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "include/klt.h"
#include "include/error.h"
#include "selectGoodFeatures_gpu.h"

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
// Compute image gradients
// ------------------------------------------------------------
__global__ void computeGradients_kernel(const unsigned char *img, float *gradx, float *grady, int width, int height){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if(x>0 && x<width-1 && y>0 && y<height-1){
        int idx = y*width + x;
        gradx[idx] = 0.5f*(img[idx+1]-img[idx-1]);
        grady[idx] = 0.5f*(img[idx+width]-img[idx-width]);
    }
}

// ------------------------------------------------------------
// Compute minimum eigenvalue over window
// ------------------------------------------------------------
__global__ void computeMinEigenvalue_kernel(const float *gradx, const float *grady, float *minEigenvalue,
                                            int width, int height, int window_hw, int window_hh, int nSkippedPixels){
    int x = (blockIdx.x*blockDim.x + threadIdx.x)*(nSkippedPixels+1);
    int y = (blockIdx.y*blockDim.y + threadIdx.y)*(nSkippedPixels+1);
    if(x>=window_hw && x<width-window_hw && y>=window_hh && y<height-window_hh){
        float gxx=0, gxy=0, gyy=0, gx, gy;
        for(int yy=y-window_hh; yy<=y+window_hh; yy++)
            for(int xx=x-window_hw; xx<=x+window_hw; xx++){
                int idx = yy*width + xx;
                gx = gradx[idx]; gy = grady[idx];
                gxx += gx*gx; gxy += gx*gy; gyy += gy*gy;
            }
        minEigenvalue[y*width+x] = _minEigenvalue_GPU(gxx,gxy,gyy);
    }
}

// ------------------------------------------------------------
// Main GPU feature selection function with timing
// ------------------------------------------------------------
int KLTSelectGoodFeatures_GPU(KLT_TrackingContext tc, unsigned char *img, int ncols, int nrows, KLT_FeatureList fl){
    int maxFeatures = MAX_FEATURES;
    unsigned char *d_img; 
    float *d_gradx, *d_grady, *d_minEigenvalues;

    cudaMalloc(&d_img, ncols*nrows*sizeof(unsigned char));
    cudaMalloc(&d_gradx, ncols*nrows*sizeof(float));
    cudaMalloc(&d_grady, ncols*nrows*sizeof(float));
    cudaMalloc(&d_minEigenvalues, ncols*nrows*sizeof(float));

    cudaMemcpy(d_img, img, ncols*nrows*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(d_minEigenvalues,0,ncols*nrows*sizeof(float));

    dim3 blockDim(16,16);
    dim3 gridDim((ncols+15)/16,(nrows+15)/16);

    // ------------------------------------------------------------
    // GPU timing
    // ------------------------------------------------------------
    cudaEvent_t start, stop;
    if(measureTime){
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
    }

    computeGradients_kernel<<<gridDim, blockDim>>>(d_img,d_gradx,d_grady,ncols,nrows);
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

    float *h_minEigenvalues = (float*)malloc(ncols*nrows*sizeof(float));
    cudaMemcpy(h_minEigenvalues,d_minEigenvalues,ncols*nrows*sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(d_img); cudaFree(d_gradx); cudaFree(d_grady); cudaFree(d_minEigenvalues);

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

    free(h_minEigenvalues);

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
