#ifndef CONVOLVE_GPU_H
#define CONVOLVE_GPU_H

#ifdef __cplusplus
extern "C" {
#endif

/* GPU wrapper functions */
void _KLTComputeGradients_gpu_wrapper(
    float* h_img, int ncols, int nrows, float sigma,
    float* h_gradx, float* h_grady);

/* Batched wrapper with CUDA streams for async pipelining */
void _KLTComputeGradients_gpu_batched_streams(
    float* h_img, int ncols, int nrows, float sigma,
    float* h_gradx, float* h_grady);

/* Stream management functions */
void _KLTInitStreams(int ncols, int nrows);
void _KLTDestroyStreams();
void _KLTSyncD2HStream();

#ifdef __cplusplus
}
#endif

#endif /* CONVOLVE_GPU_H */
