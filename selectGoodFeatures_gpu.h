#ifndef SELECTGOODFEATURES_GPU_H
#define SELECTGOODFEATURES_GPU_H

#include "klt.h"

#ifdef __cplusplus
extern "C" {
#endif

    // GPU functions
    int KLTInitCUDA();
    void KLTShutdownCUDA();
    int KLTSelectGoodFeatures_GPU(KLT_TrackingContext tc,
        unsigned char* img,
        int ncols, int nrows,
        KLT_FeatureList fl);

    // Optional verbose flag
    extern int KLT_verbose;

#ifdef __cplusplus
}
#endif

#endif