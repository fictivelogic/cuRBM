#include <cuda_runtime.h>
#include "createFinalHiddenFeatures.cuh"

__global__ void
createFinalHiddenFeaturesKernel(const float *weights,
    const float *movie_rating_probs, float* final_hidden_feature_probs,
    int num_movies, int num_hidden_features) {

    // weights[NUM_MOVIES][5][NUM_FEATURES]
    // movie_rating_probs[NUM_MOVIES][5]
    // final_hidden_feature_probs[NUM_FEATURES]
}
