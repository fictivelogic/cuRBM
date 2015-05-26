#include <cuda_runtime.h>
#include "createInitialHiddenFeatures.cuh"

__global__ void
createInitialHiddenFeaturesKernel(const float *weights,
    const int *movie_ratings, float* initial_hidden_feature_probs,
    int num_movies, int num_hidden_features) {

    // weights[NUM_MOVIES][5][NUM_FEATURES]
    // movie_ratings[NUM_MOVIES][5]
    // initial_hidden_feature_probs[NUM_FEATURES]
}
