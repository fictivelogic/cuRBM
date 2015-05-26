#include <cuda_runtime.h>
#include "createInitialHiddenFeatures.cuh"

__global__ void
createInitialHiddenFeaturesKernel(const float *weights,
    const int *movie_ratings, float* initial_hidden_feature_probs,
    int num_movies, int num_hidden_features, int num_user_ratings) {

    // weights[NUM_MOVIES][5][NUM_FEATURES]
    // movie_ratings[NUM_TRAIN_POINTS][3]
    // initial_hidden_feature_probs[NUM_FEATURES]
    unsigned int hidden_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int point_index = 0;
    float dot_prod; // Temporary, local dot product variable
    while (hidden_index < num_hidden_features) {
        dot_prod = 0.00; // Initialize the dot product to 0

        for (point_index = 0; point_index < num_user_ratings; point_index++) {
            // Indexing: weights[movie_id][rating_id][feature_id]
            // movie_id - [1, 500,000]
            // movie_id - [1, 17771]
            // rating_id - [0, 4]
            // hidden_index - [0, 99]
            user_id = *movie_ratings++;
            movie_id = *movie_ratings++;
            rating = *movie_ratings++;
            // Do the dot product
            dot_prod += weights[movie_id*5*num_hidden_features
					            + rating*num_hidden_features
					            + hidden_index]
                        * initial_hidden_feature_probs[];
        }
        // Store the dot_product result
        movie_rating_probs[movie_rating_index] = dot_prod;

        // Re-use this thread on another data point:
        hidden_index += blockDim.x * gridDim.x;
    }
}
