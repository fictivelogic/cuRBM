__global__ void
createInitialHiddenFeaturesKernel(const float *weights,
    const int *movie_ratings, float* initial_hidden_feature_probs,
    int num_movies, int num_hidden_features, int num_user_ratings) {

    // weights[NUM_MOVIES][5][NUM_FEATURES]
    // movie_ratings[NUM_TRAIN_POINTS][3]
    // initial_hidden_feature_probs[NUM_FEATURES]
    unsigned int hidden_id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int point_id = 0;
    float dot_prod; // Temporary, local dot product variable
    while (hidden_id < num_hidden_features) {
        dot_prod = 0.00; // Initialize the dot product to 0

        for (point_id = 0; point_id < num_user_ratings; point_id++) {
            // Indexing: weights[movie_id][rating][feature_id]
            // user_id - [1, 500,000]
            // movie_id - [1, 17771]
            // rating - [0, 4]
            // hidden_id - [0, 99]
            user_id = *movie_ratings++;
            movie_id = *movie_ratings++;
            rating = *movie_ratings++;
            // Do the dot product
            dot_prod += weights[movie_id*5*num_hidden_features
					            + rating*num_hidden_features
					            + hidden_id]
                        * initial_hidden_feature_probs[hidden_id];
        }
        // Store the dot_product result
        initial_hidden_feature_probs[hidden_id] = dot_prod;

        // Re-use this thread on another data point:
        hidden_id += blockDim.x * gridDim.x;
    }
}
