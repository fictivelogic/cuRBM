#ifndef NETFLIX_RBM_CREATEINITIALHIDDENFEATURES_H
#define NETFLIX_RBM_CREATEINITIALHIDDENFEATURES_H

// First step of RBM
// Get initial hidden feature probabilities from movie ratings
__global__ void createInitialHiddenFeaturesKernel(const float *weights,
                                                  const int *movie_ratings,
                                                  float* initial_hidden_probs,
                                                  int num_movies,
                                                  int num_hidden_features);

#endif //NETFLIX_RBM_CREATEINITIALHIDDENFEATURES_H
