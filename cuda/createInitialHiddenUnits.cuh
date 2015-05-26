#ifndef NETFLIX_RBM_CREATEINITIALHIDDENUNITS_H
#define NETFLIX_RBM_CREATEINITIALHIDDENUNITS_H

// First step of RBM
// Get initial hidden unit probabilities from movie ratings
__global__ void createInitialHiddenUnitsKernel(const float *weights,
                                               const float *movie_ratings,
                                               float* initial_hidden_probs,
                                               int num_movies,
                                               int num_hidden_features);

#endif //NETFLIX_RBM_CREATEINITIALHIDDENUNITS_H
