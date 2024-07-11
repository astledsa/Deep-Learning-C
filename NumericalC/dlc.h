#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "numC.h"

typedef struct {
    Tensor** tensor_array;
    int      maximum_batch;
    int      current_batch;
}Batch;

typedef struct {
    int      paramNumber;
    Tensor** parameters;

    Batch* (*forward) (Batch*, Tensor**);
}Layer;

typedef struct {
    Layer** layers;
    int     layerCount;
}Model;

typedef struct {
    Model* model;
    Batch* input;
    Batch* true;
    int    batch_size;
    int    epochs;
    double learning_rate;
}Trainer;

void free_batch (Batch* b);
void Optimize (Model* model, double learning_rate);
void Train (Trainer trainer);

