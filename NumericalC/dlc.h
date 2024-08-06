#ifndef DLC_H
#define DLC_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "numC.h"

typedef enum {
    FIRST,
    MIDDLE
}LayerPos;

typedef struct {
    Tensor** tensor_array;
    int      maximum_batch;
    int      current_batch;
}Batch;

typedef struct {
    int      paramNumber;
    Tensor** parameters;

    Batch* (*forward) (Batch*, Batch*, Tensor**, LayerPos);
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
    double* losses;
}Trainer;

void free_batch (Batch* b);
void Optimize (Model* model, double learning_rate);
void Train (Trainer trainer, int threads);
void AddInput (Batch* batch, Tensor* input);
void AddLayer (Model* model, Layer* layer);
void print_trainer_info (Trainer trainer);
void plot_wave(double* values, int size, int normalize, int logarithmic, int HEIGHT, int WIDTH);

Batch* InitBatch (Tensor* input, int batch_size);
Layer* InitLayer (Batch* (*forward) (Batch*, Batch*, Tensor**, LayerPos), int numberOfParams, int shapes[][2], char* initialization);
Model* InitModel (Layer* firstLayer);

#endif