#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "numC.h"
#include "dlc.h"

void Optimize (Model* model, double learning_rate) {
    for (int i = 0; i < model->layerCount; i++) {
        for (int j = 0; j < model->layers[i]->paramNumber; j++) {
            Update(model->layers[i]->parameters[j], learning_rate);
        }
    }
};

void free_batch (Batch* b) {
    for (int i = 0; i < b->current_batch; i++) {
        free_tensor(b->tensor_array[i]);
    }
    free_batch(b);
}

void add_input (Batch* batch, Tensor* input) {
    assert(batch->maximum_batch > batch->current_batch && "Too many inputs per batch!");
    batch->tensor_array[batch->current_batch] = input;
};

Tensor* BatchLoss (Batch* true, Batch* output) {
    int shape[2] = {1, 1};
    Tensor* loss = Zeros(shape, TRUE);

    for (int i = 0; i < output->current_batch; i++) {
        Tensor* pointLoss = Element_Pow(Sub(true->tensor_array[i], output->tensor_array[i]), 2);
        loss = Add(loss, Sum(pointLoss));

        free_tensor(pointLoss);
    }

    return loss;
}

Batch* init_batch (Tensor* input, int batch_size) {
    Batch* new_batch = (Batch*)malloc_trace(sizeof(Batch));

    new_batch->tensor_array = (Tensor**)malloc_trace(batch_size * sizeof(Tensor*));
    new_batch->maximum_batch = batch_size;
    new_batch->tensor_array[0] = input;
    new_batch->current_batch = 1;

    return new_batch;
};

Batch* ModelForward (Batch* input, Model* model) {
    Batch* output;
    for (int i = 0; i < model->layerCount; i++) {
        if (i == 0) {
            output = model->layers[i]->forward(input, model->layers[i]->parameters);
        } else {
            output = model->layers[i]->forward(output, model->layers[i]->parameters);
        }
    }

    return output;
};

void Train (Trainer trainer) {

    assert(trainer.input->current_batch == trainer.true->current_batch && "Different input and groundtruth dimensions!");

    for (int i = 0; i < trainer.epochs; i++) {
        Batch* Output = ModelForward(trainer.input, trainer.model);
        Tensor* Loss = BatchLoss(trainer.true, Output);

        Backward(Loss);
        Optimize(trainer.model, trainer.learning_rate);

        free_batch(Output);
        free_tensor(Loss);
    }

}