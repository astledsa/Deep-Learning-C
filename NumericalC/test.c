#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include "numC.h"
#include "dlc.h"

Batch* linearPass (Batch* inputs, Tensor** parameters, char* allocate) {
    Batch* output;

    if (strcmp(allocate, "t") == 0) {
        output = InitBatch(inputs->tensor_array[0], inputs->maximum_batch);
    } else {
        output = inputs;
    }

    for (int i = 0; i < inputs->current_batch; i++) {
        output->tensor_array[i] = Add(Matmul(parameters[0], inputs->tensor_array[i]), parameters[1]);
    }

    return output;
};

Batch* reluPass (Batch* inputs, Tensor** parameters, char* allocate) {
    Batch* output = inputs;

    for (int i = 0; i < inputs->current_batch; i++) {
        output->tensor_array[i] = Relu(output->tensor_array[i]);
    }

    return output;
};

int main () {
    int shape[2] = {3, 3};                                        
    double learning_rate = 0.1;                           
    Tensor* input = Random(shape, FALSE);
    Tensor* True = Eye(shape, FALSE);
    Tensor* Weight = Gaussian(shape, 0.0, 1.0, TRUE);
    Tensor* Weight2 = Gaussian(shape, 0.0, 1.0, TRUE);

    int weight_shape[2][2] = {{3, 3}, 
                              {3, 3}};  

    int dataPoints = 1000;

    Batch* X = InitBatch(input, dataPoints);                             
    Batch* y = InitBatch(True, dataPoints);                              
    
    for (int i = 0; i < dataPoints-1; i++) {
        AddInput(X, Random(shape, FALSE));                          
        AddInput(y, Copy(True));    
    }                                 

    Layer* Linear = InitLayer(linearPass, 2, weight_shape, "r"); 
    Layer* ReLu = InitLayer(reluPass, 0, weight_shape, NULL);
    Model* model = InitModel(Linear);

    AddLayer(model, ReLu);
    AddLayer(model, Linear);
                  
    Trainer trainer;                                             

    trainer.batch_size = 1;
    trainer.epochs = 20;
    trainer.input = X;
    trainer.true = y;
    trainer.model = model;
    trainer.learning_rate = 0.5;

    Train(trainer, 2);
    
}


// clock_t start_time = clock();
// clock_t end_time = clock();
// double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
// printf("Execution time: %f seconds\n", elapsed_time);
// gcc test.c dlc.c numC.c -o exec

