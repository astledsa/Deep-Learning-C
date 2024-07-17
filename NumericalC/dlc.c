#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <pthread.h>
#include "numC.h"
#include "dlc.h"

int cumulative_sum_uptil_i(int *data, int size, int index) {
  if (index < 0 || index >= size) {
    return -1; 
  }

  int sum = 0;
  for (int i = 0; i <= index; i++) {
    sum += data[i];
  }

  return sum;
}

int* split_number(int X, int n) {
    int* list = malloc_trace(n * sizeof(int));

    if (list == NULL) {
        return NULL; 
    }

    int base_value = X / n;
    int remainder = X % n;

    for (int i = 0; i < n; i++) {
        list[i] = base_value;
    }

    for (int i = 0; i < remainder; i++) {
        list[i] += 1;
    }

    return list;
}

void print_trainer_info (Trainer trainer) {
    printf("Batch Size: %d\n", trainer.batch_size);
    printf("Number of epochs: %d\n", trainer.epochs);
    printf("Learning Rate: %f\n", trainer.learning_rate);
    printf("Current Input Batch Size: %d\n", trainer.input->current_batch);
    printf("Current True Batch Size: %d\n", trainer.true->current_batch);
    printf("Number of Layers in the model: %d\n", trainer.model->layerCount);
}

void Optimize (Model* model, double learning_rate) {
    for (int i = 0; i < model->layerCount; i++) {
        for (int j = 0; j < model->layers[i]->paramNumber; j++) {
            Update(model->layers[i]->parameters[j], learning_rate);
            Zero_grad(model->layers[i]->parameters[j]);
        }
    }
};

void free_batch (Batch* b) {
    for (int i = 0; i < b->current_batch; i++) {
        free_tensor(b->tensor_array[i]);
    }
    free(b);
}

void free_layer (Layer* l) {
    for (int i = 0; i < l->paramNumber; i++) {
        free_tensor(l->parameters[i]);
    }
    free(l);
}

void free_model (Model* m) {
    for (int i = 0; i < m->layerCount; i++) {
        free_layer(m->layers[i]);
    }
    free(m);
}

void free_trainer (Trainer* t) {
    free_batch(t->input);
    free_batch(t->true);
    free_model(t->model);
    free(t);
}

void AddInput (Batch* batch, Tensor* input) {
    assert(batch->maximum_batch > batch->current_batch && "Too many inputs per batch!");
    batch->tensor_array[batch->current_batch] = input;
    batch->current_batch += 1;
};

void AddLayer (Model* model, Layer* layer) {
    model->layers = (Layer**)realloc(model->layers, (model->layerCount + 1) * sizeof(Layer));
    model->layers[model->layerCount] = layer;
    model->layerCount += 1;
}



Tensor* BatchLoss (Batch* true, Batch* output) {
    int shape[2] = {1, 1};
    Tensor* loss = Zeros(shape, TRUE);

    for (int i = 0; i < output->current_batch; i++) {
        Tensor* pointLoss = Element_Pow(Sub(true->tensor_array[i], output->tensor_array[i]), 2);
        loss = Add(loss, Sum(pointLoss));
    }

    return loss;
}

Batch* InitBatch (Tensor* input, int batch_size) {
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
            output = model->layers[i]->forward(input, model->layers[i]->parameters, "t");
        } else {
            output = model->layers[i]->forward(output, model->layers[i]->parameters, "f");
        }
    }
    return output;
};

Batch* split_batch (Batch* main_batch, int start, int stop) {
    assert (stop > start && "Incorrect split !");

    Batch* split = InitBatch(main_batch->tensor_array[start], stop - start);
    for (int i = start+1; i < stop; i++) {
        AddInput(split, main_batch->tensor_array[i]);
    }

    return split;
};

Layer* InitLayer (Batch* (*forward) (Batch*, Tensor**, char*), int numberOfParams, int shapes[][2], char* initialization) {
    Layer* layer = (Layer*)malloc_trace(sizeof(Layer));
    
    if (numberOfParams > 0) {
        Tensor** params = (Tensor**)malloc_trace(numberOfParams * sizeof(Tensor*));

        for (int i = 0; i < numberOfParams; i++) {
            int shape[2] = {shapes[i][0], shapes[i][1]};
            if (strcmp(initialization, "r") == 0) {
                params[i] = Random(shape, TRUE);
            } else {
                params[i] = Gaussian(shape, 0.0, 1.0, TRUE);
            }
        }
        layer->parameters = params;
    } else {
        layer->parameters = NULL;
    }

    layer->paramNumber = numberOfParams;
    layer->forward = forward;

    return layer;

}

Layer* copy_layer (Layer* main_layer) {
    Layer* new_layer = (Layer*)malloc_trace(sizeof(Layer));
    new_layer->parameters = (Tensor**)malloc_trace(main_layer->paramNumber * sizeof(Tensor*));

    for (int i = 0; i < main_layer->paramNumber; i++) {
        new_layer->parameters[i] = Copy(main_layer->parameters[i]);
    }

    new_layer->forward = main_layer->forward;
    new_layer->paramNumber = main_layer->paramNumber;
    return new_layer;
}

Model* InitModel (Layer* firstLayer) {
    Model* model = (Model*)malloc_trace(sizeof(Model));
    Layer** layers = (Layer**)malloc_trace(sizeof(Layer*));
    layers[0] = firstLayer;

    model->layers = layers;
    model->layerCount = 1;

    return model;
}

Model* deep_copy_model (Model* main_model) {
    Model* new_model = (Model*)malloc_trace(sizeof(Model));

    new_model->layers = (Layer**)malloc_trace(main_model->layerCount * sizeof(Layer*));
    new_model->layerCount = main_model->layerCount;

    for (int i = 0; i < new_model->layerCount; i++) {
        new_model->layers[i] = copy_layer(main_model->layers[i]);
    }

    return new_model;
}

Model* cumulativeAddModel (Model* m1, Model* m2) {
    for (int i = 0; i < m1->layerCount; i++) {
        for (int j = 0; j < m1->layers[i]->paramNumber; j++) {
            m1->layers[i]->parameters[j] = Add(m1->layers[i]->parameters[j], m2->layers[i]->parameters[j]);
        }
    }
    return m1;
}

Trainer* split_trainers (Trainer main_trainer, int threads) {
    assert(main_trainer.input->current_batch >= threads && "No. of threads exceeds No. of inputs");

    if (threads > 1) {
        Trainer* trainers = (Trainer*)malloc_trace(threads * sizeof(Trainer));
        int* input_shapes = split_number(main_trainer.input->current_batch, threads);

        for (int i = 0; i < threads; i++) {
            Trainer trainer_i;

            if (i == 0) {
                trainer_i.input = split_batch(main_trainer.input, 0, input_shapes[i]);
                trainer_i.true = split_batch(main_trainer.true, 0, input_shapes[i]);
            } else {
                trainer_i.input = split_batch(
                    main_trainer.input, 
                    cumulative_sum_uptil_i(input_shapes, threads,  i-1), 
                    cumulative_sum_uptil_i(input_shapes, threads,  i)
                );
                trainer_i.true = split_batch(
                    main_trainer.true, 
                    cumulative_sum_uptil_i(input_shapes, threads,  i-1), 
                    cumulative_sum_uptil_i(input_shapes, threads,  i)
                );
            }
            trainer_i.model = deep_copy_model(main_trainer.model);
            trainer_i.batch_size = main_trainer.batch_size;
            trainer_i.epochs = main_trainer.epochs;
            trainer_i.learning_rate = main_trainer.learning_rate;

            trainers[i] = trainer_i;
        }

        free(input_shapes);
        return trainers;
    }

    return NULL;
}



void Synchronize (Trainer main_trainer, Trainer* all_trainers, int threads) {
    printf("Synchronization Started\n");
    main_trainer.model = NULL;
    for (int i = 0; i < threads; i++) {
        if (i == 0) {
            main_trainer.model = deep_copy_model(all_trainers[i].model);
        } else {   
            main_trainer.model = cumulativeAddModel (main_trainer.model, all_trainers[i].model);
        }
        
        free_batch(all_trainers[i].input);
        free_batch(all_trainers[i].true);
        free_model(all_trainers[i].model);
    };

    printf("Done calculating sums of the models\n");
    for (int i = 0; i < main_trainer.model->layerCount; i++) {
        for (int j = 0; j < main_trainer.model->layers[i]->paramNumber; j++) {
            main_trainer.model->layers[i]->parameters[j] = Scalar(main_trainer.model->layers[i]->parameters[j], 1.0/threads);
        }
    }

    printf("Synchronization Complete.\n");
}

void* train_single_thread (void* t) {
    printf("Single thread initiliased\n");
    Trainer*  trainer = (Trainer*) t;
    assert(trainer->input->current_batch == trainer->true->current_batch && "Different input and groundtruth dimensions!");
    if (trainer->input->current_batch < trainer->input->maximum_batch) {
        printf("WARNING: underutilised Batch may lead to wasted memory.\n");
    }

    for (int i = 0; i < trainer->epochs; i++) {
        Batch* Output = ModelForward(trainer->input, trainer->model);
        Tensor* Loss = BatchLoss(trainer->true, Output);
        
        Backward(Loss);
        if (i % 10 == 0) {
            printf("Loss is %f\n", Loss->tensor_matrix->array[0]);
        }
        Optimize(trainer->model, trainer->learning_rate);
        
        free_batch(Output);
        free_tensor(Loss);
    }

    printf("Thread trained\n");

    return NULL;

}

void Train (Trainer trainer, int threads) {
    Trainer* multi_thread_trainers = split_trainers(trainer, threads);
    if (multi_thread_trainers) {

        pthread_t train_threads[threads];

        printf("Creating %d threads...\n", threads);

        for (int i = 0; i < threads; i++) {
            pthread_create(&train_threads[i], NULL, train_single_thread, (void*) &multi_thread_trainers[i]);
        }

        for (int i = 0; i < threads; i++) {
            pthread_join(train_threads[i], NULL);
        }

        printf("Threads joined\n");

        Synchronize (trainer, multi_thread_trainers, threads);

        free(multi_thread_trainers);

    } else if (threads == 0 || threads == 1) {
        Trainer* t = &trainer;
        train_single_thread((void*) t);
    }
    printf("Done!\n");
};