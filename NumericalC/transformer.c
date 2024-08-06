#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include "numC.h"
#include "dlc.h"

// Split the Q, K and V into h heads and calculate the attention
// scores, concatenate them and return
Tensor* Attention (Tensor* Q, Tensor* K, Tensor* V, int heads) {
    Tensor** AttentionScores = (Tensor**)malloc_trace(heads * sizeof(Tensor*));
    Tensor** Q_split = EqualSplit(Q, 1, heads, FREE_1);
    Tensor** K_split = EqualSplit(K, 1, heads, FREE_1);
    Tensor** V_split = EqualSplit(V, 1, heads, FREE_1);

    double d = Q_split[0]->tensor_matrix->shape[1];

    for (int i = 0; i < heads; i++) {
        AttentionScores[i] = Scalar(Matmul(Q_split[i], Transpose(K_split[i])), sqrt(d));
        Softmax(AttentionScores[i]);
        AttentionScores[i] = Matmul(AttentionScores[i], V_split[i]);

        free_tensor(Q_split[i]);
        free_tensor(K_split[i]);
        free_tensor(V_split[i]);
    }

    free(Q_split);
    free(K_split);
    free(V_split);

    Tensor* AttentionWeights = Concatenate(AttentionScores, heads, FREE_1);
    return AttentionWeights;
};

// Create the Q, K and V matrices for each matrix from the batch 
// and pass into the Attention function
Tensor* MultiHeadAttention (Tensor* Input, Tensor** parameters, int heads) {
    Tensor* Q = Matmul(Input, parameters[0]);
    Tensor* K = Matmul(Input, parameters[1]);
    Tensor* V = Matmul(Input, parameters[2]);

    Tensor* MHAWeights = Attention(Q, K, V, heads);
    Tensor* linearProjection = Matmul(MHAWeights, parameters[3]);

    return linearProjection;
};

// Apply multi-head attention to the batch of inputs
// Attention Heads = 4
Batch* transformer_forward (Batch* originalInput, Batch* output, Tensor** parameters, LayerPos Pos) {
    
    for (int i = 0; i < output->current_batch; i++) {
        if (Pos == FIRST) {
            output->tensor_array[i] = MultiHeadAttention(originalInput->tensor_array[i], parameters, 10);
        } else {
            output->tensor_array[i] = MultiHeadAttention(output->tensor_array[i], parameters, 10);
        }
    }

    return output;
};

// Add the residual connection to the output and perform
// layer normalization, it cannot be the first layer 
Batch* addnorm_forward (Batch* originalInput, Batch* output, Tensor** parameters, LayerPos Pos) {
    assert(Pos == MIDDLE);

    double epsilon = 1e-8;
    for (int i = 0; i < output->current_batch; i++) {
        output->tensor_array[i] = Add(output->tensor_array[i], originalInput->tensor_array[i]);
        Normalize(output->tensor_array[i], epsilon);
        output->tensor_array[i] = Add(Mult(output->tensor_array[i], parameters[0]), parameters[1]);
    } 
    return output;
};

// Apply two linear projections with a ReLu activation in between
// This layer cannot be the first one in the model
Batch* feed_forward (Batch* originalInput, Batch* output, Tensor** parameters, LayerPos Pos) {
    assert(Pos == MIDDLE);

    for (int i = 0; i < output->current_batch; i++) {
        output->tensor_array[i] = Relu(Add(Matmul(output->tensor_array[i], parameters[0]), parameters[1]));
        output->tensor_array[i] = Add(Matmul(output->tensor_array[i], parameters[2]), parameters[3]);
    }

    return output;
}

int main () {
    int input_shape[2] = {20, 20};

    int ffn_weights[4][2] = {{20, 20},  // W_1
                             {20, 20},  // W_2
                             {20, 20},  // W_3
                             {20, 20}}; // W_4

    int layer_norm_weights[2][2] = {{20, 20},  // γ
                                    {20, 20}}; // β

    int transformer_weights[4][2] = {{20, 40},  // W_Q
                                     {20, 40},  // W_K
                                     {20, 40},  // W_V
                                     {40, 20}}; // W_O

    Batch* Input = InitBatch(Random(input_shape, FALSE), 1);
    Batch* True = InitBatch(Eye(input_shape, FALSE), 1);

    Layer* transformer = InitLayer(transformer_forward, 4, transformer_weights, "r");
    Layer* AddNorm = InitLayer(addnorm_forward, 2, layer_norm_weights, "r");
    Layer* FFNLayer = InitLayer(feed_forward, 4, ffn_weights, "r");

    Model* ToyModel = InitModel(transformer);

    AddLayer(ToyModel, AddNorm);
    AddLayer(ToyModel, FFNLayer);
    AddLayer(ToyModel, AddNorm);

    Trainer trainer;

    trainer.batch_size = 1;
    trainer.epochs = 20;
    trainer.input = Input;
    trainer.true = True;
    trainer.learning_rate = 0.2;
    trainer.model = ToyModel;
    trainer.losses = (double*)malloc_trace(trainer.epochs * sizeof(double));;

    Train(trainer, 1);

}

// clock_t start_time = clock();
// clock_t end_time = clock();
// double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
// printf("Execution time: %f seconds\n", elapsed_time);
// gcc transformer.c dlc.c numC.c -o exec

