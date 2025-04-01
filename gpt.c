// Initialize a GPT-style model
// The hidden size is: 12,288 dimensions
// The FFN layer is: 49,152 dimensions
Model* InitGPT(int layers) {
    int input_shape[2] = {12288, 12288};
    int transformer_weights[4][2] = {{12288, 49152}, {12288, 49152}, {12288, 49152}, {49152, 12288}};
    int layer_norm_weights[2][2] = {{12288, 12288}, {12288, 12288}};
    int ffn_weights[4][2] = {{12288, 49152}, {49152, 12288}, {12288, 49152}, {49152, 12288}};

    Model* GPT = InitModel(InitLayer(transformer_forward, 4, transformer_weights, "r"));

    for (int i = 0; i < layers; i++) {
        AddLayer(GPT, InitLayer(addnorm_forward, 2, layer_norm_weights, "r"));
        AddLayer(GPT, InitLayer(feed_forward, 4, ffn_weights, "r"));
        AddLayer(GPT, InitLayer(addnorm_forward, 2, layer_norm_weights, "r"));
    }

    return GPT;
}

Batch* GenerateText(Model* GPT, Batch* input, int steps) {
    for (int i = 0; i < steps; i++) {
        Train((Trainer){GPT, input, input, 1, 1, 0.1, malloc_trace(sizeof(double))}, 1);
    }
    return input;
}

// The original GPT-3 had 96 transformer layers
int main() {
    int input_shape[2] = {20, 20};
    Batch* Input = InitBatch(Random(input_shape, FALSE), 1);
    Model* GPT = InitGPT(96);

    Batch* Output = GenerateText(GPT, Input, 2048);
    Print(Output->tensor_array[0]);
    
    return 0;
}
