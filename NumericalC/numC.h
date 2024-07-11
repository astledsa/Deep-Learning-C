#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <math.h>

// ------------------ Structs/Enums ------------------ // 

typedef enum {
    TRUE,
    FALSE
}Bool;

typedef enum {
    FREE_0,
    FREE_1,
    FREE_2,
    FREE_BOTH
}FreeFlag;

typedef enum {
    OP_NULL,
    OP_ADD,
    OP_SUBTRACT,
    OP_MULTIPLY,
    OP_MATMUL,
    OP_SIN,
    OP_COS,
    OP_TAN,
    OP_LOG,
    OP_ELE_POW,
    OP_TRANSPOSE,
    OP_SUM,
    OP_MEAN,
    OP_STD,
    OP_SCALAR
}Operations;

typedef struct {
  char *start;
  char *end;
} Range;

typedef struct {
    double* array;
    int     shape[2];
    int     stride[2];
}Matrix;

typedef struct Tensor {
    Bool           requires_grad;
    double         power;
    double         scalar;
    Matrix*        gradient;
    Matrix*        tensor_matrix;
    Operations     creation_operation;
    struct Tensor* parents[2];
}Tensor;

// ------------------ Helper ------------------ // 

void* malloc_trace (size_t size);
void free_matrix (Matrix* m);
void free_tensor (Tensor* m);
void Print (Tensor* matrix);
void Print_Matrix (Matrix* matrix);

// ------------------ Initialization ------------------ // 

Tensor* Zeros (int shape[2], Bool requires_grad);
Tensor* Ones (int shape[2], Bool requires_grad);
Tensor* Random (int shape[2], Bool requires_grad);
Tensor* Gaussian (int shape[2], double mean, double std, Bool requires_grad);
Tensor* Eye (int shape[2], Bool requires_grad);

// ------------------ Single Matrix Operations ------------------ // 

double Det (Tensor* matrix);
Tensor* Inverse (Tensor* matrix, Bool requires_grad);
Tensor* Transpose (Tensor* m);
void subset_Add (Tensor* main, Tensor* values_to_add, char* index[2]);
void subset_Sub (Tensor* main, Tensor* values_to_add, char* index[2]);
void subset_EleMul (Tensor* main, Tensor* values_to_add, char* index[2]);

// ------------------ Matrix operations ------------------ // 

Tensor* Add (Tensor* m1, Tensor* m2);
Tensor* Sub (Tensor* m1, Tensor* m2);
Tensor* Mult (Tensor* m1, Tensor* m2);
Tensor* Matmul (Tensor* m1, Tensor* m2);
Tensor* Sin (Tensor* matrix);
Tensor* Cos (Tensor* matrix);
Tensor* Tan (Tensor* matrix);
Tensor* Log (Tensor* matrix);
Tensor* Element_Pow (Tensor* matrix, double n);
Tensor* Pow (Tensor* matrix, double n);
Tensor* Scalar (Tensor* matrix, double value);

// ------------------ Scalar Operations ------------------ // 

double Min (Tensor* matrix);
double Max (Tensor* matrix);
Tensor* Sum (Tensor* matrix);
Tensor* Mean (Tensor* matrix);
Tensor* Std (Tensor* matrix);

// ------------------ Reformation ------------------ // 

void Reshape (Tensor* matrix, int shape[2]);
void Flatten (Tensor* matrix, int axis);
Tensor* Vstack (Tensor* m1, Tensor* m2);
Tensor* Hstack (Tensor* m1, Tensor* m2);
Tensor* Reduce (Tensor* m1, int axis);
Tensor* Copy (Tensor* matrix);

// ------------------ AutoGrad ------------------ // 

void Update (Tensor* Weight, double learning_rate);
void Backward (Tensor* Z);
