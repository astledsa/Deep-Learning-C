// Matrix Library for C
// Important functionlities implemented

// Matrix Creation
// -- Zeros.
// -- Ones.
// -- Normal distribution.
// -- Identity.
// -- random.

// Matrix Operations
// -- Matrix Multiplication.
// -- Transpose.
// -- Inverse.
// -- Trace.
// -- Determinant.

// Element Wise Operations
// -- Addition.
// -- Subtraction.
// -- Multiplication.
// -- Log.
// -- Sin.
// -- Cos.
// -- Tan.
// -- Power.
// -- Square Root.

// Aggregation Functions
// -- Summation.
// -- Mean.
// -- Standard Deviation.
// -- Max.
// -- Min.

// Reshaping and Indexing
// -- Reshape.
// -- Hstack.
// -- Vstack.
// -- Flatten.

// Other Functions
// -- Expand?
// -- Broadcast?
// -- Reduce.

// Utility Functions
// -- Copy.

// Functions that require backward()
// -- ADD()
// -- SUBTRACT()
// -- MULT()
// -- MATMUL()
// -- SIN()
// -- COS()
// -- TAN()
// -- LOG()
// -- Transpose()
// -- POW()
// -- Reduce()
// -- ELEMENT_POW()
// -- SUM()
// -- MEAN()
// -- STD()

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <math.h>

static size_t total_allocated_memory;

/*
 ****************************************************************************
 *                                Structs                                   *
 ****************************************************************************
 */

typedef struct {
  char *start;
  char *end;
} Range;

typedef struct {
    double* array;
    int     shape[2];
    int     stride[2];
}Matrix;

typedef struct GradNode {
    Matrix*          grad_matrix;
    struct GradNode* next;
}GradNode;

typedef struct Tensor {
    Matrix*        tensor_matrix;
    Matrix*        gradient;
    double         power;
    char*          creation_operation;
    GradNode*      backwards_grad;
    struct Tensor* parents[2];
}Tensor;

/*
 ****************************************************************************
 *                             Helper Functions                             *
 ****************************************************************************
 */

double determinant(double** matrix, int size) {
    if (size == 1) {
        return matrix[0][0];
    }
    
    if (size == 2) {
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    }
    
    double det = 0;
    double **submatrix = (double **)malloc((size) * sizeof(double *));
    for (int i = 0; i < size - 1; i++) {
        submatrix[i] = (double *)malloc((size) * sizeof(double));
    }
    
    for (int j = 0; j < size; j++) {
        int sub_i = 0;
        for (int i = 1; i < size; i++) {
            int sub_j = 0;
            for (int k = 0; k < size; k++) {
                if (k == j) continue;
                submatrix[sub_i][sub_j] = matrix[i][k];
                sub_j++;
            } 
            sub_i++;
        }
        
        det += pow(-1, j) * matrix[0][j] * determinant(submatrix, size - 1);
    }
    
    for (int i = 0; i < size - 1; i++) {
        free(submatrix[i]);
    }
    free(submatrix);
    
    return det;
}

void matrix_inverse (double** matrix, double** inverse, int N) {
    double det = determinant(matrix, N);
    
    if (det == 0) {
        printf("Matrix is singular and cannot be inverted.\n");
        return;
    }
    
    double adjoint[N][N];
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double ** sub_matrix = (double**)malloc(N * sizeof(double*));
            for (int i = 0; i < N; i++) {
                sub_matrix[i] = (double*)malloc(N * sizeof(double));
            }
            int subi = 0;
            for (int r = 0; r < N; r++) {
                if (r == i)
                    continue;
                int subj = 0;
                for (int c = 0; c < N; c++) {
                    if (c == j)
                        continue;
                    sub_matrix[subi][subj] = matrix[r][c];
                    subj++;
                }
                subi++;
            }
            adjoint[j][i] = pow(-1, i + j) * determinant(sub_matrix, N - 1);
        }
    }
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            inverse[i][j] = adjoint[i][j] / det;
        }
    }
}

void* malloc_trace (size_t size) {
    void* ptr = malloc(size);
    if (ptr != NULL) {
        printf("Allocated %zu space\n", size);
        total_allocated_memory += size;
    }
    return ptr;
}

void Print (Tensor* matrix) {
    int rows = matrix->tensor_matrix->shape[0];
    int cols = matrix->tensor_matrix->shape[1];

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%lf ", matrix->tensor_matrix->array[(i * matrix->tensor_matrix->stride[0]) + (j * matrix->tensor_matrix->stride[1])]);
        }
        printf("\n");
    }
}

void Print_Matrix (Matrix* matrix) {
    int rows = matrix->shape[0];
    int cols = matrix->shape[1];

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%lf ", matrix->array[(i * matrix->stride[0]) + (j * matrix->stride[1])]);
        }
        printf("\n");
    }
}

int* split_colon(const char *str, const int RANGE_MAX) {
    Range result;
    result.start = NULL;
    result.end = NULL;
    
    char *colon = strchr(str, ':');
    
    if (colon == NULL) {
        int* res = malloc(2 * sizeof(int));
        res[0] = atoi(str);
        res[1] = atoi(str) + 1;
        return res;
    } else if (*str == '\0') {
        printf("Invalid string format (empty string)\n");
    } else if (colon == str + strlen(str) - 1) {
        result.start = malloc((strlen(str) - 1) * sizeof(char) + 1);
        strncpy(result.start, str, strlen(str) - 1);
        result.start[strlen(str) - 1] = '\0';
        result.end = malloc(1 * sizeof(char));
        result.end[0] = 'f';
    } else {
        result.start = malloc((colon - str) * sizeof(char) + 1);
        strncpy(result.start, str, colon - str);
        result.start[colon - str] = '\0';
        result.end = malloc((strlen(str) - (colon - str) - 1) * sizeof(char) + 1);
        strncpy(result.end, colon + 1, strlen(str) - (colon - str) - 1);
        result.end[strlen(str) - (colon - str) - 1] = '\0';
    }
    int* res = malloc(2 * sizeof(int));

    if (strcmp(result.start, "f") == 1) {
        res[0] = 0;
    } else {
        res[0] = atoi(result.start);
    }
    if (strcmp(result.end, "f") == 0) {
        res[1] = RANGE_MAX;
    } else {
        res[1] = atoi(result.end);
    }

    assert (res[0] < res[1]);
    assert (res[0] > -1);
    assert (res[1] <= RANGE_MAX);

    free(result.start);
    free(result.end);

    return res;
}

double normal_random (double mean, double std) {
    double u1, u2, z1;
    do {
        u1 = (double)rand() / RAND_MAX;
        u2 = (double)rand() / RAND_MAX;
    }while (u1 <= 0 || u2 <= 0); 

    z1 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);

    return mean + std * z1;
}

int* get_index_referrence (Tensor* matrix, char* position[2]) {
    int row_max = matrix->tensor_matrix->shape[0];
    int col_max = matrix->tensor_matrix->shape[1];
    int* row_range = split_colon(position[0], row_max);
    int* col_range = split_colon(position[1], col_max);
    int row_start = row_range[0];
    int row_end = row_range[1];
    int col_start = col_range[0];
    int col_end = col_range[1];

    int* output = (int*)malloc(4 * sizeof(int));
    output[0] = row_start;
    output[1] = row_end;
    output[2] = col_start;
    output[3] = col_end;
    
    return output;
}

Matrix* empty_matrix (int shape[2]) {
    Matrix* new_matrix = (Matrix*)malloc(sizeof(Matrix));
    new_matrix->array = (double*)malloc((size_t)(shape[0] * shape[1]) * sizeof(double));
    new_matrix->shape[0] = shape[0];
    new_matrix->shape[1] = shape[1];
    new_matrix->stride[0] = shape[1];
    new_matrix->stride[1] = 1;

    return new_matrix;
}

Matrix* zero_matrix (int shape[2]) {
    Matrix* z = empty_matrix(shape);
    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++) {
            z->array[
                i * z->stride[0] +
                j * z->stride[1]
            ] = 0;
        }
    }
    return z;
}

Matrix* ones_matrix (int shape[2]) {
    Matrix* z = empty_matrix(shape);
    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++) {
            z->array[
                i * z->stride[0] +
                j * z->stride[1]
            ] = 1;
        }
    }
    return z;
}

Matrix* add_matrix (Matrix* m1, Matrix* m2) {
    Matrix* m3 = empty_matrix(m1->shape);
    for (int i = 0; i < m1->shape[0]; i++) {
        for (int j = 0; j < m2->shape[1]; j++) {
            m3->array[
                i * m3->stride[0] +
                j * m3->stride[1]
            ] = m1->array[
                i * m1->stride[0] +
                j * m1->stride[1]
            ] + m2->array[
                i * m2->stride[0] +
                j * m2->stride[1]
            ];
        }
    }
    return m3;
}

Matrix* sub_matrix (Matrix* m1, Matrix* m2) {
    Matrix* m3 = empty_matrix(m1->shape);
    for (int i = 0; i < m1->shape[0]; i++) {
        for (int j = 0; j < m2->shape[1]; j++) {
            m3->array[
                i * m3->stride[0] +
                j * m3->stride[1]
            ] = m1->array[
                i * m1->stride[0] +
                j * m1->stride[1]
            ] - m2->array[
                i * m2->stride[0] +
                j * m2->stride[1]
            ];
        }
    }

    return m3;
}

Matrix* mult_matrix (Matrix* m1, Matrix* m2) {
    Matrix* m3 = empty_matrix(m1->shape);
    for (int i = 0; i < m1->shape[0]; i++) {
        for (int j = 0; j < m2->shape[1]; j++) {
            m3->array[
                i * m3->stride[0] +
                j * m3->stride[1]
            ] = m1->array[
                i * m1->stride[0] +
                j * m1->stride[1]
            ] * m2->array[
                i * m2->stride[0] +
                j * m2->stride[1]
            ];
        }
    }

    return m3;
}

/*
 ****************************************************************************
 *                             Creation Operations                          *
 ****************************************************************************
*/

Tensor* CreateMatrix (int shape[2]) {
    assert (shape[0] > 0 && shape[1] > 0);

    Tensor* matrix = (Tensor*)malloc(sizeof(Tensor));
    matrix->gradient = zero_matrix(shape);
    matrix->tensor_matrix = empty_matrix(shape);

    matrix->power = 1;
    matrix->parents[0] = NULL;
    matrix->parents[1] = NULL;
    matrix->backwards_grad = NULL;
    matrix->creation_operation = NULL;

    return matrix;
}

Tensor* Zeros (int shape[2]) {
    Tensor* zeros = CreateMatrix(shape);
    zeros->tensor_matrix = zero_matrix(shape);

    return zeros;
}

Tensor* Ones (int shape[2]) {
    Tensor* ones = CreateMatrix(shape);
    ones->tensor_matrix = ones_matrix(shape);
    return ones;
}

Tensor* Random (int shape[2]) {
    Tensor* ones = CreateMatrix(shape);
    for (int i = 0; i < shape[0] * shape[1]; i++) {
        int rand_value = rand();
        ones->tensor_matrix->array[i] = (double)rand_value / (double)RAND_MAX;
    }
    return ones;
}

Tensor* Gaussian (int shape[2], double mean, double std) {
    srand(time(NULL));
    Tensor* normal = CreateMatrix(shape);
    for (int i = 0; i < shape[0] * shape[1]; i++) {
        int rand_value = rand();
        normal->tensor_matrix->array[i] = normal_random(mean, std);
    }
    return normal;
}

Tensor* Eye (int shape[2]) {
    assert (shape[0] == shape[1]);

    Tensor* eye = CreateMatrix(shape);
    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++) {
            if (i == j) {
                eye->tensor_matrix->array[(i * eye->tensor_matrix->stride[0]) + (j * eye->tensor_matrix->stride[1])] = 1;
            } else {
                eye->tensor_matrix->array[(i * eye->tensor_matrix->stride[0]) + (j * eye->tensor_matrix->stride[1])] = 0;
            }
        }
    }
    return eye;
}

/*
 ****************************************************************************
 *                             Matrix Operations                            *
 ****************************************************************************
*/

double Det (Tensor* matrix) {
    double ** sub_matrix = (double**)malloc(matrix->tensor_matrix->shape[0] * sizeof(double*));
    for (int i = 0; i < matrix->tensor_matrix->shape[0]; i++) {
        sub_matrix[i] = (double*)malloc(matrix->tensor_matrix->shape[0] * sizeof(double));
    }

    for (int i = 0; i < matrix->tensor_matrix->shape[0]; i++) {
        for (int j = 0; j < matrix->tensor_matrix->shape[1]; j++) {
            sub_matrix[i][j] = matrix->tensor_matrix->array[i * matrix->tensor_matrix->stride[0] + j * matrix->tensor_matrix->stride[1]];
        }
    }

    return determinant(sub_matrix, matrix->tensor_matrix->shape[0]);
}

Tensor* Inverse (Tensor* matrix) {
    double ** sub_matrix = (double**)malloc(matrix->tensor_matrix->shape[0] * sizeof(double*));
    for (int i = 0; i < matrix->tensor_matrix->shape[0]; i++) {
        sub_matrix[i] = (double*)malloc(matrix->tensor_matrix->shape[0] * sizeof(double));
    }

    double ** inverse = (double**)malloc(matrix->tensor_matrix->shape[0] * sizeof(double*));
    for (int i = 0; i < matrix->tensor_matrix->shape[0]; i++) {
        inverse[i] = (double*)malloc(matrix->tensor_matrix->shape[0] * sizeof(double));
    }

    for (int i = 0; i < matrix->tensor_matrix->shape[0]; i++) {
        for (int j = 0; j < matrix->tensor_matrix->shape[1]; j++) {
            sub_matrix[i][j] = matrix->tensor_matrix->array[i * matrix->tensor_matrix->stride[0] + j * matrix->tensor_matrix->stride[1]];
        }
    }

    matrix_inverse(sub_matrix, inverse, matrix->tensor_matrix->shape[0]);

    for (int i = 0; i < matrix->tensor_matrix->shape[0]; i++) {
        free(sub_matrix[i]);
    }
    free(sub_matrix);

    Tensor* inv = CreateMatrix(matrix->tensor_matrix->shape);
    for (int i = 0; i < matrix->tensor_matrix->shape[0]; i++) {
        for (int j = 0; j < matrix->tensor_matrix->shape[1]; j++) {
            inv->tensor_matrix->array[i * inv->tensor_matrix->stride[0] + j * inv->tensor_matrix->stride[1]] = inverse[i][j];
        }
    }

    for (int i = 0; i < matrix->tensor_matrix->shape[0]; i++) {
        free(inverse[i]);
    }
    free(inverse);

    return inv;
}

Tensor* Transpose (Tensor* m) {
    int shape[2] = {m->tensor_matrix->shape[1], m->tensor_matrix->shape[0]};
    Tensor* new_matrix = CreateMatrix(shape);
    new_matrix->creation_operation = "transpose";
    new_matrix->parents[0] = m;

    for (int i = 0; i < m->tensor_matrix->shape[0]; i++) {
        for (int j = 0; j < m->tensor_matrix->shape[1]; j++) {
            new_matrix->tensor_matrix->array[
                (i * new_matrix->tensor_matrix->stride[0]) + 
                (j * new_matrix->tensor_matrix->stride[1])
            ] = m->tensor_matrix->array[(i * m->tensor_matrix->stride[1]) + (j * m->tensor_matrix->stride[0])];
        }
    }

    return new_matrix;
}

void subset_Add (Tensor* main, Tensor* values_to_add, char* index[2]) {
    int* output = get_index_referrence(main, index);

    assert(values_to_add->tensor_matrix->shape[0] == (output[1] - output[0]));
    assert(values_to_add->tensor_matrix->shape[1] == (output[3] - output[2]));

    for (int i = output[0]; i < output[1]; i++) {
        for (int j = output[2]; j < output[3]; j++) {
            main->tensor_matrix->array[
                (i * main->tensor_matrix->stride[0]) + 
                (j * main->tensor_matrix->stride[1])
            ] += values_to_add->tensor_matrix->array[
                ((i-output[0]) * values_to_add->tensor_matrix->stride[0]) + 
                ((j-output[2]) * values_to_add->tensor_matrix->stride[1])
            ];
        }
    }

    free(output);
}

void subset_Sub (Tensor* main, Tensor* values_to_add, char* index[2]) {
    int* output = get_index_referrence(main, index);

    assert(values_to_add->tensor_matrix->shape[0] == (output[1] - output[0]));
    assert(values_to_add->tensor_matrix->shape[1] == (output[3] - output[2]));

    for (int i = output[0]; i < output[1]; i++) {
        for (int j = output[2]; j < output[3]; j++) {
            main->tensor_matrix->array[
                (i * main->tensor_matrix->stride[0]) + 
                (j * main->tensor_matrix->stride[1])
            ] -= values_to_add->tensor_matrix->array[
                ((i-output[0]) * values_to_add->tensor_matrix->stride[0]) + 
                ((j-output[2]) * values_to_add->tensor_matrix->stride[1])
            ];
        }
    }

    free(output);
}

void subset_EleMul (Tensor* main, Tensor* values_to_add, char* index[2]) {
    int* output = get_index_referrence(main, index);

    assert(values_to_add->tensor_matrix->shape[0] == (output[1] - output[0]));
    assert(values_to_add->tensor_matrix->shape[1] == (output[3] - output[2]));

    for (int i = output[0]; i < output[1]; i++) {
        for (int j = output[2]; j < output[3]; j++) {
            main->tensor_matrix->array[
                (i * main->tensor_matrix->stride[0]) + 
                (j * main->tensor_matrix->stride[1])
            ] *= values_to_add->tensor_matrix->array[
                ((i-output[0]) * values_to_add->tensor_matrix->stride[0]) + 
                ((j-output[2]) * values_to_add->tensor_matrix->stride[1])
            ];
        }
    }

    free(output);
}

/*
 ****************************************************************************
 *                        ElementWise Operations                            *
 ****************************************************************************
*/

Tensor* Add (Tensor* m1, Tensor* m2) {
    assert (m1->tensor_matrix->shape[0] == m2->tensor_matrix->shape[0] && m1->tensor_matrix->shape[1] == m2->tensor_matrix->shape[1]);

    Tensor* m3 = CreateMatrix(m1->tensor_matrix->shape);
    m3->parents[0] = m1;
    m3->parents[1] = m2;
    m3->creation_operation = "add";
    m3->tensor_matrix = add_matrix(m1->tensor_matrix, m2->tensor_matrix);

    return m3;
}

Tensor* Sub (Tensor* m1, Tensor* m2) {
    assert (m1->tensor_matrix->shape[0] == m2->tensor_matrix->shape[0] && m1->tensor_matrix->shape[1] == m2->tensor_matrix->shape[1]);

    Tensor* m3 = CreateMatrix(m1->tensor_matrix->shape);
    m3->parents[0] = m1;
    m3->parents[1] = m2;
    m3->creation_operation = "sub";
    m3->tensor_matrix = sub_matrix(m1->tensor_matrix, m2->tensor_matrix);

    return m3;
}

Tensor* Mult (Tensor* m1, Tensor* m2) {
    assert (m1->tensor_matrix->shape[0] == m2->tensor_matrix->shape[0] && m1->tensor_matrix->shape[1] == m2->tensor_matrix->shape[1]);

    Tensor* m3 = CreateMatrix(m1->tensor_matrix->shape);
    m3->parents[0] = m1;
    m3->parents[1] = m2;
    m3->creation_operation = "mult";
    m3->tensor_matrix = mult_matrix(m1->tensor_matrix, m2->tensor_matrix);

    return m3;
}

Tensor* Matmul (Tensor* m1, Tensor* m2) {
    assert (m1->tensor_matrix->shape[1] == m2->tensor_matrix->shape[0]);

    int new_shape[2] = {m1->tensor_matrix->shape[0], m2->tensor_matrix->shape[1]};
    Tensor* m3 = CreateMatrix(new_shape);
    m3->parents[0] = m1;
    m3->parents[1] = m2;
    m3->creation_operation = "matmul";

    for (int i = 0; i < m3->tensor_matrix->shape[0]; i++) {
        for (int j=0; j < m3->tensor_matrix->shape[1]; j++) {
            m3->tensor_matrix->array[i * m3->tensor_matrix->stride[0] + j * m3->tensor_matrix->stride[1]] = 0;
            for (int k = 0; k < m1->tensor_matrix->shape[1]; k++) {
                m3->tensor_matrix->array[i * m3->tensor_matrix->stride[0] + j * m3->tensor_matrix->stride[1]] += m1->tensor_matrix->array[i * m1->tensor_matrix->stride[0] + k * m1->tensor_matrix->stride[1]] *
                                                                    m2->tensor_matrix->array[k * m2->tensor_matrix->stride[0] + j * m2->tensor_matrix->stride[1]];
            }
        }
    }

    return m3;
}

Tensor* Sin (Tensor* matrix) {
    int rows = matrix->tensor_matrix->shape[0];
    int cols = matrix->tensor_matrix->shape[1];

    Tensor* new_matrix = CreateMatrix(matrix->tensor_matrix->shape);
    new_matrix->creation_operation = "sin";
    new_matrix->parents[0] = matrix;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            new_matrix->tensor_matrix->array[
                (i * new_matrix->tensor_matrix->stride[0]) + 
                (j * new_matrix->tensor_matrix->stride[1])
            ] = sin(matrix->tensor_matrix->array[(i * matrix->tensor_matrix->stride[0]) + (j * matrix->tensor_matrix->stride[1])]);
        }
    }

    return new_matrix;
}

Tensor* Cos (Tensor* matrix) {
    int rows = matrix->tensor_matrix->shape[0];
    int cols = matrix->tensor_matrix->shape[1];

    Tensor* new_matrix = CreateMatrix(matrix->tensor_matrix->shape);
    new_matrix->creation_operation = "cos";
    new_matrix->parents[0] = matrix;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            new_matrix->tensor_matrix->array[
                (i * new_matrix->tensor_matrix->stride[0]) + 
                (j * new_matrix->tensor_matrix->stride[1])
            ] = cos(matrix->tensor_matrix->array[(i * matrix->tensor_matrix->stride[0]) + (j * matrix->tensor_matrix->stride[1])]);
        }
    }

    return new_matrix;
}

Tensor* Tan (Tensor* matrix) {
    int rows = matrix->tensor_matrix->shape[0];
    int cols = matrix->tensor_matrix->shape[1];

    Tensor* new_matrix = CreateMatrix(matrix->tensor_matrix->shape);
    new_matrix->creation_operation = "tan";
    new_matrix->parents[0] = matrix;
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            assert (cos(matrix->tensor_matrix->array[
                (i * matrix->tensor_matrix->stride[0]) + 
                (j * matrix->tensor_matrix->stride[1])
            ]) != 0);
            new_matrix->tensor_matrix->array[
                (i * new_matrix->tensor_matrix->stride[0]) + 
                (j * new_matrix->tensor_matrix->stride[1])
            ] = tan(matrix->tensor_matrix->array[(i * matrix->tensor_matrix->stride[0]) + (j * matrix->tensor_matrix->stride[1])]);
        }
    }

    return new_matrix;
}

Tensor* Log (Tensor* matrix) {
    int rows = matrix->tensor_matrix->shape[0];
    int cols = matrix->tensor_matrix->shape[1];

    Tensor* new_matrix = CreateMatrix(matrix->tensor_matrix->shape);
    new_matrix->creation_operation = "log";
    new_matrix->parents[0] = matrix;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            assert (matrix->tensor_matrix->array[
                (i * matrix->tensor_matrix->stride[0]) + 
                (j * matrix->tensor_matrix->stride[1])
            ] != 0);
            new_matrix->tensor_matrix->array[
                (i * new_matrix->tensor_matrix->stride[0]) + 
                (j * new_matrix->tensor_matrix->stride[1])
            ] = log(matrix->tensor_matrix->array[(i * matrix->tensor_matrix->stride[0]) + (j * matrix->tensor_matrix->stride[1])]);
        }
    }

    return new_matrix;
}

Tensor* Element_Pow (Tensor* matrix, double n) {
    int rows = matrix->tensor_matrix->shape[0];
    int cols = matrix->tensor_matrix->shape[1];

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix->tensor_matrix->array[
                (i * matrix->tensor_matrix->stride[0]) + 
                (j * matrix->tensor_matrix->stride[1])
            ] = pow(matrix->tensor_matrix->array[(i * matrix->tensor_matrix->stride[0]) + (j * matrix->tensor_matrix->stride[1])], n);
        }
    }

    return matrix;
}

Tensor* Sqrt (Tensor* m) {
    return Element_Pow(m, 0.5);
}

Tensor* Pow (Tensor* matrix, int n) {
    Tensor* res = Ones(matrix->tensor_matrix->shape);
    while (n > 0) {
        res = Matmul(res, matrix);
        n --;
    }
    return res;
}


/*
 ****************************************************************************
 *                        Aggregration Operations                           *
 ****************************************************************************
*/

double Min (Tensor* matrix) {
    double min = matrix->tensor_matrix->array[0];
    for (int i = 1; i < (matrix->tensor_matrix->shape[0] * matrix->tensor_matrix->shape[1]); i++) {
        if (min > matrix->tensor_matrix->array[i]) {
            min = matrix->tensor_matrix->array[i];
        }
    }
    return min;
}

double Max (Tensor* matrix) {
    double max = matrix->tensor_matrix->array[0];
    for (int i = 1; i < (matrix->tensor_matrix->shape[0] * matrix->tensor_matrix->shape[1]); i++) {
        if (max < matrix->tensor_matrix->array[i]) {
            max = matrix->tensor_matrix->array[i];
        }
    }
    return max;
}

Tensor* Sum (Tensor* matrix) {
    double sum = 0;
    for (int i = 0; i < matrix->tensor_matrix->shape[0] * matrix->tensor_matrix->shape[1]; i++) {
        sum += matrix->tensor_matrix->array[i];
    }
    int shape[2] = {1, 1};

    Tensor* new_martix = CreateMatrix(shape);
    new_martix->parents[0] = matrix;
    new_martix->creation_operation = "sum";
    new_martix->tensor_matrix->array[0] = sum;

    return new_martix;
}

Tensor* Mean (Tensor* matrix) {
    double sum = 0;
    for (int i = 0; i < matrix->tensor_matrix->shape[0] * matrix->tensor_matrix->shape[1]; i++) {
        sum += matrix->tensor_matrix->array[i];
    }
    int denominator = matrix->tensor_matrix->shape[0] * matrix->tensor_matrix->shape[1];
    int shape[2] = {1, 1};
    
    Tensor* new_martix = CreateMatrix(shape);
    new_martix->parents[0] = matrix;
    new_martix->creation_operation = "mean";
    new_martix->tensor_matrix->array[0] = sum / denominator;

    return new_martix;
}

Tensor* Std (Tensor* matrix) {
    double sum = 0;
    for (int i = 0; i < matrix->tensor_matrix->shape[0] * matrix->tensor_matrix->shape[1]; i++) {
        sum += matrix->tensor_matrix->array[i];
    }
    double mean = sum / (matrix->tensor_matrix->shape[0] * matrix->tensor_matrix->shape[1]);
    double variance = 0;
    for (int i = 0; i < matrix->tensor_matrix->shape[0] * matrix->tensor_matrix->shape[1]; i++) {
        double diff = matrix->tensor_matrix->array[i] - mean;
        variance += diff * diff;
    }
    double std = sqrt(variance / ((matrix->tensor_matrix->shape[0] * matrix->tensor_matrix->shape[1]) - 1));

    int shape[2] = {1, 1};
    
    Tensor* new_martix = CreateMatrix(shape);
    new_martix->parents[0] = matrix;
    new_martix->creation_operation = "std";
    new_martix->tensor_matrix->array[0] = std;

    return new_martix;
}


/*
 ****************************************************************************
 *                              Reformation                                 *
 ****************************************************************************
*/

void Reshape (Tensor* matrix, int shape[2]) {
    assert ((matrix->tensor_matrix->shape[0] * matrix->tensor_matrix->shape[1]) == (shape[0] * shape[1]));

    matrix->tensor_matrix->shape[0] = shape[0];
    matrix->tensor_matrix->shape[1] = shape[1];
    matrix->tensor_matrix->stride[0] = shape[1];
    matrix->tensor_matrix->stride[1] = 1;
}

void Flatten (Tensor* matrix, int axis) {
    if (axis == 0) {
        int shape[2] = {matrix->tensor_matrix->shape[0] * matrix->tensor_matrix->shape[1], 1};
        Reshape(matrix, shape);
    } else {
        int shape[2] = {1, matrix->tensor_matrix->shape[0] * matrix->tensor_matrix->shape[1]};
        Reshape(matrix, shape);
    }
}

Tensor* Vstack (Tensor* m1, Tensor* m2) {
    assert (m1->tensor_matrix->shape[1] == m2->tensor_matrix->shape[1]);
    int shape[2] = {m1->tensor_matrix->shape[0] + m2->tensor_matrix->shape[0], m1->tensor_matrix->shape[1]};
    Tensor* new_matrix = CreateMatrix(shape);

    for (int i = 0; i < m2->tensor_matrix->shape[0]; i++) {
        for (int j = 0; j < m2->tensor_matrix->shape[1]; j++) {
            new_matrix->tensor_matrix->array[(i * new_matrix->tensor_matrix->stride[0]) + 
                             (j * new_matrix->tensor_matrix->stride[1])
                            ] = m2->tensor_matrix->array[(i * m2->tensor_matrix->stride[0]) + 
                                          (j * m2->tensor_matrix->stride[1])
                                        ];
        }
    }

    for (int i = m2->tensor_matrix->shape[0]; i < m2->tensor_matrix->shape[0] + m1->tensor_matrix->shape[0]; i++) {
        for (int j = 0; j < m1->tensor_matrix->shape[1]; j++) {
            new_matrix->tensor_matrix->array[(i * new_matrix->tensor_matrix->stride[0]) + 
                              (j * new_matrix->tensor_matrix->stride[1])
                            ] = m1->tensor_matrix->array[((i - m2->tensor_matrix->shape[0]) * m1->tensor_matrix->stride[0]) + 
                                          (j * m1->tensor_matrix->stride[1])
                                        ];
        }
    }

    return new_matrix;
}

Tensor* Hstack (Tensor* m1, Tensor* m2) {
    assert (m1->tensor_matrix->shape[0] == m2->tensor_matrix->shape[0]);
    int shape[2] = {m1->tensor_matrix->shape[0], m1->tensor_matrix->shape[1] + m2->tensor_matrix->shape[1]};
    Tensor* new_matrix = CreateMatrix(shape);

    for (int i = 0; i < m2->tensor_matrix->shape[0]; i++) {
        for (int j = 0; j < m2->tensor_matrix->shape[1]; j++) {
            new_matrix->tensor_matrix->array[(i * new_matrix->tensor_matrix->stride[0]) + 
                             (j * new_matrix->tensor_matrix->stride[1])
                            ] = m2->tensor_matrix->array[(i * m2->tensor_matrix->stride[0]) + 
                                          (j * m2->tensor_matrix->stride[1])
                                        ];
        }
    }

    for (int i = 0; i < m1->tensor_matrix->shape[0]; i++) {
        for (int j = m2->tensor_matrix->shape[1]; j < m1->tensor_matrix->shape[1] + m2->tensor_matrix->shape[1]; j++) {
            new_matrix->tensor_matrix->array[(i * new_matrix->tensor_matrix->stride[0]) + 
                              (j * new_matrix->tensor_matrix->stride[1])
                            ] = m1->tensor_matrix->array[(i * m1->tensor_matrix->stride[0]) + 
                                          ((j - m2->tensor_matrix->shape[1]) * m1->tensor_matrix->stride[1])
                                        ];
        }
    }

    return new_matrix;
}

Tensor* Reduce (Tensor* m1, int axis) {
    if (axis == 2) {
        return Sum(m1);
    }
    else if (axis == 0) {
        int shape[2] = {m1->tensor_matrix->shape[0], 1};
        Tensor* vector = CreateMatrix(shape);
        for (int i = 0; i < m1->tensor_matrix->shape[0]; i++) {
            double sum = 0;
            for (int j = 0; j < m1->tensor_matrix->shape[1]; j++) {
                sum += m1->tensor_matrix->array[i * m1->tensor_matrix->stride[0] + j * m1->tensor_matrix->stride[1]];
            }
            vector->tensor_matrix->array[i * vector->tensor_matrix->stride[0]] = sum;
        }
        return vector;
    } else {
        int shape[2] = {1, m1->tensor_matrix->shape[1]};
        Tensor* vector = CreateMatrix(shape);
        for (int i = 0; i < m1->tensor_matrix->shape[1]; i++) {
            double sum = 0;
            for (int j = 0; j < m1->tensor_matrix->shape[1]; j++) {
                sum += m1->tensor_matrix->array[i * m1->tensor_matrix->stride[0] + j * m1->tensor_matrix->stride[1]];
            }
            vector->tensor_matrix->array[i * vector->tensor_matrix->stride[1]] = sum;
        }
        return vector;
    }
    printf("Enter valid axis");
    return NULL;
}

/*
 ****************************************************************************
 *                              Other                                       *
 ****************************************************************************
*/

Tensor* Copy (Tensor* matrix) {
    Tensor* mat = CreateMatrix(matrix->tensor_matrix->shape);
    for (int i = 0; i < matrix->tensor_matrix->shape[0]; i++) {
        for (int j = 0; j < matrix->tensor_matrix->shape[1]; j++) {
            mat->tensor_matrix->array[i * mat->tensor_matrix->stride[0] + j * mat->tensor_matrix->stride[1]] = matrix->tensor_matrix->array[i * matrix->tensor_matrix->stride[0] + j * matrix->tensor_matrix->stride[1]];
        }
    }
    return mat;
}

/*
 ****************************************************************************
 *                              AutoGrad                                    *
 ****************************************************************************
*/

void append_grad (GradNode* head, GradNode* grad) {
    while(head->next) {
        head = head->next;
    }
    head->next = grad;
}

GradNode* initialise_grad (Matrix* grad) {
    GradNode* root_node = (GradNode*)malloc(sizeof(GradNode));

    root_node->grad_matrix = grad;
    root_node->next = NULL;

    return root_node;
}

Matrix* grad_sum (GradNode* head) {
    Matrix* sum = zero_matrix(head->grad_matrix->shape);
    GradNode* current = head;
    while (current != NULL) {
        sum = add_matrix(sum, head->grad_matrix);
        current = current->next;
    }
    return sum;
}

void backward (Tensor* Z, Matrix* backward_gradient) {
    GradNode* new_grad = initialise_grad(backward_gradient);

    if (Z->backwards_grad == NULL) {
        Z->backwards_grad = new_grad;
    } else {
        append_grad(Z->backwards_grad, new_grad);
    }

    Z->gradient = grad_sum(Z->backwards_grad);

    if (Z->parents[0] == NULL && Z->parents[1] == NULL) {
        return;
    }

    if (strcmp(Z->creation_operation, "add") == 0) {
        backward(Z->parents[0], mult_matrix(Z->gradient, ones_matrix(Z->gradient->shape)));
        backward(Z->parents[1], mult_matrix(Z->gradient, ones_matrix(Z->gradient->shape)));
    }
}


// int main () {
//     clock_t start, end;
//     double cpu_time_used;
//     start = clock();

//     int x_shape[2] = {2, 2};
//     Tensor* X = Eye(x_shape);
//     Tensor* Z = Add(X, X);
//     backward(Z, ones_matrix(Z->tensor_matrix->shape));
//     Print_Matrix(X->gradient);

//     end = clock();
//     cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
//     printf("\n");
//     printf("Total compile time: %f seconds\n", cpu_time_used);
//     return 0;
// }

// gcc numC.c -o exec