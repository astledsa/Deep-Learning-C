#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include "numC.h"

static size_t total_allocated_memory;

/*
 ****************************************************************************
 *                             Helper Functions                             *
 ****************************************************************************
 */

void* malloc_trace (size_t size) {
    void* ptr = malloc(size);
    if (ptr != NULL) {
        // printf("Allocated %zu bytes\n", size);
        total_allocated_memory += size;
    }
    return ptr;
}

void free_matrix (Matrix* m) {
    total_allocated_memory -= ((m->shape[0] * m->shape[1] * sizeof(double)) + sizeof(Matrix));
    // printf("Freed 224 bytes\n");
    free(m->array);
    free(m);
}

void free_tensor (Tensor* m) {
    total_allocated_memory -= (sizeof(Tensor));
    // printf("Freed 512 bytes\n");
    free_matrix(m->gradient);
    free_matrix(m->tensor_matrix);
    free(m);
};

double determinant(double** matrix, int size) {
    if (size == 1) {
        return matrix[0][0];
    }
    
    if (size == 2) {
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    }
    
    double det = 0;
    double **submatrix = (double **)malloc_trace((size) * sizeof(double *));
    for (int i = 0; i < size - 1; i++) {
        submatrix[i] = (double *)malloc_trace((size) * sizeof(double));
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
            double ** sub_matrix = (double**)malloc_trace(N * sizeof(double*));
            for (int i = 0; i < N; i++) {
                sub_matrix[i] = (double*)malloc_trace(N * sizeof(double));
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

void Print (Tensor* matrix) {
    int rows = matrix->tensor_matrix->shape[0];
    int cols = matrix->tensor_matrix->shape[1];

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%lf ", matrix->tensor_matrix->array[(i * matrix->tensor_matrix->stride[0]) + (j * matrix->tensor_matrix->stride[1])]);
        }
        printf("\n");
    }
    printf("\n");
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
    printf("\n");
}

int* split_colon(const char *str, const int RANGE_MAX) {
    Range result;
    result.start = NULL;
    result.end = NULL;
    
    char *colon = strchr(str, ':');
    
    if (colon == NULL) {
        int* res = malloc_trace(2 * sizeof(int));
        res[0] = atoi(str);
        res[1] = atoi(str) + 1;
        return res;
    } else if (*str == '\0') {
        printf("Invalid string format (empty string)\n");
    } else if (colon == str + strlen(str) - 1) {
        result.start = malloc_trace((strlen(str) - 1) * sizeof(char) + 1);
        strncpy(result.start, str, strlen(str) - 1);
        result.start[strlen(str) - 1] = '\0';
        result.end = malloc_trace(1 * sizeof(char));
        result.end[0] = 'f';
    } else {
        result.start = malloc_trace((colon - str) * sizeof(char) + 1);
        strncpy(result.start, str, colon - str);
        result.start[colon - str] = '\0';
        result.end = malloc_trace((strlen(str) - (colon - str) - 1) * sizeof(char) + 1);
        strncpy(result.end, colon + 1, strlen(str) - (colon - str) - 1);
        result.end[strlen(str) - (colon - str) - 1] = '\0';
    }
    int* res = malloc_trace(2 * sizeof(int));

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

    int* output = (int*)malloc_trace(4 * sizeof(int));
    output[0] = row_start;
    output[1] = row_end;
    output[2] = col_start;
    output[3] = col_end;
    
    return output;
}

Matrix* empty_matrix (int shape[2]) {
    Matrix* new_matrix = (Matrix*)malloc_trace(sizeof(Matrix));
    new_matrix->array = (double*)malloc_trace((size_t)(shape[0] * shape[1]) * sizeof(double));
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
    // printf("Ones entered\n");
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

Matrix* add_matrix (Matrix* m1, Matrix* m2, FreeFlag free) {
    // printf("Add entered\n");
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

    switch (free) {
        
        case FREE_1 :
            free_matrix(m1);
            break;
        
        case FREE_2 :
            free_matrix(m2);
            break;
        
        case FREE_BOTH :
            free_matrix(m1);
            free_matrix(m2);
            break;
        
        default:
            break;

    }

    return m3;
}

Matrix* sub_matrix (Matrix* m1, Matrix* m2, FreeFlag free) {
    // printf("Sub entered\n");
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

    switch (free) {
        
        case FREE_1 :
            free_matrix(m1);
            break;
        
        case FREE_2 :
            free_matrix(m2);
            break;
        
        case FREE_BOTH :
            free_matrix(m1);
            free_matrix(m2);
            break;
        
        default:
            break;
            
    }

    return m3;
}

Matrix* mult_matrix (Matrix* m1, Matrix* m2, FreeFlag free) {
    // printf("mult entered\n");
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

    switch (free) {
        
        case FREE_1 :
            free_matrix(m1);
            break;
        
        case FREE_2 :
            free_matrix(m2);
            break;
        
        case FREE_BOTH :
            free_matrix(m1);
            free_matrix(m2);
            break;
        
        default:
            break;
            
    }

    return m3;
}

Matrix* matmul_matrix (Matrix* m1, Matrix* m2, FreeFlag free) {
    // printf("Matmul entered\n");
    int new_shape[2] = {m1->shape[0], m2->shape[1]};
    Matrix* m3 = empty_matrix(new_shape);

    for (int i = 0; i < m3->shape[0]; i++) {
        for (int j=0; j < m3->shape[1]; j++) {
            m3->array[i * m3->stride[0] + j * m3->stride[1]] = 0;
            for (int k = 0; k < m1->shape[1]; k++) {
                m3->array[i * m3->stride[0] + j * m3->stride[1]] += m1->array[i * m1->stride[0] + k * m1->stride[1]] *
                                                                    m2->array[k * m2->stride[0] + j * m2->stride[1]];
            }
        }
    }

    switch (free) {
        
        case FREE_1 :
            free_matrix(m1);
            break;
        
        case FREE_2 :
            free_matrix(m2);
            break;
        
        case FREE_BOTH :
            free_matrix(m1);
            free_matrix(m2);
            break;
        
        default:
            break;
            
    }

    return m3;
}

Matrix* sin_matrix (Matrix* m1, FreeFlag free) {
    Matrix* m3 = empty_matrix(m1->shape);
    for (int i = 0; i < m1->shape[0]; i++) {
        for (int j = 0; j < m1->shape[1]; j++) {
            m3->array[
                (i * m3->stride[0]) + 
                (j * m3->stride[1])
            ] = sin(m1->array[(i * m1->stride[0]) + (j * m1->stride[1])]);
        }
    }

    switch (free) {
        
        case FREE_1 :
            free_matrix(m1);
            break;
        
        default:
            break;
            
    }

    return m3;
}

Matrix* cos_matrix (Matrix* m1, FreeFlag free) {
    Matrix* m3 = empty_matrix(m1->shape);
    for (int i = 0; i < m1->shape[0]; i++) {
        for (int j = 0; j < m1->shape[1]; j++) {
            m3->array[
                (i * m3->stride[0]) + 
                (j * m3->stride[1])
            ] = cos(m1->array[(i * m1->stride[0]) + (j * m1->stride[1])]);
        }
    }

    switch (free) {
        
        case FREE_1 :
            free_matrix(m1);
            break;
        
        default:
            break;
            
    }

    return m3;
}

Matrix* tan_matrix (Matrix* m1, FreeFlag free) {
    Matrix* m3 = empty_matrix(m1->shape);
    for (int i = 0; i < m1->shape[0]; i++) {
        for (int j = 0; j < m1->shape[1]; j++) {
            assert (cos(m3->array[
                (i * m3->stride[0]) + 
                (j * m3->stride[1])
            ]) != 0);
            m3->array[
                (i * m3->stride[0]) + 
                (j * m3->stride[1])
            ] = tan(m1->array[(i * m1->stride[0]) + (j * m1->stride[1])]);
        }
    }

    switch (free) {
        
        case FREE_1 :
            free_matrix(m1);
            break;
        
        default:
            break;
            
    }

    return m3;
}

Matrix* log_matrix (Matrix* m1, FreeFlag free) {
    Matrix* m3 = empty_matrix(m1->shape);
    for (int i = 0; i < m1->shape[0]; i++) {
        for (int j = 0; j < m1->shape[1]; j++) {
            assert (m3->array[
                (i * m3->stride[0]) + 
                (j * m3->stride[1])
            ] == 0);
            m3->array[
                (i * m3->stride[0]) + 
                (j * m3->stride[1])
            ] = log(m1->array[(i * m1->stride[0]) + (j * m1->stride[1])]);
        }
    }

    switch (free) {
        
        case FREE_1 :
            free_matrix(m1);
            break;
        
        default:
            break;
            
    }

    return m3;
}

Matrix* elePow_matrix (Matrix* m1, double n, FreeFlag free) {
    // printf("ele entered\n");
    Matrix* m2 = empty_matrix(m1->shape);

    for (int i = 0; i < m1->shape[0]; i++) {
        for (int j = 0; j < m1->shape[1]; j++) {
            m2->array[
                (i * m2->stride[0]) + 
                (j * m2->stride[1])
            ] = pow(m1->array[(i * m1->stride[0]) + (j * m1->stride[1])], n);
        }
    }

    switch (free) {
        
        case FREE_1 :
            free_matrix(m1);
            break;
        
        default:
            break;
            
    }

    return m2;
}

Matrix* negation (Matrix* m1, FreeFlag free) {
    // printf("negation entered\n");
    Matrix* new_matrix = empty_matrix(m1->shape);

    for (int i = 0; i < m1->shape[0]; i++) {
        for (int j = 0; j < m1->shape[1]; j++) {
            new_matrix->array[
                i * new_matrix->stride[0] +
                j * new_matrix->stride[1]
            ] = -m1->array[
                i * m1->stride[0] +
                j * m1->stride[1]
            ];
        }
    }

    switch (free) {
        
        case FREE_1 :
            free_matrix(m1);
            break;
        
        default:
            break;
            
    }

    return new_matrix;
}

Matrix* scalar_mult_matrix (Matrix* m1, double value, FreeFlag free) {
    // printf("scalar entered\n");
    Matrix* new_matrix = empty_matrix(m1->shape);

    for (int i = 0; i < m1->shape[0]; i++) {
        for (int j = 0; j < m1->shape[1]; j++) {
            new_matrix->array[
                i * new_matrix->stride[0] +
                j * new_matrix->stride[1]
            ] = m1->array[i * m1->stride[0] + j * m1->stride[1]] * value; 
        }
    }

    switch (free) {
        
        case FREE_1 :
            free_matrix(m1);
            break;
        
        default:
            break;
            
    }

    return new_matrix;
}

Matrix* transpose_matrix (Matrix* m1, FreeFlag free) {
    // printf("transpose entered\n");
    int shape[2] = {m1->shape[1], m1->shape[0]};

    Matrix* new_matrix = empty_matrix(shape);

    for (int i = 0; i < new_matrix->shape[0]; i++) {
        for (int j = 0; j < new_matrix->shape[1]; j++) {
            new_matrix->array[
                (i * new_matrix->stride[0]) + 
                (j * new_matrix->stride[1])
            ] = m1->array[(i * m1->stride[1]) + (j * m1->stride[0])];
        }
    }

    switch (free) {
        
        case FREE_1 :
            free_matrix(m1);
            break;
        
        default:
            break;
            
    }

    return new_matrix;
};

Matrix* eye_matrix (int shape[2]) {
    Matrix* new_matrix = empty_matrix(shape);

    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++) {
            if (i == j) {
                new_matrix->array[(i * new_matrix->stride[0]) + (j * new_matrix->stride[1])] = 1;
            } else {
                new_matrix->array[(i * new_matrix->stride[0]) + (j * new_matrix->stride[1])] = 0;
            }
        }
    }

    return new_matrix;
}

Matrix* relu_matrix (Matrix* m1, char* grad, FreeFlag free) {
    Matrix* new_matrix = empty_matrix(m1->shape);

    for (int i = 0; i < m1->shape[0]; i++) {
        for (int j = 0; j < m1->shape[1]; j++) {
            double value = m1->array[
                i * new_matrix->stride[0] +
                j * new_matrix->stride[1]
            ];
            if (value < 0) {
                new_matrix->array[
                    i * new_matrix->stride[0] +
                    j * new_matrix->stride[1]
                ] = 0;
            }
            if (strcmp(grad, "r") == 0 && value > 0) {
                new_matrix->array[
                    i * new_matrix->stride[0] +
                    j * new_matrix->stride[1]
                ] = value;
            } else if (value > 0) {
                new_matrix->array[
                    i * new_matrix->stride[0] +
                    j * new_matrix->stride[1]
                ] = 1;
            }
        }
    }

    switch (free) {
        
        case FREE_1 :
            free_matrix(m1);
            break;
        
        default:
            break;
            
    }

    return new_matrix;
}

/*
 ****************************************************************************
 *                             Creation Operations                          *
 ****************************************************************************
*/

Tensor* CreateTensor (int shape[2]) {
    assert (shape[0] > 0 && shape[1] > 0);

    Tensor* matrix = (Tensor*)malloc_trace(sizeof(Tensor));
    matrix->gradient = zero_matrix(shape);
    matrix->tensor_matrix = empty_matrix(shape);

    matrix->power = 1;
    matrix->scalar = 1;
    matrix->parents[0] = NULL;
    matrix->parents[1] = NULL;
    matrix->requires_grad = FALSE;
    matrix->creation_operation = OP_NULL;

    return matrix;
}

Tensor* Zeros (int shape[2], Bool requires_grad) {
    Tensor* zeros = CreateTensor(shape);
    free_matrix(zeros->tensor_matrix);
    zeros->tensor_matrix = zero_matrix(shape);
    zeros->requires_grad = requires_grad;

    return zeros;
}

Tensor* Ones (int shape[2], Bool requires_grad) {
    Tensor* ones = CreateTensor(shape);
    free_matrix(ones->tensor_matrix);
    ones->tensor_matrix = ones_matrix(shape);
    ones->requires_grad = requires_grad;

    return ones;
}

Tensor* Random (int shape[2], Bool requires_grad) {
    Tensor* ones = CreateTensor(shape);
    for (int i = 0; i < shape[0] * shape[1]; i++) {
        int rand_value = rand();
        ones->tensor_matrix->array[i] = (double)rand_value / (double)RAND_MAX;
    }
    ones->requires_grad = requires_grad;

    return ones;
}

Tensor* Gaussian (int shape[2], double mean, double std, Bool requires_grad) {
    srand(time(NULL));
    Tensor* normal = CreateTensor(shape);
    for (int i = 0; i < shape[0] * shape[1]; i++) {
        int rand_value = rand();
        normal->tensor_matrix->array[i] = normal_random(mean, std);
    }
    normal->requires_grad = requires_grad;

    return normal;
}

Tensor* Eye (int shape[2], Bool requires_grad) {
    assert (shape[0] == shape[1]);

    Tensor* eye = CreateTensor(shape);
    
    free_matrix(eye->tensor_matrix);
    eye->tensor_matrix = eye_matrix(shape);
    eye->requires_grad = requires_grad;

    return eye;
}

/*
 ****************************************************************************
 *                             Matrix Operations                            *
 ****************************************************************************
*/

double Det (Tensor* matrix) {
    double ** sub_matrix = (double**)malloc_trace(matrix->tensor_matrix->shape[0] * sizeof(double*));
    for (int i = 0; i < matrix->tensor_matrix->shape[0]; i++) {
        sub_matrix[i] = (double*)malloc_trace(matrix->tensor_matrix->shape[0] * sizeof(double));
    }

    for (int i = 0; i < matrix->tensor_matrix->shape[0]; i++) {
        for (int j = 0; j < matrix->tensor_matrix->shape[1]; j++) {
            sub_matrix[i][j] = matrix->tensor_matrix->array[i * matrix->tensor_matrix->stride[0] + j * matrix->tensor_matrix->stride[1]];
        }
    }

    return determinant(sub_matrix, matrix->tensor_matrix->shape[0]);
}

Tensor* Inverse (Tensor* matrix, Bool requires_grad) {
    double ** sub_matrix = (double**)malloc_trace(matrix->tensor_matrix->shape[0] * sizeof(double*));
    for (int i = 0; i < matrix->tensor_matrix->shape[0]; i++) {
        sub_matrix[i] = (double*)malloc_trace(matrix->tensor_matrix->shape[0] * sizeof(double));
    }

    double ** inverse = (double**)malloc_trace(matrix->tensor_matrix->shape[0] * sizeof(double*));
    for (int i = 0; i < matrix->tensor_matrix->shape[0]; i++) {
        inverse[i] = (double*)malloc_trace(matrix->tensor_matrix->shape[0] * sizeof(double));
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

    Tensor* inv = CreateTensor(matrix->tensor_matrix->shape);
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
    Tensor* new_matrix = CreateTensor(shape);
    new_matrix->creation_operation = OP_TRANSPOSE;
    new_matrix->parents[0] = m;
    new_matrix->tensor_matrix = transpose_matrix(m->tensor_matrix, 0);

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

    Tensor* m3 = CreateTensor(m1->tensor_matrix->shape);
    m3->parents[0] = m1;
    m3->parents[1] = m2;
    m3->creation_operation = OP_ADD;
    free_matrix(m3->tensor_matrix);
    m3->tensor_matrix = add_matrix(m1->tensor_matrix, m2->tensor_matrix, 0);

    if (m1->requires_grad == TRUE || m2->requires_grad == TRUE) {
        m3->requires_grad = TRUE;
    } else {
        m3->requires_grad = FALSE;
    }

    return m3;
}  

Tensor* Sub (Tensor* m1, Tensor* m2) {
    assert (m1->tensor_matrix->shape[0] == m2->tensor_matrix->shape[0] && m1->tensor_matrix->shape[1] == m2->tensor_matrix->shape[1]);

    Tensor* m3 = CreateTensor(m1->tensor_matrix->shape);
    m3->parents[0] = m1;
    m3->parents[1] = m2;
    m3->creation_operation = OP_SUBTRACT;
    free_matrix(m3->tensor_matrix);
    m3->tensor_matrix = sub_matrix(m1->tensor_matrix, m2->tensor_matrix, 0);

    if (m1->requires_grad == TRUE || m2->requires_grad == TRUE) {
        m3->requires_grad = TRUE;
    } else {
        m3->requires_grad = FALSE;
    }

    return m3;
}

Tensor* Mult (Tensor* m1, Tensor* m2) {
    assert (m1->tensor_matrix->shape[0] == m2->tensor_matrix->shape[0] && m1->tensor_matrix->shape[1] == m2->tensor_matrix->shape[1]);

    Tensor* m3 = CreateTensor(m1->tensor_matrix->shape);
    m3->parents[0] = m1;
    m3->parents[1] = m2;
    m3->creation_operation = OP_MULTIPLY;
    free_matrix(m3->tensor_matrix);
    m3->tensor_matrix = mult_matrix(m1->tensor_matrix, m2->tensor_matrix, 0);

    if (m1->requires_grad == TRUE || m2->requires_grad == TRUE) {
        m3->requires_grad = TRUE;
    } else {
        m3->requires_grad = FALSE;
    }

    return m3;
}

Tensor* Matmul (Tensor* m1, Tensor* m2) {
    assert (m1->tensor_matrix->shape[1] == m2->tensor_matrix->shape[0]);

    int new_shape[2] = {m1->tensor_matrix->shape[0], m2->tensor_matrix->shape[1]};
    Tensor* m3 = CreateTensor(new_shape);
    m3->parents[0] = m1;
    m3->parents[1] = m2;
    m3->creation_operation = OP_MATMUL;
    free_matrix(m3->tensor_matrix);
    m3->tensor_matrix = matmul_matrix(m1->tensor_matrix, m2->tensor_matrix, 0);

    if (m1->requires_grad == TRUE || m2->requires_grad == TRUE) {
        m3->requires_grad = TRUE;
    } else {
        m3->requires_grad = FALSE;
    }

    return m3;
}

Tensor* Sin (Tensor* matrix) {

    Tensor* new_matrix = CreateTensor(matrix->tensor_matrix->shape);
    new_matrix->creation_operation = OP_SIN;
    new_matrix->parents[0] = matrix;
    free_matrix(new_matrix->tensor_matrix);
    new_matrix->tensor_matrix = sin_matrix(matrix->tensor_matrix, 0);

    if (matrix->requires_grad == TRUE) {
        new_matrix->requires_grad = TRUE;
    } else {
        new_matrix->requires_grad = FALSE;
    }

    return new_matrix;
}

Tensor* Cos (Tensor* matrix) {

    Tensor* new_matrix = CreateTensor(matrix->tensor_matrix->shape);
    new_matrix->creation_operation = OP_COS;
    new_matrix->parents[0] = matrix;
    free_matrix(new_matrix->tensor_matrix);
    new_matrix->tensor_matrix = cos_matrix(matrix->tensor_matrix, 0);

    if (matrix->requires_grad == TRUE) {
        new_matrix->requires_grad = TRUE;
    } else {
        new_matrix->requires_grad = FALSE;
    }

    return new_matrix;
}

Tensor* Tan (Tensor* matrix) {

    Tensor* new_matrix = CreateTensor(matrix->tensor_matrix->shape);
    new_matrix->creation_operation = OP_TAN;
    new_matrix->parents[0] = matrix;
    free_matrix(new_matrix->tensor_matrix);
    new_matrix->tensor_matrix = tan_matrix(matrix->tensor_matrix, 0);

    if (matrix->requires_grad == TRUE) {
        new_matrix->requires_grad = TRUE;
    } else {
        new_matrix->requires_grad = FALSE;
    }

    return new_matrix;
}

Tensor* Log (Tensor* matrix) {
    int rows = matrix->tensor_matrix->shape[0];
    int cols = matrix->tensor_matrix->shape[1];

    Tensor* new_matrix = CreateTensor(matrix->tensor_matrix->shape);
    new_matrix->creation_operation = OP_LOG;
    new_matrix->parents[0] = matrix;
    free_matrix(new_matrix->tensor_matrix);
    new_matrix->tensor_matrix = log_matrix(matrix->tensor_matrix, 0);

    if (matrix->requires_grad == TRUE) {
        new_matrix->requires_grad = TRUE;
    } else {
        new_matrix->requires_grad = FALSE;
    }

    return new_matrix;
}

Tensor* Element_Pow (Tensor* matrix, double n) {
    Tensor* m2 = CreateTensor(matrix->tensor_matrix->shape);

    m2->parents[0] = matrix;
    m2->power = n;
    m2->creation_operation = OP_ELE_POW;
    free_matrix(m2->tensor_matrix);
    m2->tensor_matrix = elePow_matrix(matrix->tensor_matrix, n, 0);

    if (matrix->requires_grad == TRUE) {
        m2->requires_grad = TRUE;
    } else {
        m2->requires_grad = FALSE;
    }
    
    return m2;
}

Tensor* Sqrt (Tensor* m) {
    return Element_Pow(m, 0.5);
}

Tensor* Pow (Tensor* matrix, double n) {
    Tensor* res = Ones(matrix->tensor_matrix->shape, matrix->requires_grad);
    while (n > 0) {
        res = Matmul(res, matrix);
        n --;
    }

    if (matrix->requires_grad == TRUE) {
        res->requires_grad = TRUE;
    } else {
        res->requires_grad = FALSE;
    }

    return res;
}

Tensor* Scalar (Tensor* matrix, double value) {
    if (value == 0) {
        printf("Division by Zero error!");
        exit(EXIT_FAILURE);
    }
    Tensor* new_matrix = CreateTensor(matrix->tensor_matrix->shape);

    new_matrix->parents[0] = matrix;
    new_matrix->creation_operation = OP_SCALAR;
    free_matrix(new_matrix->tensor_matrix);
    new_matrix->tensor_matrix = scalar_mult_matrix(matrix->tensor_matrix, value, 0);

    if (matrix->requires_grad == TRUE) {
        new_matrix->requires_grad = TRUE;
    } else {
        new_matrix->requires_grad = FALSE;
    }

    new_matrix->scalar = value;

    return new_matrix;
}

Tensor* Relu (Tensor* matrix) {

    Tensor* new_matrix = CreateTensor(matrix->tensor_matrix->shape);
    new_matrix->creation_operation = OP_RELU;
    new_matrix->parents[0] = matrix;
    free_matrix(new_matrix->tensor_matrix);
    new_matrix->tensor_matrix = relu_matrix(matrix->tensor_matrix, "r", FREE_0);

    if (matrix->requires_grad == TRUE) {
        new_matrix->requires_grad = TRUE;
    } else {
        new_matrix->requires_grad = FALSE;
    }

    return new_matrix;
}

/*
 ****************************************************************************
 *                         Aggregration Operations                           *
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

    Tensor* new_matrix = CreateTensor(shape);
    new_matrix->parents[0] = matrix;
    new_matrix->creation_operation = OP_SUM;
    new_matrix->tensor_matrix->array[0] = sum;

    if (matrix->requires_grad == TRUE) {
        new_matrix->requires_grad = TRUE;
    } else {
        new_matrix->requires_grad = FALSE;
    }

    return new_matrix;
}

Tensor* Mean (Tensor* matrix) {
    double sum = 0;
    for (int i = 0; i < matrix->tensor_matrix->shape[0] * matrix->tensor_matrix->shape[1]; i++) {
        sum += matrix->tensor_matrix->array[i];
    }
    int denominator = matrix->tensor_matrix->shape[0] * matrix->tensor_matrix->shape[1];
    int shape[2] = {1, 1};
    
    Tensor* new_matrix = CreateTensor(shape);
    new_matrix->parents[0] = matrix;
    new_matrix->creation_operation = OP_MEAN;
    new_matrix->tensor_matrix->array[0] = sum / denominator;

    if (matrix->requires_grad == TRUE) {
        new_matrix->requires_grad = TRUE;
    } else {
        new_matrix->requires_grad = FALSE;
    }

    return new_matrix;
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
    
    Tensor* new_matrix = CreateTensor(shape);
    new_matrix->parents[0] = matrix;
    new_matrix->creation_operation = OP_STD;
    new_matrix->tensor_matrix->array[0] = std;

    if (matrix->requires_grad == TRUE) {
        new_matrix->requires_grad = TRUE;
    } else {
        new_matrix->requires_grad = FALSE;
    }

    return new_matrix;
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
    Tensor* new_matrix = CreateTensor(shape);

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
    Tensor* new_matrix = CreateTensor(shape);

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
        Tensor* vector = CreateTensor(shape);
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
        Tensor* vector = CreateTensor(shape);
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

Tensor* Copy (Tensor* matrix) {
    Tensor* mat = CreateTensor(matrix->tensor_matrix->shape);
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

void Zero_grad (Tensor* weight) {
    weight->gradient = zero_matrix(weight->gradient->shape);
}

void backward (Tensor* Z, Matrix* backward_gradient) {

    if (Z->requires_grad == FALSE) {
        free_matrix(backward_gradient);
        return;
    }

    Z->gradient = add_matrix(Z->gradient, backward_gradient, FREE_BOTH);

    if (Z->parents[0] == NULL && Z->parents[1] == NULL) {
        return;
    }

    switch (Z->creation_operation) {

        case OP_ADD:
            backward(Z->parents[0], mult_matrix(Z->gradient, ones_matrix(Z->gradient->shape), FREE_2));
            backward(Z->parents[1], mult_matrix(Z->gradient, ones_matrix(Z->gradient->shape), FREE_2));
            break;

        case OP_SUBTRACT:
            backward(Z->parents[0], mult_matrix(Z->gradient, ones_matrix(Z->gradient->shape), FREE_2));
            backward(Z->parents[1], mult_matrix(Z->gradient, negation(ones_matrix(Z->gradient->shape), FREE_1), FREE_2));
            break;
    
        case OP_MULTIPLY:
            backward(Z->parents[0], mult_matrix(Z->gradient, Z->parents[1]->tensor_matrix, FREE_0));
            backward(Z->parents[1], mult_matrix(Z->gradient, Z->parents[0]->tensor_matrix, FREE_0));
            break;
        
        case OP_SIN:
            backward(Z->parents[0], mult_matrix(Z->gradient, cos_matrix(Z->parents[0]->tensor_matrix, 0), FREE_2));
            break;
        
        case OP_COS:
            backward(Z->parents[0], mult_matrix(Z->gradient, negation(sin_matrix(Z->parents[0]->tensor_matrix, FREE_0), FREE_1), FREE_2));
            break;
        
        case OP_LOG:
            backward( 
                Z->parents[0], 
                mult_matrix(
                    Z->gradient, 
                    elePow_matrix(Z->parents[0]->tensor_matrix, -1.0f, FREE_0),
                    2
                )
            );
            break;
        
        case OP_ELE_POW:
            backward(
                Z->parents[0], 
                mult_matrix(
                    Z->gradient, 
                    scalar_mult_matrix(
                        elePow_matrix(Z->parents[0]->tensor_matrix, Z->power-1.0f, FREE_0), 
                        Z->power,
                        1
                    ),
                    2
                )
            );
            break;
        
        case OP_MATMUL:
            backward(Z->parents[0], matmul_matrix(Z->gradient, transpose_matrix(Z->parents[1]->tensor_matrix, FREE_0), FREE_2));
            backward(Z->parents[1], transpose_matrix(matmul_matrix(transpose_matrix(Z->gradient, FREE_0), Z->parents[0]->tensor_matrix, FREE_1), FREE_1));
            break;

        case OP_SCALAR:
            backward(Z->parents[0], scalar_mult_matrix(eye_matrix(Z->gradient->shape), Z->scalar, FREE_1));
            break;
        
        case OP_RELU:
            backward(Z->parents[0], mult_matrix(Z->gradient, relu_matrix(Z->parents[0]->tensor_matrix, "g", FREE_0), FREE_2));
            break;
        
        case OP_SUM:
            backward(Z->parents[0], scalar_mult_matrix(ones_matrix(Z->parents[0]->gradient->shape), Z->gradient->array[0], FREE_1));
            break;

        default:
            break;
    }
}

void Update (Tensor* Weight, double learning_rate) {
    Weight->tensor_matrix = scalar_mult_matrix(sub_matrix(Weight->tensor_matrix, Weight->gradient, FREE_1), learning_rate, FREE_1);
}

void Backward (Tensor* Z) {
    Matrix* back_grad = ones_matrix(Z->tensor_matrix->shape);
    backward(Z, back_grad);
}

// gcc numC.c -o exec