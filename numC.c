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
// -- Reduce
// -- Broadcast

// Utility Functions
// -- Copy

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <math.h>

static size_t total_allocated_memory;

typedef struct {
  char *start;
  char *end;
} Range;

typedef struct {
    double* array;

    int shape[2];
    int stride[2];
}Tensor;

void* malloc_trace (size_t size) {
    void* ptr = malloc(size);
    if (ptr != NULL) {
        printf("Allocated %zu space\n", size);
        total_allocated_memory += size;
    }
    return ptr;
}

void Print (Tensor* matrix) {
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

Tensor* CreateMatrix (int shape[2]) {
    assert (shape[0] > 0 && shape[1] > 0);

    Tensor* matrix = (Tensor*)malloc(sizeof(Tensor));
    matrix->array = (double*)malloc((size_t)(shape[0] * shape[1]) * sizeof(double));
    matrix->shape[0] = shape[0];
    matrix->shape[1] = shape[1];
    matrix->stride[0] = shape[1];
    matrix->stride[1] = 1;

    return matrix;
}

int* get_index_referrence (Tensor* matrix, char* position[2]) {
    int row_max = matrix->shape[0];
    int col_max = matrix->shape[1];
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

void MAT_ADD (Tensor* main, Tensor* values_to_add, char* index[2]) {
    int* output = get_index_referrence(main, index);

    assert(values_to_add->shape[0] == (output[1] - output[0]));
    assert(values_to_add->shape[1] == (output[3] - output[2]));

    for (int i = output[0]; i < output[1]; i++) {
        for (int j = output[2]; j < output[3]; j++) {
            main->array[
                (i * main->stride[0]) + 
                (j * main->stride[1])
            ] += values_to_add->array[
                ((i-output[0]) * values_to_add->stride[0]) + 
                ((j-output[2]) * values_to_add->stride[1])
            ];
        }
    }

    free(output);
}

void MAT_SUB (Tensor* main, Tensor* values_to_add, char* index[2]) {
    int* output = get_index_referrence(main, index);

    assert(values_to_add->shape[0] == (output[1] - output[0]));
    assert(values_to_add->shape[1] == (output[3] - output[2]));

    for (int i = output[0]; i < output[1]; i++) {
        for (int j = output[2]; j < output[3]; j++) {
            main->array[
                (i * main->stride[0]) + 
                (j * main->stride[1])
            ] -= values_to_add->array[
                ((i-output[0]) * values_to_add->stride[0]) + 
                ((j-output[2]) * values_to_add->stride[1])
            ];
        }
    }

    free(output);
}

void MAT_ELE_MUL (Tensor* main, Tensor* values_to_add, char* index[2]) {
    int* output = get_index_referrence(main, index);

    assert(values_to_add->shape[0] == (output[1] - output[0]));
    assert(values_to_add->shape[1] == (output[3] - output[2]));

    for (int i = output[0]; i < output[1]; i++) {
        for (int j = output[2]; j < output[3]; j++) {
            main->array[
                (i * main->stride[0]) + 
                (j * main->stride[1])
            ] *= values_to_add->array[
                ((i-output[0]) * values_to_add->stride[0]) + 
                ((j-output[2]) * values_to_add->stride[1])
            ];
        }
    }

    free(output);
}

double Min (Tensor* matrix) {
    double min = matrix->array[0];
    for (int i = 1; i < (matrix->shape[0] * matrix->shape[1]); i++) {
        if (min > matrix->array[i]) {
            min = matrix->array[i];
        }
    }
    return min;
}

double Max (Tensor* matrix) {
    double max = matrix->array[0];
    for (int i = 1; i < (matrix->shape[0] * matrix->shape[1]); i++) {
        if (max < matrix->array[i]) {
            max = matrix->array[i];
        }
    }
    return max;
}

Tensor* Zeros (int shape[2]) {
    Tensor* zeros = CreateMatrix(shape);
    for (int i = 0; i < shape[0] * shape[1]; i++) {
        zeros->array[i] = 0;
    }
    return zeros;
}

Tensor* Ones (int shape[2]) {
    Tensor* ones = CreateMatrix(shape);
    for (int i = 0; i < shape[0] * shape[1]; i++) {
        ones->array[i] = 1;
    }
    return ones;
}

Tensor* Random (int shape[2]) {
    Tensor* ones = CreateMatrix(shape);
    for (int i = 0; i < shape[0] * shape[1]; i++) {
        int rand_value = rand();
        ones->array[i] = (double)rand_value / (double)RAND_MAX;
    }
    return ones;
}

Tensor* Gaussian (int shape[2], double mean, double std) {
    srand(time(NULL));
    Tensor* normal = CreateMatrix(shape);
    for (int i = 0; i < shape[0] * shape[1]; i++) {
        int rand_value = rand();
        normal->array[i] = normal_random(mean, std);
    }
    return normal;
}

Tensor* Eye (int shape[2]) {
    assert (shape[0] == shape[1]);

    Tensor* eye = CreateMatrix(shape);
    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++) {
            if (i == j) {
                eye->array[(i * eye->stride[0]) + (j * eye->stride[1])] = 1;
            } else {
                eye->array[(i * eye->stride[0]) + (j * eye->stride[1])] = 0;
            }
        }
    }
    return eye;
}

Tensor* ADD (Tensor* m1, Tensor* m2) {
    assert (m1->shape[0] == m2->shape[0] && m1->shape[1] == m2->shape[1]);

    Tensor* m3 = CreateMatrix(m1->shape);
    for (int row = 0; row < m1->shape[0]; row++) {
        for (int col = 0; col < m1->shape[1]; col++) {
            m3->array[row * m3->stride[0] + col * m3->stride[1]] = m1->array[row * m1->stride[0] + col * m1->stride[1]]+
                                                                   m2->array[row * m2->stride[0] + col * m2->stride[1]];
        }
    }
    return m3;
}

Tensor* SUBTRACT (Tensor* m1, Tensor* m2) {
    assert (m1->shape[0] == m2->shape[0] && m1->shape[1] == m2->shape[1]);

    Tensor* m3 = CreateMatrix(m1->shape);
    for (int row = 0; row < m1->shape[0]; row++) {
        for (int col = 0; col < m1->shape[1]; col++) {
            m3->array[row * m3->stride[0] + col * m3->stride[1]] = m1->array[row * m1->stride[0] + col * m1->stride[1]]-
                                                                   m2->array[row * m2->stride[0] + col * m2->stride[1]];
        }
    }
    return m3;
}

Tensor* MULT (Tensor* m1, Tensor* m2) {
    assert (m1->shape[0] == m2->shape[0] && m1->shape[1] == m2->shape[1]);

    Tensor* m3 = CreateMatrix(m1->shape);
    for (int row = 0; row < m1->shape[0]; row++) {
        for (int col = 0; col < m1->shape[1]; col++) {
            m3->array[row * m3->stride[0] + col * m3->stride[1]] = m1->array[row * m1->stride[0] + col * m1->stride[1]]*
                                                                   m2->array[row * m2->stride[0] + col * m2->stride[1]];
        }
    }
    return m3;
}

Tensor* MATMUL (Tensor* m1, Tensor* m2) {
    assert (m1->shape[1] == m2->shape[0]);

    int new_shape[2] = {m1->shape[0], m2->shape[1]};
    Tensor* m3 = CreateMatrix(new_shape);

    for (int i = 0; i < m3->shape[0]; i++) {
        for (int j=0; j < m3->shape[1]; j++) {
            m3->array[i * m3->stride[0] + j * m3->stride[1]] = 0;
            for (int k = 0; k < m1->shape[1]; k++) {
                m3->array[i * m3->stride[0] + j * m3->stride[1]] += m1->array[i * m1->stride[0] + k * m1->stride[1]] *
                                                                    m2->array[k * m2->stride[0] + j * m2->stride[1]];
            }
        }
    }

    return m3;
}

Tensor* Transpose (Tensor* m) {
    int shape[2] = {m->shape[1], m->shape[0]};
    Tensor* new_matrix = CreateMatrix(shape);

    for (int i = 0; i < m->shape[0]; i++) {
        for (int j = 0; j < m->shape[1]; j++) {
            new_matrix->array[
                (i * new_matrix->stride[0]) + 
                (j * new_matrix->stride[1])
            ] = m->array[(i * m->stride[1]) + (j * m->stride[0])];
        }
    }

    return new_matrix;
}

Tensor* SIN (Tensor* matrix) {
    int rows = matrix->shape[0];
    int cols = matrix->shape[1];

    Tensor* new_matrix = CreateMatrix(matrix->shape);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            new_matrix->array[
                (i * new_matrix->stride[0]) + 
                (j * new_matrix->stride[1])
            ] = sin(matrix->array[(i * matrix->stride[0]) + (j * matrix->stride[1])]);
        }
    }

    return new_matrix;
}

Tensor* COS (Tensor* matrix) {
    int rows = matrix->shape[0];
    int cols = matrix->shape[1];

    Tensor* new_matrix = CreateMatrix(matrix->shape);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            new_matrix->array[
                (i * new_matrix->stride[0]) + 
                (j * new_matrix->stride[1])
            ] = cos(matrix->array[(i * matrix->stride[0]) + (j * matrix->stride[1])]);
        }
    }

    return new_matrix;
}

Tensor* TAN (Tensor* matrix) {
    int rows = matrix->shape[0];
    int cols = matrix->shape[1];

    Tensor* new_matrix = CreateMatrix(matrix->shape);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            assert (cos(matrix->array[
                (i * matrix->stride[0]) + 
                (j * matrix->stride[1])
            ]) != 0);
            new_matrix->array[
                (i * new_matrix->stride[0]) + 
                (j * new_matrix->stride[1])
            ] = tan(matrix->array[(i * matrix->stride[0]) + (j * matrix->stride[1])]);
        }
    }

    return new_matrix;
}

Tensor* LOG (Tensor* matrix) {
    int rows = matrix->shape[0];
    int cols = matrix->shape[1];

    Tensor* new_matrix = CreateMatrix(matrix->shape);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            assert (matrix->array[
                (i * matrix->stride[0]) + 
                (j * matrix->stride[1])
            ] != 0);
            new_matrix->array[
                (i * new_matrix->stride[0]) + 
                (j * new_matrix->stride[1])
            ] = log(matrix->array[(i * matrix->stride[0]) + (j * matrix->stride[1])]);
        }
    }

    return new_matrix;
}

double Trace (Tensor* m) {
    assert (m->shape[0] == m->shape[1]);

    double sum = 0;
    for (int i = 0; i < m->shape[0]; i++) {
        for (int j = 0; j < m->shape[1]; j++) {
            if (i == j) {
                sum += m->array[(i * m->stride[0]) + (j * m->stride[1])]; 
            }
        }
    }
    return sum;
}

Tensor* ELEMENT_POW (Tensor* matrix, double n) {
    int rows = matrix->shape[0];
    int cols = matrix->shape[1];

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix->array[
                (i * matrix->stride[0]) + 
                (j * matrix->stride[1])
            ] = pow(matrix->array[(i * matrix->stride[0]) + (j * matrix->stride[1])], n);
        }
    }

    return matrix;
}

Tensor* SQRT (Tensor* m) {
    return ELEMENT_POW(m, 0.5);
}

Tensor* POW (Tensor* matrix, int n) {
    Tensor* res = Ones(matrix->shape);
    while (n > 0) {
        res = MATMUL(res, matrix);
        n --;
    }
    return res;
}

double determinant(double **matrix, int size) {
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

double Det (Tensor* matrix) {
    double ** sub_matrix = (double**)malloc(matrix->shape[0] * sizeof(double*));
    for (int i = 0; i < matrix->shape[0]; i++) {
        sub_matrix[i] = (double*)malloc(matrix->shape[0] * sizeof(double));
    }

    for (int i = 0; i < matrix->shape[0]; i++) {
        for (int j = 0; j < matrix->shape[1]; j++) {
            sub_matrix[i][j] = matrix->array[i * matrix->stride[0] + j * matrix->stride[1]];
        }
    }

    return determinant(sub_matrix, matrix->shape[0]);
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

Tensor* Inverse (Tensor* matrix) {
    double ** sub_matrix = (double**)malloc(matrix->shape[0] * sizeof(double*));
    for (int i = 0; i < matrix->shape[0]; i++) {
        sub_matrix[i] = (double*)malloc(matrix->shape[0] * sizeof(double));
    }

    double ** inverse = (double**)malloc(matrix->shape[0] * sizeof(double*));
    for (int i = 0; i < matrix->shape[0]; i++) {
        inverse[i] = (double*)malloc(matrix->shape[0] * sizeof(double));
    }

    for (int i = 0; i < matrix->shape[0]; i++) {
        for (int j = 0; j < matrix->shape[1]; j++) {
            sub_matrix[i][j] = matrix->array[i * matrix->stride[0] + j * matrix->stride[1]];
        }
    }

    matrix_inverse(sub_matrix, inverse, matrix->shape[0]);

    for (int i = 0; i < matrix->shape[0]; i++) {
        free(sub_matrix[i]);
    }
    free(sub_matrix);

    Tensor* inv = CreateMatrix(matrix->shape);
    for (int i = 0; i < matrix->shape[0]; i++) {
        for (int j = 0; j < matrix->shape[1]; j++) {
            inv->array[i * inv->stride[0] + j * inv->stride[1]] = inverse[i][j];
        }
    }

    for (int i = 0; i < matrix->shape[0]; i++) {
        free(inverse[i]);
    }
    free(inverse);

    return inv;
}

Tensor* SUM (Tensor* matrix) {
    double sum = 0;
    for (int i = 0; i < matrix->shape[0] * matrix->shape[1]; i++) {
        sum += matrix->array[i];
    }
    int shape[2] = {1, 1};
    Tensor* new_martix = CreateMatrix(shape);
    new_martix->array[0] = sum;
    return new_martix;
}

Tensor* MEAN (Tensor* matrix) {
    double sum = 0;
    for (int i = 0; i < matrix->shape[0] * matrix->shape[1]; i++) {
        sum += matrix->array[i];
    }
    int denominator = matrix->shape[0] * matrix->shape[1];
    int shape[2] = {1, 1};
    Tensor* new_martix = CreateMatrix(shape);

    new_martix->array[0] = sum / denominator;
    return new_martix;
}

Tensor* STD (Tensor* matrix) {
    double sum = 0;
    for (int i = 0; i < matrix->shape[0] * matrix->shape[1]; i++) {
        sum += matrix->array[i];
    }
    double mean = sum / (matrix->shape[0] * matrix->shape[1]);
    double variance = 0;
    for (int i = 0; i < matrix->shape[0] * matrix->shape[1]; i++) {
        double diff = matrix->array[i] - mean;
        variance += diff * diff;
    }
    double std = sqrt(variance / ((matrix->shape[0] * matrix->shape[1]) - 1));

    int shape[2] = {1, 1};
    Tensor* new_martix = CreateMatrix(shape);
    new_martix->array[0] = std;
    return new_martix;
}

void Reshape (Tensor* matrix, int shape[2]) {
    assert ((matrix->shape[0] * matrix->shape[1]) == (shape[0] * shape[1]));

    matrix->shape[0] = shape[0];
    matrix->shape[1] = shape[1];
    matrix->stride[0] = shape[1];
    matrix->stride[1] = 1;
}

void Flatten (Tensor* matrix, int axis) {
    if (axis == 0) {
        int shape[2] = {matrix->shape[0] * matrix->shape[1], 1};
        Reshape(matrix, shape);
    } else {
        int shape[2] = {1, matrix->shape[0] * matrix->shape[1]};
        Reshape(matrix, shape);
    }
}

Tensor* Vstack (Tensor* m1, Tensor* m2) {
    assert (m1->shape[1] == m2->shape[1]);
    int shape[2] = {m1->shape[0] + m2->shape[0], m1->shape[1]};
    Tensor* new_matrix = CreateMatrix(shape);

    for (int i = 0; i < m2->shape[0]; i++) {
        for (int j = 0; j < m2->shape[1]; j++) {
            new_matrix->array[(i * new_matrix->stride[0]) + 
                             (j * new_matrix->stride[1])
                            ] = m2->array[(i * m2->stride[0]) + 
                                          (j * m2->stride[1])
                                        ];
        }
    }

    for (int i = m2->shape[0]; i < m2->shape[0] + m1->shape[0]; i++) {
        for (int j = 0; j < m1->shape[1]; j++) {
            new_matrix->array[(i * new_matrix->stride[0]) + 
                              (j * new_matrix->stride[1])
                            ] = m1->array[((i - m2->shape[0]) * m1->stride[0]) + 
                                          (j * m1->stride[1])
                                        ];
        }
    }

    return new_matrix;
}

Tensor* Hstack (Tensor* m1, Tensor* m2) {
    assert (m1->shape[0] == m2->shape[0]);
    int shape[2] = {m1->shape[0], m1->shape[1] + m2->shape[1]};
    Tensor* new_matrix = CreateMatrix(shape);

    for (int i = 0; i < m2->shape[0]; i++) {
        for (int j = 0; j < m2->shape[1]; j++) {
            new_matrix->array[(i * new_matrix->stride[0]) + 
                             (j * new_matrix->stride[1])
                            ] = m2->array[(i * m2->stride[0]) + 
                                          (j * m2->stride[1])
                                        ];
        }
    }

    for (int i = 0; i < m1->shape[0]; i++) {
        for (int j = m2->shape[1]; j < m1->shape[1] + m2->shape[1]; j++) {
            new_matrix->array[(i * new_matrix->stride[0]) + 
                              (j * new_matrix->stride[1])
                            ] = m1->array[(i * m1->stride[0]) + 
                                          ((j - m2->shape[1]) * m1->stride[1])
                                        ];
        }
    }

    return new_matrix;
}

// int main () {
//     clock_t start, end;
//     double cpu_time_used;

//     start = clock();

//     int shape[2] = {5, 4};
//     int newShape[2] = {5, 5};

//     Tensor* r = Random(shape);
//     Tensor* I = Eye(newShape);
//     Tensor* h = Hstack(r, I);

//     Print(h);

//     end = clock();

//     cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

//     printf("Total compile time: %f seconds\n", cpu_time_used);

//     return 0;
// }

// gcc numC.c -o exec