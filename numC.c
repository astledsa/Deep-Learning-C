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
// -- Inverse
// -- Trace.
// -- Determinant

// Element Wise Operations
// -- Addition.
// -- Subtraction.
// -- Division
// -- Multiplication.
// -- Log.
// -- Sin.
// -- Cos.
// -- Tan.
// -- Power.
// -- Square Root.

// Aggregation Functions
// -- Summation
// -- Mean
// -- Mode
// -- Median
// -- Standard Deviation
// -- Max
// -- Min

// Reshaping and Indexing
// -- Reshape
// -- Concantenate
// -- Hstack
// -- Vstack
// -- Flatten

// Utility Functions
// -- Copy

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <math.h>

typedef struct {
    double* array;

    int shape[2];
    int stride[2];
}Matrix;

void Print (Matrix* matrix) {
    int rows = matrix->shape[0];
    int cols = matrix->shape[1];

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%lf ", matrix->array[(i * matrix->stride[0]) + (j * matrix->stride[1])]);
        }
        printf("\n");
    }
}

double* get_index (Matrix* matrix, int position[2]) {
    double* element = &(matrix->array[position[0] * matrix->stride[0] + position[1] * matrix->stride[1]]);
    return element;
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

Matrix* CreateMatrix (int shape[2]) {
    assert (shape[0] > 0 && shape[1] > 0);

    Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));
    matrix->array = (double*)malloc((size_t)(shape[0] * shape[1]) * sizeof(double));
    matrix->shape[0] = shape[0];
    matrix->shape[1] = shape[1];
    matrix->stride[0] = shape[1];
    matrix->stride[1] = 1;

    return matrix;
}

Matrix* Zeros (int shape[2]) {
    Matrix* zeros = CreateMatrix(shape);
    for (int i = 0; i < shape[0] * shape[1]; i++) {
        zeros->array[i] = 0;
    }
    return zeros;
}

Matrix* Ones (int shape[2]) {
    Matrix* ones = CreateMatrix(shape);
    for (int i = 0; i < shape[0] * shape[1]; i++) {
        ones->array[i] = 1;
    }
    return ones;
}

Matrix* Random (int shape[2]) {
    Matrix* ones = CreateMatrix(shape);
    for (int i = 0; i < shape[0] * shape[1]; i++) {
        int rand_value = rand();
        ones->array[i] = (double)rand_value / (double)RAND_MAX;
    }
    return ones;
}

Matrix* Gaussian (int shape[2], double mean, double std) {
    srand(time(NULL));
    Matrix* normal = CreateMatrix(shape);
    for (int i = 0; i < shape[0] * shape[1]; i++) {
        int rand_value = rand();
        normal->array[i] = normal_random(mean, std);
    }
    return normal;
}

Matrix* Eye (int shape[2]) {
    assert (shape[0] == shape[1]);

    Matrix* eye = CreateMatrix(shape);
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

Matrix* ADD (Matrix* m1, Matrix* m2) {
    assert (m1->shape[0] == m2->shape[0] && m1->shape[1] == m2->shape[1]);

    Matrix* m3 = CreateMatrix(m1->shape);
    for (int row = 0; row < m1->shape[0]; row++) {
        for (int col = 0; col < m1->shape[1]; col++) {
            m3->array[row * m3->stride[0] + col * m3->stride[1]] = m1->array[row * m1->stride[0] + col * m1->stride[1]]+
                                                                   m2->array[row * m2->stride[0] + col * m2->stride[1]];
        }
    }
    return m3;
}

Matrix* SUBTRACT (Matrix* m1, Matrix* m2) {
    assert (m1->shape[0] == m2->shape[0] && m1->shape[1] == m2->shape[1]);

    Matrix* m3 = CreateMatrix(m1->shape);
    for (int row = 0; row < m1->shape[0]; row++) {
        for (int col = 0; col < m1->shape[1]; col++) {
            m3->array[row * m3->stride[0] + col * m3->stride[1]] = m1->array[row * m1->stride[0] + col * m1->stride[1]]-
                                                                   m2->array[row * m2->stride[0] + col * m2->stride[1]];
        }
    }
    return m3;
}

Matrix* MULT (Matrix* m1, Matrix* m2) {
    assert (m1->shape[0] == m2->shape[0] && m1->shape[1] == m2->shape[1]);

    Matrix* m3 = CreateMatrix(m1->shape);
    for (int row = 0; row < m1->shape[0]; row++) {
        for (int col = 0; col < m1->shape[1]; col++) {
            m3->array[row * m3->stride[0] + col * m3->stride[1]] = m1->array[row * m1->stride[0] + col * m1->stride[1]]*
                                                                   m2->array[row * m2->stride[0] + col * m2->stride[1]];
        }
    }
    return m3;
}

Matrix* MATMUL (Matrix* m1, Matrix* m2) {
    assert (m1->shape[1] == m2->shape[0]);

    int new_shape[2] = {m1->shape[0], m2->shape[1]};
    Matrix* m3 = CreateMatrix(new_shape);

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

Matrix* Transpose (Matrix* m) {
    int temp = m->shape[0];
    int temp2 = m->stride[0];
    m->shape[0] = m->shape[1];
    m->stride[0] = m->stride[1];
    m->stride[1] = temp2;
    m->shape[1] = temp;

    return m;
}

Matrix* SIN (Matrix* matrix) {
    int rows = matrix->shape[0];
    int cols = matrix->shape[1];

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix->array[
                (i * matrix->stride[0]) + 
                (j * matrix->stride[1])
            ] = sin(matrix->array[(i * matrix->stride[0]) + (j * matrix->stride[1])]);
        }
    }

    return matrix;
}

Matrix* COS (Matrix* matrix) {
    int rows = matrix->shape[0];
    int cols = matrix->shape[1];

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix->array[
                (i * matrix->stride[0]) + 
                (j * matrix->stride[1])
            ] = cos(matrix->array[(i * matrix->stride[0]) + (j * matrix->stride[1])]);
        }
    }

    return matrix;
}

Matrix* TAN (Matrix* matrix) {
    int rows = matrix->shape[0];
    int cols = matrix->shape[1];

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            assert (cos(matrix->array[
                (i * matrix->stride[0]) + 
                (j * matrix->stride[1])
            ]) != 0);
            matrix->array[
                (i * matrix->stride[0]) + 
                (j * matrix->stride[1])
            ] = tan(matrix->array[(i * matrix->stride[0]) + (j * matrix->stride[1])]);
        }
    }

    return matrix;
}

Matrix* LOG (Matrix* matrix) {
    int rows = matrix->shape[0];
    int cols = matrix->shape[1];

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            assert (matrix->array[
                (i * matrix->stride[0]) + 
                (j * matrix->stride[1])
            ] != 0);
            matrix->array[
                (i * matrix->stride[0]) + 
                (j * matrix->stride[1])
            ] = log(matrix->array[(i * matrix->stride[0]) + (j * matrix->stride[1])]);
        }
    }

    return matrix;
}

double Trace (Matrix* m) {
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

Matrix* ELEMENT_POW (Matrix* matrix, double n) {
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

Matrix* SQRT (Matrix* m) {
    return ELEMENT_POW(m, 0.5);
}

Matrix* POW (Matrix* matrix, int n) {
    Matrix* res = Ones(matrix->shape);
    while (n > 0) {
        res = MATMUL(res, matrix);
        n --;
    }
    return res;
}



int main() {
    // srand(time(NULL));

    // int shape[2] = {5, 5};

    // Matrix* I = Random(shape);

    // Print(I);
    // printf("\n");
    // Print(LOG(I));
}

// gcc numC.c -o exec