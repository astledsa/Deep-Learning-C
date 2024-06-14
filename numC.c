// Matrix Library for C
// Important functionlities implemented

// Matrix Creation
// -- Zeros
// -- Ones
// -- Normal distribution
// -- Identity
// -- random

// Matrix Operations
// -- Matrix Multiplication
// -- Transpose
// -- Inverse
// -- Trace
// -- Determinant

// Element Wise Operations
// -- Addition
// -- Subtraction
// -- Division
// -- Multiplication
// -- Log
// -- Sin
// -- Cos
// -- Tan
// -- Power
// -- Square Root

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
    matrix->array = (double*)malloc(sizeof(shape[0] * shape[1]));
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
    srand(time(NULL));
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

// gcc numC.c -o exec