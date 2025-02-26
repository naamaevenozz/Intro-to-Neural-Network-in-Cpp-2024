// Matrix.h
#ifndef MATRIX_H
#define MATRIX_H
#include <cstdlib>
#include <iostream>
#include <valarray>
#include <stdexcept>
#include <cmath>

#define ACCURACY1 1e-9
#define ACCURACY2 0.1
#define RANGE_ERROR "Index out of range"
struct matrix_dims {
    int rows, cols;
};
class Matrix {
private:
    int _rows;
    int _cols;
    float **_arr;

public:
    Matrix(int rows, int cols);

    // Default constructor
    Matrix();

    // Copy constructor
    Matrix(const Matrix &other);

    // Destructor
    ~Matrix();

    // Get number of rows
    int get_rows() const;

    int get_cols() const;

    Matrix& transpose();

    Matrix& vectorize();

    void plain_print() const;

    Matrix dot(Matrix &other) const;

    double norm() const;

    Matrix rref() const ;

    int argmax()const;

    double sum()const;

    Matrix &operator+=(const Matrix &other);

    Matrix operator+(const Matrix &other) const;

    Matrix &operator=(const Matrix &other);

    Matrix operator*(const Matrix &other) const;

    Matrix operator*(float c) const;

    friend Matrix operator*(float c, const Matrix &other);

    float operator()(int i, int j) const;

    float &operator()(int i, int j);

    float operator[](int m) const;

    float &operator[](int m);

    friend std::ostream& operator<<(std::ostream &os,
            const Matrix &matrix);

    friend std::istream &operator>>(std::istream &in,
            Matrix &matrix);

    void norm_row(Matrix &mat, int row, int lead, int num_cols) const;

    void swap_rows(Matrix &mat, int row1,
                   int row2, int lead, int num_cols) const;
    };

#endif //MATRIX_H