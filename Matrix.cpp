#include <cstdlib>
#include <iostream>
#include <valarray>
#include <stdexcept>
#include "Matrix.h"



Matrix::Matrix(int rows, int cols) :
_rows(rows), _cols(cols), _arr(new float*[_rows]) {
    if (_rows <= 0 || _cols <= 0) {
        throw std::invalid_argument
        ("Error: Row and column sizes must be non-negative");
    }
    for (int i = 0; i < _rows; i++) {
        _arr[i] = new float[cols];
        if (_arr[i] == nullptr) {
            for (int k = 0; k < i; ++k) {
                delete[] _arr[k];
            }
            delete[] _arr;
            throw std::bad_alloc();
        }
        for (int j = 0; j < _cols; ++j) {
            _arr[i][j] = 0;
        }
    }
}

Matrix::Matrix() : Matrix(1, 1) {
}

Matrix::Matrix(const Matrix &other) :
_rows(other._rows), _cols(other._cols),
_arr(new float*[_rows]) {
    for (int i = 0; i < _rows; ++i) {
        _arr[i] = new float[_cols];
        if (_arr[i] == nullptr) {
            for (int k = 0; k < i; ++k) {
                delete[] _arr[k];
            }
            delete[] _arr;
            throw std::bad_alloc();
        }
        for (int j = 0; j < _cols; ++j) {
            _arr[i][j] = other._arr[i][j];
        }
    }
}

Matrix::~Matrix() {
    for (int i = 0; i < _rows; ++i) {
        delete[] _arr[i];
    }
    delete[] _arr;
}

int Matrix::get_rows() const {
    return _rows;
}

int Matrix::get_cols() const {
    return _cols;
}

Matrix& Matrix::transpose() {
    float **new_arr = new float*[_cols];
    for (int i = 0; i < _cols; ++i) {
        new_arr[i] = new float[_rows];
        if (new_arr[i] == nullptr) {
            for (int k = 0; k < i; ++k) {
                delete[] new_arr[k];
            }
            delete[] new_arr;
            throw std::bad_alloc();
        }
        for (int j = 0; j < _rows; ++j) {
            new_arr[i][j] = _arr[j][i];
        }
    }
    for (int i = 0; i < _rows; ++i) {
        delete[] _arr[i];
    }
    delete[] _arr;

    _arr = new_arr;
    std::swap(_rows, _cols);

    return *this;
}


Matrix& Matrix::vectorize() {
    float *new_arr = new float[_rows * _cols];
    int index = 0;
    for (int i = 0; i < _rows; ++i) {
        for (int j = 0; j < _cols; ++j) {
            new_arr[index++] = _arr[i][j];
        }
    }

    for (int i = 0; i < _rows; ++i) {
        delete[] _arr[i];
    }
    delete[] _arr;

    _arr = new float*[index];
    for (int i = 0; i < index; ++i) {
        _arr[i] = new float[1];
        if (_arr[i] == nullptr) {
            for (int k = 0; k < i; ++k) {
                delete[] _arr[k];
            }
            delete[] _arr;
            delete[] new_arr;
            throw std::bad_alloc();
        }
        _arr[i][0] = new_arr[i];
    }

    _rows = index;
    _cols = 1;

    delete[] new_arr;

    return *this;
}


void Matrix::plain_print() const{
    for (int i = 0; i < _rows; i++)
    {
        for (int j = 0; j < _cols; j++)
        {
            std::cout << _arr[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

Matrix Matrix::dot(Matrix &other)const {
    if((_rows!=other._rows)||(_cols!=other._cols))
    {
        throw std::runtime_error
        ("Error: The matrices must have the same dimension.");
    }
    Matrix *new_matrix = new Matrix(_rows, _cols);
    for (int i = 0; i < _rows; ++i) {
        for (int j = 0; j < _cols; ++j) {
            new_matrix->_arr[i][j] = _arr[i][j] * other._arr[i][j];
        }
    }
    return *new_matrix;
}

double Matrix::norm() const {
    double sum = 0;
    for (int i = 0; i < _rows; ++i) {
        for (int j = 0; j < _cols; ++j) {
            sum += std::pow(_arr[i][j], 2);
        }
    }
    return std::sqrt(sum);
}

void Matrix::swap_rows(Matrix& mat, int row1, int row2,
                       int lead, int num_cols) const {
    if (row1 != row2) {
        for (int col = lead; col < num_cols; ++col) {
            std::swap(mat._arr[row1][col],
                      mat._arr[row2][col]);
        }
    }
}

void Matrix::norm_row(Matrix& mat, int row, int lead,
                      int num_cols) const {
    float lead_val = mat._arr[row][lead];
    for (int col = lead; col < num_cols; col++) {
        mat._arr[row][col] /= lead_val;
    }
}

Matrix Matrix::rref() const {
    Matrix result(*this);
    int lead_num = 0;
    int num_rows = result.get_rows();
    int num_cols = result.get_cols();

    for (int r = 0; r < num_rows; ++r) {
        if (lead_num >= num_cols) {
            break;
        }
        int i = r;

        while (i < num_rows && std::abs
        (result._arr[i][lead_num]) < ACCURACY1) {
            i++;
        }
        if (i == num_rows) {
            lead_num++;
            r--;
            continue;
        }

        swap_rows(result, r, i, lead_num, num_cols);
        norm_row(result, r, lead_num, num_cols);

        for (int j = 0; j < num_rows; ++j) {
            if (j != r) {
                float val = result._arr[j][lead_num];
                for (int k = lead_num; k < num_cols; ++k) {
                    result._arr[j][k] -= val * result._arr[r][k];
                }
            }
        }
        lead_num++;
    }

    return result;
}

int Matrix::argmax() const{
    float max = _arr[0][0];
    int max_index = 0;
    for (int i = 0; i < _rows; ++i) {
        for (int j = 0; j < _cols; ++j) {
            if (_arr[i][j] > max) {
                max = _arr[i][j];
                max_index = (i * _cols) + j;
            }
        }
    }
    return max_index;
}

double Matrix::sum()const {
    float sum = 0;
    for (int i = 0; i < _rows; ++i) {
        for (int j = 0; j < _cols; ++j) {
            sum += _arr[i][j];
        }
    }
    return sum;
}

Matrix &Matrix::operator+=(const Matrix &other) {
    for (int i = 0; i < _rows; ++i) {
        for (int j = 0; j < _cols; ++j) {
            _arr[i][j] += other._arr[i][j];
        }
    }
    return *this;
}

Matrix Matrix::operator+(const Matrix &other) const {
    if((_rows!=other._rows)||_cols!=other._cols)
    {
        throw std::runtime_error
        ("Error: The row and column sizes of the two matrices must be equal");
    }
    Matrix result(_rows,_cols);
    for (int i = 0; i < _rows; ++i) {
        for (int j = 0; j < _cols; ++j) {
            result._arr[i][j] = _arr[i][j] + other._arr[i][j];
        }
    }
    return result;
}

Matrix &Matrix::operator=(const Matrix &other) {
    if (this == &other) {
        return *this;
    }
    for (int i = 0; i < _rows; ++i) {
        delete[] _arr[i];
    }
    delete[] _arr;

    _rows = other._rows;
    _cols = other._cols;
    _arr = new float*[_rows];

    for (int i = 0; i < _rows; ++i) {
        _arr[i] = new float[_cols];
        if (_arr[i] == nullptr) {
            for (int k = 0; k < i; ++k) {
                delete[] _arr[k];
            }
            delete[] _arr;
            throw std::bad_alloc();
        }
        for (int j = 0; j < _cols; ++j) {
            _arr[i][j] = other._arr[i][j];
        }
    }
    return *this;
}

Matrix Matrix::operator*(const Matrix &other) const
{
    if(_cols!=other._rows)
    {
        throw std::runtime_error
        ("Error: Matrices cannot be multiplied");
    }
    Matrix result(_rows,other._cols);
    for (int i = 0; i < _rows; ++i) {
        for (int j = 0; j < other._cols; ++j) {
            float total = 0;
            for (int k = 0; k < _cols; ++k) {
                total += _arr[i][k] * other._arr[k][j];
            }
            result._arr[i][j] = total;
        }
    }
    return result;
}

Matrix Matrix::operator*(const float c) const
{
    Matrix result(_rows,_cols);
    for (int i = 0; i < _rows; ++i) {
        for (int j = 0; j <_cols; ++j) {
            result._arr[i][j] = _arr[i][j]*c;
        }
    }
    return result;
}
Matrix  operator*(const float c,const Matrix &other)
{
    Matrix result(other._rows,other._cols);
    for (int i = 0; i < other._rows; ++i) {
        for (int j = 0; j <other._cols; ++j) {
            result._arr[i][j] = other._arr[i][j]*c;
        }
    }
    return result;
}
float Matrix::operator()(int i, int j) const
{
    if(i<0||i>_rows-1||j<0||j>_cols-1)
    {
        throw std::out_of_range(RANGE_ERROR);
    }
    return _arr[i][j];
}
float &Matrix::operator()(int i, int j)
{
    if(i<0||i>_rows-1||j<0||j>_cols-1)
    {
        throw std::out_of_range(RANGE_ERROR);
    }
    return _arr[i][j];
}

float &Matrix::operator[](int m)
{
    if(m<0||m>=_rows*_cols)
    {
        throw std::out_of_range(RANGE_ERROR);
    }
    return _arr[m/_cols][m%_cols];
}
float Matrix::operator[](int m)const
{
    if(m<0||m>=_rows*_cols)
    {
        throw std::out_of_range(RANGE_ERROR);
    }
    return _arr[m/_cols][m%_cols];
}


std::ostream& operator<<(std::ostream& os,
                         const Matrix& matrix) {
    for (int i = 0; i < matrix._rows; i++) {
        for (int j = 0; j < matrix._cols; j++) {
            if (matrix(i, j) > ACCURACY2) {
                os << "**";
            } else {
                os << "  ";
            }
        }
        os << std::endl;
    }
    return os;
}
std::istream& operator>>(std::istream& in, Matrix& matrix) {
    int total_elements = matrix._cols * matrix._rows;
    float* temp_buffer = new float[total_elements];

    in.read(reinterpret_cast<char*>(temp_buffer),
            total_elements * sizeof(float));

    if (!in) {
        delete[] temp_buffer;
        throw std::runtime_error
        ("Error: Failed to read matrix data from stream.");
    }

    if (static_cast<std::streamsize>
    (total_elements * sizeof(float)) != in.gcount()) {
        delete[] temp_buffer;
        throw std::runtime_error
        ("Error: Stream does not contain enough data to fill the matrix.");
    }

    int buffer_index = 0;
    for (int row = 0; row < matrix._rows; ++row) {
        for (int col = 0; col < matrix._cols; ++col) {
            matrix(row, col) = temp_buffer[buffer_index++];
        }
    }

    delete[] temp_buffer;
    return in;
}

