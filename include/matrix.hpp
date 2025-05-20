/**
 * @file matrix.hpp
 * @brief Definition of the templated matrix<T> class.
 *
 * This header declares the matrix<T> class, a lightweight and flexible
 * C++ matrix implementation supporting basic linear algebra operations,
 * element access, and utility functions (zeros, ones, eye, etc.).
 *
 * Include this file to use the matrix library.
 *
 * @author edybostina
 * @date 2025-05-20 at 17:20
 * @version 1.1
 * @note This is a work in progress and may not be fully functional.
 *       The library is intended for educational purposes and may
 *       not be suitable for production use.
 * 
 * @license MIT License
 */


#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <iomanip>
#include <algorithm>
#include <random>
#include <limits>
#include <string>
#include <fstream>

using namespace std;

template <typename T>
class matrix {
    private:
        vector<vector<T>> _data;
        int _rows, _cols;
    public:
        matrix(int rows, int cols) : _rows(rows), _cols(cols) {
            _data.resize(rows, vector<T>(cols));
        }

        matrix(const vector<vector<T>>& data) : _data(data), _rows(data.size()), _cols(data[0].size()) {}

        int rows() const { return _rows; }
        int cols() const { return _cols; }
        vector<vector<T>> data() const { return _data; }

        // Access operator

        vector<T>& operator()(int index);
        const vector<T>& operator()(int index) const;

        T& operator()(int row, int col);
        const T& operator()(int row, int col) const;

        // Basic initialization

        static matrix<T> zeros(int rows, int cols);
        static matrix<T> ones(int rows, int cols);
        static matrix<T> eye(int rows, int cols);
        static matrix<T> random(int rows, int cols, T min, T max);

        // File I/O

        template <typename U>
        friend ostream& operator<<(ostream& os, const matrix<U>& m);

        template <typename U>
        friend istream& operator>>(istream& is, matrix<U>& m);

        // Casting

        template <typename U>
        explicit operator matrix<U>() const {
            matrix<U> result(_rows, _cols);
            for (int i = 0; i < _rows; ++i) {
                for (int j = 0; j < _cols; ++j) {
                    result(i, j) = static_cast<U>((*this)(i, j));
                }
            }
            return result;
        }

        // Arithmetic operators
        // Matrix x Matrix

        matrix<T> operator+(const matrix<T>& other) const;
        matrix<T> operator-(const matrix<T>& other) const;
        matrix<T> operator*(const matrix<T>& other) const;
        matrix<T> operator+=(const matrix<T>& other);
        matrix<T> operator-=(const matrix<T>& other);
        matrix<T> operator*=(const matrix<T>& other);
        bool operator==(const matrix<T>& other);
        bool operator!=(const matrix<T>& other);

        // Matrix x Scalar

        matrix<T> operator+(const T& scalar) const;
        matrix<T> operator-(const T& scalar) const;
        matrix<T> operator*(const T& scalar) const;
        matrix<T> operator/(const T& scalar) const;
        matrix<T> operator+=(const T& scalar);
        matrix<T> operator-=(const T& scalar);
        matrix<T> operator*=(const T& scalar);
        matrix<T> operator/=(const T& scalar);

        // Matrix functions
      
        T determinant() const;
        T trace() const;
        matrix<T> transpose() const;
        matrix<T> cofactor() const;
        matrix<T> minor(int row, int col) const;
        matrix<T> adjoint() const;
        matrix<double> inverse() const;

        void swapRows(int row1, int row2);
        void swapCols(int col1, int col2);

        void resize(int rows, int cols);
        
    
};


// ==================================================
// ==================== FIle I/O ====================
// ==================================================

// Output stream operator
template <typename T>
ostream& operator<<(ostream& os, const matrix<T>& m) {
    for (int i = 0; i < m.rows(); ++i) {
        for (int j = 0; j < m.cols(); ++j) {
            os << m(i, j) << " ";
        }
        os << endl;
    }
    return os;
}
// Input stream operator
template <typename T>
istream& operator>>(std::istream& is, matrix<T>& m) {
    for (int i = 0; i < m.rows(); ++i) {
        for (int j = 0; j < m.cols(); ++j) {
            is >> m(i, j);
        }
    }
    return is;
}


// =========================================================
// ==================== Acces operators ====================
// =========================================================

// matrix(index) returns the i-th row of the matrix
template <typename T>
vector<T>& matrix<T>::operator()(int index) {
    if (index < 0 || index >= _rows) {
        throw out_of_range("Index out of range");
    }
    return _data[index];
}
// const matrix(index)
template <typename T>
const vector<T>& matrix<T>::operator()(int index) const {
    if (index < 0 || index >= _rows) {
        throw out_of_range("Index out of range");
    }
    return _data[index];
}

// matrix(row, col) returns the element at (row, col)
template <typename T>
T& matrix<T>::operator()(int row, int col) {
    if (row < 0 || row >= _rows || col < 0 || col >= _cols) {
        throw out_of_range("Index out of range");
    }
    return _data[row][col];
}
// const matrix(row, col)
template <typename T>
const T& matrix<T>::operator()(int row, int col) const {
    if (row < 0 || row >= _rows || col < 0 || col >= _cols) {
        throw out_of_range("Index out of range");
    }
    return _data[row][col];
}


// ==============================================================
// ==================== Basic initialization ====================
// ==============================================================

// Matrix full of zeros
template <typename T>
matrix<T> matrix<T>::zeros(int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        throw invalid_argument("Matrix dimensions must be positive");
    }
    return matrix<T>(vector<vector<T>>(rows, vector<T>(cols, 0)));
}
// Matrix full of ones
template <typename T>
matrix<T> matrix<T>::ones(int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        throw invalid_argument("Matrix dimensions must be positive");
    }
    return matrix<T>(vector<vector<T>>(rows, vector<T>(cols, 1)));
}
// Identity matrix
template <typename T>
matrix<T> matrix<T>::eye(int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        throw invalid_argument("Matrix dimensions must be positive");
    }

    matrix<T> result(rows, cols);
    for (int i = 0; i < min(rows, cols); ++i) {
        result(i, i) = 1;
    }
    return result;
}

// Random initialization
// This function uses the <random> lib to generate random numbers
// between 0 and 1
template <typename T>
matrix<T> matrix<T>::random(int rows, int cols, T min, T max) {
    if (rows <= 0 || cols <= 0) {
        throw invalid_argument("Matrix dimensions must be positive");
    }
    if (min >= max) {
        throw invalid_argument("Invalid range for random numbers");
    }
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<T> dis(min, max);
    matrix<T> result(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(i, j) = dis(gen);
        }
    }
    return result;
}

// ==============================================================
// ==================== Arithmetic operators ====================
// ==============================================================

// Matrix and Matrix

// Matrix addition
template <typename T>
matrix<T> matrix<T>::operator+(const matrix<T>& other) const {
    if (_rows != other._rows || _cols != other._cols) {
        throw invalid_argument("Matrix dimensions do not match for addition");
    }
    matrix<T> result(_rows, _cols);
    for (int i = 0; i < _rows; ++i) {
        for (int j = 0; j < _cols; ++j) {
            result(i, j) = (*this)(i, j) + other(i, j);
        }
    }
    return result;
}
// Matrix subtraction
template <typename T>
matrix<T> matrix<T>::operator-(const matrix<T>& other) const {
    if (_rows != other._rows || _cols != other._cols) {
        throw invalid_argument("Matrix dimensions do not match for subtraction");
    }
    matrix<T> result(_rows, _cols);
    for (int i = 0; i < _rows; ++i) {
        for (int j = 0; j < _cols; ++j) {
            result(i, j) = (*this)(i, j) - other(i, j);
        }
    }
    return result;
}
// Matrix multiplication
template <typename T>
matrix<T> matrix<T>::operator*(const matrix<T>& other) const {
    if (_cols != other._rows) {
        throw invalid_argument("Matrix dimensions do not match for multiplication");
    }
    matrix<T> result(_rows, other._cols);
    for (int i = 0; i < _rows; ++i) {
        for (int j = 0; j < other._cols; ++j) {
            result(i, j) = 0;
            for (int k = 0; k < _cols; ++k) {
                result(i, j) += (*this)(i, k) * other(k, j);
            }
        }
    }
    return result;
}
// Matrix addition assignment
template <typename T>
matrix<T> matrix<T>::operator+=(const matrix<T>& other) {
    if (_rows != other._rows || _cols != other._cols) {
        throw invalid_argument("Matrix dimensions do not match for addition assignment");
    }
    for (int i = 0; i < _rows; ++i) {
        for (int j = 0; j < _cols; ++j) {
            (*this)(i, j) += other(i, j);
        }
    }
    return *this;
}
// Matrix subtraction assignment
template <typename T>
matrix<T> matrix<T>::operator-=(const matrix<T>& other) {
    if (_rows != other._rows || _cols != other._cols) {
        throw invalid_argument("Matrix dimensions do not match for subtraction assignment");
    }
    for (int i = 0; i < _rows; ++i) {
        for (int j = 0; j < _cols; ++j) {
            (*this)(i, j) -= other(i, j);
        }
    }
    return *this;
}
// Matrix multiplication assignment
template <typename T>
matrix<T> matrix<T>::operator*=(const matrix<T>& other) {
    if (_cols != other._rows) {
        throw invalid_argument("Matrix dimensions do not match for multiplication assignment");
    }
    matrix<T> result = (*this) * other;
    *this = result;
    return *this;
}
// Matrix equality operator
template <typename T>
bool matrix<T>::operator==(const matrix<T>& other) {
    if (_rows != other._rows || _cols != other._cols) {
        return false;
    }
    for (int i = 0; i < _rows; ++i) {
        for (int j = 0; j < _cols; ++j) {
            if ((*this)(i, j) != other(i, j)) {
                return false;
            }
        }
    }
    return true;
}
// Matrix inequality operator
template <typename T>
bool matrix<T>::operator!=(const matrix<T>& other) {
    return !(*this == other);
}

// Matrix and Scalar

// Matrix addition with scalar
template <typename T>
matrix<T> matrix<T>::operator+(const T& scalar) const {
    matrix<T> result(_rows, _cols);
    for (int i = 0; i < _rows; ++i) {
        for (int j = 0; j < _cols; ++j) {
            result(i, j) = (*this)(i, j) + scalar;
        }
    }
    return result;
}
// Matrix subtraction with scalar
template <typename T>
matrix<T> matrix<T>::operator-(const T& scalar) const {
    matrix<T> result(_rows, _cols);
    for (int i = 0; i < _rows; ++i) {
        for (int j = 0; j < _cols; ++j) {
            result(i, j) = (*this)(i, j) - scalar;
        }
    }
    return result;
}
// Matrix multiplication with scalar
template <typename T>
matrix<T> matrix<T>::operator*(const T& scalar) const {
    matrix<T> result(_rows, _cols);
    for (int i = 0; i < _rows; ++i) {
        for (int j = 0; j < _cols; ++j) {
            result(i, j) = (*this)(i, j) * scalar;
        }
    }
    return result;
}
// Matrix division by scalar
template <typename T>
matrix<T> matrix<T>::operator/(const T& scalar) const {
    if (scalar == 0) {
        throw invalid_argument("Division by zero");
    }
    matrix<T> result(_rows, _cols);
    for (int i = 0; i < _rows; ++i) {
        for (int j = 0; j < _cols; ++j) {
            result(i, j) = (*this)(i, j) / scalar;
        }
    }
    return result;
}

// Matrix addition assignment with scalar
template <typename T>
matrix<T> matrix<T>::operator+=(const T& scalar) {
    for (int i = 0; i < _rows; ++i) {
        for (int j = 0; j < _cols; ++j) {
            (*this)(i, j) += scalar;
        }
    }
    return *this;
}
// Matrix subtraction assignment with scalar
template <typename T>
matrix<T> matrix<T>::operator-=(const T& scalar) {
    for (int i = 0; i < _rows; ++i) {
        for (int j = 0; j < _cols; ++j) {
            (*this)(i, j) -= scalar;
        }
    }
    return *this;
}
// Matrix multiplication assignment with scalar
template <typename T>
matrix<T> matrix<T>::operator*=(const T& scalar) {
    for (int i = 0; i < _rows; ++i) {
        for (int j = 0; j < _cols; ++j) {
            (*this)(i, j) *= scalar;
        }
    }
    return *this;
}
// Matrix division assignment by scalar
template <typename T>
matrix<T> matrix<T>::operator/=(const T& scalar) {
    if (scalar == 0) {
        throw invalid_argument("Division by zero");
    }
    for (int i = 0; i < _rows; ++i) {
        for (int j = 0; j < _cols; ++j) {
            (*this)(i, j) /= scalar;
        }
    }
    return *this;
}

// ==========================================================
// ==================== Matrix functions ====================
// ==========================================================

// Determinant
template <typename T>
T matrix<T>::determinant() const {
    if (_rows != _cols) {
        throw invalid_argument("Matrix must be square to compute determinant");
    }
    double det = 1.0;
    matrix<double> temp = (matrix<double>)*this;
    for (int i = 0; i < _rows; ++i) {
        int pivot = i;
        for (int j = i + 1; j < _rows; ++j) {
            if (abs(temp(j, i)) > abs(temp(pivot, i))) {
                pivot = j;
            }
        }
        if (pivot != i) {
            swap(temp(i), temp(pivot));
            det *= -1;
        }
        if (abs(temp(i, i)) < numeric_limits<double>::epsilon()) {
            return 0;
        }
        det *= temp(i, i);
        for (int j = i + 1; j < _rows; ++j) {
            double factor = temp(j, i) / temp(i, i);
            for (int k = i + 1; k < _cols; ++k) {
                temp(j, k) -= factor * temp(i, k);
            }
        }
    }
    return det;
}

// Trace
template <typename T>
T matrix<T>::trace() const {
    if (_rows != _cols) {
        throw invalid_argument("Matrix must be square to compute trace");
    }
    T sum = 0;
    for (int i = 0; i < _rows; ++i) {
        sum += (*this)(i, i);
    }
    return sum;
}

// Transpose
template <typename T>
matrix<T> matrix<T>::transpose() const {
    matrix<T> result(_cols, _rows);
    for (int i = 0; i < _rows; ++i) {
        for (int j = 0; j < _cols; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

// Cofactor
template <typename T>
matrix<T> matrix<T>::cofactor() const {
    matrix<T> result(_rows, _cols);
    for (int i = 0; i < _rows; ++i) {
        for (int j = 0; j < _cols; ++j) {
           matrix<T> minor = this->minor(i, j);
            result(i, j) = ((i + j) % 2 == 0 ? 1 : -1) * minor.determinant();
        }
    }
    return result;
}

// Minor
template <typename T>
matrix<T> matrix<T>::minor(int row, int col) const {
    if (row < 0 || row >= _rows || col < 0 || col >= _cols) {
        throw out_of_range("Index out of range");
    }
    matrix<T> result(_rows - 1, _cols - 1);
    for (int i = 0; i < _rows; ++i) {
        if (i == row) continue;
        for (int j = 0; j < _cols; ++j) {
            if (j == col) continue;
            result(i < row ? i : i - 1, j < col ? j : j - 1) = (*this)(i, j);
        }
    }
    return result;
}

// Adjoint
template <typename T>
matrix<T> matrix<T>::adjoint() const {
    if (_rows != _cols) {
        throw invalid_argument("Matrix must be square to compute adjoint");
    }
    matrix<T> result(_rows, _cols);
    result = this->cofactor().transpose();
    return result;
}

// Inverse
template <typename T>
matrix<double> matrix<T>::inverse() const {
    if (_rows != _cols) {
        throw invalid_argument("Matrix must be square to compute inverse");
    }
    double temp;
    matrix<double> augmented(_rows, 2 * _cols);
    matrix<double> result(_rows, _cols);
    for (int i = 0; i < _rows; ++i) {
        for (int j = 0; j < _cols; ++j) {
            augmented(i, j) = (*this)(i, j);
        }
        for (int j = _cols; j < 2 * _cols; ++j) {
            if (i == j - _cols) {
                augmented(i, j) = 1;
            } else {
                augmented(i, j) = 0;
            }
        }
    }
    for (int i = _rows - 1; i > 0; i--) {
        if (augmented(i-1, 0) < augmented(i, 0)) {
            swap(augmented(i), augmented(i-1));
        }
    }

    for (int i = 0; i < _rows; ++i) {
        for (int j = 0; j < _cols; ++j) {
            if (j != i) {
                temp = augmented(j, i) / augmented(i, i);
                for (int k = 0; k < 2 * _cols; ++k) {
                    augmented(j, k) -= augmented(i, k) * temp;
                }
            }
        }
    }
    for (int i = 0; i < _rows; ++i) {
        temp = augmented(i, i);
        for (int j = 0; j < 2 * _cols; ++j) {
            augmented(i, j) /= temp;
        }
    }
    for (int i = 0; i < _rows; ++i) {
        for (int j = 0; j < _cols; ++j) {
            result(i, j) = augmented(i, j + _cols);
        }
    }
    return result;
}
// Swap rows
template <typename T>
void matrix<T>::swapRows(int row1, int row2) {
    if (row1 < 0 || row1 >= _rows || row2 < 0 || row2 >= _rows) {
        throw out_of_range("Row index out of range");
    }
    swap(_data[row1], _data[row2]);
}
// Swap columns
template <typename T>
void matrix<T>::swapCols(int col1, int col2) {
    if (col1 < 0 || col1 >= _cols || col2 < 0 || col2 >= _cols) {
        throw out_of_range("Column index out of range");
    }
    for (int i = 0; i < _rows; ++i) {
        swap(_data[i][col1], _data[i][col2]);
    }
}
// Resize matrix
template <typename T>
void matrix<T>::resize(int rows, int cols) {
    if (rows < 0 || cols < 0) {
        throw invalid_argument("Matrix dimensions must be non-negative");
    }
    _data.resize(rows);
    for (int i = 0; i < rows; ++i) {
        _data[i].resize(cols);
    }
    _rows = rows;
    _cols = cols;
}



