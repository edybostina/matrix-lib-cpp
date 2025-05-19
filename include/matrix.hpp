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
 * @date 2025-05-20 at 1 AM lol
 * @version 1.0
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
        // TODO: implement random initialization
        // matrix<T> rand(int rows, int cols);

        // File I/O
        template <typename U>
        friend ostream& operator<<(ostream& os, const matrix<U>& m);

        template <typename U>
        friend istream& operator>>(istream& is, matrix<U>& m);



        // Arithmetic operators
        // Matrix x Matrix
        matrix<T> operator+(const matrix<T>& other) const;
        matrix<T> operator-(const matrix<T>& other) const;
        matrix<T> operator*(const matrix<T>& other) const;
        matrix<T> operator+=(const matrix<T>& other);
        matrix<T> operator-=(const matrix<T>& other);
        matrix<T> operator*=(const matrix<T>& other);

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
        // TODO: fix the inverse function
        // for some stupid reason it doesn't work
        // and I am too tired to fix it rn.
        T determinant() const;
        matrix<T> transpose() const;
        matrix<T> cofactor(int row, int col) const;
        matrix<T> minor(int row, int col) const;
        matrix<T> adjoint() const;
        matrix<T> inverse() const;
    
};

// ==================== FIle I/O ====================
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


// ==================== Acces operators ====================
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


// ==================== Basic initialization ====================
// Matrix full of zeros
template <typename T>
matrix<T> matrix<T>::zeros(int rows, int cols) {
    return matrix<T>(vector<vector<T>>(rows, vector<T>(cols, 0)));
}
// Matrix full of ones
template <typename T>
matrix<T> matrix<T>::ones(int rows, int cols) {
    return matrix<T>(vector<vector<T>>(rows, vector<T>(cols, 1)));
}
// Identity matrix
template <typename T>
matrix<T> matrix<T>::eye(int rows, int cols) {
    matrix<T> result(rows, cols);
    for (int i = 0; i < min(rows, cols); ++i) {
        result(i, i) = 1;
    }
    return result;
}

// TODO: implement random initialization

// Random initialization
// This function uses the <random> lib to generate random numbers
// between 0 and 1
// template <typename T>
// matrix<T> matrix<T>::rand(int rows, int cols) {
//     srand(time(NULL));
//     matrix<T> result(rows, cols);
//     for (int i = 0; i < rows; ++i) {
//         for (int j = 0; j < cols; ++j) {
//             result(i, j) = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
//         }
//     }
//     return result;
// }


// ==================== Arithmetic operators ====================
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

// ==================== Matrix functions ====================
// Determinant
template <typename T>
T matrix<T>::determinant() const {
    if (_rows != _cols) {
        throw invalid_argument("Matrix must be square to compute determinant");
    }
    matrix<T> temp = *this;
    double det = 1.0;
    int swap_count = 0;
    for (int i = 0; i < _rows; ++i) {
        int max_row = i;
        for (int k = i + 1; k < _rows; ++k) {
            if (abs(temp(k, i)) > abs(temp(max_row, i))) {
                max_row = k;
            }
        }
        if (i != max_row) {
            swap_count++;
            for (int j = 0; j < _cols; ++j) {
                swap(temp(i, j), temp(max_row, j));
            }
        }

        // Tolerance check for zero pivot
        if (abs(temp(i, i)) < 1e-12) {
            return 0;
        }

        for (int k = i + 1; k < _rows; ++k) {
            double factor = temp(k, i) / temp(i, i);
            for (int j = i; j < _cols; ++j) {
                temp(k, j) -= factor * temp(i, j);
            }
        }
    }

    for (int i = 0; i < _rows; ++i) {
        det *= temp(i, i);
    }
    return (swap_count % 2 == 0) ? det : -det;
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
matrix<T> matrix<T>::cofactor(int row, int col) const {
    matrix<T> result(_rows - 1, _cols - 1);
    int i = 0, j = 0;
    for (int r = 0; r < _rows; ++r) {
        for (int c = 0; c < _cols; ++c) {
            if (r != row && c != col) {
                result(i, j) = (*this)(r, c);
                j++;
                if (j == _cols - 1) {
                    j = 0;
                    i++;
                }
            }
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
    if (_rows == 1) {
        result(0, 0) = 1;
        return result;
    }
    
    int sign = 1;
    for (int i = 0; i < _rows; ++i) {
        for (int j = 0; j < _cols; ++j) {
            result(j, i) = sign * cofactor(i, j).determinant();
            sign = -sign;
        }
    }
    return result;
}

// Inverse
template <typename T>
matrix<T> matrix<T>::inverse() const {
    if (_rows != _cols) {
        throw invalid_argument("Matrix is not square");
    }
    double det = determinant();
    if (abs(det) < 1e-12) {
        throw invalid_argument("Matrix is singular and cannot be inverted");
    }
    matrix<T> adj = adjoint();
    matrix<T> inv(_rows, _cols);
    for (int i = 0; i < _rows; ++i) {
        for (int j = 0; j < _cols; ++j) {
            inv(i, j) = adj(i, j) / det;
        }
    }
    return inv;
}



