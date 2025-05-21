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
 * @date 2025-05-21 at 2 AM
 * @version 0.1.2
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

template <typename T>
class matrix {
    private:
        int _rows = 0;
        int _cols = 0;
        std::vector<std::vector<T>> _data;
    public:
        constexpr matrix() noexcept = default;

        explicit matrix(int rows, int cols)
            : _rows(rows), _cols(cols), _data(rows, std::vector<T>(cols)) {}

        matrix(const std::vector<std::vector<T>>& init)
            : _rows(init.size()), _cols(init.empty() ? 0 : init[0].size()), _data(init) {}
        matrix(std::initializer_list<std::initializer_list<T>> init) {
            _rows = init.size();
            _cols = init.begin()->size();
            _data.reserve(_rows);
            for (const auto& row : init) {
                _data.emplace_back(row);
            }
        }

         // Rule of Five
        matrix(const matrix&) = default;
        matrix(matrix&&) noexcept = default;
        matrix& operator=(const matrix&) = default;
        matrix& operator=(matrix&&) noexcept = default;
        ~matrix() = default;

        [[nodiscard]] constexpr int rows() const noexcept { return _rows; }
        [[nodiscard]] constexpr int cols() const noexcept { return _cols; }
        [[nodiscard]] constexpr int size() const noexcept { return _rows * _cols; }


        // Access operator

        std::vector<T>& operator()(int index);
        const std::vector<T>& operator()(int index) const;

        T& operator()(int row, int col);
        const T& operator()(int row, int col) const;

        // Casting

        template <typename U>
        [[nodiscard]] explicit operator matrix<U>() const {
            matrix<U> result(_rows, _cols);
            for (int i = 0; i < _rows; ++i) {
                for (int j = 0; j < _cols; ++j) {
                    result(i, j) = static_cast<U>((*this)(i, j));
                }
            }
            return result;
        }

        // Factory methods

        [[nodiscard]] static matrix<T> zeros(int rows, int cols);
        [[nodiscard]] static matrix<T> ones(int rows, int cols);
        [[nodiscard]] static matrix<T> eye(int rows, int cols);
        [[nodiscard]] static matrix<T> random(int rows, int cols, T min, T max);

        // File I/O

        template <typename U>
        friend std::ostream& operator<<(std::ostream& os, const matrix<U>& m);

        template <typename U>
        friend std::istream& operator>>(std::istream& is, matrix<U>& m);


        // Arithmetic operators
        // Matrix x Matrix

        [[nodiscard]] matrix<T> operator+(const matrix<T>& other) const;
        [[nodiscard]] matrix<T> operator-(const matrix<T>& other) const;
        [[nodiscard]] matrix<T> operator*(const matrix<T>& other) const;

        matrix<T> operator+=(const matrix<T>& other);
        matrix<T> operator-=(const matrix<T>& other);
        matrix<T> operator*=(const matrix<T>& other);

        [[nodiscard]] bool operator==(const matrix<T>& other) const noexcept;
        [[nodiscard]] bool operator!=(const matrix<T>& other) const noexcept;

        // Matrix x Scalar

        [[nodiscard]] matrix<T> operator+(const T& scalar) const;
        [[nodiscard]] matrix<T> operator-(const T& scalar) const;
        [[nodiscard]] matrix<T> operator*(const T& scalar) const;
        [[nodiscard]] matrix<T> operator/(const T& scalar) const;

        matrix<T> operator+=(const T& scalar);
        matrix<T> operator-=(const T& scalar);
        matrix<T> operator*=(const T& scalar);
        matrix<T> operator/=(const T& scalar);

        // Matrix functions
      
        [[nodiscard]] T determinant() const;
        [[nodiscard]] T trace() const;
        [[nodiscard]] matrix<T> transpose() const;
        [[nodiscard]] matrix<T> cofactor() const;
        [[nodiscard]] matrix<T> minor(int row, int col) const;
        [[nodiscard]] matrix<T> adjoint() const;
        [[nodiscard]] matrix<double> inverse() const;

        void swapRows(int row1, int row2);
        void swapCols(int col1, int col2);
        void resize(int rows, int cols);
};


// ==================================================
// ==================== FIle I/O ====================
// ==================================================

// Output stream operator
template <typename T>
std::ostream& operator<<(std::ostream& os, const matrix<T>& m) {
    for (int i = 0; i < m.rows(); ++i) {
        for (int j = 0; j < m.cols(); ++j) {
            os << m(i, j) << " ";
        }
        os << "\n";
    }
    return os;
}
// Input stream operator
template <typename T>
std::istream& operator>>(std::istream& is, matrix<T>& m) {
    int rows, cols;
    is >> rows >> cols;
    m = matrix<T>(rows, cols);
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
std::vector<T>& matrix<T>::operator()(int index) {
    if (index < 0 || index >= _rows) {
        throw std::out_of_range("Index out of range");
    }
    return _data[index];
}
// const matrix(index)
template <typename T>
const std::vector<T>& matrix<T>::operator()(int index) const {
    if (index < 0 || index >= _rows) {
        throw std::out_of_range("Index out of range");
    }
    return _data[index];
}

// matrix(row, col) returns the element at (row, col)
template <typename T>
T& matrix<T>::operator()(int row, int col) {
    if (row < 0 || row >= _rows || col < 0 || col >= _cols) {
        throw std::out_of_range("Index out of range");
    }
    return _data[row][col];
}
// const matrix(row, col)
template <typename T>
const T& matrix<T>::operator()(int row, int col) const {
    if (row < 0 || row >= _rows || col < 0 || col >= _cols) {
        throw std::out_of_range("Index out of range");
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
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    return matrix<T>(std::vector<std::vector<T>>(rows, std::vector<T>(cols, 0)));
}
// Matrix full of ones
template <typename T>
matrix<T> matrix<T>::ones(int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    return matrix<T>(std::vector<std::vector<T>>(rows, std::vector<T>(cols, 1)));
}
// Identity matrix
template <typename T>
matrix<T> matrix<T>::eye(int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }

    matrix<T> result(rows, cols);
    for (int i = 0; i < std::min(rows, cols); ++i) {
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
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    if (min >= max) {
        throw std::invalid_argument("Invalid range for random numbers");
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(min, max);
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
        throw std::invalid_argument("Matrix dimensions do not match for addition");
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
        throw std::invalid_argument("Matrix dimensions do not match for subtraction");
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
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
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
        throw std::invalid_argument("Matrix dimensions do not match for addition assignment");
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
        throw std::invalid_argument("Matrix dimensions do not match for subtraction assignment");
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
        throw std::invalid_argument("Matrix dimensions do not match for multiplication assignment");
    }
    matrix<T> result = (*this) * other;
    *this = result;
    return *this;
}
// Matrix equality operator
template <typename T>
bool matrix<T>::operator==(const matrix<T>& other) const noexcept{ 
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
bool matrix<T>::operator!=(const matrix<T>& other) const noexcept {
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
        throw std::invalid_argument("Division by zero");
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
        throw std::invalid_argument("Division by zero");
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
        throw std::invalid_argument("Matrix must be square to compute determinant");
    }
    if (_rows == 1) {
        return (*this)(0, 0);
    }
    if (_rows == 2) {
        return (*this)(0, 0) * (*this)(1, 1) - (*this)(0, 1) * (*this)(1, 0);
    }
    if (_rows == 3) {
        return (*this)(0, 0) * ((*this)(1, 1) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 1)) -
               (*this)(0, 1) * ((*this)(1, 0) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 0)) +
               (*this)(0, 2) * ((*this)(1, 0) * (*this)(2, 1) - (*this)(1, 1) * (*this)(2, 0));
    }
    // For larger matrices, use Gaussian elimination
    double det = 1.0;
    matrix<double> temp = (matrix<double>)*this;
    for (int i = 0; i < _rows; ++i) {
        int pivot = i;
        for (int j = i + 1; j < _rows; ++j) {
            if (std::abs(temp(j, i)) > std::abs(temp(pivot, i))) {
                pivot = j;
            }
        }
        if (pivot != i) {
            swap(temp(i), temp(pivot));
            det *= -1;
        }
        if (std::abs(temp(i, i)) < std::numeric_limits<double>::epsilon()) {
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
        throw std::invalid_argument("Matrix must be square to compute trace");
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
    matrix<T> result(_rows - 1, _cols - 1);
    for (int i = 0, r = 0; i < _rows; ++i) {
        if (i == row) continue;
        for (int j = 0, c = 0; j < _cols; ++j) {
            if (j == col) continue;
            result(r, c) = (*this)(i, j);
            ++c;
        }
        ++r;
    }
    return result;
}


// Adjoint
template <typename T>
matrix<T> matrix<T>::adjoint() const {
    if (_rows != _cols) {
        throw std::invalid_argument("Matrix must be square to compute adjoint");
    }
    matrix<T> result(_rows, _cols);
    result = this->cofactor().transpose();
    return result;
}

// Inverse
template <typename T>
matrix<double> matrix<T>::inverse() const {
    if (_rows != _cols) {
        throw std::invalid_argument("Matrix must be square to compute inverse");
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
        if (augmented(i, i) == 0) {
            throw std::invalid_argument("Matrix is singular and cannot be inverted");
        }
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
        throw std::out_of_range("Row index out of range");
    }
    swap(_data[row1], _data[row2]);
}
// Swap columns
template <typename T>
void matrix<T>::swapCols(int col1, int col2) {
    if (col1 < 0 || col1 >= _cols || col2 < 0 || col2 >= _cols) {
        throw std::out_of_range("Column index out of range");
    }
    for (int i = 0; i < _rows; ++i) {
        swap(_data[i][col1], _data[i][col2]);
    }
}
// Resize matrix
template <typename T>
void matrix<T>::resize(int rows, int cols) {
    if (rows < 0 || cols < 0) {
        throw std::invalid_argument("Matrix dimensions must be non-negative");
    }
    _data.resize(rows);
    for (int i = 0; i < rows; ++i) {
        _data[i].resize(cols);
    }
    _rows = rows;
    _cols = cols;
}



