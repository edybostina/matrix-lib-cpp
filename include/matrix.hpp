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
 * @date 2025-06-07 at 14:20
 * @version 0.1.5
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
#include <fstream>
#include <utility>
#include <thread>
#include <future>

template <typename T>
class matrix
{
private:
    int _rows = 0;
    int _cols = 0;
    std::vector<T> _data;

    int _index(int row, int col) const { return row * _cols + col; }

    std::vector<T> _to_row_vector(const std::vector<std::vector<T>> &init) const
    {
        std::vector<T> row_vector;
        for (const auto &row : init)
        {
            row_vector.insert(row_vector.end(), row.begin(), row.end());
        }
        return row_vector;
    }

public:
    constexpr matrix() noexcept = default;

    explicit matrix(int rows, int cols)
        : _rows(rows), _cols(cols), _data(rows * cols, T(0)) {}

    matrix(const std::vector<std::vector<T>> &init)
        : _rows(init.size()), _cols(init.empty() ? 0 : init[0].size()), _data(_to_row_vector(init)) {}

    matrix(std::initializer_list<std::initializer_list<T>> init)
    {
        _rows = init.size();
        _cols = init.begin()->size();
        _data.reserve(_rows * _cols);
        for (const auto &row : init)
        {
            if (row.size() != (size_t)_cols)
            {
                throw std::invalid_argument("All rows must have the same number of columns");
            }
            _data.insert(_data.end(), row.begin(), row.end());
        }
    }

    // Rule of Five
    matrix(const matrix &) = default;
    matrix(matrix &&) noexcept = default;
    matrix &operator=(const matrix &) = default;
    matrix &operator=(matrix &&) noexcept = default;
    ~matrix() = default;

    [[nodiscard]] constexpr int rows() const noexcept { return _rows; }
    [[nodiscard]] constexpr int cols() const noexcept { return _cols; }
    [[nodiscard]] constexpr int size() const noexcept { return _rows * _cols; }

    // Access operator

    std::vector<T> &operator()(int index);
    const std::vector<T> &operator()(int index) const;

    T &operator()(int row, int col);
    const T &operator()(int row, int col) const;

    auto begin() { return _data.begin(); }
    auto end() { return _data.end(); }
    auto begin() const { return _data.begin(); }
    auto end() const { return _data.end(); }

    // Access operator for row and column

    [[nodiscard]] matrix<T> row(int index) const;
    [[nodiscard]] matrix<T> col(int index) const;

    // Casting

    template <typename U>
    [[nodiscard]] explicit operator matrix<U>() const
    {
        matrix<U> result(_rows, _cols);
        for (int i = 0; i < _rows; ++i)
        {
            for (int j = 0; j < _cols; ++j)
            {
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
    friend std::ostream &operator<<(std::ostream &os, const matrix<U> &m);

    template <typename U>
    friend std::istream &operator>>(std::istream &is, matrix<U> &m);

    // Arithmetic operators
    // Matrix x Matrix

    [[nodiscard]] matrix<T> operator+(const matrix<T> &other) const;
    [[nodiscard]] matrix<T> operator-(const matrix<T> &other) const;
    [[nodiscard]] matrix<T> operator*(const matrix<T> &other) const;

    matrix<T> operator+=(const matrix<T> &other);
    matrix<T> operator-=(const matrix<T> &other);
    matrix<T> operator*=(const matrix<T> &other);

    [[nodiscard]] bool operator==(const matrix<T> &other) const noexcept;
    [[nodiscard]] bool operator!=(const matrix<T> &other) const noexcept;

    // Matrix x Scalar

    [[nodiscard]] matrix<T> operator+(const T &scalar) const;
    [[nodiscard]] matrix<T> operator-(const T &scalar) const;
    [[nodiscard]] matrix<T> operator*(const T &scalar) const;
    [[nodiscard]] matrix<T> operator/(const T &scalar) const;

    matrix<T> operator+=(const T &scalar);
    matrix<T> operator-=(const T &scalar);
    matrix<T> operator*=(const T &scalar);
    matrix<T> operator/=(const T &scalar);

    // Scalar x Matrix
    [[nodiscard]] friend matrix<T> operator+(const T &scalar, const matrix<T> &m)
    {
        return m + scalar;
    }
    [[nodiscard]] friend matrix<T> operator-(const T &scalar, const matrix<T> &m)
    {
        return m - scalar;
    }
    [[nodiscard]] friend matrix<T> operator*(const T &scalar, const matrix<T> &m)
    {
        return m * scalar;
    }
    [[nodiscard]] friend matrix<T> operator/(const T &scalar, const matrix<T> &m)
    {
        return m / scalar;
    }
    friend matrix<T> operator+=(const T &scalar, matrix<T> &m)
    {
        return m += scalar;
    }
    friend matrix<T> operator-=(const T &scalar, matrix<T> &m)
    {
        return m -= scalar;
    }
    friend matrix<T> operator*=(const T &scalar, matrix<T> &m)
    {
        return m *= scalar;
    }
    friend matrix<T> operator/=(const T &scalar, matrix<T> &m)
    {
        return m /= scalar;
    }

    // Matrix functions

    [[nodiscard]] double determinant() const;
    [[nodiscard]] T trace() const;
    [[nodiscard]] matrix<T> transpose() const;
    [[nodiscard]] matrix<T> cofactor() const;
    [[nodiscard]] matrix<T> minor(int row, int col) const;
    [[nodiscard]] matrix<T> adjoint() const;
    [[nodiscard]] matrix<double> inverse() const;
    [[nodiscard]] double norm(int p) const;
    [[nodiscard]] int rank() const;
    [[nodiscard]] matrix<double> gaussian_elimination() const;

    void swapRows(int row1, int row2);
    void swapCols(int col1, int col2);
    void resize(int rows, int cols);

    [[nodiscard]] matrix<T> submatrix(int top_corner_x, int top_corner_y, int bottom_corner_x, int bottom_corner_y) const;

    // Numeric methods

    [[nodiscard]] std::pair<matrix<double>, matrix<double>> LU_decomposition() const;
    [[nodiscard]] std::pair<matrix<double>, matrix<double>> QR_decomposition() const;
    [[nodiscard]] matrix<double> eigenvalues() const;
    [[nodiscard]] matrix<double> eigenvectors() const;
};

// ==================================================
// ==================== FIle I/O ====================
// ==================================================

// Output stream operator
template <typename T>
std::ostream &operator<<(std::ostream &os, const matrix<T> &m)
{
    for (int i = 0; i < m.rows(); ++i)
    {
        for (int j = 0; j < m.cols(); ++j)
        {
            os << m(i, j) << " ";
        }
        os << "\n";
    }
    return os;
}
// Input stream operator
template <typename T>
std::istream &operator>>(std::istream &is, matrix<T> &m)
{
    int rows, cols;
    is >> rows >> cols;
    m = matrix<T>(rows, cols);
    for (int i = 0; i < m.rows(); ++i)
    {
        for (int j = 0; j < m.cols(); ++j)
        {
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
std::vector<T> &matrix<T>::operator()(int index)
{
    if (index < 0 || index >= _rows)
    {
        throw std::out_of_range("Index out of range");
    }
    std::vector<T> &row = _data;
    row.resize(_cols);
    std::copy(_data.begin() + index * _cols, _data.begin() + (index + 1) * _cols, row.begin());
    return row;
}
// const matrix(index)
template <typename T>
const std::vector<T> &matrix<T>::operator()(int index) const
{
    if (index < 0 || index >= _rows)
    {
        throw std::out_of_range("Index out of range");
    }
    return std::vector<T>(_data.begin() + index * _cols, _data.begin() + (index + 1) * _cols);
}

// matrix(row, col) returns the element at (row, col)
template <typename T>
T &matrix<T>::operator()(int row, int col)
{
    if (row < 0 || row >= _rows || col < 0 || col >= _cols)
    {
        throw std::out_of_range("Index out of range");
    }
    return _data[_index(row, col)];
}
// const matrix(row, col)
template <typename T>
const T &matrix<T>::operator()(int row, int col) const
{
    if (row < 0 || row >= _rows || col < 0 || col >= _cols)
    {
        throw std::out_of_range("Index out of range");
    }
    return _data[_index(row, col)];
}

// matrix.row(i) returns the i-th row of the matrix
template <typename T>
matrix<T> matrix<T>::row(int index) const
{
    if (index < 0 || index >= _rows)
    {
        throw std::out_of_range("Index out of range");
    }
    matrix<T> result(1, _cols);
    for (int j = 0; j < _cols; ++j)
    {
        result(0, j) = _data[_index(index, j)];
    }
    return result;
}
// matrix.col(j) returns the j-th column of the matrix
template <typename T>
matrix<T> matrix<T>::col(int index) const
{
    if (index < 0 || index >= _cols)
    {
        throw std::out_of_range("Index out of range");
    }
    matrix<T> result(_rows, 1);
    for (int i = 0; i < _rows; ++i)
    {
        result(i, 0) = _data[_index(i, index)];
    }
    return result;
}

// ==============================================================
// ==================== Basic initialization ====================
// ==============================================================

// Matrix full of zeros
template <typename T>
matrix<T> matrix<T>::zeros(int rows, int cols)
{
    if (rows <= 0 || cols <= 0)
    {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    return matrix<T>(std::vector<std::vector<T>>(rows, std::vector<T>(cols, 0)));
}
// Matrix full of ones
template <typename T>
matrix<T> matrix<T>::ones(int rows, int cols)
{
    if (rows <= 0 || cols <= 0)
    {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    return matrix<T>(std::vector<std::vector<T>>(rows, std::vector<T>(cols, 1)));
}
// Identity matrix
template <typename T>
matrix<T> matrix<T>::eye(int rows, int cols)
{
    if (rows <= 0 || cols <= 0)
    {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }

    matrix<T> result(rows, cols);
    for (int i = 0; i < std::min(rows, cols); ++i)
    {
        result(i, i) = 1;
    }
    return result;
}

// Random initialization
// This function uses the <random> lib to generate random numbers
// between 0 and 1
template <typename T>
matrix<T> matrix<T>::random(int rows, int cols, T min, T max)
{
    if (rows <= 0 || cols <= 0)
    {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    if (min >= max)
    {
        throw std::invalid_argument("Invalid range for random numbers");
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(min, max);
    matrix<T> result(rows, cols);
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
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
matrix<T> matrix<T>::operator+(const matrix<T> &other) const
{
    if (_rows != other._rows || _cols != other._cols)
    {
        throw std::invalid_argument("Matrix dimensions do not match for addition");
    }

    matrix<T> result(_rows, _cols);

    // Number of threads to use
    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](int start_row, int end_row)
    {
        for (int i = start_row; i < end_row; ++i)
        {
            for (int j = 0; j < _cols; ++j)
            {
                result(i, j) = (*this)(i, j) + other(i, j);
            }
        }
    };

    int rows_per_thread = _rows / num_threads;
    int leftover = _rows % num_threads;

    int start = 0;
    for (int t = 0; t < num_threads; ++t)
    {
        int end = start + rows_per_thread + (t < leftover ? 1 : 0);
        threads[t] = std::thread(worker, start, end);
        start = end;
    }

    for (auto &t : threads)
    {
        if (t.joinable())
            t.join();
    }

    return result;
}

// Matrix subtraction
template <typename T>
matrix<T> matrix<T>::operator-(const matrix<T> &other) const
{
    if (_rows != other._rows || _cols != other._cols)
    {
        throw std::invalid_argument("Matrix dimensions do not match for addition");
    }

    matrix<T> result(_rows, _cols);

    // Number of threads to use
    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](int start_row, int end_row)
    {
        for (int i = start_row; i < end_row; ++i)
        {
            for (int j = 0; j < _cols; ++j)
            {
                result(i, j) = (*this)(i, j) - other(i, j);
            }
        }
    };

    int rows_per_thread = _rows / num_threads;
    int leftover = _rows % num_threads;

    int start = 0;
    for (int t = 0; t < num_threads; ++t)
    {
        int end = start + rows_per_thread + (t < leftover ? 1 : 0);
        threads[t] = std::thread(worker, start, end);
        start = end;
    }

    for (auto &t : threads)
    {
        if (t.joinable())
            t.join();
    }

    return result;
}

// Matrix multiplication
template <typename T>
matrix<T> matrix<T>::operator*(const matrix<T> &other) const
{
    if (_cols != other._rows)
    {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    }

    matrix<T> result(_rows, other._cols);
    auto worker = [&](int row_start, int row_end)
    {
        for (int i = row_start; i < row_end; ++i)
        {
            for (int j = 0; j < other._cols; ++j)
            {
                T sum = 0;
                for (int k = 0; k < _cols; ++k)
                {
                    sum += (*this)(i, k) * other(k, j);
                }
                result(i, j) = sum;
            }
        }
    };

    int num_threads = std::thread::hardware_concurrency();
    int chunk_size = (_rows + num_threads - 1) / num_threads;

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t)
    {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, _rows);
        if (start < end)
        {
            threads.emplace_back(worker, start, end);
        }
    }

    for (auto &t : threads)
    {
        t.join();
    }

    return result;
}

// Matrix addition assignment
template <typename T>
matrix<T> matrix<T>::operator+=(const matrix<T> &other)
{
    if (_rows != other._rows || _cols != other._cols)
    {
        throw std::invalid_argument("Matrix dimensions do not match for addition assignment");
    }

    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](int start_row, int end_row)
    {
        for (int i = start_row; i < end_row; ++i)
        {
            for (int j = 0; j < _cols; ++j)
            {
                (*this)(i, j) += other(i, j);
            }
        }
    };

    int rows_per_thread = _rows / num_threads;
    int leftover = _rows % num_threads;

    int start = 0;
    for (int t = 0; t < num_threads; ++t)
    {
        int end = start + rows_per_thread + (t < leftover ? 1 : 0);
        threads[t] = std::thread(worker, start, end);
        start = end;
    }

    for (auto &t : threads)
    {
        if (t.joinable())
            t.join();
    }

    return *this;
}
// Matrix subtraction assignment
template <typename T>
matrix<T> matrix<T>::operator-=(const matrix<T> &other)
{
    if (_rows != other._rows || _cols != other._cols)
    {
        throw std::invalid_argument("Matrix dimensions do not match for subtraction assignment");
    }
    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](int start_row, int end_row)
    {
        for (int i = start_row; i < end_row; ++i)
        {
            for (int j = 0; j < _cols; ++j)
            {
                (*this)(i, j) -= other(i, j);
            }
        }
    };

    int rows_per_thread = _rows / num_threads;
    int leftover = _rows % num_threads;

    int start = 0;
    for (int t = 0; t < num_threads; ++t)
    {
        int end = start + rows_per_thread + (t < leftover ? 1 : 0);
        threads[t] = std::thread(worker, start, end);
        start = end;
    }

    for (auto &t : threads)
    {
        if (t.joinable())
            t.join();
    }

    return *this;
}
// Matrix multiplication assignment
template <typename T>
matrix<T> matrix<T>::operator*=(const matrix<T> &other)
{
    if (_cols != other._rows)
    {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication assignment");
    }
    matrix<T> result = (*this) * other;
    *this = result;
    return *this;
}
// Matrix equality operator
template <typename T>
bool matrix<T>::operator==(const matrix<T> &other) const noexcept
{
    if (_rows != other._rows || _cols != other._cols)
    {
        return false;
    }
    for (int i = 0; i < _rows; ++i)
    {
        for (int j = 0; j < _cols; ++j)
        {
            if ((*this)(i, j) != other(i, j))
            {
                return false;
            }
        }
    }
    return true;
}
// Matrix inequality operator
template <typename T>
bool matrix<T>::operator!=(const matrix<T> &other) const noexcept
{
    return !(*this == other);
}

// Matrix and Scalar

// Matrix addition with scalar
template <typename T>
matrix<T> matrix<T>::operator+(const T &scalar) const
{
    matrix<T> result(_rows, _cols);

    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](int start_row, int end_row)
    {
        for (int i = start_row; i < end_row; ++i)
        {
            for (int j = 0; j < _cols; ++j)
            {
                result(i, j) = (*this)(i, j) + scalar;
            }
        }
    };

    int rows_per_thread = _rows / num_threads;
    int leftover = _rows % num_threads;

    int start = 0;
    for (int t = 0; t < num_threads; ++t)
    {
        int end = start + rows_per_thread + (t < leftover ? 1 : 0);
        threads[t] = std::thread(worker, start, end);
        start = end;
    }

    for (auto &t : threads)
    {
        if (t.joinable())
            t.join();
    }

    return result;
}
// Matrix subtraction with scalar
template <typename T>
matrix<T> matrix<T>::operator-(const T &scalar) const
{
    matrix<T> result(_rows, _cols);

    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](int start_row, int end_row)
    {
        for (int i = start_row; i < end_row; ++i)
        {
            for (int j = 0; j < _cols; ++j)
            {
                result(i, j) = (*this)(i, j) - scalar;
            }
        }
    };

    int rows_per_thread = _rows / num_threads;
    int leftover = _rows % num_threads;

    int start = 0;
    for (int t = 0; t < num_threads; ++t)
    {
        int end = start + rows_per_thread + (t < leftover ? 1 : 0);
        threads[t] = std::thread(worker, start, end);
        start = end;
    }

    for (auto &t : threads)
    {
        if (t.joinable())
            t.join();
    }

    return result;
}
// Matrix multiplication with scalar
template <typename T>
matrix<T> matrix<T>::operator*(const T &scalar) const
{
    matrix<T> result(_rows, _cols);

    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](int start_row, int end_row)
    {
        for (int i = start_row; i < end_row; ++i)
        {
            for (int j = 0; j < _cols; ++j)
            {
                result(i, j) = (*this)(i, j) * scalar;
            }
        }
    };

    int rows_per_thread = _rows / num_threads;
    int leftover = _rows % num_threads;

    int start = 0;
    for (int t = 0; t < num_threads; ++t)
    {
        int end = start + rows_per_thread + (t < leftover ? 1 : 0);
        threads[t] = std::thread(worker, start, end);
        start = end;
    }

    for (auto &t : threads)
    {
        if (t.joinable())
            t.join();
    }

    return result;
}
// Matrix division by scalar
template <typename T>
matrix<T> matrix<T>::operator/(const T &scalar) const
{
    if (scalar == 0)
    {
        throw std::invalid_argument("Division by zero");
    }
    matrix<T> result(_rows, _cols);

    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](int start_row, int end_row)
    {
        for (int i = start_row; i < end_row; ++i)
        {
            for (int j = 0; j < _cols; ++j)
            {
                result(i, j) = (*this)(i, j) / scalar;
            }
        }
    };

    int rows_per_thread = _rows / num_threads;
    int leftover = _rows % num_threads;

    int start = 0;
    for (int t = 0; t < num_threads; ++t)
    {
        int end = start + rows_per_thread + (t < leftover ? 1 : 0);
        threads[t] = std::thread(worker, start, end);
        start = end;
    }

    for (auto &t : threads)
    {
        if (t.joinable())
            t.join();
    }

    return result;
}

// Matrix addition assignment with scalar
template <typename T>
matrix<T> matrix<T>::operator+=(const T &scalar)
{
    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](int start_row, int end_row)
    {
        for (int i = start_row; i < end_row; ++i)
        {
            for (int j = 0; j < _cols; ++j)
            {
                (*this)(i, j) += scalar;
            }
        }
    };

    int rows_per_thread = _rows / num_threads;
    int leftover = _rows % num_threads;

    int start = 0;
    for (int t = 0; t < num_threads; ++t)
    {
        int end = start + rows_per_thread + (t < leftover ? 1 : 0);
        threads[t] = std::thread(worker, start, end);
        start = end;
    }

    for (auto &t : threads)
    {
        if (t.joinable())
            t.join();
    }

    return *this;
}
// Matrix subtraction assignment with scalar
template <typename T>
matrix<T> matrix<T>::operator-=(const T &scalar)
{
    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](int start_row, int end_row)
    {
        for (int i = start_row; i < end_row; ++i)
        {
            for (int j = 0; j < _cols; ++j)
            {
                (*this)(i, j) -= scalar;
            }
        }
    };

    int rows_per_thread = _rows / num_threads;
    int leftover = _rows % num_threads;

    int start = 0;
    for (int t = 0; t < num_threads; ++t)
    {
        int end = start + rows_per_thread + (t < leftover ? 1 : 0);
        threads[t] = std::thread(worker, start, end);
        start = end;
    }

    for (auto &t : threads)
    {
        if (t.joinable())
            t.join();
    }

    return *this;
}
// Matrix multiplication assignment with scalar
template <typename T>
matrix<T> matrix<T>::operator*=(const T &scalar)
{
    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](int start_row, int end_row)
    {
        for (int i = start_row; i < end_row; ++i)
        {
            for (int j = 0; j < _cols; ++j)
            {
                (*this)(i, j) *= scalar;
            }
        }
    };

    int rows_per_thread = _rows / num_threads;
    int leftover = _rows % num_threads;

    int start = 0;
    for (int t = 0; t < num_threads; ++t)
    {
        int end = start + rows_per_thread + (t < leftover ? 1 : 0);
        threads[t] = std::thread(worker, start, end);
        start = end;
    }

    for (auto &t : threads)
    {
        if (t.joinable())
            t.join();
    }

    return *this;
}
// Matrix division assignment by scalar
template <typename T>
matrix<T> matrix<T>::operator/=(const T &scalar)
{
    if (scalar == 0)
    {
        throw std::invalid_argument("Division by zero");
    }
    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](int start_row, int end_row)
    {
        for (int i = start_row; i < end_row; ++i)
        {
            for (int j = 0; j < _cols; ++j)
            {
                (*this)(i, j) /= scalar;
            }
        }
    };

    int rows_per_thread = _rows / num_threads;
    int leftover = _rows % num_threads;

    int start = 0;
    for (int t = 0; t < num_threads; ++t)
    {
        int end = start + rows_per_thread + (t < leftover ? 1 : 0);
        threads[t] = std::thread(worker, start, end);
        start = end;
    }

    for (auto &t : threads)
    {
        if (t.joinable())
            t.join();
    }

    return *this;
}

// ==========================================================
// ==================== Matrix functions ====================
// ==========================================================

// Determinant
template <typename T>
double matrix<T>::determinant() const
{
    if (_rows != _cols)
    {
        throw std::invalid_argument("Matrix must be square to compute determinant");
    }
    if (_rows == 1)
    {
        return (*this)(0, 0);
    }
    if (_rows == 2)
    {
        return (*this)(0, 0) * (*this)(1, 1) - (*this)(0, 1) * (*this)(1, 0);
    }
    if (_rows == 3)
    {
        return (*this)(0, 0) * ((*this)(1, 1) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 1)) -
               (*this)(0, 1) * ((*this)(1, 0) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 0)) +
               (*this)(0, 2) * ((*this)(1, 0) * (*this)(2, 1) - (*this)(1, 1) * (*this)(2, 0));
    }
    // For larger matrices, use Gaussian elimination
    double det = 1.0;
    matrix<double> temp = (matrix<double>)*this;
    for (int i = 0; i < _rows; ++i)
    {
        int pivot = i;
        for (int j = i + 1; j < _rows; ++j)
        {
            if (std::abs(temp(j, i)) > std::abs(temp(pivot, i)))
            {
                pivot = j;
            }
        }
        if (pivot != i)
        {
            swap(temp(i), temp(pivot));
            det *= -1;
        }
        if (std::abs(temp(i, i)) < std::numeric_limits<double>::epsilon())
        {
            return 0;
        }
        det *= temp(i, i);
        for (int j = i + 1; j < _rows; ++j)
        {
            double factor = temp(j, i) / temp(i, i);
            for (int k = i + 1; k < _cols; ++k)
            {
                temp(j, k) -= factor * temp(i, k);
            }
        }
    }
    return det;
}

// Trace
template <typename T>
T matrix<T>::trace() const
{
    if (_rows != _cols)
    {
        throw std::invalid_argument("Matrix must be square to compute trace");
    }
    T sum = 0;
    for (int i = 0; i < _rows; ++i)
    {
        sum += (*this)(i, i);
    }
    return sum;
}

// Transpose
template <typename T>
matrix<T> matrix<T>::transpose() const
{
    matrix<T> result(_cols, _rows);
    for (int i = 0; i < _rows; ++i)
    {
        for (int j = 0; j < _cols; ++j)
        {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

// Cofactor
template <typename T>
matrix<T> matrix<T>::cofactor() const
{
    matrix<T> result(_rows, _cols);
    for (int i = 0; i < _rows; ++i)
    {
        for (int j = 0; j < _cols; ++j)
        {
            matrix<T> minor = this->minor(i, j);
            result(i, j) = ((i + j) % 2 == 0 ? 1 : -1) * minor.determinant();
        }
    }
    return result;
}

// Minor
template <typename T>
matrix<T> matrix<T>::minor(int row, int col) const
{
    matrix<T> result(_rows - 1, _cols - 1);
    for (int i = 0, r = 0; i < _rows; ++i)
    {
        if (i == row)
            continue;
        for (int j = 0, c = 0; j < _cols; ++j)
        {
            if (j == col)
                continue;
            result(r, c) = (*this)(i, j);
            ++c;
        }
        ++r;
    }
    return result;
}

// Adjoint
template <typename T>
matrix<T> matrix<T>::adjoint() const
{
    if (_rows != _cols)
    {
        throw std::invalid_argument("Matrix must be square to compute adjoint");
    }
    matrix<T> result(_rows, _cols);
    result = this->cofactor().transpose();
    return result;
}

// Inverse
template <typename T>
matrix<double> matrix<T>::inverse() const
{
    if (_rows != _cols)
    {
        throw std::invalid_argument("Matrix must be square to compute inverse");
    }
    double temp;
    matrix<double> augmented(_rows, 2 * _cols);
    matrix<double> result(_rows, _cols);
    for (int i = 0; i < _rows; ++i)
    {
        for (int j = 0; j < _cols; ++j)
        {
            augmented(i, j) = (*this)(i, j);
        }
        for (int j = _cols; j < 2 * _cols; ++j)
        {
            if (i == j - _cols)
            {
                augmented(i, j) = 1;
            }
            else
            {
                augmented(i, j) = 0;
            }
        }
    }
    for (int i = _rows - 1; i > 0; i--)
    {
        if (augmented(i - 1, 0) < augmented(i, 0))
        {
            augmented.swapRows(i - 1, i);
        }
    }

    for (int i = 0; i < _rows; ++i)
    {
        if (augmented(i, i) == 0)
        {
            throw std::invalid_argument("Matrix is singular and cannot be inverted");
        }
        for (int j = 0; j < _cols; ++j)
        {
            if (j != i)
            {
                temp = augmented(j, i) / augmented(i, i);
                for (int k = 0; k < 2 * _cols; ++k)
                {
                    augmented(j, k) -= augmented(i, k) * temp;
                }
            }
        }
    }
    for (int i = 0; i < _rows; ++i)
    {
        temp = augmented(i, i);
        for (int j = 0; j < 2 * _cols; ++j)
        {
            augmented(i, j) /= temp;
        }
    }
    for (int i = 0; i < _rows; ++i)
    {
        for (int j = 0; j < _cols; ++j)
        {
            result(i, j) = augmented(i, j + _cols);
        }
    }
    return result;
}

// Norm
template <typename T>
double matrix<T>::norm(int p) const
{
    if (p < 1)
    {
        throw std::invalid_argument("Norm order must be greater than or equal to 1");
    }
    double norm = 0;
    for (int i = 0; i < _rows; ++i)
    {
        for (int j = 0; j < _cols; ++j)
        {
            norm += std::pow(std::abs((*this)(i, j)), p);
        }
    }
    return std::pow(norm, 1.0 / p);
}

// Swap rows
template <typename T>
void matrix<T>::swapRows(int row1, int row2)
{
    if (row1 < 0 || row1 >= _rows || row2 < 0 || row2 >= _rows)
    {
        throw std::out_of_range("Row index out of range");
    }
    for (int j = 0; j < _cols; ++j)
    {
        _data[_index(row1, j)] = _data[_index(row1, j)] + _data[_index(row2, j)];
        _data[_index(row2, j)] = _data[_index(row1, j)] - _data[_index(row2, j)];
        _data[_index(row1, j)] = _data[_index(row1, j)] - _data[_index(row2, j)];
    }
}
// Swap columns
template <typename T>
void matrix<T>::swapCols(int col1, int col2)
{
    if (col1 < 0 || col1 >= _cols || col2 < 0 || col2 >= _cols)
    {
        throw std::out_of_range("Column index out of range");
    }
    for (int i = 0; i < _rows; ++i)
    {
        _data[_index(i, col1)] = _data[_index(i, col1)] + _data[_index(i, col2)];
        _data[_index(i, col2)] = _data[_index(i, col1)] - _data[_index(i, col2)];
        _data[_index(i, col1)] = _data[_index(i, col1)] - _data[_index(i, col2)];
    }
}
// Resize matrix
template <typename T>
void matrix<T>::resize(int rows, int cols)
{
    if (rows < 0 || cols < 0)
    {
        throw std::invalid_argument("Matrix dimensions must be non-negative");
    }
    if (rows == _rows && cols == _cols)
    {
        return; // No change needed
    }
    std::vector<T> new_data(rows * cols);
    for (int i = 0; i < std::min(_rows, rows); ++i)
    {
        for (int j = 0; j < std::min(_cols, cols); ++j)
        {
            new_data[i * cols + j] = (*this)(i, j);
        }
    }
    _data = std::move(new_data);
    _rows = rows;
    _cols = cols;
}

// Submatrix
template <typename T>
matrix<T> matrix<T>::submatrix(int top_corner_x, int top_corner_y, int bottom_corner_x, int bottom_corner_y) const
{
    if (top_corner_x < 0 || top_corner_x >= _rows || bottom_corner_x < 0 || bottom_corner_x >= _rows ||
        top_corner_y < 0 || top_corner_y >= _cols || bottom_corner_y < 0 || bottom_corner_y >= _cols)
    {
        throw std::out_of_range("Submatrix indices out of range");
    }
    if (top_corner_x > bottom_corner_x || top_corner_y > bottom_corner_y)
    {
        throw std::invalid_argument("Invalid submatrix indices");
    }
    matrix<T> result(bottom_corner_x - top_corner_x + 1, bottom_corner_y - top_corner_y + 1);
    for (int i = top_corner_x; i <= bottom_corner_x; ++i)
    {
        for (int j = top_corner_y; j <= bottom_corner_y; ++j)
        {
            result(i - top_corner_x, j - top_corner_y) = (*this)(i, j);
        }
    }
    return result;
}

template <typename T>
int matrix<T>::rank() const
{
    matrix<double> gaussian = this->gaussian_elimination();
    int rank = 0;
    for (int i = 0; i < _rows; ++i)
    {
        bool non_zero_row = false;
        for (int j = 0; j < _cols; ++j)
        {
            if (std::abs(gaussian(i, j)) > std::numeric_limits<double>::epsilon())
            {
                non_zero_row = true;
                break;
            }
        }
        if (non_zero_row)
        {
            ++rank;
        }
    }
    return rank;
}

template <typename T>
matrix<double> matrix<T>::gaussian_elimination() const
{
    if (_rows == 0 || _cols == 0)
    {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    matrix<double> result = (matrix<double>)*this;
    for (int i = 0; i < _rows; ++i)
    {
        int pivot_row = i;
        for (int j = i + 1; j < _rows; ++j)
        {
            if (std::abs(result(j, i)) > std::abs(result(pivot_row, i)))
            {
                pivot_row = j;
            }
        }
        if (std::abs(result(pivot_row, i)) < std::numeric_limits<double>::epsilon())
        {
            continue;
        }
        if (pivot_row != i)
        {
            result.swapRows(i, pivot_row);
        }
        for (int j = i + 1; j < _rows; ++j)
        {
            double factor = result(j, i) / result(i, i);
            for (int k = i; k < _cols; ++k)
            {
                result(j, k) -= factor * result(i, k);
            }
        }
    }
    return result;
}

// =========================================================
// ==================== Numeric Methods ====================
// =========================================================

// LU decomposition
// This function uses Crout's algorithm to compute the LU decomposition
// L and U are stored in a pair of matrices
// L is a lower triangular matrix
// U is an upper triangular matrix
// returns make_pair(L, U)

// TODO: FIXXXXXXXXXXXXX !!!!!!!!!!!!!!
// not thread safe yet

// template <typename T>
// std::pair<matrix<double>, matrix<double>> matrix<T>::LU_decomposition() const
// {
//     if (_rows != _cols)
//     {
//         throw std::invalid_argument("Matrix must be square for LU decomposition");
//     }

//     matrix<double> L(_rows, _cols);
//     matrix<double> U = matrix<double>::eye(_rows, _cols);

//     for (int p = 0; p < _rows; ++p)
//     {
//         std::vector<std::future<void>> futures;

//         for (int i = 0; i < p; ++i)
//         {
//             futures.emplace_back(std::async(std::launch::async, [&, i]()
//                                             {
//                 U(i, p) = (*this)(i, p) - (L.row(i) * U.col(p))(0, 0);
//                 U(i, p) /= L(i, i); }));
//         }

//         for (auto &f : futures)
//             f.get();
//         futures.clear();

//         for (int i = p; i < _rows; ++i)
//         {
//             futures.emplace_back(std::async(std::launch::async, [&, i]()
//                                             { L(i, p) = (double)(*this)(i, p) - (L.row(i) * U.col(p))(0, 0); }));
//         }

//         for (auto &f : futures)
//             f.get();
//     }

//     return std::make_pair(L, U);
// }

// LU decomposition
// This function uses Crout's algorithm to compute the LU decomposition
// L and U are stored in a pair of matrices
// L is a lower triangular matrix
// U is an upper triangular matrix
// returns make_pair(L, U)
template <typename T>
std::pair<matrix<double>, matrix<double>> matrix<T>::LU_decomposition() const
{
    if (_rows != _cols)
    {
        throw std::invalid_argument("Matrix must be square for LU decomposition");
    }
    matrix<double> L(_rows, _cols);
    matrix<double> U = matrix<double>::eye(_rows, _cols);
    for (int p = 0; p < _rows; ++p)
    {
        for (int i = 0; i < p; ++i)
        {
            U(i, p) = (*this)(i, p) - (L.row(i) * U.col(p))(0, 0);
            U(i, p) /= L(i, i);
        }
        for (int i = p; i < _rows; ++i)
        {
            L(i, p) = (double)(*this)(i, p) - (L.row(i) * U.col(p))(0, 0);
        }
    }

    return std::make_pair(L, U);
}

// QR decomposition
// This function uses the Householder reflection method to compute the QR decomposition
// Q is an orthogonal matrix
// R is an upper triangular matrix
// returns make_pair(Q, R)

template <typename T>
std::pair<matrix<double>, matrix<double>> matrix<T>::QR_decomposition() const
{
    matrix<double> Q = (matrix<double>)*this;
    matrix<double> R(_rows, _cols);

    double norm0 = this->col(0).norm(2);
    for (int i = 0; i < _rows; ++i)
        Q(i, 0) = (*this)(i, 0) / norm0;

    std::vector<std::future<void>> futures;
    for (int j = 0; j < _cols; ++j)
    {
        futures.emplace_back(std::async(std::launch::async, [&, j]()
                                        { R(0, j) = (Q.col(0).transpose() * (matrix<double>)(*this))(0, j); }));
    }
    for (auto &f : futures)
        f.get();

    for (int i = 0; i < _cols - 1; ++i)
    {
        futures.clear();
        for (int j = i + 1; j < _cols; ++j)
        {
            futures.emplace_back(std::async(std::launch::async, [&, i, j]()
                                            {
                matrix<double> new_col = Q.col(j);
                new_col -= (Q.col(j).transpose() * Q.col(i))(0, 0) * Q.col(i);
                for (int k = 0; k < _rows; ++k)
                    Q(k, j) = new_col(k, 0); }));
        }
        for (auto &f : futures)
            f.get();

        double norm = Q.col(i + 1).norm(2);
        for (int k = 0; k < _rows; ++k)
            Q(k, i + 1) = Q(k, i + 1) / norm;

        futures.clear();
        for (int k = 0; k < _cols; ++k)
        {
            futures.emplace_back(std::async(std::launch::async, [&, i, k]()
                                            { R(i + 1, k) = (Q.col(i + 1).transpose() * (matrix<double>)(*this))(0, k); }));
        }
        for (auto &f : futures)
            f.get();
    }

    return std::make_pair(Q, R);
}

// Eigenvalues
// Currently, this function uses the QR algorithm to compute the eigenvalues
// Does not work for complex numbers yet.
template <typename T>
matrix<double> matrix<T>::eigenvalues() const
{
    if (_rows != _cols)
    {
        throw std::invalid_argument("Matrix must be square to compute eigenvalues");
    }

    matrix<double> eigenvalues = (matrix<double>)(*this);
    for (int iter = 0; iter < 100; iter++)
    {
        matrix<double> Q, R;
        std::tie(Q, R) = eigenvalues.QR_decomposition();
        eigenvalues = R * Q;
    }
    matrix<double> result(_rows, 1);
    for (int i = 0; i < _rows; ++i)
    {
        result(i, 0) = eigenvalues(i, i);
    }
    return result;
}

// Eigenvectors
// Currently, this function uses the QR algorithm to compute the eigenvectors
// Does not work for complex numbers yet.
template <typename T>
matrix<double> matrix<T>::eigenvectors() const
{
    if (_rows != _cols)
    {
        throw std::invalid_argument("Matrix must be square to compute eigenvalues");
    }

    matrix<double> eigenvalues = (matrix<double>)(*this);
    matrix<double> eigenvectors = matrix<double>::eye(_rows, _cols);
    for (int iter = 0; iter < 100; iter++)
    {
        matrix<double> Q, R;
        std::tie(Q, R) = eigenvalues.QR_decomposition();
        eigenvalues = R * Q;
        eigenvectors = eigenvectors * Q;
    }
    return eigenvectors;
}
