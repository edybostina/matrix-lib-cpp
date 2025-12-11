#pragma once

// Template implementations for matrix_core.hpp
// This file is included at the end of matrix_core.hpp

#include <sstream>
#include <stdexcept>

// ============================================================================
// Access Operators Implementation
// ============================================================================

// matrix(index) returns the i-th row of the matrix
template <typename T>
std::vector<T> &matrix<T>::operator()(size_t index)
{
    if (index >= _rows)
    {
        std::ostringstream oss;
        oss << "Row index " << index << " out of range [0, " << _rows << ")";
        throw std::out_of_range(oss.str());
    }
    std::vector<T> &row = _data;
    row.resize(_cols);
    std::copy(_data.begin() + index * _cols, _data.begin() + (index + 1) * _cols, row.begin());
    return row;
}

// const matrix(index)
template <typename T>
const std::vector<T> &matrix<T>::operator()(size_t index) const
{
    if (index >= _rows)
    {
        std::ostringstream oss;
        oss << "Row index " << index << " out of range [0, " << _rows << ")";
        throw std::out_of_range(oss.str());
    }
    std::vector<T> &row = const_cast<std::vector<T> &>(_data);
    row.resize(_cols);
    std::copy(_data.begin() + index * _cols, _data.begin() + (index + 1) * _cols, row.begin());
    return row;
}

// matrix(row, col) returns the element at (row, col)
template <typename T>
T &matrix<T>::operator()(size_t row, size_t col)
{
    if (row >= _rows || col >= _cols)
    {
        std::ostringstream oss;
        oss << "Index (" << row << ", " << col << ") out of range for matrix of size "
            << _rows << "x" << _cols;
        throw std::out_of_range(oss.str());
    }
    return _data[_index(row, col)];
}

// const matrix(row, col)
template <typename T>
const T &matrix<T>::operator()(size_t row, size_t col) const
{
    if (row >= _rows || col >= _cols)
    {
        std::ostringstream oss;
        oss << "Index (" << row << ", " << col << ") out of range for matrix of size "
            << _rows << "x" << _cols;
        throw std::out_of_range(oss.str());
    }
    return _data[_index(row, col)];
}

// ============================================================================
// Row and Column Access
// ============================================================================

// matrix.row(i) returns the i-th row of the matrix
template <typename T>
matrix<T> matrix<T>::row(size_t index) const
{
    if (index >= _rows)
    {
        std::ostringstream oss;
        oss << "Row index " << index << " out of range [0, " << _rows << ")";
        throw std::out_of_range(oss.str());
    }
    matrix<T> result(1, _cols);
    for (size_t j = 0; j < _cols; ++j)
    {
        result(0, j) = _data[_index(index, j)];
    }
    return result;
}

// matrix.col(j) returns the j-th column of the matrix
template <typename T>
matrix<T> matrix<T>::col(size_t index) const
{
    if (index >= _cols)
    {
        std::ostringstream oss;
        oss << "Column index " << index << " out of range [0, " << _cols << ")";
        throw std::out_of_range(oss.str());
    }
    matrix<T> result(_rows, 1);
    for (size_t i = 0; i < _rows; ++i)
    {
        result(i, 0) = _data[_index(i, index)];
    }
    return result;
}

// ============================================================================
// Factory Methods Implementation
// ============================================================================

// Matrix full of zeros
template <typename T>
matrix<T> matrix<T>::zeros(size_t rows, size_t cols)
{
    if (rows == 0 || cols == 0)
    {
        throw std::invalid_argument("Matrix dimensions must be positive (non-zero)");
    }
    return matrix<T>(std::vector<std::vector<T>>(rows, std::vector<T>(cols, T(0))));
}

// Matrix full of ones
template <typename T>
matrix<T> matrix<T>::ones(size_t rows, size_t cols)
{
    if (rows == 0 || cols == 0)
    {
        throw std::invalid_argument("Matrix dimensions must be positive (non-zero)");
    }
    return matrix<T>(std::vector<std::vector<T>>(rows, std::vector<T>(cols, T(1))));
}

// Identity matrix
template <typename T>
matrix<T> matrix<T>::eye(size_t rows, size_t cols)
{
    if (rows == 0 || cols == 0)
    {
        throw std::invalid_argument("Matrix dimensions must be positive (non-zero)");
    }

    matrix<T> result(rows, cols);
    size_t min_dim = std::min(rows, cols);
    for (size_t i = 0; i < min_dim; ++i)
    {
        result(i, i) = T(1);
    }
    return result;
}

// Random initialization
template <typename T>
matrix<T> matrix<T>::random(size_t rows, size_t cols, T min, T max)
{
    if (rows == 0 || cols == 0)
    {
        throw std::invalid_argument("Matrix dimensions must be positive (non-zero)");
    }
    if (min >= max)
    {
        std::ostringstream oss;
        oss << "Invalid range for random numbers: min (" << min << ") >= max (" << max << ")";
        throw std::invalid_argument(oss.str());
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(min, max);
    matrix<T> result(rows, cols);
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            result(i, j) = dis(gen);
        }
    }
    return result;
}
