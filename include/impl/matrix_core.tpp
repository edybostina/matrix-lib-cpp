#pragma once

// Template implementations for matrix_core.hpp
// This file is included at the end of matrix_core.hpp

// ============================================================================
// Access Operators Implementation
// ============================================================================

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

// ============================================================================
// Row and Column Access
// ============================================================================

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

// ============================================================================
// Factory Methods Implementation
// ============================================================================

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
