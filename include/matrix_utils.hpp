#pragma once

#include "matrix_core.hpp"

// Power of a matrix
template <typename T>
matrix<T> matrix<T>::pow(const int &power) const
{
    if (_rows != _cols)
    {
        throw std::invalid_argument("Matrix must be square to compute power");
    }
    matrix<T> result = eye(_rows, _cols);
    matrix<T> A = *this;
    int p = power;
    while (p > 0)
    {
        if ((p & 1) != 0) // p is odd
        {
            result *= A;
        }
        A = A * A;
        p /= 2;
    }
    return result;
}

// Exponential power of a matrix (e^matrix)
template <typename T>
matrix<double> matrix<T>::exponential_pow(int max_iter) const
{
    if (_rows != _cols)
    {
        throw std::invalid_argument("Matrix must be square to compute exponential power");
    }
    matrix<double> result = (matrix<double>)eye(_rows, _cols);
    matrix<double> term = (matrix<double>)eye(_rows, _cols);
    // use the taylor series expansion for e^A
    for (int iter = 1; iter <= max_iter; ++iter)
    {
        term = term * ((matrix<double>)(*this)) / iter;
        result = result + term;
    }
    return result;
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

// Set submatrix to a given matrix
template <typename T>
void matrix<T>::set_submatrix(int top_corner_x, int top_corner_y, const matrix<T> &submatrix)
{
    if (top_corner_x < 0 || top_corner_x >= _rows || top_corner_y < 0 || top_corner_y >= _cols ||
        top_corner_x + submatrix.rows() > _rows || top_corner_y + submatrix.cols() > _cols)
    {
        throw std::out_of_range("Submatrix indices out of range");
    }
    for (int i = 0; i < submatrix.rows(); ++i)
    {
        for (int j = 0; j < submatrix.cols(); ++j)
        {
            (*this)(top_corner_x + i, top_corner_y + j) = submatrix(i, j);
        }
    }
}

// Extract diagonal elements
template <typename T>
std::vector<T> matrix<T>::diagonal(int k) const
{
    if (_rows != _cols)
    {
        throw std::invalid_argument("Matrix must be square to extract diagonal");
    }
    if (k < -_rows + 1 || k > _cols - 1)
    {
        throw std::out_of_range("Diagonal index out of range");
    }
    std::vector<T> diag;
    diag.reserve(_rows - std::abs(k));
    for (int i = 0; i < _rows; ++i)
    {
        int j = i + k;
        if (j >= 0 && j < _cols)
        {
            diag.push_back((*this)(i, j));
        }
    }
    return diag;
}

// Extract anti-diagonal elements
template <typename T>
std::vector<T> matrix<T>::anti_diagonal(int k) const
{
    if (_rows != _cols)
    {
        throw std::invalid_argument("Matrix must be square to extract anti-diagonal");
    }
    if (k < -_rows + 1 || k > _cols - 1)
    {
        throw std::out_of_range("Anti-diagonal index out of range");
    }
    std::vector<T> anti_diag;
    anti_diag.reserve(_rows - std::abs(k));
    for (int i = 0; i < _rows; ++i)
    {
        int j = _cols - 1 - (i + k);
        if (j >= 0 && j < _cols)
        {
            anti_diag.push_back((*this)(i, j));
        }
    }
    return anti_diag;
}

// Set diagonal elements
template <typename T>
void matrix<T>::set_diagonal(const std::vector<T> &diag, int k)
{
    if (_rows != _cols || (int)diag.size() != _rows - std::abs(k))
    {
        throw std::invalid_argument("Matrix must be square and diagonal vector must match size");
    }
    if (k < -_rows + 1 || k > _cols - 1)
    {
        throw std::out_of_range("Diagonal index out of range");
    }
    for (int i = 0; i < _rows; ++i)
    {
        int j = i + k;
        if (j >= 0 && j < _cols)
        {
            (*this)(i, j) = diag[i];
        }
    }
}

// Set anti-diagonal elements
template <typename T>
void matrix<T>::set_anti_diagonal(const std::vector<T> &anti_diag, int k)
{
    if (_rows != _cols || (int)anti_diag.size() != _rows - std::abs(k))
    {
        throw std::invalid_argument("Matrix must be square and anti-diagonal vector must match size");
    }
    if (k < -_rows + 1 || k > _cols - 1)
    {
        throw std::out_of_range("Anti-diagonal index out of range");
    }
    for (int i = 0; i < _rows; ++i)
    {
        int j = _cols - 1 - (i + k);
        if (j >= 0 && j < _cols)
        {
            (*this)(i, j) = anti_diag[i];
        }
    }
}