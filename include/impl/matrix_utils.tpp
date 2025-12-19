#pragma once

// Template implementations for matrix_utils.hpp
// This file is included at the end of matrix_utils.hpp

#include "matrix_core.hpp"

#include <sstream>

/**
 * @brief Computes matrix power A^p using binary exponentiation.
 * @param power The exponent (must be non-negative)
 * @return Matrix raised to the given power
 * @throws std::invalid_argument if matrix is not square
 * @details O(n³ log p) where n is matrix dimension and p is the power
 */
template <typename T>
matrix<T> matrix<T>::pow(const int& power) const
{
    if (_rows != _cols)
    {
        std::ostringstream oss;
        oss << "Matrix must be square to compute power, but got " << _rows << "x" << _cols;
        throw std::invalid_argument(oss.str());
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

/**
 * @brief Computes matrix exponential e^A using Taylor series expansion.
 * @param max_iter Maximum number of iterations for series convergence (default varies)
 * @return Matrix exponential as double-precision matrix
 * @throws std::invalid_argument if matrix is not square
 * @details O(n³ × max_iter) where n is matrix dimension
 */
template <typename T>
matrix<double> matrix<T>::exponential_pow(int max_iter) const
{
    if (_rows != _cols)
    {
        std::ostringstream oss;
        oss << "Matrix must be square to compute exponential power, but got " << _rows << "x" << _cols;
        throw std::invalid_argument(oss.str());
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

/**
 * @brief Swaps two rows in-place using XOR swap algorithm.
 * @param row1 Index of first row
 * @param row2 Index of second row
 * @throws std::out_of_range if row indices are out of bounds
 * @details O(n) where n is number of columns
 */
template <typename T>
void matrix<T>::swapRows(size_t row1, size_t row2)
{
    if (row1 >= _rows || row2 >= _rows)
    {
        std::ostringstream oss;
        oss << "Row index " << (row1 >= _rows ? row1 : row2) << " out of range [0, " << _rows << ")";
        throw std::out_of_range(oss.str());
    }
    for (size_t j = 0; j < _cols; ++j)
    {
        _data[_index(row1, j)] = _data[_index(row1, j)] + _data[_index(row2, j)];
        _data[_index(row2, j)] = _data[_index(row1, j)] - _data[_index(row2, j)];
        _data[_index(row1, j)] = _data[_index(row1, j)] - _data[_index(row2, j)];
    }
}
/**
 * @brief Swaps two columns in-place using XOR swap algorithm.
 * @param col1 Index of first column
 * @param col2 Index of second column
 * @throws std::out_of_range if column indices are out of bounds
 * @details O(m) where m is number of rows
 */
template <typename T>
void matrix<T>::swapCols(size_t col1, size_t col2)
{
    if (col1 >= _cols || col2 >= _cols)
    {
        std::ostringstream oss;
        oss << "Column index " << (col1 >= _cols ? col1 : col2) << " out of range [0, " << _cols << ")";
        throw std::out_of_range(oss.str());
    }
    for (size_t i = 0; i < _rows; ++i)
    {
        _data[_index(i, col1)] = _data[_index(i, col1)] + _data[_index(i, col2)];
        _data[_index(i, col2)] = _data[_index(i, col1)] - _data[_index(i, col2)];
        _data[_index(i, col1)] = _data[_index(i, col1)] - _data[_index(i, col2)];
    }
}
/**
 * @brief Resizes the matrix, preserving existing elements where possible.
 * @param rows New number of rows
 * @param cols New number of columns
 * @note New elements are zero-initialized; truncated elements are discarded
 * @details O(mn) where m,n are min(old,new) dimensions
 */
template <typename T>
void matrix<T>::resize(size_t rows, size_t cols)
{
    if (rows == _rows && cols == _cols)
    {
        return; // No change needed
    }
    std::vector<T> new_data(rows * cols);
    for (size_t i = 0; i < std::min(_rows, rows); ++i)
    {
        for (size_t j = 0; j < std::min(_cols, cols); ++j)
        {
            new_data[i * cols + j] = (*this)(i, j);
        }
    }
    _data = std::move(new_data);
    _rows = rows;
    _cols = cols;
}

/**
 * @brief Extracts a rectangular submatrix from this matrix.
 * @param top_corner_x Top-left row index (inclusive)
 * @param top_corner_y Top-left column index (inclusive)
 * @param bottom_corner_x Bottom-right row index (inclusive)
 * @param bottom_corner_y Bottom-right column index (inclusive)
 * @return New matrix containing the specified submatrix
 * @throws std::out_of_range if indices are out of bounds
 * @throws std::invalid_argument if top corner > bottom corner
 * @details O(mn) where m,n are submatrix dimensions
 */
template <typename T>
matrix<T> matrix<T>::submatrix(size_t top_corner_x, size_t top_corner_y, size_t bottom_corner_x,
                               size_t bottom_corner_y) const
{
    if (top_corner_x >= _rows || bottom_corner_x >= _rows || top_corner_y >= _cols || bottom_corner_y >= _cols)
    {
        std::ostringstream oss;
        oss << "Submatrix indices out of range: matrix is " << _rows << "x" << _cols << ", requested corners ("
            << top_corner_x << "," << top_corner_y << ") to (" << bottom_corner_x << "," << bottom_corner_y << ")";
        throw std::out_of_range(oss.str());
    }
    if (top_corner_x > bottom_corner_x || top_corner_y > bottom_corner_y)
    {
        std::ostringstream oss;
        oss << "Invalid submatrix indices: top corner (" << top_corner_x << "," << top_corner_y
            << ") must be <= bottom corner (" << bottom_corner_x << "," << bottom_corner_y << ")";
        throw std::invalid_argument(oss.str());
    }
    matrix<T> result(bottom_corner_x - top_corner_x + 1, bottom_corner_y - top_corner_y + 1);
    for (size_t i = top_corner_x; i <= bottom_corner_x; ++i)
    {
        for (size_t j = top_corner_y; j <= bottom_corner_y; ++j)
        {
            result(i - top_corner_x, j - top_corner_y) = (*this)(i, j);
        }
    }
    return result;
}

/**
 * @brief Copies a submatrix into this matrix at the specified position.
 * @param top_corner_x Row index where submatrix placement begins
 * @param top_corner_y Column index where submatrix placement begins
 * @param submatrix The matrix to copy into this matrix
 * @throws std::out_of_range if submatrix extends beyond matrix bounds
 * @details O(mn) where m,n are submatrix dimensions
 */
template <typename T>
void matrix<T>::set_submatrix(size_t top_corner_x, size_t top_corner_y, const matrix<T>& submatrix)
{
    if (top_corner_x >= _rows || top_corner_y >= _cols || top_corner_x + submatrix.rows() > _rows ||
        top_corner_y + submatrix.cols() > _cols)
    {
        std::ostringstream oss;
        oss << "Submatrix placement out of range: matrix is " << _rows << "x" << _cols << ", submatrix is "
            << submatrix.rows() << "x" << submatrix.cols() << ", placement at (" << top_corner_x << "," << top_corner_y
            << ")";
        throw std::out_of_range(oss.str());
    }
    for (size_t i = 0; i < submatrix.rows(); ++i)
    {
        for (size_t j = 0; j < submatrix.cols(); ++j)
        {
            (*this)(top_corner_x + i, top_corner_y + j) = submatrix(i, j);
        }
    }
}

/**
 * @brief Extracts diagonal elements from the matrix.
 * @param k Diagonal offset (0=main, >0=upper, <0=lower)
 * @return Vector containing diagonal elements
 * @throws std::invalid_argument if matrix is not square
 * @throws std::out_of_range if k is out of valid range
 * @details O(n) where n is matrix dimension
 */
template <typename T>
std::vector<T> matrix<T>::diagonal(int k) const
{
    if (_rows != _cols)
    {
        std::ostringstream oss;
        oss << "Matrix must be square to extract diagonal, but got " << _rows << "x" << _cols;
        throw std::invalid_argument(oss.str());
    }
    if (k < -static_cast<int>(_rows) + 1 || k > static_cast<int>(_cols) - 1)
    {
        std::ostringstream oss;
        oss << "Diagonal index " << k << " out of range [" << -static_cast<int>(_rows) + 1 << ", "
            << static_cast<int>(_cols) - 1 << "]";
        throw std::out_of_range(oss.str());
    }
    std::vector<T> diag;
    diag.reserve(_rows - std::abs(k));
    for (size_t i = 0; i < _rows; ++i)
    {
        int j = static_cast<int>(i) + k;
        if (j >= 0 && j < static_cast<int>(_cols))
        {
            diag.push_back((*this)(i, j));
        }
    }
    return diag;
}

/**
 * @brief Extracts anti-diagonal elements from the matrix.
 * @param k Anti-diagonal offset (0=main, >0=upper, <0=lower)
 * @return Vector containing anti-diagonal elements
 * @throws std::invalid_argument if matrix is not square
 * @throws std::out_of_range if k is out of valid range
 * @details O(n) where n is matrix dimension
 */
template <typename T>
std::vector<T> matrix<T>::anti_diagonal(int k) const
{
    if (_rows != _cols)
    {
        std::ostringstream oss;
        oss << "Matrix must be square to extract anti-diagonal, but got " << _rows << "x" << _cols;
        throw std::invalid_argument(oss.str());
    }
    if (k < -static_cast<int>(_rows) + 1 || k > static_cast<int>(_cols) - 1)
    {
        std::ostringstream oss;
        oss << "Anti-diagonal index " << k << " out of range [" << -static_cast<int>(_rows) + 1 << ", "
            << static_cast<int>(_cols) - 1 << "]";
        throw std::out_of_range(oss.str());
    }
    std::vector<T> anti_diag;
    anti_diag.reserve(_rows - std::abs(k));
    for (size_t i = 0; i < _rows; ++i)
    {
        int j = static_cast<int>(_cols) - 1 - (static_cast<int>(i) + k);
        if (j >= 0 && j < static_cast<int>(_cols))
        {
            anti_diag.push_back((*this)(i, j));
        }
    }
    return anti_diag;
}

/**
 * @brief Sets diagonal elements in the matrix.
 * @param diag Vector of values to set on the diagonal
 * @param k Diagonal offset (0=main, >0=upper, <0=lower)
 * @throws std::invalid_argument if matrix is not square or vector size mismatch
 * @throws std::out_of_range if k is out of valid range
 * @details O(n) where n is matrix dimension
 */
template <typename T>
void matrix<T>::set_diagonal(const std::vector<T>& diag, int k)
{
    if (_rows != _cols)
    {
        std::ostringstream oss;
        oss << "Matrix must be square to set diagonal, but got " << _rows << "x" << _cols;
        throw std::invalid_argument(oss.str());
    }
    size_t expected_size = _rows - std::abs(k);
    if (diag.size() != expected_size)
    {
        std::ostringstream oss;
        oss << "Diagonal vector size mismatch: expected " << expected_size << ", got " << diag.size();
        throw std::invalid_argument(oss.str());
    }
    if (k < -static_cast<int>(_rows) + 1 || k > static_cast<int>(_cols) - 1)
    {
        std::ostringstream oss;
        oss << "Diagonal index " << k << " out of range [" << -static_cast<int>(_rows) + 1 << ", "
            << static_cast<int>(_cols) - 1 << "]";
        throw std::out_of_range(oss.str());
    }
    for (size_t i = 0; i < _rows; ++i)
    {
        int j = i + k;
        if (j >= 0 && j < static_cast<int>(_cols))
        {
            (*this)(i, j) = diag[i];
        }
    }
}

/**
 * @brief Sets anti-diagonal elements in the matrix.
 * @param anti_diag Vector of values to set on the anti-diagonal
 * @param k Anti-diagonal offset (0=main, >0=upper, <0=lower)
 * @throws std::invalid_argument if matrix is not square or vector size mismatch
 * @throws std::out_of_range if k is out of valid range
 * @details O(n) where n is matrix dimension
 */
template <typename T>
void matrix<T>::set_anti_diagonal(const std::vector<T>& anti_diag, int k)
{
    if (_rows != _cols)
    {
        std::ostringstream oss;
        oss << "Matrix must be square to set anti-diagonal, but got " << _rows << "x" << _cols;
        throw std::invalid_argument(oss.str());
    }
    size_t expected_size = _rows - std::abs(k);
    if (anti_diag.size() != expected_size)
    {
        std::ostringstream oss;
        oss << "Anti-diagonal vector size mismatch: expected " << expected_size << ", got " << anti_diag.size();
        throw std::invalid_argument(oss.str());
    }
    if (k < -static_cast<int>(_rows) + 1 || k > static_cast<int>(_cols) - 1)
    {
        std::ostringstream oss;
        oss << "Anti-diagonal index " << k << " out of range [" << -static_cast<int>(_rows) + 1 << ", "
            << static_cast<int>(_cols) - 1 << "]";
        throw std::out_of_range(oss.str());
    }
    for (size_t i = 0; i < _rows; ++i)
    {
        int j = _cols - 1 - (static_cast<int>(i) + k);
        if (j >= 0 && j < static_cast<int>(_cols))
        {
            (*this)(i, j) = anti_diag[i];
        }
    }
}