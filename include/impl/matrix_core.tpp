#pragma once

// Template implementations for matrix_core.hpp
// This file is included at the end of matrix_core.hpp

#include <sstream>
#include <stdexcept>

// ============================================================================
// Access Operators Implementation
// ============================================================================

/**
 * @brief Returns the i-th row as a vector.
 *
 * @param index Row index
 * @return Reference to row vector
 * @throws std::out_of_range If index is out of bounds
 * @details Time O(n), Space O(n)
 */
template <typename T>
std::vector<T>& matrix<T>::operator()(size_t index)
{
    if (index >= _rows)
    {
        std::ostringstream oss;
        oss << "Row index " << index << " out of range [0, " << _rows << ")";
        throw std::out_of_range(oss.str());
    }
    std::vector<T>& row = _data;
    row.resize(_cols);
    std::copy(_data.begin() + index * _cols, _data.begin() + (index + 1) * _cols, row.begin());
    return row;
}

/**
 * @brief Returns the i-th row as a const vector.
 *
 * @param index Row index
 * @return Const reference to row vector
 * @throws std::out_of_range If index is out of bounds
 * @details Time O(n), Space O(n)
 */
template <typename T>
const std::vector<T>& matrix<T>::operator()(size_t index) const
{
    if (index >= _rows)
    {
        std::ostringstream oss;
        oss << "Row index " << index << " out of range [0, " << _rows << ")";
        throw std::out_of_range(oss.str());
    }
    std::vector<T>& row = const_cast<std::vector<T>&>(_data);
    row.resize(_cols);
    std::copy(_data.begin() + index * _cols, _data.begin() + (index + 1) * _cols, row.begin());
    return row;
}

/**
 * @brief Returns reference to element at (row, col).
 *
 * @param row Row index
 * @param col Column index
 * @return Reference to element
 * @throws std::out_of_range If indices are out of bounds
 * @details Time O(1), Space O(1)
 */
template <typename T>
T& matrix<T>::operator()(size_t row, size_t col)
{
    if (row >= _rows || col >= _cols)
    {
        std::ostringstream oss;
        oss << "Index (" << row << ", " << col << ") out of range for matrix of size " << _rows << "x" << _cols;
        throw std::out_of_range(oss.str());
    }
    return _data[_index(row, col)];
}

/**
 * @brief Returns const reference to element at (row, col).
 *
 * @param row Row index
 * @param col Column index
 * @return Const reference to element
 * @throws std::out_of_range If indices are out of bounds
 * @details Time O(1), Space O(1)
 */
template <typename T>
const T& matrix<T>::operator()(size_t row, size_t col) const
{
    if (row >= _rows || col >= _cols)
    {
        std::ostringstream oss;
        oss << "Index (" << row << ", " << col << ") out of range for matrix of size " << _rows << "x" << _cols;
        throw std::out_of_range(oss.str());
    }
    return _data[_index(row, col)];
}

// ============================================================================
// Row and Column Access
// ============================================================================

/**
 * @brief Extracts the i-th row as a 1xn matrix.
 *
 * @param index Row index
 * @return 1xn matrix containing the row
 * @throws std::out_of_range If index is out of bounds
 * @details Time O(n), Space O(n)
 */
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

/**
 * @brief Extracts the j-th column as an mx1 matrix.
 *
 * @param index Column index
 * @return mx1 matrix containing the column
 * @throws std::out_of_range If index is out of bounds
 * @details Time O(m), Space O(m)
 */
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

/**
 * @brief Creates a matrix filled with zeros.
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Matrix of zeros
 * @throws std::invalid_argument If dimensions are zero
 * @details Time O(m*n), Space O(m*n)
 */
template <typename T>
matrix<T> matrix<T>::zeros(size_t rows, size_t cols)
{
    if (rows == 0 || cols == 0)
    {
        throw std::invalid_argument("Matrix dimensions must be positive (non-zero)");
    }
    return matrix<T>(std::vector<std::vector<T>>(rows, std::vector<T>(cols, T(0))));
}

/**
 * @brief Creates a matrix filled with ones.
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Matrix of ones
 * @throws std::invalid_argument If dimensions are zero
 * @details Time O(m*n), Space O(m*n)
 */
template <typename T>
matrix<T> matrix<T>::ones(size_t rows, size_t cols)
{
    if (rows == 0 || cols == 0)
    {
        throw std::invalid_argument("Matrix dimensions must be positive (non-zero)");
    }
    return matrix<T>(std::vector<std::vector<T>>(rows, std::vector<T>(cols, T(1))));
}

/**
 * @brief Creates an identity matrix (1s on diagonal, 0s elsewhere).
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Identity matrix
 * @throws std::invalid_argument If dimensions are zero
 * @details Time O(min(m,n)), Space O(m*n)
 */
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

/**
 * @brief Creates a matrix with random values in specified range.
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @param min Minimum value (inclusive)
 * @param max Maximum value (exclusive for floats, inclusive for integers)
 * @return Matrix with random values
 * @throws std::invalid_argument If dimensions are zero or min >= max
 * @details Time O(m*n), Space O(m*n). Uses thread-local Mersenne Twister for
 *          performance and thread safety. Automatically selects int or float
 *          distribution based on template type.
 */
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

    thread_local std::mt19937 gen(std::random_device{}());

    matrix<T> result(rows, cols);

    if constexpr (std::is_integral_v<T>)
    {
        std::uniform_int_distribution<T> dis(min, max);
        for (size_t i = 0; i < rows * cols; ++i)
        {
            result._data[i] = dis(gen);
        }
    }
    else
    {
        std::uniform_real_distribution<T> dis(min, max);
        for (size_t i = 0; i < rows * cols; ++i)
        {
            result._data[i] = dis(gen);
        }
    }

    return result;
}
