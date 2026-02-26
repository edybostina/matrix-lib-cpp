#pragma once

// Template implementations for matrix_core.hpp
// This file is included at the end of matrix_core.hpp

#include <sstream>
#include <stdexcept>

// ============================================================================
// Constructors Implementation
// ============================================================================

/**
 * @brief Construct matrix with given dimensions, zero-initialized.
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @details Time O(m*n), Space O(m*n)
 */
template <typename T>
matrix<T>::matrix(size_t rows, size_t cols) : _rows(rows), _cols(cols), _data(rows * cols, T(0))
{
}

/**
 * @brief Construct from 2D vector.
 *
 * @param init 2D vector containing matrix data
 * @details Time O(m*n), Space O(m*n)
 */
template <typename T>
matrix<T>::matrix(const std::vector<std::vector<T>>& init)
    : _rows(init.size()), _cols(init.empty() ? 0 : init[0].size()), _data(_to_row_vector(init))
{
}

/**
 * @brief Construct from initializer list.
 *
 * @param init Nested initializer list
 * @throws std::invalid_argument If rows have different sizes
 * @details Time O(m*n), Space O(m*n)
 */
template <typename T>
matrix<T>::matrix(std::initializer_list<std::initializer_list<T>> init)
{
    _rows = init.size();
    _cols = init.begin()->size();
    _data.reserve(_rows * _cols);
    for (const auto& row : init)
    {
        if (row.size() != (size_t)_cols)
        {
            throw std::invalid_argument("All rows must have the same number of columns");
        }
        _data.insert(_data.end(), row.begin(), row.end());
    }
}

// ============================================================================
// Access Operators Implementation
// ============================================================================

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
// Unary Operators Implementation
// ============================================================================

/**
 * @brief Unary negation operator.
 *
 * @return New matrix with all elements negated
 * @details Time O(m*n), Space O(m*n)
 */
template <typename T>
matrix<T> matrix<T>::operator-() const
{
    matrix<T> result(_rows, _cols);
    for (size_t i = 0; i < size(); ++i)
    {
        result._data[i] = -_data[i];
    }
    return result;
}

/**
 * @brief Unary plus operator (identity).
 *
 * @return Copy of the matrix
 * @details Time O(m*n), Space O(m*n)
 */
template <typename T>
matrix<T> matrix<T>::operator+() const
{
    return *this;
}

// ============================================================================
// Row and Column Access - Proxy-Based Implementation
// ============================================================================

/**
 * @brief Returns a proxy object for accessing and modifying a row.
 *
 * @param index Row index
 * @return row_proxy object for the specified row
 * @throws std::out_of_range If index is out of bounds
 * @details Time O(1), Space O(1)
 */
template <typename T>
row_proxy<T> matrix<T>::row(size_t index)
{
    if (index >= _rows)
    {
        std::ostringstream oss;
        oss << "Row index " << index << " out of range [0, " << _rows << ")";
        throw std::out_of_range(oss.str());
    }
    return row_proxy<T>(*this, index);
}

/**
 * @brief Returns a const proxy object for read-only row access.
 *
 * @param index Row index
 * @return const_row_proxy object for the specified row
 * @throws std::out_of_range If index is out of bounds
 * @details Time O(1), Space O(1)
 */
template <typename T>
const_row_proxy<T> matrix<T>::row(size_t index) const
{
    if (index >= _rows)
    {
        std::ostringstream oss;
        oss << "Row index " << index << " out of range [0, " << _rows << ")";
        throw std::out_of_range(oss.str());
    }
    return const_row_proxy<T>(*this, index);
}

/**
 * @brief Returns a proxy object for accessing and modifying a column.
 *
 * @param index Column index
 * @return col_proxy object for the specified column
 * @throws std::out_of_range If index is out of bounds
 * @details Time O(1), Space O(1)
 */
template <typename T>
col_proxy<T> matrix<T>::col(size_t index)
{
    if (index >= _cols)
    {
        std::ostringstream oss;
        oss << "Column index " << index << " out of range [0, " << _cols << ")";
        throw std::out_of_range(oss.str());
    }
    return col_proxy<T>(*this, index);
}

/**
 * @brief Returns a const proxy object for read-only column access.
 *
 * @param index Column index
 * @return const_col_proxy object for the specified column
 * @throws std::out_of_range If index is out of bounds
 * @details Time O(1), Space O(1)
 */
template <typename T>
const_col_proxy<T> matrix<T>::col(size_t index) const
{
    if (index >= _cols)
    {
        std::ostringstream oss;
        oss << "Column index " << index << " out of range [0, " << _cols << ")";
        throw std::out_of_range(oss.str());
    }
    return const_col_proxy<T>(*this, index);
}

// ============================================================================
// Legacy Row and Column Access (Deprecated)
// ============================================================================

/**
 * @brief Legacy method: Extracts the i-th row as a vector.
 * @deprecated Use row() proxy instead
 */
template <typename T>
std::vector<T> matrix<T>::get_row(size_t index) const
{
    if (index >= _rows)
    {
        std::ostringstream oss;
        oss << "Row index " << index << " out of range [0, " << _rows << ")";
        throw std::out_of_range(oss.str());
    }
    std::vector<T> result(_cols);
    std::copy(_data.begin() + index * _cols, _data.begin() + (index + 1) * _cols, result.begin());
    return result;
}

/**
 * @brief Legacy method: Extracts the j-th column as a vector.
 * @deprecated Use col() proxy instead
 */
template <typename T>
std::vector<T> matrix<T>::get_col(size_t index) const
{
    if (index >= _cols)
    {
        std::ostringstream oss;
        oss << "Column index " << index << " out of range [0, " << _cols << ")";
        throw std::out_of_range(oss.str());
    }
    std::vector<T> result(_rows);
    for (size_t i = 0; i < _rows; ++i)
    {
        result[i] = _data[_index(i, index)];
    }
    return result;
}

/**
 * @brief Legacy method: Sets the i-th row from a vector.
 * @deprecated Use row() = vec instead
 */
template <typename T>
void matrix<T>::set_row(size_t index, const std::vector<T>& values)
{
    if (index >= _rows)
    {
        std::ostringstream oss;
        oss << "Row index " << index << " out of range [0, " << _rows << ")";
        throw std::out_of_range(oss.str());
    }
    if (values.size() != _cols)
    {
        std::ostringstream oss;
        oss << "Vector size " << values.size() << " doesn't match column count " << _cols;
        throw std::invalid_argument(oss.str());
    }
    std::copy(values.begin(), values.end(), _data.begin() + index * _cols);
}

/**
 * @brief Legacy method: Sets the j-th column from a vector.
 * @deprecated Use col() = vec instead
 */
template <typename T>
void matrix<T>::set_col(size_t index, const std::vector<T>& values)
{
    if (index >= _cols)
    {
        std::ostringstream oss;
        oss << "Column index " << index << " out of range [0, " << _cols << ")";
        throw std::out_of_range(oss.str());
    }
    if (values.size() != _rows)
    {
        std::ostringstream oss;
        oss << "Vector size " << values.size() << " doesn't match row count " << _rows;
        throw std::invalid_argument(oss.str());
    }
    for (size_t i = 0; i < _rows; ++i)
    {
        _data[_index(i, index)] = values[i];
    }
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
