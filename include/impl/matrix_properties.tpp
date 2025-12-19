#pragma once

// Template implementations for matrix_properties.hpp
// This file is included at the end of matrix_properties.hpp

/**
 * @brief Checks if the matrix is symmetric (A == A^T).
 * @return true if symmetric, false otherwise (non-square matrices are not
 * symmetric)
 * @details O(n²)
 */
template <typename T>
bool matrix<T>::is_symmetric() const
{
    if (_rows != _cols)
    {
        return false;
    }
    for (size_t i = 0; i < _rows; ++i)
    {
        for (size_t j = i + 1; j < _cols; ++j)
        {
            if ((*this)(i, j) != (*this)(j, i))
            {
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief Checks if the matrix is diagonal (all off-diagonal elements are zero).
 * @return true if diagonal, false otherwise (non-square matrices are not
 * diagonal)
 * @details O(n²)
 */
template <typename T>
bool matrix<T>::is_diagonal() const
{
    if (_rows != _cols)
    {
        return false;
    }
    for (size_t i = 0; i < _rows; ++i)
    {
        for (size_t j = 0; j < _cols; ++j)
        {
            if (i != j && (*this)(i, j) != 0)
            {
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief Checks if the matrix is lower triangular (all elements above the
 * diagonal are zero).
 * @return true if lower triangular, false otherwise (non-square matrices are
 * not triangular)
 * @details O(n²)
 */
template <typename T>
bool matrix<T>::is_lower_triangular() const
{
    if (_rows != _cols)
    {
        return false;
    }
    for (size_t i = 0; i < _rows; ++i)
    {
        for (size_t j = i + 1; j < _cols; ++j)
        {
            if ((*this)(i, j) != 0)
            {
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief Checks if the matrix is upper triangular (all elements below the
 * diagonal are zero).
 * @return true if upper triangular, false otherwise (non-square matrices are
 * not triangular)
 * @details O(n²)
 */
template <typename T>
bool matrix<T>::is_upper_triangular() const
{
    if (_rows != _cols)
    {
        return false;
    }
    for (size_t i = 0; i < _rows; ++i)
    {
        for (size_t j = 0; j < i; ++j)
        {
            if ((*this)(i, j) != 0)
            {
                return false;
            }
        }
    }
    return true;
}