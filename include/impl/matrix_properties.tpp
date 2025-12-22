#pragma once

// Template implementations for matrix_properties.hpp
// This file is included at the end of matrix_properties.hpp

/**
 * @brief Checks if the matrix is symmetric (A == A^T).
 * @return true if symmetric, false otherwise (non-square matrices are not
 * symmetric)
 * @details O(n²)
 * @note Uses tolerance-based comparison for floating-point types
 */
template <typename T>
bool matrix<T>::is_symmetric() const
{
    if (_rows != _cols)
    {
        return false;
    }

    // Use tolerance for floating-point comparison
    constexpr double tolerance = 1e-10;

    for (size_t i = 0; i < _rows; ++i)
    {
        for (size_t j = i + 1; j < _cols; ++j)
        {
            T diff = (*this)(i, j) - (*this)(j, i);
            T abs_diff = diff < 0 ? -diff : diff;

            if constexpr (std::is_floating_point_v<T>)
            {
                if (abs_diff > tolerance)
                {
                    return false;
                }
            }
            else
            {
                if (diff != 0)
                {
                    return false;
                }
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
 * @note Uses tolerance-based comparison for floating-point types
 */
template <typename T>
bool matrix<T>::is_diagonal() const
{
    if (_rows != _cols)
    {
        return false;
    }

    constexpr double tolerance = 1e-10;

    for (size_t i = 0; i < _rows; ++i)
    {
        for (size_t j = 0; j < _cols; ++j)
        {
            if (i != j)
            {
                T value = (*this)(i, j);
                T abs_value = value < 0 ? -value : value;

                if constexpr (std::is_floating_point_v<T>)
                {
                    if (abs_value > tolerance)
                    {
                        return false;
                    }
                }
                else
                {
                    if (value != 0)
                    {
                        return false;
                    }
                }
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
 * @note Uses tolerance-based comparison for floating-point types
 */
template <typename T>
bool matrix<T>::is_lower_triangular() const
{
    if (_rows != _cols)
    {
        return false;
    }

    constexpr double tolerance = 1e-10;

    for (size_t i = 0; i < _rows; ++i)
    {
        for (size_t j = i + 1; j < _cols; ++j)
        {
            T value = (*this)(i, j);
            T abs_value = value < 0 ? -value : value;

            if constexpr (std::is_floating_point_v<T>)
            {
                if (abs_value > tolerance)
                {
                    return false;
                }
            }
            else
            {
                if (value != 0)
                {
                    return false;
                }
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
 * @note Uses tolerance-based comparison for floating-point types
 */
template <typename T>
bool matrix<T>::is_upper_triangular() const
{
    if (_rows != _cols)
    {
        return false;
    }

    constexpr double tolerance = 1e-10;

    for (size_t i = 0; i < _rows; ++i)
    {
        for (size_t j = 0; j < i; ++j)
        {
            T value = (*this)(i, j);
            T abs_value = value < 0 ? -value : value;

            if constexpr (std::is_floating_point_v<T>)
            {
                if (abs_value > tolerance)
                {
                    return false;
                }
            }
            else
            {
                if (value != 0)
                {
                    return false;
                }
            }
        }
    }
    return true;
}

/**
 * @brief Checks if the matrix is orthogonal (A * A^T == I).
 * @return true if orthogonal, false otherwise (non-square matrices are not
 * orthogonal)
 * @details O(n³)
 */
template <typename T>
bool matrix<T>::is_orthogonal(double tolerance) const
{
    if (_rows != _cols)
    {
        return false;
    }
    matrix<T> product = (*this) * this->transpose();
    matrix<T> identity = matrix<T>::eye(_rows, _cols);
    for (size_t i = 0; i < _rows; ++i)
    {
        for (size_t j = 0; j < _cols; ++j)
        {
            T absolute = product(i, j) - identity(i, j);
            absolute = absolute < 0 ? -absolute : absolute;
            if (absolute > tolerance)
            {
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief Checks if the matrix is singular (determinant is zero).
 * @return true if singular, false otherwise (non-square matrices are not
 * singular)
 * @details O(n³)
 */
template <typename T>
bool matrix<T>::is_singular(double tolerance) const
{
    if (_rows != _cols)
    {
        return false;
    }
    double det = this->determinant();
    return std::abs(det) < tolerance;
}

/**
 * @brief Checks if the matrix is idempotent (A * A == A).
 * @return true if idempotent, false otherwise (non-square matrices are not
 * idempotent)
 * @details O(n³)
 */
template <typename T>
bool matrix<T>::is_idempotent(double tolerance) const
{
    if (_rows != _cols)
    {
        return false;
    }
    matrix<T> squared = (*this) * (*this);
    for (size_t i = 0; i < _rows; ++i)
    {
        for (size_t j = 0; j < _cols; ++j)
        {
            T absolute = squared(i, j) - (*this)(i, j);
            absolute = absolute < 0 ? -absolute : absolute;
            if (absolute > tolerance)
            {
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief Checks if the matrix is nilpotent (A^k == 0 for some k).
 * @param k Exponent to test
 * @return true if nilpotent, false otherwise (non-square matrices are not
 * nilpotent)
 * @details O(k*n³)
 */
template <typename T>
bool matrix<T>::is_nilpotent(size_t k, double tolerance) const
{
    if (_rows != _cols)
    {
        return false;
    }
    matrix<T> power = (*this);
    for (size_t exp = 1; exp < k; ++exp)
    {
        power = power * (*this);
    }
    for (size_t i = 0; i < _rows; ++i)
    {
        for (size_t j = 0; j < _cols; ++j)
        {
            T absolute = power(i, j);
            absolute = absolute < 0 ? -absolute : absolute;
            if (absolute > tolerance)
            {
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief Checks if the matrix is involutory (A * A == I).
 * @return true if involutory, false otherwise (non-square matrices are not
 * involutory)
 * @details O(n³)
 */
template <typename T>
bool matrix<T>::is_involutory(double tolerance) const
{
    if (_rows != _cols)
    {
        return false;
    }

    matrix<T> squared = (*this) * (*this);
    matrix<T> identity = matrix<T>::eye(_rows, _cols);
    for (size_t i = 0; i < _rows; ++i)
    {
        for (size_t j = 0; j < _cols; ++j)
        {
            T absolute = squared(i, j) - identity(i, j);
            absolute = absolute < 0 ? -absolute : absolute;
            if (absolute > tolerance)
            {
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief Checks if the matrix is positive definite (all eigenvalues > 0).
 *
 * Uses Sylvester's criterion: A symmetric matrix is positive definite if and only if
 * all leading principal minors are positive. For non-symmetric matrices, this attempts
 * to verify using Cholesky-like approach for efficiency.
 *
 * @return true if positive definite, false otherwise (non-square matrices are not
 * positive definite)
 * @details O(n³) using Cholesky-inspired approach
 * @note For accurate results, matrix should be symmetric. Non-symmetric matrices may
 *       give false positives/negatives.
 */
template <typename T>
bool matrix<T>::is_positive_definite() const
{
    if (_rows != _cols)
    {
        return false;
    }

    if (!this->is_symmetric())
    {
        return false;
    }

    matrix<double> A = static_cast<matrix<double>>(*this);

    for (size_t k = 0; k < _rows; ++k)
    {
        if (A(k, k) <= 0)
        {
            return false;
        }

        double sqrt_akk = std::sqrt(A(k, k));

        for (size_t i = k + 1; i < _rows; ++i)
        {
            A(i, k) /= sqrt_akk;

            for (size_t j = k + 1; j <= i; ++j)
            {
                A(i, j) -= A(i, k) * A(j, k);
            }
        }
    }

    return true;
}

/**
 * @brief Checks if the matrix is negative definite (all eigenvalues < 0).
 *
 * A matrix is negative definite if -A is positive definite.
 *
 * @return true if negative definite, false otherwise (non-square matrices are not
 * negative definite)
 * @details O(n³)
 * @note For accurate results, matrix should be symmetric.
 */
template <typename T>
bool matrix<T>::is_negative_definite() const
{
    if (_rows != _cols)
    {
        return false;
    }

    matrix<T> neg_A(this->_rows, this->_cols);
    for (size_t i = 0; i < _rows; ++i)
    {
        for (size_t j = 0; j < _cols; ++j)
        {
            neg_A(i, j) = -(*this)(i, j);
        }
    }

    return neg_A.is_positive_definite();
}