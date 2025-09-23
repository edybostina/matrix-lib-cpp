#pragma once
#include "matrix_core.hpp"


// Check if the matrix is symmetric
template <typename T>
bool matrix<T>::is_symmetric() const
{
    if (_rows != _cols)
    {
        return false;
    }
    for (int i = 0; i < _rows; ++i)
    {
        for (int j = i + 1; j < _cols; ++j)
        {
            if ((*this)(i, j) != (*this)(j, i))
            {
                return false;
            }
        }
    }
    return true;
}

// Check if the matrix is diagonal
template <typename T>
bool matrix<T>::is_diagonal() const
{
    if (_rows != _cols)
    {
        return false;
    }
    for (int i = 0; i < _rows; ++i)
    {
        for (int j = 0; j < _cols; ++j)
        {
            if (i != j && (*this)(i, j) != 0)
            {
                return false;
            }
        }
    }
    return true;
}

// Check if the matrix is lower triangular
template <typename T>
bool matrix<T>::is_lower_triangular() const
{
    if (_rows != _cols)
    {
        return false;
    }
    for (int i = 0; i < _rows; ++i)
    {
        for (int j = i + 1; j < _cols; ++j)
        {
            if ((*this)(i, j) != 0)
            {
                return false;
            }
        }
    }
    return true;
}

// Check if the matrix is upper triangular
template <typename T>
bool matrix<T>::is_upper_triangular() const
{
    if (_rows != _cols)
    {
        return false;
    }
    for (int i = 0; i < _rows; ++i)
    {
        for (int j = 0; j < i; ++j)
        {
            if ((*this)(i, j) != 0)
            {
                return false;
            }
        }
    }
    return true;
}