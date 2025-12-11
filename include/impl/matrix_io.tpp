#pragma once

// Template implementations for matrix_io.hpp
// This file is included at the end of matrix_io.hpp

#include <stdexcept>

template <typename T>
std::ostream &operator<<(std::ostream &os, const matrix<T> &m)
{
    for (size_t i = 0; i < m.rows(); ++i)
    {
        for (size_t j = 0; j < m.cols(); ++j)
        {
            os << m(i, j) << " ";
        }
        os << "\n";
    }
    return os;
}
template <typename T>
std::istream &operator>>(std::istream &is, matrix<T> &m)
{
    size_t rows, cols;
    is >> rows >> cols;
    m = matrix<T>(rows, cols);
    for (size_t i = 0; i < m.rows(); ++i)
    {
        for (size_t j = 0; j < m.cols(); ++j)
        {
            is >> m(i, j);
        }
    }
    return is;
}