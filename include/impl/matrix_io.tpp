#pragma once

// Template implementations for matrix_io.hpp
// This file is included at the end of matrix_io.hpp

#include <stdexcept>

/**
 * @brief Outputs matrix to stream in readable format.
 *
 * @param os Output stream
 * @param m Matrix to output
 * @return Reference to output stream
 * @details Time O(m*n), Space O(1)
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const matrix<T>& m)
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

/**
 * @brief Reads matrix from input stream.
 *
 * Expects format: rows cols followed by matrix elements row by row.
 *
 * @param is Input stream
 * @param m Matrix to populate
 * @return Reference to input stream
 * @details Time O(m*n), Space O(m*n)
 */
template <typename T>
std::istream& operator>>(std::istream& is, matrix<T>& m)
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