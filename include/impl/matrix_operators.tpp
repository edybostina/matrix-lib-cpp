#pragma once

// Template implementations for matrix_operators.hpp
// This file is included at the end of matrix_operators.hpp

// ============================================================================
// Matrix-Matrix Arithmetic Operations
// ============================================================================

// Matrix addition
template <typename T>
matrix<T> matrix<T>::operator+(const matrix<T> &other) const
{
    if (_rows != other._rows || _cols != other._cols)
    {
        throw std::invalid_argument("Matrix dimensions do not match for addition");
    }

    matrix<T> result(_rows, _cols);

    // Number of threads to use
    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](int start_row, int end_row)
    {
        for (int i = start_row; i < end_row; ++i)
        {
            for (int j = 0; j < _cols; ++j)
            {
                result(i, j) = (*this)(i, j) + other(i, j);
            }
        }
    };

    int rows_per_thread = _rows / num_threads;
    int leftover = _rows % num_threads;

    int start = 0;
    for (int t = 0; t < num_threads; ++t)
    {
        int end = start + rows_per_thread + (t < leftover ? 1 : 0);
        threads[t] = std::thread(worker, start, end);
        start = end;
    }

    for (auto &t : threads)
    {
        if (t.joinable())
            t.join();
    }

    return result;
}

// Matrix subtraction
template <typename T>
matrix<T> matrix<T>::operator-(const matrix<T> &other) const
{
    if (_rows != other._rows || _cols != other._cols)
    {
        throw std::invalid_argument("Matrix dimensions do not match for addition");
    }

    matrix<T> result(_rows, _cols);

    // Number of threads to use
    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](int start_row, int end_row)
    {
        for (int i = start_row; i < end_row; ++i)
        {
            for (int j = 0; j < _cols; ++j)
            {
                result(i, j) = (*this)(i, j) - other(i, j);
            }
        }
    };

    int rows_per_thread = _rows / num_threads;
    int leftover = _rows % num_threads;

    int start = 0;
    for (int t = 0; t < num_threads; ++t)
    {
        int end = start + rows_per_thread + (t < leftover ? 1 : 0);
        threads[t] = std::thread(worker, start, end);
        start = end;
    }

    for (auto &t : threads)
    {
        if (t.joinable())
            t.join();
    }

    return result;
}

// Matrix multiplication
template <typename T>
matrix<T> matrix<T>::operator*(const matrix<T> &other) const
{
    if (_cols != other._rows)
    {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    }

    matrix<T> result(_rows, other._cols);
    auto worker = [&](int row_start, int row_end)
    {
        for (int i = row_start; i < row_end; ++i)
        {
            for (int j = 0; j < other._cols; ++j)
            {
                T sum = 0;
                for (int k = 0; k < _cols; ++k)
                {
                    sum += (*this)(i, k) * other(k, j);
                }
                result(i, j) = sum;
            }
        }
    };

    int num_threads = std::thread::hardware_concurrency();
    int chunk_size = (_rows + num_threads - 1) / num_threads;

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t)
    {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, _rows);
        if (start < end)
        {
            threads.emplace_back(worker, start, end);
        }
    }

    for (auto &t : threads)
    {
        t.join();
    }

    return result;
}

// Matrix addition assignment
template <typename T>
matrix<T> matrix<T>::operator+=(const matrix<T> &other)
{
    if (_rows != other._rows || _cols != other._cols)
    {
        throw std::invalid_argument("Matrix dimensions do not match for addition assignment");
    }

    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](int start_row, int end_row)
    {
        for (int i = start_row; i < end_row; ++i)
        {
            for (int j = 0; j < _cols; ++j)
            {
                (*this)(i, j) += other(i, j);
            }
        }
    };

    int rows_per_thread = _rows / num_threads;
    int leftover = _rows % num_threads;

    int start = 0;
    for (int t = 0; t < num_threads; ++t)
    {
        int end = start + rows_per_thread + (t < leftover ? 1 : 0);
        threads[t] = std::thread(worker, start, end);
        start = end;
    }

    for (auto &t : threads)
    {
        if (t.joinable())
            t.join();
    }

    return *this;
}
// Matrix subtraction assignment
template <typename T>
matrix<T> matrix<T>::operator-=(const matrix<T> &other)
{
    if (_rows != other._rows || _cols != other._cols)
    {
        throw std::invalid_argument("Matrix dimensions do not match for subtraction assignment");
    }
    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](int start_row, int end_row)
    {
        for (int i = start_row; i < end_row; ++i)
        {
            for (int j = 0; j < _cols; ++j)
            {
                (*this)(i, j) -= other(i, j);
            }
        }
    };

    int rows_per_thread = _rows / num_threads;
    int leftover = _rows % num_threads;

    int start = 0;
    for (int t = 0; t < num_threads; ++t)
    {
        int end = start + rows_per_thread + (t < leftover ? 1 : 0);
        threads[t] = std::thread(worker, start, end);
        start = end;
    }

    for (auto &t : threads)
    {
        if (t.joinable())
            t.join();
    }

    return *this;
}
// Matrix multiplication assignment
template <typename T>
matrix<T> matrix<T>::operator*=(const matrix<T> &other)
{
    if (_cols != other._rows)
    {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication assignment");
    }
    matrix<T> result = (*this) * other;
    *this = result;
    return *this;
}
// Matrix equality operator
template <typename T>
bool matrix<T>::operator==(const matrix<T> &other) const noexcept
{
    if (_rows != other._rows || _cols != other._cols)
    {
        return false;
    }
    for (int i = 0; i < _rows; ++i)
    {
        for (int j = 0; j < _cols; ++j)
        {
            if ((*this)(i, j) != other(i, j))
            {
                return false;
            }
        }
    }
    return true;
}
// Matrix inequality operator
template <typename T>
bool matrix<T>::operator!=(const matrix<T> &other) const noexcept
{
    return !(*this == other);
}

// Hadamard product
template <typename T>
matrix<T> matrix<T>::hadamard(const matrix<T> &other) const
{
    if (_rows != other.rows() || _cols != other.cols())
    {
        throw std::invalid_argument("Matrix dimensions do not match for the Hadamard product");
    }
    matrix<T> result(_rows, _cols);

    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](int start_row, int end_row)
    {
        for (int i = start_row; i < end_row; ++i)
        {
            for (int j = 0; j < _cols; ++j)
            {
                result(i, j) = (*this)(i, j) * other(i, j);
            }
        }
    };

    int rows_per_thread = _rows / num_threads;
    int leftover = _rows % num_threads;

    int start = 0;
    for (int t = 0; t < num_threads; ++t)
    {
        int end = start + rows_per_thread + (t < leftover ? 1 : 0);
        threads[t] = std::thread(worker, start, end);
        start = end;
    }

    for (auto &t : threads)
    {
        if (t.joinable())
            t.join();
    }

    return result;
}

// Matrix and Scalar

// Matrix addition with scalar
template <typename T>
matrix<T> matrix<T>::operator+(const T &scalar) const
{
    matrix<T> result(_rows, _cols);

    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](int start_row, int end_row)
    {
        for (int i = start_row; i < end_row; ++i)
        {
            for (int j = 0; j < _cols; ++j)
            {
                result(i, j) = (*this)(i, j) + scalar;
            }
        }
    };

    int rows_per_thread = _rows / num_threads;
    int leftover = _rows % num_threads;

    int start = 0;
    for (int t = 0; t < num_threads; ++t)
    {
        int end = start + rows_per_thread + (t < leftover ? 1 : 0);
        threads[t] = std::thread(worker, start, end);
        start = end;
    }

    for (auto &t : threads)
    {
        if (t.joinable())
            t.join();
    }

    return result;
}
// Matrix subtraction with scalar
template <typename T>
matrix<T> matrix<T>::operator-(const T &scalar) const
{
    matrix<T> result(_rows, _cols);

    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](int start_row, int end_row)
    {
        for (int i = start_row; i < end_row; ++i)
        {
            for (int j = 0; j < _cols; ++j)
            {
                result(i, j) = (*this)(i, j) - scalar;
            }
        }
    };

    int rows_per_thread = _rows / num_threads;
    int leftover = _rows % num_threads;

    int start = 0;
    for (int t = 0; t < num_threads; ++t)
    {
        int end = start + rows_per_thread + (t < leftover ? 1 : 0);
        threads[t] = std::thread(worker, start, end);
        start = end;
    }

    for (auto &t : threads)
    {
        if (t.joinable())
            t.join();
    }

    return result;
}
// Matrix multiplication with scalar
template <typename T>
matrix<T> matrix<T>::operator*(const T &scalar) const
{
    matrix<T> result(_rows, _cols);

    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](int start_row, int end_row)
    {
        for (int i = start_row; i < end_row; ++i)
        {
            for (int j = 0; j < _cols; ++j)
            {
                result(i, j) = (*this)(i, j) * scalar;
            }
        }
    };

    int rows_per_thread = _rows / num_threads;
    int leftover = _rows % num_threads;

    int start = 0;
    for (int t = 0; t < num_threads; ++t)
    {
        int end = start + rows_per_thread + (t < leftover ? 1 : 0);
        threads[t] = std::thread(worker, start, end);
        start = end;
    }

    for (auto &t : threads)
    {
        if (t.joinable())
            t.join();
    }

    return result;
}
// Matrix division by scalar
template <typename T>
matrix<T> matrix<T>::operator/(const T &scalar) const
{
    if (scalar == 0)
    {
        throw std::invalid_argument("Division by zero");
    }
    matrix<T> result(_rows, _cols);

    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](int start_row, int end_row)
    {
        for (int i = start_row; i < end_row; ++i)
        {
            for (int j = 0; j < _cols; ++j)
            {
                result(i, j) = (*this)(i, j) / scalar;
            }
        }
    };

    int rows_per_thread = _rows / num_threads;
    int leftover = _rows % num_threads;

    int start = 0;
    for (int t = 0; t < num_threads; ++t)
    {
        int end = start + rows_per_thread + (t < leftover ? 1 : 0);
        threads[t] = std::thread(worker, start, end);
        start = end;
    }

    for (auto &t : threads)
    {
        if (t.joinable())
            t.join();
    }

    return result;
}

// Matrix addition assignment with scalar
template <typename T>
matrix<T> matrix<T>::operator+=(const T &scalar)
{
    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](int start_row, int end_row)
    {
        for (int i = start_row; i < end_row; ++i)
        {
            for (int j = 0; j < _cols; ++j)
            {
                (*this)(i, j) += scalar;
            }
        }
    };

    int rows_per_thread = _rows / num_threads;
    int leftover = _rows % num_threads;

    int start = 0;
    for (int t = 0; t < num_threads; ++t)
    {
        int end = start + rows_per_thread + (t < leftover ? 1 : 0);
        threads[t] = std::thread(worker, start, end);
        start = end;
    }

    for (auto &t : threads)
    {
        if (t.joinable())
            t.join();
    }

    return *this;
}
// Matrix subtraction assignment with scalar
template <typename T>
matrix<T> matrix<T>::operator-=(const T &scalar)
{
    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](int start_row, int end_row)
    {
        for (int i = start_row; i < end_row; ++i)
        {
            for (int j = 0; j < _cols; ++j)
            {
                (*this)(i, j) -= scalar;
            }
        }
    };

    int rows_per_thread = _rows / num_threads;
    int leftover = _rows % num_threads;

    int start = 0;
    for (int t = 0; t < num_threads; ++t)
    {
        int end = start + rows_per_thread + (t < leftover ? 1 : 0);
        threads[t] = std::thread(worker, start, end);
        start = end;
    }

    for (auto &t : threads)
    {
        if (t.joinable())
            t.join();
    }

    return *this;
}
// Matrix multiplication assignment with scalar
template <typename T>
matrix<T> matrix<T>::operator*=(const T &scalar)
{
    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](int start_row, int end_row)
    {
        for (int i = start_row; i < end_row; ++i)
        {
            for (int j = 0; j < _cols; ++j)
            {
                (*this)(i, j) *= scalar;
            }
        }
    };

    int rows_per_thread = _rows / num_threads;
    int leftover = _rows % num_threads;

    int start = 0;
    for (int t = 0; t < num_threads; ++t)
    {
        int end = start + rows_per_thread + (t < leftover ? 1 : 0);
        threads[t] = std::thread(worker, start, end);
        start = end;
    }

    for (auto &t : threads)
    {
        if (t.joinable())
            t.join();
    }

    return *this;
}
// Matrix division assignment by scalar
template <typename T>
matrix<T> matrix<T>::operator/=(const T &scalar)
{
    if (scalar == 0)
    {
        throw std::invalid_argument("Division by zero");
    }
    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](int start_row, int end_row)
    {
        for (int i = start_row; i < end_row; ++i)
        {
            for (int j = 0; j < _cols; ++j)
            {
                (*this)(i, j) /= scalar;
            }
        }
    };

    int rows_per_thread = _rows / num_threads;
    int leftover = _rows % num_threads;

    int start = 0;
    for (int t = 0; t < num_threads; ++t)
    {
        int end = start + rows_per_thread + (t < leftover ? 1 : 0);
        threads[t] = std::thread(worker, start, end);
        start = end;
    }

    for (auto &t : threads)
    {
        if (t.joinable())
            t.join();
    }

    return *this;
}
