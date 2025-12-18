#pragma once

// Template implementations for matrix_operators.hpp
// This file is included at the end of matrix_operators.hpp

#ifdef MATRIX_USE_BLAS
extern "C"
{
    // BLAS declarations
    void dgemm_(const char *transa, const char *transb, const int *m, const int *n, const int *k,
                const double *alpha, const double *a, const int *lda, const double *b, const int *ldb,
                const double *beta, double *c, const int *ldc);
    void sgemm_(const char *transa, const char *transb, const int *m, const int *n, const int *k,
                const float *alpha, const float *a, const int *lda, const float *b, const int *ldb,
                const float *beta, float *c, const int *ldc);
}
#endif

#ifdef MATRIX_USE_SIMD
#include <immintrin.h>
#endif

// ============================================================================
// Matrix-Matrix Arithmetic Operations
// ============================================================================

// Matrix addition
template <typename T>
matrix<T> matrix<T>::operator+(const matrix<T> &other) const
{
    if (_rows != other._rows || _cols != other._cols)
    {
        std::ostringstream oss;
        oss << "Matrix dimensions do not match for addition: "
            << _rows << "x" << _cols << " vs " << other._rows << "x" << other._cols;
        throw std::invalid_argument(oss.str());
    }

    matrix<T> result(_rows, _cols);

    // Number of threads to use
    const size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](size_t start_row, size_t end_row)
    {
        for (size_t i = start_row; i < end_row; ++i)
        {
            for (size_t j = 0; j < _cols; ++j)
            {
                result(i, j) = (*this)(i, j) + other(i, j);
            }
        }
    };

    size_t rows_per_thread = _rows / num_threads;
    size_t leftover = _rows % num_threads;

    size_t start = 0;
    for (size_t t = 0; t < num_threads; ++t)
    {
        size_t end = start + rows_per_thread + (t < leftover ? 1 : 0);
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
        std::ostringstream oss;
        oss << "Matrix dimensions do not match for subtraction: "
            << _rows << "x" << _cols << " vs " << other._rows << "x" << other._cols;
        throw std::invalid_argument(oss.str());
    }

    matrix<T> result(_rows, _cols);

    // Number of threads to use
    const size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](size_t start_row, size_t end_row)
    {
        for (size_t i = start_row; i < end_row; ++i)
        {
            for (size_t j = 0; j < _cols; ++j)
            {
                result(i, j) = (*this)(i, j) - other(i, j);
            }
        }
    };

    size_t rows_per_thread = _rows / num_threads;
    size_t leftover = _rows % num_threads;

    size_t start = 0;
    for (size_t t = 0; t < num_threads; ++t)
    {
        size_t end = start + rows_per_thread + (t < leftover ? 1 : 0);
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

// Matrix multiplication - Cache-optimized blocked algorithm
template <typename T>
matrix<T> matrix<T>::operator*(const matrix<T> &other) const
{
    if (_cols != other._rows)
    {
        std::ostringstream oss;
        oss << "Matrix dimensions do not match for multiplication: "
            << _rows << "x" << _cols << " * " << other._rows << "x" << other._cols
            << " (columns of first matrix " << _cols << " must equal rows of second matrix " << other._rows << ")";
        throw std::invalid_argument(oss.str());
    }

    matrix<T> result(_rows, other._cols);
    constexpr size_t BLOCK_SIZE = 64;
    constexpr size_t MIN_PARALLEL_SIZE = 256;

    std::fill(result._data.begin(), result._data.end(), T(0));

    const T *a_ptr = this->_data.data();
    const T *b_ptr = other._data.data();
    T *c_ptr = result._data.data();

    if (_rows >= MIN_PARALLEL_SIZE && other._cols >= MIN_PARALLEL_SIZE)
    {
        auto worker = [&](size_t i_start, size_t i_end)
        {
            for (size_t ii = i_start; ii < i_end; ii += BLOCK_SIZE)
            {
                size_t i_block_end = std::min(ii + BLOCK_SIZE, i_end);

                for (size_t jj = 0; jj < other._cols; jj += BLOCK_SIZE)
                {
                    size_t j_block_end = std::min(jj + BLOCK_SIZE, other._cols);

                    for (size_t kk = 0; kk < _cols; kk += BLOCK_SIZE)
                    {
                        size_t k_block_end = std::min(kk + BLOCK_SIZE, _cols);

                        for (size_t i = ii; i < i_block_end; ++i)
                        {
                            for (size_t k = kk; k < k_block_end; ++k)
                            {
                                T a_ik = a_ptr[i * _cols + k];
                                size_t result_row_offset = i * other._cols;
                                size_t other_row_offset = k * other._cols;

#ifdef MATRIX_USE_SIMD
                                // SIMD-optimized inner loop for double
                                if constexpr (std::is_same_v<T, double>)
                                {
                                    size_t j = jj;
                                    __m256d a_vec = _mm256_set1_pd(a_ik);
                                    for (; j + 3 < j_block_end; j += 4)
                                    {
                                        __m256d c = _mm256_loadu_pd(&c_ptr[result_row_offset + j]);
                                        __m256d b = _mm256_loadu_pd(&b_ptr[other_row_offset + j]);
                                        c = _mm256_fmadd_pd(a_vec, b, c);
                                        _mm256_storeu_pd(&c_ptr[result_row_offset + j], c);
                                    }
                                    for (; j < j_block_end; ++j)
                                    {
                                        c_ptr[result_row_offset + j] += a_ik * b_ptr[other_row_offset + j];
                                    }
                                }
                                // SIMD-optimized inner loop for float
                                else if constexpr (std::is_same_v<T, float>)
                                {
                                    size_t j = jj;
                                    __m256 a_vec = _mm256_set1_ps(a_ik);
                                    for (; j + 7 < j_block_end; j += 8)
                                    {
                                        __m256 c = _mm256_loadu_ps(&c_ptr[result_row_offset + j]);
                                        __m256 b = _mm256_loadu_ps(&b_ptr[other_row_offset + j]);
                                        c = _mm256_fmadd_ps(a_vec, b, c);
                                        _mm256_storeu_ps(&c_ptr[result_row_offset + j], c);
                                    }
                                    for (; j < j_block_end; ++j)
                                    {
                                        c_ptr[result_row_offset + j] += a_ik * b_ptr[other_row_offset + j];
                                    }
                                }
                                else
#endif
                                {
                                    // Manual unrolling for non-SIMD types
                                    size_t j = jj;
                                    for (; j + 3 < j_block_end; j += 4)
                                    {
                                        c_ptr[result_row_offset + j] += a_ik * b_ptr[other_row_offset + j];
                                        c_ptr[result_row_offset + j + 1] += a_ik * b_ptr[other_row_offset + j + 1];
                                        c_ptr[result_row_offset + j + 2] += a_ik * b_ptr[other_row_offset + j + 2];
                                        c_ptr[result_row_offset + j + 3] += a_ik * b_ptr[other_row_offset + j + 3];
                                    }
                                    for (; j < j_block_end; ++j)
                                    {
                                        c_ptr[result_row_offset + j] += a_ik * b_ptr[other_row_offset + j];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };

        size_t num_threads = std::thread::hardware_concurrency();
        size_t chunk_size = (_rows + num_threads - 1) / num_threads;

        std::vector<std::thread> threads;
        for (size_t t = 0; t < num_threads; ++t)
        {
            size_t start = t * chunk_size;
            size_t end = std::min(start + chunk_size, _rows);
            if (start < end)
            {
                threads.emplace_back(worker, start, end);
            }
        }

        for (auto &t : threads)
        {
            t.join();
        }
    }
    else
    {
        for (size_t ii = 0; ii < _rows; ii += BLOCK_SIZE)
        {
            size_t i_block_end = std::min(ii + BLOCK_SIZE, _rows);

            for (size_t jj = 0; jj < other._cols; jj += BLOCK_SIZE)
            {
                size_t j_block_end = std::min(jj + BLOCK_SIZE, other._cols);

                for (size_t kk = 0; kk < _cols; kk += BLOCK_SIZE)
                {
                    size_t k_block_end = std::min(kk + BLOCK_SIZE, _cols);

                    for (size_t i = ii; i < i_block_end; ++i)
                    {
                        for (size_t k = kk; k < k_block_end; ++k)
                        {
                            T a_ik = a_ptr[i * _cols + k];
                            size_t result_row_offset = i * other._cols;
                            size_t other_row_offset = k * other._cols;

#ifdef MATRIX_USE_SIMD
                            // SIMD-optimized inner loop for double
                            if constexpr (std::is_same_v<T, double>)
                            {
                                size_t j = jj;
                                __m256d a_vec = _mm256_set1_pd(a_ik);
                                for (; j + 3 < j_block_end; j += 4)
                                {
                                    __m256d c = _mm256_loadu_pd(&c_ptr[result_row_offset + j]);
                                    __m256d b = _mm256_loadu_pd(&b_ptr[other_row_offset + j]);
                                    c = _mm256_fmadd_pd(a_vec, b, c);
                                    _mm256_storeu_pd(&c_ptr[result_row_offset + j], c);
                                }
                                for (; j < j_block_end; ++j)
                                {
                                    c_ptr[result_row_offset + j] += a_ik * b_ptr[other_row_offset + j];
                                }
                            }
                            // SIMD-optimized inner loop for float
                            else if constexpr (std::is_same_v<T, float>)
                            {
                                size_t j = jj;
                                __m256 a_vec = _mm256_set1_ps(a_ik);
                                for (; j + 7 < j_block_end; j += 8)
                                {
                                    __m256 c = _mm256_loadu_ps(&c_ptr[result_row_offset + j]);
                                    __m256 b = _mm256_loadu_ps(&b_ptr[other_row_offset + j]);
                                    c = _mm256_fmadd_ps(a_vec, b, c);
                                    _mm256_storeu_ps(&c_ptr[result_row_offset + j], c);
                                }
                                for (; j < j_block_end; ++j)
                                {
                                    c_ptr[result_row_offset + j] += a_ik * b_ptr[other_row_offset + j];
                                }
                            }
                            else
#endif
                            {
                                // Manual unrolling for non-SIMD types
                                size_t j = jj;
                                for (; j + 3 < j_block_end; j += 4)
                                {
                                    c_ptr[result_row_offset + j] += a_ik * b_ptr[other_row_offset + j];
                                    c_ptr[result_row_offset + j + 1] += a_ik * b_ptr[other_row_offset + j + 1];
                                    c_ptr[result_row_offset + j + 2] += a_ik * b_ptr[other_row_offset + j + 2];
                                    c_ptr[result_row_offset + j + 3] += a_ik * b_ptr[other_row_offset + j + 3];
                                }
                                for (; j < j_block_end; ++j)
                                {
                                    c_ptr[result_row_offset + j] += a_ik * b_ptr[other_row_offset + j];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return result;
}

// Matrix addition assignment
template <typename T>
matrix<T> matrix<T>::operator+=(const matrix<T> &other)
{
    if (_rows != other._rows || _cols != other._cols)
    {
        std::ostringstream oss;
        oss << "Matrix dimensions do not match for addition assignment: "
            << _rows << "x" << _cols << " vs " << other._rows << "x" << other._cols;
        throw std::invalid_argument(oss.str());
    }

    const size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](size_t start_row, size_t end_row)
    {
        for (size_t i = start_row; i < end_row; ++i)
        {
            for (size_t j = 0; j < _cols; ++j)
            {
                (*this)(i, j) += other(i, j);
            }
        }
    };

    size_t rows_per_thread = _rows / num_threads;
    size_t leftover = _rows % num_threads;

    size_t start = 0;
    for (size_t t = 0; t < num_threads; ++t)
    {
        size_t end = start + rows_per_thread + (t < leftover ? 1 : 0);
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
        std::ostringstream oss;
        oss << "Matrix dimensions do not match for subtraction assignment: "
            << _rows << "x" << _cols << " vs " << other._rows << "x" << other._cols;
        throw std::invalid_argument(oss.str());
    }
    const size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](size_t start_row, size_t end_row)
    {
        for (size_t i = start_row; i < end_row; ++i)
        {
            for (size_t j = 0; j < _cols; ++j)
            {
                (*this)(i, j) -= other(i, j);
            }
        }
    };

    size_t rows_per_thread = _rows / num_threads;
    size_t leftover = _rows % num_threads;

    size_t start = 0;
    for (size_t t = 0; t < num_threads; ++t)
    {
        size_t end = start + rows_per_thread + (t < leftover ? 1 : 0);
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
        std::ostringstream oss;
        oss << "Matrix dimensions do not match for multiplication assignment: "
            << _rows << "x" << _cols << " *= " << other._rows << "x" << other._cols
            << " (columns of first matrix " << _cols << " must equal rows of second matrix " << other._rows << ")";
        throw std::invalid_argument(oss.str());
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
    for (size_t i = 0; i < _rows; ++i)
    {
        for (size_t j = 0; j < _cols; ++j)
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
        std::ostringstream oss;
        oss << "Matrix dimensions do not match for Hadamard product: "
            << _rows << "x" << _cols << " vs " << other.rows() << "x" << other.cols();
        throw std::invalid_argument(oss.str());
    }
    matrix<T> result(_rows, _cols);

    const size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](size_t start_row, size_t end_row)
    {
        for (size_t i = start_row; i < end_row; ++i)
        {
            for (size_t j = 0; j < _cols; ++j)
            {
                result(i, j) = (*this)(i, j) * other(i, j);
            }
        }
    };

    size_t rows_per_thread = _rows / num_threads;
    size_t leftover = _rows % num_threads;

    size_t start = 0;
    for (size_t t = 0; t < num_threads; ++t)
    {
        size_t end = start + rows_per_thread + (t < leftover ? 1 : 0);
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

    const size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](size_t start_row, size_t end_row)
    {
        for (size_t i = start_row; i < end_row; ++i)
        {
            for (size_t j = 0; j < _cols; ++j)
            {
                result(i, j) = (*this)(i, j) + scalar;
            }
        }
    };

    size_t rows_per_thread = _rows / num_threads;
    size_t leftover = _rows % num_threads;

    size_t start = 0;
    for (size_t t = 0; t < num_threads; ++t)
    {
        size_t end = start + rows_per_thread + (t < leftover ? 1 : 0);
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

    const size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](size_t start_row, size_t end_row)
    {
        for (size_t i = start_row; i < end_row; ++i)
        {
            for (size_t j = 0; j < _cols; ++j)
            {
                result(i, j) = (*this)(i, j) - scalar;
            }
        }
    };

    size_t rows_per_thread = _rows / num_threads;
    size_t leftover = _rows % num_threads;

    size_t start = 0;
    for (size_t t = 0; t < num_threads; ++t)
    {
        size_t end = start + rows_per_thread + (t < leftover ? 1 : 0);
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

    const size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](size_t start_row, size_t end_row)
    {
        for (size_t i = start_row; i < end_row; ++i)
        {
            for (size_t j = 0; j < _cols; ++j)
            {
                result(i, j) = (*this)(i, j) * scalar;
            }
        }
    };

    size_t rows_per_thread = _rows / num_threads;
    size_t leftover = _rows % num_threads;

    size_t start = 0;
    for (size_t t = 0; t < num_threads; ++t)
    {
        size_t end = start + rows_per_thread + (t < leftover ? 1 : 0);
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

    const size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](size_t start_row, size_t end_row)
    {
        for (size_t i = start_row; i < end_row; ++i)
        {
            for (size_t j = 0; j < _cols; ++j)
            {
                result(i, j) = (*this)(i, j) / scalar;
            }
        }
    };

    size_t rows_per_thread = _rows / num_threads;
    size_t leftover = _rows % num_threads;

    size_t start = 0;
    for (size_t t = 0; t < num_threads; ++t)
    {
        size_t end = start + rows_per_thread + (t < leftover ? 1 : 0);
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
    const size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](size_t start_row, size_t end_row)
    {
        for (size_t i = start_row; i < end_row; ++i)
        {
            for (size_t j = 0; j < _cols; ++j)
            {
                (*this)(i, j) += scalar;
            }
        }
    };

    size_t rows_per_thread = _rows / num_threads;
    size_t leftover = _rows % num_threads;

    size_t start = 0;
    for (size_t t = 0; t < num_threads; ++t)
    {
        size_t end = start + rows_per_thread + (t < leftover ? 1 : 0);
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
    const size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](size_t start_row, size_t end_row)
    {
        for (size_t i = start_row; i < end_row; ++i)
        {
            for (size_t j = 0; j < _cols; ++j)
            {
                (*this)(i, j) -= scalar;
            }
        }
    };

    size_t rows_per_thread = _rows / num_threads;
    size_t leftover = _rows % num_threads;

    size_t start = 0;
    for (size_t t = 0; t < num_threads; ++t)
    {
        size_t end = start + rows_per_thread + (t < leftover ? 1 : 0);
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
    const size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](size_t start_row, size_t end_row)
    {
        for (size_t i = start_row; i < end_row; ++i)
        {
            for (size_t j = 0; j < _cols; ++j)
            {
                (*this)(i, j) *= scalar;
            }
        }
    };

    size_t rows_per_thread = _rows / num_threads;
    size_t leftover = _rows % num_threads;

    size_t start = 0;
    for (size_t t = 0; t < num_threads; ++t)
    {
        size_t end = start + rows_per_thread + (t < leftover ? 1 : 0);
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
    const size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](size_t start_row, size_t end_row)
    {
        for (size_t i = start_row; i < end_row; ++i)
        {
            for (size_t j = 0; j < _cols; ++j)
            {
                (*this)(i, j) /= scalar;
            }
        }
    };

    size_t rows_per_thread = _rows / num_threads;
    size_t leftover = _rows % num_threads;

    size_t start = 0;
    for (size_t t = 0; t < num_threads; ++t)
    {
        size_t end = start + rows_per_thread + (t < leftover ? 1 : 0);
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

// ============================================================================
// BLAS-optimized specializations for double and float
// ============================================================================

#ifdef MATRIX_USE_BLAS

// BLAS-optimized matrix multiplication for double
template <>
inline matrix<double> matrix<double>::operator*(const matrix<double> &other) const
{
    if (_cols != other._rows)
    {
        std::ostringstream oss;
        oss << "Matrix dimensions do not match for multiplication: "
            << _rows << "x" << _cols << " * " << other._rows << "x" << other._cols;
        throw std::invalid_argument(oss.str());
    }

    matrix<double> result(_rows, other._cols);
    
    // BLAS dgemm: C = alpha*A*B + beta*C
    // We use row-major order, so we compute C^T = B^T * A^T
    char trans_no = 'N';
    int m = static_cast<int>(other._cols);  // columns of B (rows of result)
    int n = static_cast<int>(_rows);         // rows of A (columns of result)
    int k = static_cast<int>(_cols);         // columns of A = rows of B
    double alpha = 1.0;
    double beta = 0.0;
    int lda = static_cast<int>(other._cols); // leading dimension of B
    int ldb = static_cast<int>(_cols);       // leading dimension of A
    int ldc = static_cast<int>(other._cols); // leading dimension of C
    
    dgemm_(&trans_no, &trans_no, &m, &n, &k,
           &alpha, other._data.data(), &lda,
           this->_data.data(), &ldb,
           &beta, result._data.data(), &ldc);
    
    return result;
}

// BLAS-optimized matrix multiplication for float
template <>
inline matrix<float> matrix<float>::operator*(const matrix<float> &other) const
{
    if (_cols != other._rows)
    {
        std::ostringstream oss;
        oss << "Matrix dimensions do not match for multiplication: "
            << _rows << "x" << _cols << " * " << other._rows << "x" << other._cols;
        throw std::invalid_argument(oss.str());
    }

    matrix<float> result(_rows, other._cols);
    
    // BLAS sgemm: C = alpha*A*B + beta*C
    char trans_no = 'N';
    int m = static_cast<int>(other._cols);
    int n = static_cast<int>(_rows);
    int k = static_cast<int>(_cols);
    float alpha = 1.0f;
    float beta = 0.0f;
    int lda = static_cast<int>(other._cols);
    int ldb = static_cast<int>(_cols);
    int ldc = static_cast<int>(other._cols);
    
    sgemm_(&trans_no, &trans_no, &m, &n, &k,
           &alpha, other._data.data(), &lda,
           this->_data.data(), &ldb,
           &beta, result._data.data(), &ldc);
    
    return result;
}

#endif // MATRIX_USE_BLAS
