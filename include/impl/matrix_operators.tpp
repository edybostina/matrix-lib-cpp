#pragma once

// Template implementations for matrix_operators.hpp
// This file is included at the end of matrix_operators.hpp

#ifdef MATRIX_USE_BLAS
extern "C"
{
    // BLAS declarations
    void dgemm_(const char* transa, const char* transb, const int* m, const int* n, const int* k, const double* alpha,
                const double* a, const int* lda, const double* b, const int* ldb, const double* beta, double* c,
                const int* ldc);
    void sgemm_(const char* transa, const char* transb, const int* m, const int* n, const int* k, const float* alpha,
                const float* a, const int* lda, const float* b, const int* ldb, const float* beta, float* c,
                const int* ldc);
}
#endif

#ifdef MATRIX_USE_SIMD
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
// x86/x64 - use AVX2
#include <immintrin.h>
#define MATRIX_USE_AVX2
#elif defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
// ARM64 - use NEON
#include <arm_neon.h>
#define MATRIX_USE_NEON
#endif
#endif

// ============================================================================
// Matrix-Matrix Arithmetic Operations
// ============================================================================

/**
 * @brief Element-wise matrix addition (A + B).
 *
 * Optimizations: SIMD (AVX2/NEON), multi-threading (>10k elements), direct
 * memory access.
 *
 * @param other Matrix to add
 * @return New matrix with element-wise sum
 * @throws std::invalid_argument If dimensions don't match
 * @details Time O(m*n), Space O(m*n)
 */
template <typename T>
matrix<T> matrix<T>::operator+(const matrix<T>& other) const
{
    if (_rows != other._rows || _cols != other._cols)
    {
        std::ostringstream oss;
        oss << "Matrix dimensions do not match for addition: " << _rows << "x" << _cols << " vs " << other._rows << "x"
            << other._cols;
        throw std::invalid_argument(oss.str());
    }

    matrix<T> result(_rows, _cols);

    const size_t total_elements = _rows * _cols;
    constexpr size_t MIN_PARALLEL_SIZE = 10000;

    const T* a_ptr = this->_data.data();
    const T* b_ptr = other._data.data();
    T* result_ptr = result._data.data();

    if (total_elements >= MIN_PARALLEL_SIZE)
    {
        const size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        const size_t elements_per_thread = (total_elements + num_threads - 1) / num_threads;

        auto worker = [&](size_t start_idx, size_t end_idx)
        {
            size_t i = start_idx;

#ifdef MATRIX_USE_SIMD
            if constexpr (std::is_same_v<T, double>)
            {
#ifdef MATRIX_USE_AVX2
                for (; i + 3 < end_idx; i += 4)
                {
                    __m256d a = _mm256_loadu_pd(&a_ptr[i]);
                    __m256d b = _mm256_loadu_pd(&b_ptr[i]);
                    __m256d c = _mm256_add_pd(a, b);
                    _mm256_storeu_pd(&result_ptr[i], c);
                }
#elif defined(MATRIX_USE_NEON)
                for (; i + 1 < end_idx; i += 2)
                {
                    float64x2_t a = vld1q_f64(&a_ptr[i]);
                    float64x2_t b = vld1q_f64(&b_ptr[i]);
                    float64x2_t c = vaddq_f64(a, b);
                    vst1q_f64(&result_ptr[i], c);
                }
#endif
            }
            else if constexpr (std::is_same_v<T, float>)
            {
#ifdef MATRIX_USE_AVX2
                for (; i + 7 < end_idx; i += 8)
                {
                    __m256 a = _mm256_loadu_ps(&a_ptr[i]);
                    __m256 b = _mm256_loadu_ps(&b_ptr[i]);
                    __m256 c = _mm256_add_ps(a, b);
                    _mm256_storeu_ps(&result_ptr[i], c);
                }
#elif defined(MATRIX_USE_NEON)
                for (; i + 3 < end_idx; i += 4)
                {
                    float32x4_t a = vld1q_f32(&a_ptr[i]);
                    float32x4_t b = vld1q_f32(&b_ptr[i]);
                    float32x4_t c = vaddq_f32(a, b);
                    vst1q_f32(&result_ptr[i], c);
                }
#endif
            }
#endif
            for (; i + 3 < end_idx; i += 4)
            {
                result_ptr[i] = a_ptr[i] + b_ptr[i];
                result_ptr[i + 1] = a_ptr[i + 1] + b_ptr[i + 1];
                result_ptr[i + 2] = a_ptr[i + 2] + b_ptr[i + 2];
                result_ptr[i + 3] = a_ptr[i + 3] + b_ptr[i + 3];
            }
            for (; i < end_idx; ++i)
            {
                result_ptr[i] = a_ptr[i] + b_ptr[i];
            }
        };

        for (size_t t = 0; t < num_threads; ++t)
        {
            size_t start = t * elements_per_thread;
            size_t end = std::min(start + elements_per_thread, total_elements);
            if (start < end)
            {
                threads.emplace_back(worker, start, end);
            }
        }

        for (auto& t : threads)
        {
            t.join();
        }
    }
    else
    {
        size_t i = 0;

#ifdef MATRIX_USE_SIMD
        if constexpr (std::is_same_v<T, double>)
        {
#ifdef MATRIX_USE_AVX2
            for (; i + 3 < total_elements; i += 4)
            {
                __m256d a = _mm256_loadu_pd(&a_ptr[i]);
                __m256d b = _mm256_loadu_pd(&b_ptr[i]);
                __m256d c = _mm256_add_pd(a, b);
                _mm256_storeu_pd(&result_ptr[i], c);
            }
#elif defined(MATRIX_USE_NEON)
            for (; i + 1 < total_elements; i += 2)
            {
                float64x2_t a = vld1q_f64(&a_ptr[i]);
                float64x2_t b = vld1q_f64(&b_ptr[i]);
                float64x2_t c = vaddq_f64(a, b);
                vst1q_f64(&result_ptr[i], c);
            }
#endif
        }
        else if constexpr (std::is_same_v<T, float>)
        {
#ifdef MATRIX_USE_AVX2
            for (; i + 7 < total_elements; i += 8)
            {
                __m256 a = _mm256_loadu_ps(&a_ptr[i]);
                __m256 b = _mm256_loadu_ps(&b_ptr[i]);
                __m256 c = _mm256_add_ps(a, b);
                _mm256_storeu_ps(&result_ptr[i], c);
            }
#elif defined(MATRIX_USE_NEON)
            for (; i + 3 < total_elements; i += 4)
            {
                float32x4_t a = vld1q_f32(&a_ptr[i]);
                float32x4_t b = vld1q_f32(&b_ptr[i]);
                float32x4_t c = vaddq_f32(a, b);
                vst1q_f32(&result_ptr[i], c);
            }
#endif
        }
#endif
        for (; i + 3 < total_elements; i += 4)
        {
            result_ptr[i] = a_ptr[i] + b_ptr[i];
            result_ptr[i + 1] = a_ptr[i + 1] + b_ptr[i + 1];
            result_ptr[i + 2] = a_ptr[i + 2] + b_ptr[i + 2];
            result_ptr[i + 3] = a_ptr[i + 3] + b_ptr[i + 3];
        }
        for (; i < total_elements; ++i)
        {
            result_ptr[i] = a_ptr[i] + b_ptr[i];
        }
    }

    return result;
}

/**
 * @brief Element-wise matrix subtraction (A - B).
 *
 * Optimizations: SIMD (AVX2/NEON), multi-threading (>10k elements), direct
 * memory access.
 *
 * @param other Matrix to subtract
 * @return New matrix with element-wise difference
 * @throws std::invalid_argument If dimensions don't match
 * @details Time O(m*n), Space O(m*n)
 */
template <typename T>
matrix<T> matrix<T>::operator-(const matrix<T>& other) const
{
    if (_rows != other._rows || _cols != other._cols)
    {
        std::ostringstream oss;
        oss << "Matrix dimensions do not match for subtraction: " << _rows << "x" << _cols << " vs " << other._rows
            << "x" << other._cols;
        throw std::invalid_argument(oss.str());
    }

    matrix<T> result(_rows, _cols);

    const size_t total_elements = _rows * _cols;
    constexpr size_t MIN_PARALLEL_SIZE = 10000;

    const T* a_ptr = this->_data.data();
    const T* b_ptr = other._data.data();
    T* result_ptr = result._data.data();

    if (total_elements >= MIN_PARALLEL_SIZE)
    {
        const size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        const size_t elements_per_thread = (total_elements + num_threads - 1) / num_threads;

        auto worker = [&](size_t start_idx, size_t end_idx)
        {
            size_t i = start_idx;

#ifdef MATRIX_USE_SIMD
            if constexpr (std::is_same_v<T, double>)
            {
#ifdef MATRIX_USE_AVX2
                for (; i + 3 < end_idx; i += 4)
                {
                    __m256d a = _mm256_loadu_pd(&a_ptr[i]);
                    __m256d b = _mm256_loadu_pd(&b_ptr[i]);
                    __m256d c = _mm256_sub_pd(a, b);
                    _mm256_storeu_pd(&result_ptr[i], c);
                }
#elif defined(MATRIX_USE_NEON)
                for (; i + 1 < end_idx; i += 2)
                {
                    float64x2_t a = vld1q_f64(&a_ptr[i]);
                    float64x2_t b = vld1q_f64(&b_ptr[i]);
                    float64x2_t c = vsubq_f64(a, b);
                    vst1q_f64(&result_ptr[i], c);
                }
#endif
            }
            else if constexpr (std::is_same_v<T, float>)
            {
#ifdef MATRIX_USE_AVX2
                for (; i + 7 < end_idx; i += 8)
                {
                    __m256 a = _mm256_loadu_ps(&a_ptr[i]);
                    __m256 b = _mm256_loadu_ps(&b_ptr[i]);
                    __m256 c = _mm256_sub_ps(a, b);
                    _mm256_storeu_ps(&result_ptr[i], c);
                }
#elif defined(MATRIX_USE_NEON)
                for (; i + 3 < end_idx; i += 4)
                {
                    float32x4_t a = vld1q_f32(&a_ptr[i]);
                    float32x4_t b = vld1q_f32(&b_ptr[i]);
                    float32x4_t c = vsubq_f32(a, b);
                    vst1q_f32(&result_ptr[i], c);
                }
#endif
            }
#endif

            for (; i + 3 < end_idx; i += 4)
            {
                result_ptr[i] = a_ptr[i] - b_ptr[i];
                result_ptr[i + 1] = a_ptr[i + 1] - b_ptr[i + 1];
                result_ptr[i + 2] = a_ptr[i + 2] - b_ptr[i + 2];
                result_ptr[i + 3] = a_ptr[i + 3] - b_ptr[i + 3];
            }
            for (; i < end_idx; ++i)
            {
                result_ptr[i] = a_ptr[i] - b_ptr[i];
            }
        };

        for (size_t t = 0; t < num_threads; ++t)
        {
            size_t start = t * elements_per_thread;
            size_t end = std::min(start + elements_per_thread, total_elements);
            if (start < end)
            {
                threads.emplace_back(worker, start, end);
            }
        }

        for (auto& t : threads)
        {
            t.join();
        }
    }
    else
    {
        size_t i = 0;

#ifdef MATRIX_USE_SIMD
        if constexpr (std::is_same_v<T, double>)
        {
#ifdef MATRIX_USE_AVX2
            for (; i + 3 < total_elements; i += 4)
            {
                __m256d a = _mm256_loadu_pd(&a_ptr[i]);
                __m256d b = _mm256_loadu_pd(&b_ptr[i]);
                __m256d c = _mm256_sub_pd(a, b);
                _mm256_storeu_pd(&result_ptr[i], c);
            }
#elif defined(MATRIX_USE_NEON)
            for (; i + 1 < total_elements; i += 2)
            {
                float64x2_t a = vld1q_f64(&a_ptr[i]);
                float64x2_t b = vld1q_f64(&b_ptr[i]);
                float64x2_t c = vsubq_f64(a, b);
                vst1q_f64(&result_ptr[i], c);
            }
#endif
        }
        else if constexpr (std::is_same_v<T, float>)
        {
#ifdef MATRIX_USE_AVX2
            for (; i + 7 < total_elements; i += 8)
            {
                __m256 a = _mm256_loadu_ps(&a_ptr[i]);
                __m256 b = _mm256_loadu_ps(&b_ptr[i]);
                __m256 c = _mm256_sub_ps(a, b);
                _mm256_storeu_ps(&result_ptr[i], c);
            }
#elif defined(MATRIX_USE_NEON)
            for (; i + 3 < total_elements; i += 4)
            {
                float32x4_t a = vld1q_f32(&a_ptr[i]);
                float32x4_t b = vld1q_f32(&b_ptr[i]);
                float32x4_t c = vsubq_f32(a, b);
                vst1q_f32(&result_ptr[i], c);
            }
#endif
        }
#endif

        for (; i + 3 < total_elements; i += 4)
        {
            result_ptr[i] = a_ptr[i] - b_ptr[i];
            result_ptr[i + 1] = a_ptr[i + 1] - b_ptr[i + 1];
            result_ptr[i + 2] = a_ptr[i + 2] - b_ptr[i + 2];
            result_ptr[i + 3] = a_ptr[i + 3] - b_ptr[i + 3];
        }
        for (; i < total_elements; ++i)
        {
            result_ptr[i] = a_ptr[i] - b_ptr[i];
        }
    }

    return result;
}

/**
 * @brief Matrix multiplication (A * B).
 *
 * Optimizations: Blocked multiplication, SIMD (AVX2/NEON), multi-threading
 * (>256x256), direct memory access.
 *
 * @param other Matrix to multiply
 * @return New matrix with the product
 * @throws std::invalid_argument If dimensions don't match
 * @details Time O(m*n*p), Space O(m*p)
 */
template <typename T>
matrix<T> matrix<T>::operator*(const matrix<T>& other) const
{
    if (_cols != other._rows)
    {
        std::ostringstream oss;
        oss << "Matrix dimensions do not match for multiplication: " << _rows << "x" << _cols << " * " << other._rows
            << "x" << other._cols << " (columns of first matrix " << _cols << " must equal rows of second matrix "
            << other._rows << ")";
        throw std::invalid_argument(oss.str());
    }

    matrix<T> result(_rows, other._cols);
    constexpr size_t BLOCK_SIZE = 64;
    constexpr size_t MIN_PARALLEL_SIZE = 256;

    std::fill(result._data.begin(), result._data.end(), T(0));

    const T* a_ptr = this->_data.data();
    const T* b_ptr = other._data.data();
    T* c_ptr = result._data.data();

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
#ifdef MATRIX_USE_AVX2
                                    __m256d a_vec = _mm256_set1_pd(a_ik);
                                    for (; j + 3 < j_block_end; j += 4)
                                    {
                                        __m256d c = _mm256_loadu_pd(&c_ptr[result_row_offset + j]);
                                        __m256d b = _mm256_loadu_pd(&b_ptr[other_row_offset + j]);
                                        c = _mm256_fmadd_pd(a_vec, b, c);
                                        _mm256_storeu_pd(&c_ptr[result_row_offset + j], c);
                                    }
#elif defined(MATRIX_USE_NEON)
                                    float64x2_t a_vec = vdupq_n_f64(a_ik);
                                    for (; j + 1 < j_block_end; j += 2)
                                    {
                                        float64x2_t c = vld1q_f64(&c_ptr[result_row_offset + j]);
                                        float64x2_t b = vld1q_f64(&b_ptr[other_row_offset + j]);
                                        c = vfmaq_f64(c, a_vec, b);
                                        vst1q_f64(&c_ptr[result_row_offset + j], c);
                                    }
#endif
                                    for (; j < j_block_end; ++j)
                                    {
                                        c_ptr[result_row_offset + j] += a_ik * b_ptr[other_row_offset + j];
                                    }
                                }
                                // SIMD-optimized inner loop for float
                                else if constexpr (std::is_same_v<T, float>)
                                {
                                    size_t j = jj;
#ifdef MATRIX_USE_AVX2
                                    __m256 a_vec = _mm256_set1_ps(a_ik);
                                    for (; j + 7 < j_block_end; j += 8)
                                    {
                                        __m256 c = _mm256_loadu_ps(&c_ptr[result_row_offset + j]);
                                        __m256 b = _mm256_loadu_ps(&b_ptr[other_row_offset + j]);
                                        c = _mm256_fmadd_ps(a_vec, b, c);
                                        _mm256_storeu_ps(&c_ptr[result_row_offset + j], c);
                                    }
#elif defined(MATRIX_USE_NEON)
                                    float32x4_t a_vec = vdupq_n_f32(a_ik);
                                    for (; j + 3 < j_block_end; j += 4)
                                    {
                                        float32x4_t c = vld1q_f32(&c_ptr[result_row_offset + j]);
                                        float32x4_t b = vld1q_f32(&b_ptr[other_row_offset + j]);
                                        c = vfmaq_f32(c, a_vec, b);
                                        vst1q_f32(&c_ptr[result_row_offset + j], c);
                                    }
#endif
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

        for (auto& t : threads)
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
#ifdef MATRIX_USE_AVX2
                                __m256d a_vec = _mm256_set1_pd(a_ik);
                                for (; j + 3 < j_block_end; j += 4)
                                {
                                    __m256d c = _mm256_loadu_pd(&c_ptr[result_row_offset + j]);
                                    __m256d b = _mm256_loadu_pd(&b_ptr[other_row_offset + j]);
                                    c = _mm256_fmadd_pd(a_vec, b, c);
                                    _mm256_storeu_pd(&c_ptr[result_row_offset + j], c);
                                }
#elif defined(MATRIX_USE_NEON)
                                float64x2_t a_vec = vdupq_n_f64(a_ik);
                                for (; j + 1 < j_block_end; j += 2)
                                {
                                    float64x2_t c = vld1q_f64(&c_ptr[result_row_offset + j]);
                                    float64x2_t b = vld1q_f64(&b_ptr[other_row_offset + j]);
                                    c = vfmaq_f64(c, a_vec, b);
                                    vst1q_f64(&c_ptr[result_row_offset + j], c);
                                }
#endif
                                for (; j < j_block_end; ++j)
                                {
                                    c_ptr[result_row_offset + j] += a_ik * b_ptr[other_row_offset + j];
                                }
                            }
                            // SIMD-optimized inner loop for float
                            else if constexpr (std::is_same_v<T, float>)
                            {
                                size_t j = jj;
#ifdef MATRIX_USE_AVX2
                                __m256 a_vec = _mm256_set1_ps(a_ik);
                                for (; j + 7 < j_block_end; j += 8)
                                {
                                    __m256 c = _mm256_loadu_ps(&c_ptr[result_row_offset + j]);
                                    __m256 b = _mm256_loadu_ps(&b_ptr[other_row_offset + j]);
                                    c = _mm256_fmadd_ps(a_vec, b, c);
                                    _mm256_storeu_ps(&c_ptr[result_row_offset + j], c);
                                }
#elif defined(MATRIX_USE_NEON)
                                float32x4_t a_vec = vdupq_n_f32(a_ik);
                                for (; j + 3 < j_block_end; j += 4)
                                {
                                    float32x4_t c = vld1q_f32(&c_ptr[result_row_offset + j]);
                                    float32x4_t b = vld1q_f32(&b_ptr[other_row_offset + j]);
                                    c = vfmaq_f32(c, a_vec, b);
                                    vst1q_f32(&c_ptr[result_row_offset + j], c);
                                }
#endif
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

/**
 * @brief Element-wise matrix addition assignment (A += B).
 *
 * Optimizations: SIMD (AVX2/NEON), multi-threading (>10k elements), direct
 * memory access.
 *
 * @param other Matrix to add
 * @return Reference to this matrix after addition
 * @throws std::invalid_argument If dimensions don't match
 * @details Time O(m*n), Space O(1)
 */
template <typename T>
matrix<T> matrix<T>::operator+=(const matrix<T>& other)
{
    if (_rows != other._rows || _cols != other._cols)
    {
        std::ostringstream oss;
        oss << "Matrix dimensions do not match for addition assignment: " << _rows << "x" << _cols << " vs "
            << other._rows << "x" << other._cols;
        throw std::invalid_argument(oss.str());
    }

    const size_t total_elements = _rows * _cols;
    constexpr size_t MIN_PARALLEL_SIZE = 10000;

    const T* a_ptr = this->_data.data();
    const T* b_ptr = other._data.data();
    T* result_ptr = this->_data.data();

    if (total_elements >= MIN_PARALLEL_SIZE)
    {
        const size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        const size_t elements_per_thread = (total_elements + num_threads - 1) / num_threads;

        auto worker = [&](size_t start_idx, size_t end_idx)
        {
            size_t i = start_idx;

#ifdef MATRIX_USE_SIMD
            if constexpr (std::is_same_v<T, double>)
            {
#ifdef MATRIX_USE_AVX2
                for (; i + 3 < end_idx; i += 4)
                {
                    __m256d a = _mm256_loadu_pd(&a_ptr[i]);
                    __m256d b = _mm256_loadu_pd(&b_ptr[i]);
                    __m256d c = _mm256_add_pd(a, b);
                    _mm256_storeu_pd(&result_ptr[i], c);
                }
#elif defined(MATRIX_USE_NEON)
                for (; i + 1 < end_idx; i += 2)
                {
                    float64x2_t a = vld1q_f64(&a_ptr[i]);
                    float64x2_t b = vld1q_f64(&b_ptr[i]);
                    float64x2_t c = vaddq_f64(a, b);
                    vst1q_f64(&result_ptr[i], c);
                }
#endif
            }
            else if constexpr (std::is_same_v<T, float>)
            {
#ifdef MATRIX_USE_AVX2
                for (; i + 7 < end_idx; i += 8)
                {
                    __m256 a = _mm256_loadu_ps(&a_ptr[i]);
                    __m256 b = _mm256_loadu_ps(&b_ptr[i]);
                    __m256 c = _mm256_add_ps(a, b);
                    _mm256_storeu_ps(&result_ptr[i], c);
                }
#elif defined(MATRIX_USE_NEON)
                for (; i + 3 < end_idx; i += 4)
                {
                    float32x4_t a = vld1q_f32(&a_ptr[i]);
                    float32x4_t b = vld1q_f32(&b_ptr[i]);
                    float32x4_t c = vaddq_f32(a, b);
                    vst1q_f32(&result_ptr[i], c);
                }
#endif
            }
#endif
            for (; i + 3 < end_idx; i += 4)
            {
                result_ptr[i] = a_ptr[i] + b_ptr[i];
                result_ptr[i + 1] = a_ptr[i + 1] + b_ptr[i + 1];
                result_ptr[i + 2] = a_ptr[i + 2] + b_ptr[i + 2];
                result_ptr[i + 3] = a_ptr[i + 3] + b_ptr[i + 3];
            }
            for (; i < end_idx; ++i)
            {
                result_ptr[i] = a_ptr[i] + b_ptr[i];
            }
        };

        for (size_t t = 0; t < num_threads; ++t)
        {
            size_t start = t * elements_per_thread;
            size_t end = std::min(start + elements_per_thread, total_elements);
            if (start < end)
            {
                threads.emplace_back(worker, start, end);
            }
        }

        for (auto& t : threads)
        {
            t.join();
        }
    }
    else
    {
        size_t i = 0;

#ifdef MATRIX_USE_SIMD
        if constexpr (std::is_same_v<T, double>)
        {
#ifdef MATRIX_USE_AVX2
            for (; i + 3 < total_elements; i += 4)
            {
                __m256d a = _mm256_loadu_pd(&a_ptr[i]);
                __m256d b = _mm256_loadu_pd(&b_ptr[i]);
                __m256d c = _mm256_add_pd(a, b);
                _mm256_storeu_pd(&result_ptr[i], c);
            }
#elif defined(MATRIX_USE_NEON)
            for (; i + 1 < total_elements; i += 2)
            {
                float64x2_t a = vld1q_f64(&a_ptr[i]);
                float64x2_t b = vld1q_f64(&b_ptr[i]);
                float64x2_t c = vaddq_f64(a, b);
                vst1q_f64(&result_ptr[i], c);
            }
#endif
        }
        else if constexpr (std::is_same_v<T, float>)
        {
#ifdef MATRIX_USE_AVX2
            for (; i + 7 < total_elements; i += 8)
            {
                __m256 a = _mm256_loadu_ps(&a_ptr[i]);
                __m256 b = _mm256_loadu_ps(&b_ptr[i]);
                __m256 c = _mm256_add_ps(a, b);
                _mm256_storeu_ps(&result_ptr[i], c);
            }
#elif defined(MATRIX_USE_NEON)
            for (; i + 3 < total_elements; i += 4)
            {
                float32x4_t a = vld1q_f32(&a_ptr[i]);
                float32x4_t b = vld1q_f32(&b_ptr[i]);
                float32x4_t c = vaddq_f32(a, b);
                vst1q_f32(&result_ptr[i], c);
            }
#endif
        }
#endif
        for (; i + 3 < total_elements; i += 4)
        {
            result_ptr[i] = a_ptr[i] + b_ptr[i];
            result_ptr[i + 1] = a_ptr[i + 1] + b_ptr[i + 1];
            result_ptr[i + 2] = a_ptr[i + 2] + b_ptr[i + 2];
            result_ptr[i + 3] = a_ptr[i + 3] + b_ptr[i + 3];
        }
        for (; i < total_elements; ++i)
        {
            result_ptr[i] = a_ptr[i] + b_ptr[i];
        }
    }
    return *this;
}

/**
 * @brief Element-wise matrix subtraction assignment (A -= B).
 *
 * Optimizations: SIMD (AVX2/NEON), multi-threading (>10k elements), direct
 * memory access.
 *
 * @param other Matrix to subtract
 * @return Reference to this matrix after subtraction
 * @throws std::invalid_argument If dimensions don't match
 * @details Time O(m*n), Space O(1)
 */
template <typename T>
matrix<T> matrix<T>::operator-=(const matrix<T>& other)
{
    if (_rows != other._rows || _cols != other._cols)
    {
        std::ostringstream oss;
        oss << "Matrix dimensions do not match for subtraction assignment: " << _rows << "x" << _cols << " vs "
            << other._rows << "x" << other._cols;
        throw std::invalid_argument(oss.str());
    }

    const size_t total_elements = _rows * _cols;
    constexpr size_t MIN_PARALLEL_SIZE = 10000;

    const T* a_ptr = this->_data.data();
    const T* b_ptr = other._data.data();
    T* result_ptr = this->_data.data();

    if (total_elements >= MIN_PARALLEL_SIZE)
    {
        const size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        const size_t elements_per_thread = (total_elements + num_threads - 1) / num_threads;

        auto worker = [&](size_t start_idx, size_t end_idx)
        {
            size_t i = start_idx;

#ifdef MATRIX_USE_SIMD
            if constexpr (std::is_same_v<T, double>)
            {
#ifdef MATRIX_USE_AVX2
                for (; i + 3 < end_idx; i += 4)
                {
                    __m256d a = _mm256_loadu_pd(&a_ptr[i]);
                    __m256d b = _mm256_loadu_pd(&b_ptr[i]);
                    __m256d c = _mm256_sub_pd(a, b);
                    _mm256_storeu_pd(&result_ptr[i], c);
                }
#elif defined(MATRIX_USE_NEON)
                for (; i + 1 < end_idx; i += 2)
                {
                    float64x2_t a = vld1q_f64(&a_ptr[i]);
                    float64x2_t b = vld1q_f64(&b_ptr[i]);
                    float64x2_t c = vsubq_f64(a, b);
                    vst1q_f64(&result_ptr[i], c);
                }
#endif
            }
            else if constexpr (std::is_same_v<T, float>)
            {
#ifdef MATRIX_USE_AVX2
                for (; i + 7 < end_idx; i += 8)
                {
                    __m256 a = _mm256_loadu_ps(&a_ptr[i]);
                    __m256 b = _mm256_loadu_ps(&b_ptr[i]);
                    __m256 c = _mm256_sub_ps(a, b);
                    _mm256_storeu_ps(&result_ptr[i], c);
                }
#elif defined(MATRIX_USE_NEON)
                for (; i + 3 < end_idx; i += 4)
                {
                    float32x4_t a = vld1q_f32(&a_ptr[i]);
                    float32x4_t b = vld1q_f32(&b_ptr[i]);
                    float32x4_t c = vsubq_f32(a, b);
                    vst1q_f32(&result_ptr[i], c);
                }
#endif
            }
#endif
            for (; i + 3 < end_idx; i += 4)
            {
                result_ptr[i] = a_ptr[i] - b_ptr[i];
                result_ptr[i + 1] = a_ptr[i + 1] - b_ptr[i + 1];
                result_ptr[i + 2] = a_ptr[i + 2] - b_ptr[i + 2];
                result_ptr[i + 3] = a_ptr[i + 3] - b_ptr[i + 3];
            }
            for (; i < end_idx; ++i)
            {
                result_ptr[i] = a_ptr[i] - b_ptr[i];
            }
        };

        for (size_t t = 0; t < num_threads; ++t)
        {
            size_t start = t * elements_per_thread;
            size_t end = std::min(start + elements_per_thread, total_elements);
            if (start < end)
            {
                threads.emplace_back(worker, start, end);
            }
        }

        for (auto& t : threads)
        {
            t.join();
        }
    }
    else
    {
        size_t i = 0;

#ifdef MATRIX_USE_SIMD
        if constexpr (std::is_same_v<T, double>)
        {
#ifdef MATRIX_USE_AVX2
            for (; i + 3 < total_elements; i += 4)
            {
                __m256d a = _mm256_loadu_pd(&a_ptr[i]);
                __m256d b = _mm256_loadu_pd(&b_ptr[i]);
                __m256d c = _mm256_sub_pd(a, b);
                _mm256_storeu_pd(&result_ptr[i], c);
            }
#elif defined(MATRIX_USE_NEON)
            for (; i + 1 < total_elements; i += 2)
            {
                float64x2_t a = vld1q_f64(&a_ptr[i]);
                float64x2_t b = vld1q_f64(&b_ptr[i]);
                float64x2_t c = vsubq_f64(a, b);
                vst1q_f64(&result_ptr[i], c);
            }
#endif
        }
        else if constexpr (std::is_same_v<T, float>)
        {
#ifdef MATRIX_USE_AVX2
            for (; i + 7 < total_elements; i += 8)
            {
                __m256 a = _mm256_loadu_ps(&a_ptr[i]);
                __m256 b = _mm256_loadu_ps(&b_ptr[i]);
                __m256 c = _mm256_sub_ps(a, b);
                _mm256_storeu_ps(&result_ptr[i], c);
            }
#elif defined(MATRIX_USE_NEON)
            for (; i + 3 < total_elements; i += 4)
            {
                float32x4_t a = vld1q_f32(&a_ptr[i]);
                float32x4_t b = vld1q_f32(&b_ptr[i]);
                float32x4_t c = vsubq_f32(a, b);
                vst1q_f32(&result_ptr[i], c);
            }
#endif
        }
#endif
        for (; i + 3 < total_elements; i += 4)
        {
            result_ptr[i] = a_ptr[i] - b_ptr[i];
            result_ptr[i + 1] = a_ptr[i + 1] - b_ptr[i + 1];
            result_ptr[i + 2] = a_ptr[i + 2] - b_ptr[i + 2];
            result_ptr[i + 3] = a_ptr[i + 3] - b_ptr[i + 3];
        }
        for (; i < total_elements; ++i)
        {
            result_ptr[i] = a_ptr[i] - b_ptr[i];
        }
    }
    return *this;
}

/**
 * @brief Element-wise matrix multiplication assignment (A *= B).
 *
 * Optimizations: SIMD (AVX2/NEON), multi-threading (>1k elements), direct
 * memory access.
 *
 * @param other Matrix to multiply
 * @return Reference to this matrix after multiplication
 * @throws std::invalid_argument If dimensions don't match
 * @details Time O(m*n), Space O(1)
 */

template <typename T>
matrix<T> matrix<T>::operator*=(const matrix<T>& other)
{
    if (_cols != other._rows)
    {
        std::ostringstream oss;
        oss << "Matrix dimensions do not match for multiplication assignment: " << _rows << "x" << _cols
            << " *= " << other._rows << "x" << other._cols << " (columns of first matrix " << _cols
            << " must equal rows of second matrix " << other._rows << ")";
        throw std::invalid_argument(oss.str());
    }
    matrix<T> result = (*this) * other;
    *this = result;
    return *this;
}

/**
 * @brief Matrix equality operator.
 *
 * @param other Matrix to compare
 * @return true if matrices are equal, false otherwise
 * @details Time O(m*n), Space O(1)
 */
template <typename T>
bool matrix<T>::operator==(const matrix<T>& other) const noexcept
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

/**
 * @brief Matrix inequality operator.
 *
 * @param other Matrix to compare
 * @return true if matrices are not equal, false otherwise
 * @details Time O(m*n), Space O(1)
 */
template <typename T>
bool matrix<T>::operator!=(const matrix<T>& other) const noexcept
{
    return !(*this == other);
}

/**
 * @brief Element-wise Hadamard product (A ⊙ B).
 *
 * Optimizations: SIMD (AVX2/NEON), multi-threading (>10k elements), direct
 * memory access.
 *
 * @param other Matrix to multiply
 * @return New matrix with the Hadamard product
 * @throws std::invalid_argument If dimensions don't match
 * @details Time O(m*n), Space O(m*n)
 */
template <typename T>
matrix<T> matrix<T>::hadamard(const matrix<T>& other) const
{
    if (_rows != other.rows() || _cols != other.cols())
    {
        std::ostringstream oss;
        oss << "Matrix dimensions do not match for Hadamard product: " << _rows << "x" << _cols << " vs "
            << other.rows() << "x" << other.cols();
        throw std::invalid_argument(oss.str());
    }
    matrix<T> result(_rows, _cols);

    const size_t total_elements = _rows * _cols;
    constexpr size_t MIN_PARALLEL_SIZE = 10000;

    const T* a_ptr = this->_data.data();
    const T* b_ptr = other._data.data();
    T* result_ptr = result._data.data();

    if (total_elements >= MIN_PARALLEL_SIZE)
    {
        const size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        const size_t elements_per_thread = (total_elements + num_threads - 1) / num_threads;

        auto worker = [&](size_t start_idx, size_t end_idx)
        {
            for (size_t i = start_idx; i < end_idx; ++i)
            {
                result_ptr[i] = a_ptr[i] * b_ptr[i];
            }
        };

        for (size_t t = 0; t < num_threads; ++t)
        {
            size_t start = t * elements_per_thread;
            size_t end = std::min(start + elements_per_thread, total_elements);
            if (start < end)
            {
                threads.emplace_back(worker, start, end);
            }
        }

        for (auto& t : threads)
        {
            t.join();
        }
    }
    else
    {
        for (size_t i = 0; i < total_elements; ++i)
        {
            result_ptr[i] = a_ptr[i] * b_ptr[i];
        }
    }

    return result;
}

// Matrix and Scalar

/**
 * @brief Matrix addition with scalar
 * @param scalar Scalar to add
 * @return New matrix with the result
 * @details Time O(m*n), Space O(m*n)
 */
template <typename T>
matrix<T> matrix<T>::operator+(const T& scalar) const
{
    matrix<T> result(_rows, _cols);
    const size_t total_elements = _rows * _cols;
    constexpr size_t MIN_PARALLEL_SIZE = 10000;

    const T* a_ptr = this->_data.data();
    T* result_ptr = result._data.data();

    if (total_elements >= MIN_PARALLEL_SIZE)
    {
        const size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        const size_t elements_per_thread = (total_elements + num_threads - 1) / num_threads;

        auto worker = [&](size_t start_idx, size_t end_idx)
        {
            for (size_t i = start_idx; i < end_idx; ++i)
            {
                result_ptr[i] = a_ptr[i] + scalar;
            }
        };

        for (size_t t = 0; t < num_threads; ++t)
        {
            size_t start = t * elements_per_thread;
            size_t end = std::min(start + elements_per_thread, total_elements);
            if (start < end)
            {
                threads.emplace_back(worker, start, end);
            }
        }

        for (auto& t : threads)
        {
            t.join();
        }
    }
    else
    {
        for (size_t i = 0; i < total_elements; ++i)
        {
            result_ptr[i] = a_ptr[i] + scalar;
        }
    }

    return result;
}

/**
 * @brief Matrix subtraction with scalar
 *
 * @param scalar Scalar to subtract
 * @return New matrix with the result
 * @details Time O(m*n), Space O(m*n)
 */
template <typename T>
matrix<T> matrix<T>::operator-(const T& scalar) const
{
    matrix<T> result(_rows, _cols);
    const size_t total_elements = _rows * _cols;
    constexpr size_t MIN_PARALLEL_SIZE = 10000;

    const T* a_ptr = this->_data.data();
    T* result_ptr = result._data.data();

    if (total_elements >= MIN_PARALLEL_SIZE)
    {
        const size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        const size_t elements_per_thread = (total_elements + num_threads - 1) / num_threads;

        auto worker = [&](size_t start_idx, size_t end_idx)
        {
            for (size_t i = start_idx; i < end_idx; ++i)
            {
                result_ptr[i] = a_ptr[i] - scalar;
            }
        };

        for (size_t t = 0; t < num_threads; ++t)
        {
            size_t start = t * elements_per_thread;
            size_t end = std::min(start + elements_per_thread, total_elements);
            if (start < end)
            {
                threads.emplace_back(worker, start, end);
            }
        }

        for (auto& t : threads)
        {
            t.join();
        }
    }
    else
    {
        for (size_t i = 0; i < total_elements; ++i)
        {
            result_ptr[i] = a_ptr[i] - scalar;
        }
    }

    return result;
}

/**
 * @brief Matrix multiplication with scalar
 *
 * @param scalar Scalar to multiply
 * @return New matrix with the result
 * @details Time O(m*n), Space O(m*n)
 */
template <typename T>
matrix<T> matrix<T>::operator*(const T& scalar) const
{
    matrix<T> result(_rows, _cols);
    const size_t total_elements = _rows * _cols;
    constexpr size_t MIN_PARALLEL_SIZE = 10000;

    const T* a_ptr = this->_data.data();
    T* result_ptr = result._data.data();

    if (total_elements >= MIN_PARALLEL_SIZE)
    {
        const size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        const size_t elements_per_thread = (total_elements + num_threads - 1) / num_threads;

        auto worker = [&](size_t start_idx, size_t end_idx)
        {
            for (size_t i = start_idx; i < end_idx; ++i)
            {
                result_ptr[i] = a_ptr[i] * scalar;
            }
        };

        for (size_t t = 0; t < num_threads; ++t)
        {
            size_t start = t * elements_per_thread;
            size_t end = std::min(start + elements_per_thread, total_elements);
            if (start < end)
            {
                threads.emplace_back(worker, start, end);
            }
        }

        for (auto& t : threads)
        {
            t.join();
        }
    }
    else
    {
        for (size_t i = 0; i < total_elements; ++i)
        {
            result_ptr[i] = a_ptr[i] * scalar;
        }
    }

    return result;
}

/**
 * @brief Matrix division by scalar
 *
 * @param scalar Scalar to divide
 * @return New matrix with the result
 * @details Time O(m*n), Space O(m*n)
 */
template <typename T>
matrix<T> matrix<T>::operator/(const T& scalar) const
{
    if (scalar == 0)
    {
        throw std::invalid_argument("Division by zero");
    }
    matrix<T> result(_rows, _cols);
    const size_t total_elements = _rows * _cols;
    constexpr size_t MIN_PARALLEL_SIZE = 10000;

    const T* a_ptr = this->_data.data();
    T* result_ptr = result._data.data();

    if (total_elements >= MIN_PARALLEL_SIZE)
    {
        const size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        const size_t elements_per_thread = (total_elements + num_threads - 1) / num_threads;

        auto worker = [&](size_t start_idx, size_t end_idx)
        {
            for (size_t i = start_idx; i < end_idx; ++i)
            {
                result_ptr[i] = a_ptr[i] / scalar;
            }
        };

        for (size_t t = 0; t < num_threads; ++t)
        {
            size_t start = t * elements_per_thread;
            size_t end = std::min(start + elements_per_thread, total_elements);
            if (start < end)
            {
                threads.emplace_back(worker, start, end);
            }
        }

        for (auto& t : threads)
        {
            t.join();
        }
    }
    else
    {
        for (size_t i = 0; i < total_elements; ++i)
        {
            result_ptr[i] = a_ptr[i] / scalar;
        }
    }

    return result;
}

/**
 * @brief Matrix addition assignment with scalar
 *
 * @param scalar Scalar to add
 * @return New matrix with the result
 * @details Time O(m*n), Space O(m*n)
 */
template <typename T>
matrix<T> matrix<T>::operator+=(const T& scalar)
{
    const size_t total_elements = _rows * _cols;
    constexpr size_t MIN_PARALLEL_SIZE = 10000;

    T* data_ptr = this->_data.data();

    if (total_elements >= MIN_PARALLEL_SIZE)
    {
        const size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        const size_t elements_per_thread = (total_elements + num_threads - 1) / num_threads;

        auto worker = [&](size_t start_idx, size_t end_idx)
        {
            for (size_t i = start_idx; i < end_idx; ++i)
            {
                data_ptr[i] += scalar;
            }
        };

        for (size_t t = 0; t < num_threads; ++t)
        {
            size_t start = t * elements_per_thread;
            size_t end = std::min(start + elements_per_thread, total_elements);
            if (start < end)
            {
                threads.emplace_back(worker, start, end);
            }
        }

        for (auto& t : threads)
        {
            t.join();
        }
    }
    else
    {
        for (size_t i = 0; i < total_elements; ++i)
        {
            data_ptr[i] += scalar;
        }
    }

    return *this;
}

/**
 * @brief Matrix subtraction assignment with scalar
 *
 * @param scalar Scalar to subtract
 * @return New matrix with the result
 * @details Time O(m*n), Space O(m*n)
 */
template <typename T>
matrix<T> matrix<T>::operator-=(const T& scalar)
{
    const size_t total_elements = _rows * _cols;
    constexpr size_t MIN_PARALLEL_SIZE = 10000;

    T* data_ptr = this->_data.data();

    if (total_elements >= MIN_PARALLEL_SIZE)
    {
        const size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        const size_t elements_per_thread = (total_elements + num_threads - 1) / num_threads;

        auto worker = [&](size_t start_idx, size_t end_idx)
        {
            for (size_t i = start_idx; i < end_idx; ++i)
            {
                data_ptr[i] -= scalar;
            }
        };

        for (size_t t = 0; t < num_threads; ++t)
        {
            size_t start = t * elements_per_thread;
            size_t end = std::min(start + elements_per_thread, total_elements);
            if (start < end)
            {
                threads.emplace_back(worker, start, end);
            }
        }

        for (auto& t : threads)
        {
            t.join();
        }
    }
    else
    {
        for (size_t i = 0; i < total_elements; ++i)
        {
            data_ptr[i] -= scalar;
        }
    }

    return *this;
}

/**
 * @brief Matrix multiplication assignment with scalar
 *
 * @param scalar Scalar to multiply
 * @return New matrix with the result
 * @details Time O(m*n), Space O(m*n)
 */
template <typename T>
matrix<T> matrix<T>::operator*=(const T& scalar)
{
    const size_t total_elements = _rows * _cols;
    constexpr size_t MIN_PARALLEL_SIZE = 10000;

    T* data_ptr = this->_data.data();

    if (total_elements >= MIN_PARALLEL_SIZE)
    {
        const size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        const size_t elements_per_thread = (total_elements + num_threads - 1) / num_threads;

        auto worker = [&](size_t start_idx, size_t end_idx)
        {
            for (size_t i = start_idx; i < end_idx; ++i)
            {
                data_ptr[i] *= scalar;
            }
        };

        for (size_t t = 0; t < num_threads; ++t)
        {
            size_t start = t * elements_per_thread;
            size_t end = std::min(start + elements_per_thread, total_elements);
            if (start < end)
            {
                threads.emplace_back(worker, start, end);
            }
        }

        for (auto& t : threads)
        {
            t.join();
        }
    }
    else
    {
        for (size_t i = 0; i < total_elements; ++i)
        {
            data_ptr[i] *= scalar;
        }
    }

    return *this;
}

/**
 * @brief Matrix division assignment with scalar
 *
 * @param scalar Scalar to divide
 * @return New matrix with the result
 * @details Time O(m*n), Space O(m*n)
 */
template <typename T>
matrix<T> matrix<T>::operator/=(const T& scalar)
{
    if (scalar == 0)
    {
        throw std::invalid_argument("Division by zero");
    }
    const size_t total_elements = _rows * _cols;
    constexpr size_t MIN_PARALLEL_SIZE = 10000;

    T* data_ptr = this->_data.data();

    if (total_elements >= MIN_PARALLEL_SIZE)
    {
        const size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        const size_t elements_per_thread = (total_elements + num_threads - 1) / num_threads;

        auto worker = [&](size_t start_idx, size_t end_idx)
        {
            for (size_t i = start_idx; i < end_idx; ++i)
            {
                data_ptr[i] /= scalar;
            }
        };

        for (size_t t = 0; t < num_threads; ++t)
        {
            size_t start = t * elements_per_thread;
            size_t end = std::min(start + elements_per_thread, total_elements);
            if (start < end)
            {
                threads.emplace_back(worker, start, end);
            }
        }

        for (auto& t : threads)
        {
            t.join();
        }
    }
    else
    {
        for (size_t i = 0; i < total_elements; ++i)
        {
            data_ptr[i] /= scalar;
        }
    }

    return *this;
}

// ============================================================================
// BLAS-optimized specializations for double and float
// ============================================================================

#ifdef MATRIX_USE_BLAS

// BLAS-optimized matrix multiplication for double
template <>
inline matrix<double> matrix<double>::operator*(const matrix<double>& other) const
{
    if (_cols != other._rows)
    {
        std::ostringstream oss;
        oss << "Matrix dimensions do not match for multiplication: " << _rows << "x" << _cols << " * " << other._rows
            << "x" << other._cols;
        throw std::invalid_argument(oss.str());
    }

    matrix<double> result(_rows, other._cols);

    // BLAS dgemm: C = alpha*A*B + beta*C
    // We use row-major order, so we compute C^T = B^T * A^T
    char trans_no = 'N';
    int m = static_cast<int>(other._cols); // columns of B (rows of result)
    int n = static_cast<int>(_rows);       // rows of A (columns of result)
    int k = static_cast<int>(_cols);       // columns of A = rows of B
    double alpha = 1.0;
    double beta = 0.0;
    int lda = static_cast<int>(other._cols); // leading dimension of B
    int ldb = static_cast<int>(_cols);       // leading dimension of A
    int ldc = static_cast<int>(other._cols); // leading dimension of C

    dgemm_(&trans_no, &trans_no, &m, &n, &k, &alpha, other._data.data(), &lda, this->_data.data(), &ldb, &beta,
           result._data.data(), &ldc);

    return result;
}

// BLAS-optimized matrix multiplication for float
template <>
inline matrix<float> matrix<float>::operator*(const matrix<float>& other) const
{
    if (_cols != other._rows)
    {
        std::ostringstream oss;
        oss << "Matrix dimensions do not match for multiplication: " << _rows << "x" << _cols << " * " << other._rows
            << "x" << other._cols;
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

    sgemm_(&trans_no, &trans_no, &m, &n, &k, &alpha, other._data.data(), &lda, this->_data.data(), &ldb, &beta,
           result._data.data(), &ldc);

    return result;
}

#endif // MATRIX_USE_BLAS
