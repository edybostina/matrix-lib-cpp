#pragma once

// Template implementations for matrix_algorithms.hpp
// This file is included at the end of matrix_algorithms.hpp

/**
 * @brief Computes the determinant of a square matrix.
 *
 * Optimizations: Direct formulas for 1x1, 2x2, 3x3; Gaussian elimination for
 * larger matrices.
 *
 * @return Determinant value as double
 * @throws std::invalid_argument If matrix is not square
 * @details Time O(n³) for n>3, O(1) for n≤3
 */
template <typename T>
double matrix<T>::determinant() const
{
    if (_rows != _cols)
    {
        std::ostringstream oss;
        oss << "Matrix is " << _rows << "x" << _cols << ", must be square to compute determinant";
        throw std::invalid_argument(oss.str());
    }
    if (_rows == 1)
    {
        return (*this)(0, 0);
    }
    if (_rows == 2)
    {
        return (*this)(0, 0) * (*this)(1, 1) - (*this)(0, 1) * (*this)(1, 0);
    }
    if (_rows == 3)
    {
        return (*this)(0, 0) * ((*this)(1, 1) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 1)) -
               (*this)(0, 1) * ((*this)(1, 0) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 0)) +
               (*this)(0, 2) * ((*this)(1, 0) * (*this)(2, 1) - (*this)(1, 1) * (*this)(2, 0));
    }

    double det = 1.0;
    matrix<double> temp = (matrix<double>)*this;
    for (size_t i = 0; i < _rows; ++i)
    {
        size_t pivot = i;
        for (size_t j = i + 1; j < _rows; ++j)
        {
            if (std::abs(temp(j, i)) > std::abs(temp(pivot, i)))
            {
                pivot = j;
            }
        }
        if (pivot != i)
        {
            temp.swap_rows(i, pivot);
            det *= -1;
        }
        if (std::abs(temp(i, i)) < std::numeric_limits<double>::epsilon())
        {
            return 0;
        }
        det *= temp(i, i);
        for (size_t j = i + 1; j < _rows; ++j)
        {
            double factor = temp(j, i) / temp(i, i);
            for (size_t k = i + 1; k < _cols; ++k)
            {
                temp(j, k) -= factor * temp(i, k);
            }
        }
    }
    return det;
}

/**
 * @brief Computes the trace of a square matrix.
 *
 * Optimizations: SIMD for diagonal access (AVX2/NEON), direct memory access.
 *
 * @return Trace value as T
 * @throws std::invalid_argument If matrix is not square
 * @details Time O(n), Space O(1)
 */
template <typename T>
T matrix<T>::trace() const
{
    if (_rows != _cols)
    {
        std::ostringstream oss;
        oss << "Matrix is " << _rows << "x" << _cols << ", must be square to compute trace";
        throw std::invalid_argument(oss.str());
    }

#ifdef MATRIX_USE_SIMD
    if constexpr (std::is_same<T, double>::value)
    {
#if defined(__AVX2__)
        double sum = 0.0;
        const double* data_ptr = _data.data();
        const size_t stride = _cols + 1;
        size_t i = 0;

        __m256d sum_vec = _mm256_setzero_pd();

        for (; i + 3 < _rows; i += 4)
        {
            __m256d vals = _mm256_set_pd(data_ptr[(i + 3) * stride], data_ptr[(i + 2) * stride],
                                         data_ptr[(i + 1) * stride], data_ptr[i * stride]);
            sum_vec = _mm256_add_pd(sum_vec, vals);
        }

        // Horizontal sum
        double temp[4];
        _mm256_storeu_pd(temp, sum_vec);
        sum = temp[0] + temp[1] + temp[2] + temp[3];

        // Handle remaining diagonal elements
        for (; i < _rows; ++i)
        {
            sum += data_ptr[i * stride];
        }

        return static_cast<T>(sum);
#elif defined(__ARM_NEON)
        double sum = 0.0;
        const double* data_ptr = _data.data();
        const size_t stride = _cols + 1;
        size_t i = 0;

        float64x2_t sum_vec = vdupq_n_f64(0.0);

        // Process 2 diagonal elements at a time
        for (; i + 1 < _rows; i += 2)
        {
            float64x2_t vals = {data_ptr[i * stride], data_ptr[(i + 1) * stride]};
            sum_vec = vaddq_f64(sum_vec, vals);
        }

        // Extract sum
        sum = vgetq_lane_f64(sum_vec, 0) + vgetq_lane_f64(sum_vec, 1);

        // Handle remaining diagonal elements
        for (; i < _rows; ++i)
        {
            sum += data_ptr[i * stride];
        }

        return static_cast<T>(sum);
#endif
    }
    // SIMD optimization for float type
    else if constexpr (std::is_same<T, float>::value)
    {
#if defined(__AVX2__)
        float sum = 0.0f;
        const float* data_ptr = _data.data();
        const size_t stride = _cols + 1;
        size_t i = 0;

        __m256 sum_vec = _mm256_setzero_ps();

        // Process 8 diagonal elements at a time
        for (; i + 7 < _rows; i += 8)
        {
            __m256 vals =
                _mm256_set_ps(data_ptr[(i + 7) * stride], data_ptr[(i + 6) * stride], data_ptr[(i + 5) * stride],
                              data_ptr[(i + 4) * stride], data_ptr[(i + 3) * stride], data_ptr[(i + 2) * stride],
                              data_ptr[(i + 1) * stride], data_ptr[i * stride]);
            sum_vec = _mm256_add_ps(sum_vec, vals);
        }

        // Horizontal sum
        float temp[8];
        _mm256_storeu_ps(temp, sum_vec);
        sum = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];

        // Handle remaining diagonal elements
        for (; i < _rows; ++i)
        {
            sum += data_ptr[i * stride];
        }

        return static_cast<T>(sum);
#elif defined(__ARM_NEON)
        float sum = 0.0f;
        const float* data_ptr = _data.data();
        const size_t stride = _cols + 1;
        size_t i = 0;

        float32x4_t sum_vec = vdupq_n_f32(0.0f);

        // Process 4 diagonal elements at a time
        for (; i + 3 < _rows; i += 4)
        {
            float32x4_t vals = {data_ptr[i * stride], data_ptr[(i + 1) * stride], data_ptr[(i + 2) * stride],
                                data_ptr[(i + 3) * stride]};
            sum_vec = vaddq_f32(sum_vec, vals);
        }

        // Extract sum
        float temp[4];
        vst1q_f32(temp, sum_vec);
        sum = temp[0] + temp[1] + temp[2] + temp[3];

        // Handle remaining diagonal elements
        for (; i < _rows; ++i)
        {
            sum += data_ptr[i * stride];
        }

        return static_cast<T>(sum);
#endif
    }
#endif

    // Fallback: scalar implementation
    T sum = 0;
    for (size_t i = 0; i < _rows; ++i)
    {
        sum += (*this)(i, i);
    }
    return sum;
}

/**
 * @brief Computes the transpose of the matrix.
 *
 * Optimizations: Cache blocking (tiling) + multi-threading for large matrices.
 *
 * @return Transposed matrix
 * @details Time O(m*n), Space O(m*n)
 */
template <typename T>
matrix<T> matrix<T>::transpose() const
{
    matrix<T> result(_cols, _rows);
    const size_t total_elements = _rows * _cols;
    constexpr size_t MIN_PARALLEL_SIZE = 10000;
    constexpr size_t BLOCK_SIZE = 64; // Cache-friendly block size

    if (total_elements >= MIN_PARALLEL_SIZE)
    {
        size_t num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0)
            num_threads = 4;

        std::vector<std::thread> threads;

        // Divide work by row blocks
        size_t row_blocks = (_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        size_t blocks_per_thread = (row_blocks + num_threads - 1) / num_threads;

        for (size_t t = 0; t < num_threads; ++t)
        {
            threads.emplace_back(
                [=, &result]()
                {
                    size_t start_block = t * blocks_per_thread;
                    size_t end_block = std::min(start_block + blocks_per_thread, row_blocks);

                    for (size_t block_i = start_block; block_i < end_block; ++block_i)
                    {
                        size_t i_start = block_i * BLOCK_SIZE;
                        size_t i_end = std::min(i_start + BLOCK_SIZE, _rows);

                        for (size_t block_j = 0; block_j < (_cols + BLOCK_SIZE - 1) / BLOCK_SIZE; ++block_j)
                        {
                            size_t j_start = block_j * BLOCK_SIZE;
                            size_t j_end = std::min(j_start + BLOCK_SIZE, _cols);

                            for (size_t i = i_start; i < i_end; ++i)
                            {
                                for (size_t j = j_start; j < j_end; ++j)
                                {
                                    result(j, i) = (*this)(i, j);
                                }
                            }
                        }
                    }
                });
        }

        for (auto& thread : threads)
        {
            thread.join();
        }
    }
    else
    {
        // Cache blocking for small matrices (no threading)
        for (size_t ii = 0; ii < _rows; ii += BLOCK_SIZE)
        {
            for (size_t jj = 0; jj < _cols; jj += BLOCK_SIZE)
            {
                for (size_t i = ii; i < std::min(ii + BLOCK_SIZE, _rows); ++i)
                {
                    for (size_t j = jj; j < std::min(jj + BLOCK_SIZE, _cols); ++j)
                    {
                        result(j, i) = (*this)(i, j);
                    }
                }
            }
        }
    }
    return result;
}

/** * @brief Computes the cofactor matrix.
 *
 * Optimizations: Multi-threading for matrices >= 4x4 (embarrassingly parallel).
 *
 * @return Cofactor matrix
 * @details Time O(n^4), Space O(n^2)
 */
template <typename T>
matrix<T> matrix<T>::cofactor() const
{
    matrix<T> result(_rows, _cols);

    // Parallelize for matrices 4x4 and larger
    constexpr size_t MIN_SIZE_FOR_PARALLEL = 4;

    if (_rows >= MIN_SIZE_FOR_PARALLEL)
    {
        size_t num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0)
            num_threads = 4;

        std::vector<std::thread> threads;
        size_t rows_per_thread = (_rows + num_threads - 1) / num_threads;

        for (size_t t = 0; t < num_threads; ++t)
        {
            threads.emplace_back(
                [=, &result]()
                {
                    size_t start_row = t * rows_per_thread;
                    size_t end_row = std::min(start_row + rows_per_thread, _rows);

                    for (size_t i = start_row; i < end_row; ++i)
                    {
                        for (size_t j = 0; j < _cols; ++j)
                        {
                            matrix<T> minor_mat = this->minor(i, j);
                            result(i, j) = ((i + j) % 2 == 0 ? 1 : -1) * minor_mat.determinant();
                        }
                    }
                });
        }

        for (auto& thread : threads)
        {
            thread.join();
        }
    }
    else
    {
        for (size_t i = 0; i < _rows; ++i)
        {
            for (size_t j = 0; j < _cols; ++j)
            {
                matrix<T> minor_mat = this->minor(i, j);
                result(i, j) = ((i + j) % 2 == 0 ? 1 : -1) * minor_mat.determinant();
            }
        }
    }

    return result;
}

/** * @brief Computes the minor matrix by removing specified row and column.
 *
 * @param row Row index to remove
 * @param col Column index to remove
 * @return Minor matrix
 * @details Time O(n^2), Space O(n^2)
 */
template <typename T>
matrix<T> matrix<T>::minor(size_t row, size_t col) const
{
    matrix<T> result(_rows - 1, _cols - 1);
    for (size_t i = 0, r = 0; i < _rows; ++i)
    {
        if (i == row)
            continue;
        for (size_t j = 0, c = 0; j < _cols; ++j)
        {
            if (j == col)
                continue;
            result(r, c) = (*this)(i, j);
            ++c;
        }
        ++r;
    }
    return result;
}

/** * @brief Computes the adjoint (adjugate) of the matrix.
 *
 * @return Adjoint matrix
 * @throws std::invalid_argument If matrix is not square
 * @details Time O(n^4), Space O(n^2)
 */
template <typename T>
matrix<T> matrix<T>::adjoint() const
{
    if (_rows != _cols)
    {
        std::ostringstream oss;
        oss << "Matrix is " << _rows << "x" << _cols << ", must be square to compute adjoint";
        throw std::invalid_argument(oss.str());
    }
    matrix<T> result(_rows, _cols);
    result = this->cofactor().transpose();
    return result;
}

/** * @brief Computes the inverse of the matrix.
 *
 * Optimizations: Uses LU decomposition + parallel column solving for better
 * performance and numerical stability.
 *
 * @return Inverse matrix
 * @throws std::invalid_argument If matrix is not square or singular
 * @details Time O(n^3), Space O(n^2)
 */
template <typename T>
matrix<double> matrix<T>::inverse() const
{
    if (_rows != _cols)
    {
        std::ostringstream oss;
        oss << "Matrix is " << _rows << "x" << _cols << ", must be square to compute inverse";
        throw std::invalid_argument(oss.str());
    }

    // Use LU decomposition for better numerical stability
    matrix<double> L, U;
    try
    {
        std::tie(L, U) = this->LU_decomposition();
    }
    catch (const std::exception&)
    {
        throw std::invalid_argument("Matrix is singular and cannot be inverted");
    }

    // Check for singularity
    const double eps = std::numeric_limits<double>::epsilon() * 100;
    for (size_t i = 0; i < _rows; ++i)
    {
        if (std::abs(L(i, i)) < eps || std::abs(U(i, i)) < eps)
        {
            throw std::invalid_argument("Matrix is singular and cannot be inverted");
        }
    }

    matrix<double> inv(_rows, _cols);

    // Solve for each column of the inverse in parallel
    constexpr size_t MIN_SIZE_FOR_PARALLEL = 8;

    if (_rows >= MIN_SIZE_FOR_PARALLEL)
    {
        size_t num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0)
            num_threads = 4;

        std::vector<std::thread> threads;
        size_t cols_per_thread = (_cols + num_threads - 1) / num_threads;

        for (size_t t = 0; t < num_threads; ++t)
        {
            threads.emplace_back(
                [=, &inv, &L, &U]()
                {
                    size_t start_col = t * cols_per_thread;
                    size_t end_col = std::min(start_col + cols_per_thread, _cols);

                    for (size_t k = start_col; k < end_col; ++k)
                    {
                        // Create k-th unit vector
                        matrix<double> e(_rows, 1);
                        e(k, 0) = 1.0;

                        // Forward substitution: solve Ly = e
                        matrix<double> y(_rows, 1);
                        for (size_t i = 0; i < _rows; ++i)
                        {
                            double sum = e(i, 0);
                            for (size_t j = 0; j < i; ++j)
                            {
                                sum -= L(i, j) * y(j, 0);
                            }
                            y(i, 0) = sum / L(i, i);
                        }

                        // Backward substitution: solve Ux = y
                        matrix<double> x(_rows, 1);
                        for (int i = static_cast<int>(_rows) - 1; i >= 0; --i)
                        {
                            double sum = y(i, 0);
                            for (size_t j = i + 1; j < _cols; ++j)
                            {
                                sum -= U(i, j) * x(j, 0);
                            }
                            x(i, 0) = sum / U(i, i);
                        }

                        // Store column in result
                        for (size_t i = 0; i < _rows; ++i)
                        {
                            inv(i, k) = x(i, 0);
                        }
                    }
                });
        }

        for (auto& thread : threads)
        {
            thread.join();
        }
    }
    else
    {
        // Serial computation for small matrices
        for (size_t k = 0; k < _cols; ++k)
        {
            // Create k-th unit vector
            matrix<double> e(_rows, 1);
            e(k, 0) = 1.0;

            // Forward substitution: solve Ly = e
            matrix<double> y(_rows, 1);
            for (size_t i = 0; i < _rows; ++i)
            {
                double sum = e(i, 0);
                for (size_t j = 0; j < i; ++j)
                {
                    sum -= L(i, j) * y(j, 0);
                }
                y(i, 0) = sum / L(i, i);
            }

            // Backward substitution: solve Ux = y
            matrix<double> x(_rows, 1);
            for (int i = static_cast<int>(_rows) - 1; i >= 0; --i)
            {
                double sum = y(i, 0);
                for (size_t j = i + 1; j < _cols; ++j)
                {
                    sum -= U(i, j) * x(j, 0);
                }
                x(i, 0) = sum / U(i, i);
            }

            // Store column in result
            for (size_t i = 0; i < _rows; ++i)
            {
                inv(i, k) = x(i, 0);
            }
        }
    }

    return inv;
}

/**
 * @brief Computes the p-norm of the matrix.
 *
 * Optimizations: SIMD (AVX2/NEON) for p=2 (Frobenius norm), direct memory access.
 *
 * @param p Norm order (must be >= 1)
 * @return Norm value as double
 * @throws std::invalid_argument If p < 1
 * @details Time O(m*n), Space O(1)
 */
template <typename T>
double matrix<T>::norm(int p) const
{
    if (p < 1)
    {
        throw std::invalid_argument("Norm order must be greater than or equal to 1");
    }
    if (p == 2)
    {
        double sum = 0.0;
        const size_t total = _rows * _cols;

#ifdef MATRIX_USE_SIMD
        if constexpr (std::is_same<T, double>::value)
        {
#if defined(__AVX2__)
            const double* data_ptr = _data.data();
            size_t i = 0;

            __m256d sum_vec = _mm256_setzero_pd();
            for (; i + 3 < total; i += 4)
            {
                __m256d vals = _mm256_loadu_pd(&data_ptr[i]);
                sum_vec = _mm256_fmadd_pd(vals, vals, sum_vec);
            }

            double temp[4];
            _mm256_storeu_pd(temp, sum_vec);
            sum = temp[0] + temp[1] + temp[2] + temp[3];

            for (; i < total; ++i)
            {
                sum += data_ptr[i] * data_ptr[i];
            }

            return std::sqrt(sum);
#elif defined(__ARM_NEON)
            const double* data_ptr = _data.data();
            size_t i = 0;

            float64x2_t sum_vec = vdupq_n_f64(0.0);
            for (; i + 1 < total; i += 2)
            {
                float64x2_t vals = vld1q_f64(&data_ptr[i]);
                sum_vec = vfmaq_f64(sum_vec, vals, vals);
            }

            sum = vgetq_lane_f64(sum_vec, 0) + vgetq_lane_f64(sum_vec, 1);

            for (; i < total; ++i)
            {
                sum += data_ptr[i] * data_ptr[i];
            }

            return std::sqrt(sum);
#endif
        }
        else if constexpr (std::is_same<T, float>::value)
        {
#if defined(__AVX2__)
            const float* data_ptr = _data.data();
            size_t i = 0;

            __m256 sum_vec = _mm256_setzero_ps();
            for (; i + 7 < total; i += 8)
            {
                __m256 vals = _mm256_loadu_ps(&data_ptr[i]);
                sum_vec = _mm256_fmadd_ps(vals, vals, sum_vec);
            }

            // Horizontal sum
            float temp[8];
            _mm256_storeu_ps(temp, sum_vec);
            sum = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];

            // Handle remaining elements
            for (; i < total; ++i)
            {
                sum += data_ptr[i] * data_ptr[i];
            }

            return std::sqrt(sum);
#elif defined(__ARM_NEON)
            const float* data_ptr = _data.data();
            size_t i = 0;

            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            for (; i + 3 < total; i += 4)
            {
                float32x4_t vals = vld1q_f32(&data_ptr[i]);
                sum_vec = vfmaq_f32(sum_vec, vals, vals);
            }

            float temp[4];
            vst1q_f32(temp, sum_vec);
            sum = temp[0] + temp[1] + temp[2] + temp[3];

            for (; i < total; ++i)
            {
                sum += data_ptr[i] * data_ptr[i];
            }

            return std::sqrt(sum);
#endif
        }
#endif

        for (size_t i = 0; i < total; ++i)
        {
            double val = static_cast<double>(_data[i]);
            sum += val * val;
        }
        return std::sqrt(sum);
    }

    double norm = 0;
    for (size_t i = 0; i < _rows; ++i)
    {
        for (size_t j = 0; j < _cols; ++j)
        {
            T absolute = (*this)(i, j) < 0 ? -(*this)(i, j) : (*this)(i, j);
            norm += std::pow(absolute, p);
        }
    }
    return std::pow(norm, 1.0 / p);
}

/**
 * @brief Computes the rank of the matrix.
 *
 * Optimizations: Uses Gaussian elimination with numerical tolerance for
 * stability.
 *
 * @return Rank as size_t
 * @details Time O(m*n*min(m,n)), Space O(m*n)
 */
template <typename T>
size_t matrix<T>::rank() const
{
    matrix<double> gaussian = this->gaussian_elimination();

    double max_val = 0.0;
    for (size_t i = 0; i < _rows; ++i)
    {
        for (size_t j = 0; j < _cols; ++j)
        {
            max_val = std::max(max_val, std::abs(gaussian(i, j)));
        }
    }

    double tolerance = std::max(1e-10 * max_val, std::numeric_limits<double>::epsilon());

    size_t rank = 0;
    for (size_t i = 0; i < _rows; ++i)
    {
        bool non_zero_row = false;
        for (size_t j = 0; j < _cols; ++j)
        {
            if (std::abs(gaussian(i, j)) > tolerance)
            {
                non_zero_row = true;
                break;
            }
        }
        if (non_zero_row)
        {
            ++rank;
        }
    }
    return rank;
}

/**
 * @brief Performs Gaussian elimination to obtain row echelon form.
 *
 * Optimizations: Partial pivoting for numerical stability.
 *
 * @return Matrix in row echelon form
 * @throws std::invalid_argument If matrix has zero dimensions
 * @details Time O(m*n*min(m,n)), Space O(m*n)
 */
template <typename T>
matrix<double> matrix<T>::gaussian_elimination() const
{
    if (_rows == 0 || _cols == 0)
    {
        std::ostringstream oss;
        oss << "Matrix dimensions must be positive, got " << _rows << "x" << _cols;
        throw std::invalid_argument(oss.str());
    }
    matrix<double> result = (matrix<double>)*this;
    for (size_t i = 0; i < _rows; ++i)
    {
        size_t pivot_row = i;
        for (size_t j = i + 1; j < _rows; ++j)
        {
            if (std::abs(result(j, i)) > std::abs(result(pivot_row, i)))
            {
                pivot_row = j;
            }
        }
        if (std::abs(result(pivot_row, i)) < std::numeric_limits<double>::epsilon())
        {
            continue;
        }
        if (pivot_row != i)
        {
            result.swap_rows(i, pivot_row);
        }
        for (size_t j = i + 1; j < _rows; ++j)
        {
            double factor = result(j, i) / result(i, i);
            for (size_t k = i; k < _cols; ++k)
            {
                result(j, k) -= factor * result(i, k);
            }
        }
    }
    return result;
}

/**
 * @brief Computes LU decomposition (A = LU).
 *
 * @return Pair of (L, U) where L is lower triangular, U is upper triangular
 * @throws std::invalid_argument If matrix is not square
 * @details Time O(n³), Space O(n²)
 */
template <typename T>
std::pair<matrix<double>, matrix<double>> matrix<T>::LU_decomposition() const
{
    if (_rows != _cols)
    {
        std::ostringstream oss;
        oss << "Matrix is " << _rows << "x" << _cols << ", must be square for LU decomposition";
        throw std::invalid_argument(oss.str());
    }
    matrix<double> L(_rows, _cols);
    matrix<double> U = matrix<double>::eye(_rows, _cols);
    for (size_t p = 0; p < _rows; ++p)
    {
        for (size_t i = 0; i < p; ++i)
        {
            U(i, p) = (*this)(i, p) - (L.row(i).to_matrix() * U.col(p).to_matrix())(0, 0);
            U(i, p) /= L(i, i);
        }
        for (size_t i = p; i < _rows; ++i)
        {
            L(i, p) = (double)(*this)(i, p) - (L.row(i).to_matrix() * U.col(p).to_matrix())(0, 0);
        }
    }

    return std::make_pair(L, U);
}

/**
 * @brief Computes QR decomposition using Gram-Schmidt process (A = QR).
 *
 * Optimizations: Modified Gram-Schmidt for numerical stability.
 *
 * @return Pair of (Q, R) where Q is orthogonal, R is upper triangular
 * @details Time O(m*n²), Space O(m*n)
 */
template <typename T>
std::pair<matrix<double>, matrix<double>> matrix<T>::QR_decomposition() const
{
    matrix<double> A = static_cast<matrix<double>>(*this);
    matrix<double> Q(_rows, _cols);
    matrix<double> R(_cols, _cols);

    // Modified Gram-Schmidt process
    for (size_t j = 0; j < _cols; ++j)
    {
        // Copy column j of A to column j of Q
        for (size_t i = 0; i < _rows; ++i)
        {
            Q(i, j) = A(i, j);
        }

        // Orthogonalize against all previous columns
        for (size_t k = 0; k < j; ++k)
        {
            // R[k,j] = Q[:,k]^T * Q[:,j]
            double dot = 0.0;
            for (size_t i = 0; i < _rows; ++i)
            {
                dot += Q(i, k) * Q(i, j);
            }
            R(k, j) = dot;

            // Q[:,j] = Q[:,j] - R[k,j] * Q[:,k]
            for (size_t i = 0; i < _rows; ++i)
            {
                Q(i, j) -= R(k, j) * Q(i, k);
            }
        }

        // Normalize column j
        double norm = 0.0;
        for (size_t i = 0; i < _rows; ++i)
        {
            norm += Q(i, j) * Q(i, j);
        }
        norm = std::sqrt(norm);

        R(j, j) = norm;

        if (norm > std::numeric_limits<double>::epsilon())
        {
            for (size_t i = 0; i < _rows; ++i)
            {
                Q(i, j) /= norm;
            }
        }
    }

    return std::make_pair(Q, R);
}

/**
 * @brief Computes eigenvalues using QR algorithm with Wilkinson shift.
 *
 * Implements the QR algorithm with the following improvements:
 * - Wilkinson shift for faster convergence (quadratic vs. linear)
 * - Deflation to isolate converged eigenvalues
 * - Early termination when off-diagonal elements are sufficiently small
 * - Better numerical stability through adaptive tolerance
 *
 * For symmetric matrices, consider using specialized algorithms in the future.
 *
 * @param max_iter Maximum iterations for convergence (default 1000)
 * @return Column matrix containing eigenvalues (sorted by magnitude, descending)
 * @throws std::invalid_argument If matrix is not square
 * @details Time O(max_iter * n³), Space O(n²)
 * @note Does not support complex eigenvalues yet; for matrices with complex
 *       eigenvalues, results may be inaccurate or fail to converge
 */
template <typename T>
matrix<double> matrix<T>::eigenvalues(int max_iter) const
{
    if (_rows != _cols)
    {
        std::ostringstream oss;
        oss << "Matrix is " << _rows << "x" << _cols << ", must be square to compute eigenvalues";
        throw std::invalid_argument(oss.str());
    }

    if (_rows == 1)
    {
        matrix<double> result(1, 1);
        result(0, 0) = static_cast<double>((*this)(0, 0));
        return result;
    }

    matrix<double> A = static_cast<matrix<double>>(*this);

    double frobenius_norm = 0.0;
    for (size_t i = 0; i < _rows; ++i)
    {
        for (size_t j = 0; j < _cols; ++j)
        {
            frobenius_norm += A(i, j) * A(i, j);
        }
    }
    frobenius_norm = std::sqrt(frobenius_norm);

    const double eps = std::numeric_limits<double>::epsilon();
    const double tol = std::max(eps * frobenius_norm, 1e-12);

    size_t n = _rows;
    std::vector<double> eigenvals;
    eigenvals.reserve(n);

    int total_iter = 0;
    while (n > 1 && total_iter < max_iter)
    {
        bool converged = false;

        if (std::abs(A(n - 1, n - 2)) < tol)
        {
            eigenvals.push_back(A(n - 1, n - 1));
            --n;
            converged = true;
        }
        else if (n > 2 && std::abs(A(n - 2, n - 3)) < tol)
        {
            double a = A(n - 2, n - 2);
            double b = A(n - 2, n - 1);
            double c = A(n - 1, n - 2);
            double d = A(n - 1, n - 1);

            double trace = a + d;
            double det = a * d - b * c;
            double discriminant = trace * trace - 4.0 * det;

            if (discriminant >= 0)
            {
                double sqrt_disc = std::sqrt(discriminant);
                eigenvals.push_back((trace + sqrt_disc) / 2.0);
                eigenvals.push_back((trace - sqrt_disc) / 2.0);
            }
            else
            {
                eigenvals.push_back(trace / 2.0);
                eigenvals.push_back(trace / 2.0);
            }

            n -= 2;
            converged = true;
        }

        if (!converged)
        {
            double shift = 0.0;
            if (n >= 2)
            {
                double a = A(n - 2, n - 2);
                double _ = A(n - 2, n - 1);
                double c = A(n - 1, n - 2);
                double d = A(n - 1, n - 1);

                double delta = (a - d) / 2.0;
                double sign = (delta >= 0) ? 1.0 : -1.0;

                shift = d - sign * c * c / (std::abs(delta) + std::sqrt(delta * delta + c * c));
            }
            else
            {
                shift = A(n - 1, n - 1);
            }

            matrix<double> A_sub(n, n);
            for (size_t i = 0; i < n; ++i)
            {
                for (size_t j = 0; j < n; ++j)
                {
                    A_sub(i, j) = A(i, j);
                }
                A_sub(i, i) -= shift;
            }

            matrix<double> Q, R;
            std::tie(Q, R) = A_sub.QR_decomposition();

            A_sub = R * Q;
            for (size_t i = 0; i < n; ++i)
            {
                A_sub(i, i) += shift;
            }

            for (size_t i = 0; i < n; ++i)
            {
                for (size_t j = 0; j < n; ++j)
                {
                    A(i, j) = A_sub(i, j);
                }
            }

            ++total_iter;
        }
    }

    if (n == 1)
    {
        eigenvals.push_back(A(0, 0));
    }
    else if (n == 2)
    {
        double a = A(0, 0);
        double b = A(0, 1);
        double c = A(1, 0);
        double d = A(1, 1);

        double trace = a + d;
        double det = a * d - b * c;
        double discriminant = trace * trace - 4.0 * det;

        if (discriminant >= 0)
        {
            double sqrt_disc = std::sqrt(discriminant);
            eigenvals.push_back((trace + sqrt_disc) / 2.0);
            eigenvals.push_back((trace - sqrt_disc) / 2.0);
        }
        else
        {
            eigenvals.push_back(trace / 2.0);
            eigenvals.push_back(trace / 2.0);
        }
    }

    std::sort(eigenvals.begin(), eigenvals.end(), [](double a, double b) { return std::abs(a) > std::abs(b); });

    matrix<double> result(_rows, 1);
    for (size_t i = 0; i < _rows; ++i)
    {
        result(i, 0) = eigenvals[i];
    }

    return result;
}

/**
 * @brief Computes eigenvectors using inverse iteration for each eigenvalue.
 *
 * Optimizations: Parallel computation of independent eigenvectors for large matrices.
 *
 * For each eigenvalue from eigenvalues(), computes the corresponding eigenvector
 * using inverse iteration with shift. This is more reliable than the QR method
 * for tracking eigenvectors, especially for matrices with close or repeated eigenvalues.
 *
 * @param max_iter Maximum iterations for convergence (default 1000)
 * @return Matrix with eigenvectors as columns
 * @throws std::invalid_argument If matrix is not square
 * @details Time O(n * max_iter * n²), Space O(n²)
 * @note Does not support complex eigenvectors yet; for matrices with complex
 *       eigenvalues, results may be inaccurate
 */
template <typename T>
matrix<double> matrix<T>::eigenvectors(int max_iter) const
{
    if (_rows != _cols)
    {
        std::ostringstream oss;
        oss << "Matrix is " << _rows << "x" << _cols << ", must be square to compute eigenvectors";
        throw std::invalid_argument(oss.str());
    }

    if (_rows == 1)
    {
        matrix<double> result(1, 1);
        result(0, 0) = 1.0;
        return result;
    }

    matrix<double> eigenvals = this->eigenvalues(max_iter);
    matrix<double> A = static_cast<matrix<double>>(*this);
    matrix<double> V(_rows, _cols);

    const double eps = 1e-10;

    auto compute_eigenvector = [&](size_t k)
    {
        double lambda = eigenvals(k, 0);

        matrix<double> A_shifted = A;
        for (size_t i = 0; i < _rows; ++i)
        {
            A_shifted(i, i) -= lambda;
        }

        matrix<double> v(_rows, 1);
        for (size_t i = 0; i < _rows; ++i)
        {
            v(i, 0) = (i == k % _rows) ? 1.0 : ((i + k) % 3 == 0 ? 0.5 : 0.1);
        }

        for (size_t j = 0; j < k; ++j)
        {
            double dot = 0.0;
            for (size_t i = 0; i < _rows; ++i)
            {
                dot += v(i, 0) * V(i, j);
            }
            for (size_t i = 0; i < _rows; ++i)
            {
                v(i, 0) -= dot * V(i, j);
            }
        }

        double norm = 0.0;
        for (size_t i = 0; i < _rows; ++i)
        {
            norm += v(i, 0) * v(i, 0);
        }
        norm = std::sqrt(norm);
        if (norm > eps)
        {
            for (size_t i = 0; i < _rows; ++i)
            {
                v(i, 0) /= norm;
            }
        }

        for (int iter = 0; iter < 50; ++iter)
        {
            matrix<double> A_reg = A_shifted;
            for (size_t i = 0; i < _rows; ++i)
            {
                A_reg(i, i) += eps * 100;
            }

            matrix<double> augmented(_rows, _cols + 1);
            for (size_t i = 0; i < _rows; ++i)
            {
                for (size_t j = 0; j < _cols; ++j)
                {
                    augmented(i, j) = A_reg(i, j);
                }
                augmented(i, _cols) = v(i, 0);
            }

            for (size_t i = 0; i < _rows; ++i)
            {
                size_t pivot = i;
                for (size_t j = i + 1; j < _rows; ++j)
                {
                    if (std::abs(augmented(j, i)) > std::abs(augmented(pivot, i)))
                    {
                        pivot = j;
                    }
                }

                if (pivot != i)
                {
                    for (size_t j = 0; j <= _cols; ++j)
                    {
                        std::swap(augmented(i, j), augmented(pivot, j));
                    }
                }

                for (size_t j = i + 1; j < _rows; ++j)
                {
                    if (std::abs(augmented(i, i)) > eps)
                    {
                        double factor = augmented(j, i) / augmented(i, i);
                        for (size_t l = i; l <= _cols; ++l)
                        {
                            augmented(j, l) -= factor * augmented(i, l);
                        }
                    }
                }
            }

            matrix<double> v_new(_rows, 1);
            for (int i = _rows - 1; i >= 0; --i)
            {
                double sum = augmented(i, _cols);
                for (size_t j = i + 1; j < _cols; ++j)
                {
                    sum -= augmented(i, j) * v_new(j, 0);
                }
                v_new(i, 0) = std::abs(augmented(i, i)) > eps ? sum / augmented(i, i) : 0.0;
            }

            for (size_t j = 0; j < k; ++j)
            {
                double dot = 0.0;
                for (size_t i = 0; i < _rows; ++i)
                {
                    dot += v_new(i, 0) * V(i, j);
                }
                for (size_t i = 0; i < _rows; ++i)
                {
                    v_new(i, 0) -= dot * V(i, j);
                }
            }

            norm = 0.0;
            for (size_t i = 0; i < _rows; ++i)
            {
                norm += v_new(i, 0) * v_new(i, 0);
            }
            norm = std::sqrt(norm);

            if (norm < eps)
            {
                break;
            }

            for (size_t i = 0; i < _rows; ++i)
            {
                v_new(i, 0) /= norm;
            }

            double diff = 0.0;
            for (size_t i = 0; i < _rows; ++i)
            {
                double d = v_new(i, 0) - v(i, 0);
                diff += d * d;
            }

            v = v_new;

            if (std::sqrt(diff) < eps * 10)
            {
                break;
            }
        }

        for (size_t i = 0; i < _rows; ++i)
        {
            V(i, k) = v(i, 0);
        }
    };

    for (size_t k = 0; k < _rows; ++k)
    {
        compute_eigenvector(k);
    }

    return V;
}

/**
 * @brief Computes Singular Value Decomposition (SVD) of the matrix.
 * @return Tuple of (U, Σ, V^T) where U and V are orthogonal matrices and Σ is a diagonal matrix of singular values.
 * @throws std::invalid_argument If matrix has zero dimensions
 * @details Time O(m*n*min(m,n)), Space O(m*n)
 * @note Uses eigendecomposition of A^T*A or A*A^T for computing singular vectors
 */
template <typename T>
std::tuple<matrix<double>, matrix<double>, matrix<double>> matrix<T>::SVD() const
{
    if (_rows == 0 || _cols == 0)
    {
        std::ostringstream oss;
        oss << "Matrix dimensions must be positive, got " << _rows << "x" << _cols;
        throw std::invalid_argument(oss.str());
    }

    matrix<double> A = static_cast<matrix<double>>(*this);
    matrix<double> At = A.transpose();

    const double eps = 1e-10;

    matrix<double> AtA = At * A;

    matrix<double> eigenvals_mat = AtA.eigenvalues();
    matrix<double> V = AtA.eigenvectors();

    std::vector<double> singular_values(_cols);
    for (size_t i = 0; i < _cols; ++i)
    {
        double eigenvalue = eigenvals_mat(i, 0);
        singular_values[i] = (eigenvalue > eps) ? std::sqrt(std::abs(eigenvalue)) : 0.0;
    }

    matrix<double> Sigma(_rows, _cols);
    for (size_t i = 0; i < std::min(_rows, _cols); ++i)
    {
        Sigma(i, i) = singular_values[i];
    }

    matrix<double> U(_rows, _rows);
    for (size_t i = 0; i < std::min(_rows, _cols); ++i)
    {
        if (singular_values[i] > eps)
        {
            for (size_t j = 0; j < _rows; ++j)
            {
                double sum = 0.0;
                for (size_t k = 0; k < _cols; ++k)
                {
                    sum += A(j, k) * V(k, i);
                }
                U(j, i) = sum / singular_values[i];
            }

            double norm = 0.0;
            for (size_t j = 0; j < _rows; ++j)
            {
                norm += U(j, i) * U(j, i);
            }
            norm = std::sqrt(norm);

            if (norm > eps)
            {
                for (size_t j = 0; j < _rows; ++j)
                {
                    U(j, i) /= norm;
                }
            }
        }
    }

    if (_rows > _cols)
    {
        for (size_t i = _cols; i < _rows; ++i)
        {
            for (size_t basis_idx = 0; basis_idx < _rows; ++basis_idx)
            {
                for (size_t j = 0; j < _rows; ++j)
                {
                    U(j, i) = (j == basis_idx) ? 1.0 : 0.0;
                }

                for (size_t k = 0; k < i; ++k)
                {
                    double dot = 0.0;
                    for (size_t j = 0; j < _rows; ++j)
                    {
                        dot += U(j, i) * U(j, k);
                    }
                    for (size_t j = 0; j < _rows; ++j)
                    {
                        U(j, i) -= dot * U(j, k);
                    }
                }

                double norm = 0.0;
                for (size_t j = 0; j < _rows; ++j)
                {
                    norm += U(j, i) * U(j, i);
                }
                norm = std::sqrt(norm);

                if (norm > eps)
                {
                    for (size_t j = 0; j < _rows; ++j)
                    {
                        U(j, i) /= norm;
                    }
                    break;
                }
            }
        }
    }

    return std::make_tuple(U, Sigma, V.transpose());
}

/**
 * @brief Solves the linear system Ax = b using LU decomposition.
 * @param b Right-hand side vector
 * @return Solution vector x
 * @throws std::invalid_argument If matrix is not square or dimensions mismatch
 * @details Time O(n³), Space O(n²)
 */
template <typename T>
matrix<double> matrix<T>::solve(const matrix<double>& b) const
{
    if (_rows != _cols)
    {
        std::ostringstream oss;
        oss << "Matrix is " << _rows << "x" << _cols << ", must be square to solve linear system";
        throw std::invalid_argument(oss.str());
    }
    if (b._rows != _rows || b._cols != 1)
    {
        std::ostringstream oss;
        oss << "Right-hand side vector dimensions " << b._rows << "x" << b._cols << " do not match matrix dimensions "
            << _rows << "x" << _cols;
        throw std::invalid_argument(oss.str());
    }
    matrix<double> L, U;
    std::tie(L, U) = this->LU_decomposition();
    matrix<double> y(_rows, 1);
    for (size_t i = 0; i < _rows; ++i)
    {
        double sum = b(i, 0);
        for (size_t j = 0; j < i; ++j)
        {
            sum -= L(i, j) * y(j, 0);
        }
        y(i, 0) = sum / L(i, i);
    }
    matrix<double> x(_rows, 1);
    for (int i = _rows - 1; i >= 0; --i)
    {
        double sum = y(i, 0);
        for (size_t j = i + 1; j < _cols; ++j)
        {
            sum -= U(i, j) * x(j, 0);
        }
        x(i, 0) = sum / U(i, i);
    }
    return x;
}
