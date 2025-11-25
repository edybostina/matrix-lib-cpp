#pragma once

#include "matrix_core.hpp"
#include "matrix_simd.hpp"

// SIMD-optimized matrix operations
// These are used when xsimd is available and T is float or double

// SIMD Matrix addition
template <typename T>
matrix<T> matrix_add_simd(const matrix<T> &a, const matrix<T> &b)
{
    matrix<T> result(a.rows(), a.cols());
    matrix_simd::simd_add(a.data_ptr(), b.data_ptr(), result.data_ptr(), a.size());
    return result;
}

// SIMD Matrix subtraction
template <typename T>
matrix<T> matrix_sub_simd(const matrix<T> &a, const matrix<T> &b)
{
    matrix<T> result(a.rows(), a.cols());
    matrix_simd::simd_sub(a.data_ptr(), b.data_ptr(), result.data_ptr(), a.size());
    return result;
}

// SIMD Hadamard product
template <typename T>
matrix<T> matrix_hadamard_simd(const matrix<T> &a, const matrix<T> &b)
{
    matrix<T> result(a.rows(), a.cols());
    matrix_simd::simd_mul(a.data_ptr(), b.data_ptr(), result.data_ptr(), a.size());
    return result;
}

// SIMD Matrix multiplication (using SIMD dot products)
template <typename T>
matrix<T> matrix_mul_simd(const matrix<T> &a, const matrix<T> &b)
{
    matrix<T> result(a.rows(), b.cols());
    
    std::vector<T> col_buffer(a.cols());
    
    for (int i = 0; i < a.rows(); ++i)
    {
        for (int j = 0; j < b.cols(); ++j)
        {
            for (int k = 0; k < b.rows(); ++k)
            {
                col_buffer[k] = b(k, j);
            }
            result(i, j) = matrix_simd::simd_dot(a.data_ptr() + i * a.cols(), 
                                                  col_buffer.data(), 
                                                  a.cols());
        }
    }
    
    return result;
}

// SIMD Matrix multiplication with multithreading
template <typename T>
matrix<T> matrix_mul_simd_threaded(const matrix<T> &a, const matrix<T> &b)
{
    matrix<T> result(a.rows(), b.cols());
    
    auto worker = [&](int row_start, int row_end)
    {
        std::vector<T> col_buffer(a.cols());
        
        for (int i = row_start; i < row_end; ++i)
        {
            for (int j = 0; j < b.cols(); ++j)
            {
                for (int k = 0; k < b.rows(); ++k)
                {
                    col_buffer[k] = b(k, j);
                }
                result(i, j) = matrix_simd::simd_dot(a.data_ptr() + i * a.cols(), 
                                                      col_buffer.data(), 
                                                      a.cols());
            }
        }
    };
    
    int num_threads = std::thread::hardware_concurrency();
    int chunk_size = (a.rows() + num_threads - 1) / num_threads;
    
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t)
    {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, a.rows());
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

// SIMD Scalar addition
template <typename T>
matrix<T> matrix_add_scalar_simd(const matrix<T> &a, const T &scalar)
{
    matrix<T> result(a.rows(), a.cols());
    matrix_simd::simd_add_scalar(a.data_ptr(), scalar, result.data_ptr(), a.size());
    return result;
}

// SIMD Scalar multiplication
template <typename T>
matrix<T> matrix_mul_scalar_simd(const matrix<T> &a, const T &scalar)
{
    matrix<T> result(a.rows(), a.cols());
    matrix_simd::simd_mul_scalar(a.data_ptr(), scalar, result.data_ptr(), a.size());
    return result;
}
