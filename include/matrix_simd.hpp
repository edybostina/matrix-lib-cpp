#pragma once

#include "matrix_fwd.hpp"

#ifdef __has_include
#if __has_include(<xsimd/xsimd.hpp>)
#define HAS_XSIMD 1
#include <xsimd/xsimd.hpp>
#else
#define HAS_XSIMD 0
#endif
#else
#define HAS_XSIMD 0
#endif

namespace matrix_simd
{

#if HAS_XSIMD

    // SIMD-optimized element-wise addition for float/double
    template <typename T>
    inline void simd_add(const T *a, const T *b, T *result, int size)
    {
        using batch = xsimd::batch<T>;
        constexpr size_t simd_size = batch::size;

        const int vec_size = (size / simd_size) * simd_size;

        for (int i = 0; i < vec_size; i += simd_size)
        {
            auto va = batch::load_unaligned(&a[i]);
            auto vb = batch::load_unaligned(&b[i]);
            auto vr = va + vb;
            vr.store_unaligned(&result[i]);
        }

        for (int i = vec_size; i < size; ++i)
        {
            result[i] = a[i] + b[i];
        }
    }

    // SIMD-optimized element-wise subtraction
    template <typename T>
    inline void simd_sub(const T *a, const T *b, T *result, int size)
    {
        using batch = xsimd::batch<T>;
        constexpr size_t simd_size = batch::size;

        const int vec_size = (size / simd_size) * simd_size;

        for (int i = 0; i < vec_size; i += simd_size)
        {
            auto va = batch::load_unaligned(&a[i]);
            auto vb = batch::load_unaligned(&b[i]);
            auto vr = va - vb;
            vr.store_unaligned(&result[i]);
        }

        for (int i = vec_size; i < size; ++i)
        {
            result[i] = a[i] - b[i];
        }
    }

    // SIMD-optimized element-wise multiplication (Hadamard product)
    template <typename T>
    inline void simd_mul(const T *a, const T *b, T *result, int size)
    {
        using batch = xsimd::batch<T>;
        constexpr size_t simd_size = batch::size;

        const int vec_size = (size / simd_size) * simd_size;

        for (int i = 0; i < vec_size; i += simd_size)
        {
            auto va = batch::load_unaligned(&a[i]);
            auto vb = batch::load_unaligned(&b[i]);
            auto vr = va * vb;
            vr.store_unaligned(&result[i]);
        }

        for (int i = vec_size; i < size; ++i)
        {
            result[i] = a[i] * b[i];
        }
    }

    // SIMD-optimized scalar addition
    template <typename T>
    inline void simd_add_scalar(const T *a, T scalar, T *result, int size)
    {
        using batch = xsimd::batch<T>;
        constexpr size_t simd_size = batch::size;

        const int vec_size = (size / simd_size) * simd_size;
        auto vs = batch(scalar);

        for (int i = 0; i < vec_size; i += simd_size)
        {
            auto va = batch::load_unaligned(&a[i]);
            auto vr = va + vs;
            vr.store_unaligned(&result[i]);
        }

        for (int i = vec_size; i < size; ++i)
        {
            result[i] = a[i] + scalar;
        }
    }

    // SIMD-optimized scalar multiplication
    template <typename T>
    inline void simd_mul_scalar(const T *a, T scalar, T *result, int size)
    {
        using batch = xsimd::batch<T>;
        constexpr size_t simd_size = batch::size;

        const int vec_size = (size / simd_size) * simd_size;
        auto vs = batch(scalar);

        for (int i = 0; i < vec_size; i += simd_size)
        {
            auto va = batch::load_unaligned(&a[i]);
            auto vr = va * vs;
            vr.store_unaligned(&result[i]);
        }

        for (int i = vec_size; i < size; ++i)
        {
            result[i] = a[i] * scalar;
        }
    }

    // SIMD-optimized dot product for matrix multiplication
    template <typename T>
    inline T simd_dot(const T *a, const T *b, int size)
    {
        using batch = xsimd::batch<T>;
        constexpr size_t simd_size = batch::size;

        const int vec_size = (size / simd_size) * simd_size;

        auto sum_vec = batch(T(0));
        for (int i = 0; i < vec_size; i += simd_size)
        {
            auto va = batch::load_unaligned(&a[i]);
            auto vb = batch::load_unaligned(&b[i]);
            sum_vec = xsimd::fma(va, vb, sum_vec); // fused multiply-add
        }

        T sum = xsimd::reduce_add(sum_vec);

        for (int i = vec_size; i < size; ++i)
        {
            sum += a[i] * b[i];
        }

        return sum;
    }

    // Check if type is supported by SIMD
    template <typename T>
    struct is_simd_supported
    {
        static constexpr bool value = std::is_same<T, float>::value ||
                                      std::is_same<T, double>::value;
    };

#else
    template <typename T>
    inline void simd_add(const T *a, const T *b, T *result, int size)
    {
        for (int i = 0; i < size; ++i)
        {
            result[i] = a[i] + b[i];
        }
    }

    template <typename T>
    inline void simd_sub(const T *a, const T *b, T *result, int size)
    {
        for (int i = 0; i < size; ++i)
        {
            result[i] = a[i] - b[i];
        }
    }

    template <typename T>
    inline void simd_mul(const T *a, const T *b, T *result, int size)
    {
        for (int i = 0; i < size; ++i)
        {
            result[i] = a[i] * b[i];
        }
    }

    template <typename T>
    inline void simd_add_scalar(const T *a, T scalar, T *result, int size)
    {
        for (int i = 0; i < size; ++i)
        {
            result[i] = a[i] + scalar;
        }
    }

    template <typename T>
    inline void simd_mul_scalar(const T *a, T scalar, T *result, int size)
    {
        for (int i = 0; i < size; ++i)
        {
            result[i] = a[i] * scalar;
        }
    }

    template <typename T>
    inline T simd_dot(const T *a, const T *b, int size)
    {
        T sum = T(0);
        for (int i = 0; i < size; ++i)
        {
            sum += a[i] * b[i];
        }
        return sum;
    }

    template <typename T>
    struct is_simd_supported
    {
        static constexpr bool value = false;
    };

#endif

} // namespace matrix_simd
