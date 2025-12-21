#include <chrono>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "../include/matrix.hpp"

using namespace std;

template <typename Func>
double benchmark(Func func, int warmup = 2, int iterations = 5)
{
    for (int i = 0; i < warmup; ++i)
    {
        func();
    }

    vector<double> times;
    times.reserve(iterations);

    for (int i = 0; i < iterations; ++i)
    {
        auto start = chrono::high_resolution_clock::now();
        func();
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        times.push_back(elapsed.count());
    }

    sort(times.begin(), times.end());
    return times[times.size() / 2];
}

void print_time(const string& label, double time_sec, int width = 45)
{
    cout << "  " << setw(width) << left << label << ": ";

    if (time_sec < 1e-6)
    {
        cout << fixed << setprecision(2) << (time_sec * 1e9) << " ns\n";
    }
    else if (time_sec < 1e-3)
    {
        cout << fixed << setprecision(2) << (time_sec * 1e6) << " µs\n";
    }
    else if (time_sec < 1.0)
    {
        cout << fixed << setprecision(1) << (time_sec * 1e3) << " ms\n";
    }
    else
    {
        cout << fixed << setprecision(2) << time_sec << " s\n";
    }
}

double calculate_gflops(size_t operations, double time_seconds)
{
    return (operations / 1e9) / time_seconds;
}

matrix<double> naive_matmul(const matrix<double>& A, const matrix<double>& B)
{
    size_t rows = A.rows();
    size_t cols = B.cols();
    size_t inner = A.cols();
    matrix<double> C(rows, cols);

    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            C(i, j) = 0;
            for (size_t k = 0; k < inner; ++k)
            {
                C(i, j) += A(i, k) * B(k, j);
            }
        }
    }
    return C;
}

void benchmark_size(int size)
{
    cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    cout << "  Matrix Size: " << size << "x" << size << "\n";
    cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";

    matrix<double> A = matrix<double>::random(size, size, -10.0, 10.0);
    matrix<double> B = matrix<double>::random(size, size, -10.0, 10.0);

    int warmup = size > 1000 ? 1 : 2;
    int iters = size > 1000 ? 2 : 5;

    cout << "\n▸ Matrix Multiplication (A x B)\n";

    double naive_time = 0;
    if (size <= 500)
    {
        naive_time = benchmark([&]() { auto C = naive_matmul(A, B); }, warmup, iters);
        print_time("Naive O(n³)", naive_time);
    }

    double opt_time = benchmark([&]() { auto C = A * B; }, warmup, iters);
    print_time("Optimized (tiled + SIMD + MT)", opt_time);

    size_t ops = 2ULL * size * size * size;
    double gflops = calculate_gflops(ops, opt_time);
    cout << "  → Performance: " << fixed << setprecision(2) << gflops << " GFLOP/s";

    if (naive_time > 0)
    {
        double speedup = naive_time / opt_time;
        cout << " (" << fixed << setprecision(1) << speedup << "x speedup)";
    }
    cout << "\n";

    cout << "\n▸ Matrix Addition (A + B)\n";
    double add_time = benchmark([&]() { auto C = A + B; }, warmup, iters);
    print_time("Optimized (SIMD + MT)", add_time);

    cout << "\n▸ Transpose (Aᵀ)\n";
    double trans_time = benchmark([&]() { auto C = A.transpose(); }, warmup, iters);
    print_time("Optimized", trans_time);

    if (size <= 800)
    {
        cout << "\n▸ LU Decomposition\n";
        double lu_time = benchmark([&]() { auto [L, U] = A.LU_decomposition(); }, warmup, max(2, iters - 2));
        print_time("Gaussian elimination", lu_time);
    }

    if (size <= 500)
    {
        cout << "\n▸ Determinant\n";
        double det_time = benchmark([&]() { double _ = A.determinant(); }, warmup, max(2, iters - 2));
        print_time("LU-based", det_time);
    }
}

void print_header()
{
    cout << "\n";
    cout << "╔════════════════════════════════════════════════════════════════════════╗\n";
    cout << "║         MATRIX LIBRARY PERFORMANCE BENCHMARK - CORE OPERATIONS         ║\n";
    cout << "╚════════════════════════════════════════════════════════════════════════╝\n";

    cout << "\nSystem Configuration:\n";
    cout << "  Compiler: ";
#if defined(__clang__)
    cout << "Clang " << __clang_major__ << "." << __clang_minor__;
#elif defined(__GNUC__)
    cout << "GCC " << __GNUC__ << "." << __GNUC_MINOR__;
#elif defined(_MSC_VER)
    cout << "MSVC " << _MSC_VER;
#else
    cout << "Unknown";
#endif
    cout << "\n";

    cout << "  SIMD:     ";
#ifdef MATRIX_USE_SIMD
#if defined(__AVX2__)
    cout << "AVX2";
#elif defined(__ARM_NEON)
    cout << "ARM NEON";
#else
    cout << "Generic";
#endif
#else
    cout << "Disabled";
#endif
    cout << "\n";

    cout << "  BLAS:     ";
#ifdef MATRIX_USE_BLAS
    cout << "Enabled";
#else
    cout << "Disabled";
#endif
    cout << "\n";
}

int main()
{
    print_header();

    vector<int> sizes = {100, 200, 500, 1000, 2000};

    for (int size : sizes)
    {
        benchmark_size(size);
    }

    cout << "\n╔════════════════════════════════════════════════════════════════════════╗\n";
    cout << "║                          BENCHMARK COMPLETE                            ║\n";
    cout << "╚════════════════════════════════════════════════════════════════════════╝\n\n";

    return 0;
}
