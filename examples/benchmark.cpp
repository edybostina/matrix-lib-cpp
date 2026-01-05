#include <chrono>
#include <iomanip>
#include <iostream>
#include <algorithm>

#include "../include/matrix.hpp"

using namespace std;

template <typename Func>
double benchmark(Func func, int iterations = 5)
{
    // Warmup
    func();

    vector<double> times;
    for (int i = 0; i < iterations; ++i)
    {
        auto start = chrono::high_resolution_clock::now();
        func();
        auto end = chrono::high_resolution_clock::now();
        times.push_back(chrono::duration<double>(end - start).count());
    }

    sort(times.begin(), times.end());
    return times[times.size() / 2];
}

string format_time(double seconds)
{
    char buf[32];
    if (seconds < 1e-6)
        snprintf(buf, sizeof(buf), "%6.2f ns", seconds * 1e9);
    else if (seconds < 1e-3)
        snprintf(buf, sizeof(buf), "%6.2f µs", seconds * 1e6);
    else if (seconds < 1.0)
        snprintf(buf, sizeof(buf), "%6.1f ms", seconds * 1e3);
    else
        snprintf(buf, sizeof(buf), "%6.2f s", seconds);
    return string(buf);
}

int main()
{
    cout << "\nMatrix Library Benchmark (v" << MATRIX_LIB_VERSION << ")\n";
    cout << "─────────────────────────────────────────────\n";

#ifdef MATRIX_USE_SIMD
    cout << "SIMD: Enabled  ";
#else
    cout << "SIMD: Disabled  ";
#endif
#ifdef MATRIX_USE_BLAS
    cout << "BLAS: Enabled\n\n";
#else
    cout << "BLAS: Disabled\n\n";
#endif

    vector<int> sizes = {128, 256, 512, 1024, 2048, 4096, 8192};

    cout << setw(8) << "Size" << setw(12) << "MatMul" << setw(12) << "Add" << setw(12) << "Transpose" << setw(10)
         << "GFLOP/s" << "\n";
    cout << string(54, '-') << "\n";

    for (int n : sizes)
    {
        auto A = matrix<double>::random(n, n, -1.0, 1.0);
        auto B = matrix<double>::random(n, n, -1.0, 1.0);

        int iters = n < 512 ? 5 : 3;

        double mul_time = benchmark([&]() { auto C = A * B; }, iters);
        double add_time = benchmark([&]() { auto C = A + B; }, iters);
        double trans_time = benchmark([&]() { auto C = A.transpose(); }, iters);

        double gflops = (2.0 * n * n * n / 1e9) / mul_time;

        cout << setw(8) << n << setw(12) << format_time(mul_time) << setw(12) << format_time(add_time) << setw(12)
             << format_time(trans_time) << setw(10) << fixed << setprecision(1) << gflops << "\n";
    }

    // Additional operations for smaller matrices
    cout << "\nOther Operations (512x512):\n";
    cout << string(54, '-') << "\n";

    auto M = matrix<double>::random(512, 512, -1.0, 1.0);

    double det_time = benchmark([&]() { auto _ = M.determinant(); }, 3);
    double inv_time = benchmark([&]() { auto I = M.inverse(); }, 3);
    auto [L, U] = M.LU_decomposition();
    double lu_time = benchmark([&]() { auto x = M.LU_decomposition(); }, 3);

    cout << "  Determinant:        " << format_time(det_time) << "\n";
    cout << "  Inverse:            " << format_time(inv_time) << "\n";
    cout << "  LU Decomposition:   " << format_time(lu_time) << "\n";

    cout << "\n";
    return 0;
}
