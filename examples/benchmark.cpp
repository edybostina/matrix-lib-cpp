#include <iostream>
#include <chrono>
#include <iomanip>
#include "../include/matrix.hpp"

using namespace std;

template <typename Func>
double benchmark(Func func, const std::string &label, int warmup = 2, int iterations = 10)
{
    for (int i = 0; i < warmup; ++i)
    {
        func();
    }

    vector<double> times;
    for (int i = 0; i < iterations; ++i)
    {
        auto start = chrono::high_resolution_clock::now();
        func();
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, std::nano> elapsed = end - start;
        times.push_back(elapsed.count() / 1e9);
    }

    double min_time = *min_element(times.begin(), times.end());

    cout << setw(45) << left << label << ": ";
    if (min_time < 1e-6)
    {
        cout << fixed << setprecision(5) << (min_time * 1e9) << " ns\n";
    }
    else if (min_time < 1e-3)
    {
        cout << fixed << setprecision(3) << (min_time * 1e6) << " µs\n";
    }
    else if (min_time < 1.0)
    {
        cout << fixed << setprecision(3) << (min_time * 1e3) << " ms\n";
    }
    else
    {
        cout << fixed << setprecision(6) << min_time << " s\n";
    }

    return min_time;
}

matrix<double> naive_matrix_multiplication(const matrix<double> &A, const matrix<double> &B)
{
    int rows = A.rows();
    int cols = B.cols();
    matrix<double> C(rows, cols);
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            C(i, j) = 0;
            for (size_t k = 0; k < A.cols(); ++k)
            {
                C(i, j) += A(i, k) * B(k, j);
            }
        }
    }
    return C;
}

matrix<double> naive_matrix_addition(const matrix<double> &A, const matrix<double> &B)
{
    int rows = A.rows();
    int cols = A.cols();
    matrix<double> C(rows, cols);
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            C(i, j) = A(i, j) + B(i, j);
        }
    }
    return C;
}

pair<matrix<double>, matrix<double>> naive_LU_decomposition(const matrix<double> &A)
{
    matrix<double> L(A.rows(), A.cols());
    matrix<double> U = matrix<double>::eye(A.rows(), A.cols());
    for (size_t p = 0; p < A.rows(); ++p)
    {
        for (size_t i = 0; i < p; ++i)
        {
            U(i, p) = A(i, p) - (L.row(i) * U.col(p))(0, 0);
            U(i, p) /= L(i, i);
        }
        for (size_t i = p; i < A.rows(); ++i)
        {
            L(i, p) = A(i, p) - (L.row(i) * U.col(p))(0, 0);
        }
    }

    return std::make_pair(L, U);
}

void print_header(const string &title)
{
    cout << "\n"
         << string(70, '=') << "\n";
    cout << "  " << title << "\n";
    cout << string(70, '=') << "\n";
}

void print_speedup(double baseline, double optimized, const string &label)
{
    if (optimized == 0.0 || baseline == 0.0)
    {
        return;
    }
    double speedup = baseline / optimized;
    cout << "  -> " << label << " speedup: " << fixed << setprecision(2)
         << speedup << "x\n";
}

int main()
{
    cout << "\n";
    cout << "====================== MATRIX LIBRARY PERFORMANCE BENCHMARK ======================\n\n";

    vector<int> sizes = {50, 100, 200, 400, 600, 800, 1000};

    for (int size : sizes)
    {
        print_header("Matrix Size: " + to_string(size) + "x" + to_string(size));

        matrix<double> A = matrix<double>::random(size, size, -10.0, 10.0);
        matrix<double> B = matrix<double>::random(size, size, -10.0, 10.0);

        cout << "\n[ Matrix Addition ]\n";

        double add_naive = benchmark([&]()
                                     { matrix<double> C = naive_matrix_addition(A, B); }, "Naive (single-threaded)");

        double add_threaded = benchmark([&]()
                                        { matrix<double> C = A + B; }, "Multithreaded");

        print_speedup(add_naive, add_threaded, "Multithreaded vs Naive");

        cout << "\n[ Hadamard Product (element-wise multiplication) ]\n";

        double had_naive = benchmark([&]()
                                     {
            int rows = A.rows();
            int cols = A.cols();
            matrix<double> C(rows, cols);
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    C(i, j) = A(i, j) * B(i, j);
                }
            } }, "Naive (single-threaded)");

        double had_threaded = benchmark([&]()
                                        { matrix<double> C = A.hadamard(B); }, "Multithreaded");

        print_speedup(had_naive, had_threaded, "Multithreaded vs Naive");

        cout << "\n[ Scalar Multiplication ]\n";

        double scalar = 3.14159;

        double scalar_naive = benchmark([&]()
                                        {
            int rows = A.rows();
            int cols = A.cols();
            matrix<double> C(rows, cols);
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    C(i, j) = A(i, j) * scalar;
                }
            } }, "Naive (single-threaded)");

        double scalar_threaded = benchmark([&]()
                                           { matrix<double> C = A * scalar; }, "Multithreaded");

        print_speedup(scalar_naive, scalar_threaded, "Multithreaded vs Naive");

        cout << "\n[ Matrix Multiplication ]\n";

        double mul_naive = benchmark([&]()
                                     { matrix<double> C = naive_matrix_multiplication(A, B); }, "Naive (single-threaded)", 0, 2);

        double mul_threaded = benchmark([&]()
                                        { matrix<double> C = A * B; }, "Multithreaded", 0, 2);

        print_speedup(mul_naive, mul_threaded, "Multithreaded vs Naive");
    }

    cout << "\n"
         << string(70, '=') << "\n";
    cout << "Benchmark completed!\n";
    cout << string(70, '=') << "\n\n";

    return 0;
}
