#include <iostream>
#include <chrono>
#include "../include/matrix.hpp"

using namespace std;

template <typename Func>
void benchmark(Func func, const std::string &label)
{
    auto start = chrono::high_resolution_clock::now();
    func();
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << label << " took " << elapsed.count() << " seconds\n";
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
            for (int k = 0; k < A.cols(); ++k)
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
    for (int p = 0; p < A.rows(); ++p)
    {
        for (int i = 0; i < p; ++i)
        {
            U(i, p) = A(i, p) - (L.row(i) * U.col(p))(0, 0);
            U(i, p) /= L(i, i);
        }
        for (int i = p; i < A.rows(); ++i)
        {
            L(i, p) = A(i, p) - (L.row(i) * U.col(p))(0, 0);
        }
    }

    return std::make_pair(L, U);
}

// Have fun with this
int main()
{
    matrix<double> A = matrix<double>::random(1000, 1000, -10.0, 10.0);
    matrix<double> B = matrix<double>::random(1000, 1000, -10.0, 10.0);

    benchmark([&]()
              {
                  matrix<double> C = naive_matrix_multiplication(A, B);
              },
              "Single-threaded version");

    benchmark([&]()
              { matrix<double> C = A * B; },
              "Multithread");
}