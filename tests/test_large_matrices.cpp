#include "../include/matrix.hpp"
#include "test_framework.hpp"

using namespace std;

TEST(LargeMatrices, Inverse_Parallel_10x10)
{
    auto A = matrix<double>::eye(10, 10);
    for (size_t i = 0; i < 10; ++i)
    {
        A(i, i) = static_cast<double>(i + 1);
    }
    auto B = A.inverse();
    auto I = A * B;

    for (size_t i = 0; i < 10; ++i)
    {
        ASSERT_TRUE(abs(I(i, i) - 1.0) < 1e-9);
    }
}

TEST(LargeMatrices, Transpose_CacheBlocked_Large)
{
    auto A = matrix<double>::random(500, 400, 0.0, 1.0);
    auto B = A.transpose();
    ASSERT_TRUE(B.rows() == 400);
    ASSERT_TRUE(B.cols() == 500);

    ASSERT_TRUE(abs(B(0, 0) - A(0, 0)) < 1e-12);
    ASSERT_TRUE(abs(B(399, 499) - A(499, 399)) < 1e-12);
}

TEST(LargeMatrices, Transpose_DoubleTranspose_Large)
{
    auto A = matrix<double>::random(300, 250, 0.0, 1.0);
    auto B = A.transpose().transpose();
    ASSERT_TRUE((A - B).norm(2) < 1e-10);
}

TEST(LargeMatrices, Norm_SIMD_Double)
{
    auto A = matrix<double>::ones(1000, 1000);
    double n = A.norm(2);
    ASSERT_TRUE(abs(n - 1000.0) < 1e-9);
}

TEST(LargeMatrices, Norm_SIMD_Float)
{
    auto A = matrix<float>::ones(1000, 1000);
    double n = A.norm(2);
    ASSERT_TRUE(abs(n - 1000.0) < 1e-6);
}

TEST(LargeMatrices, Norm_SIMD_NonSquare)
{
    auto A = matrix<double>::ones(500, 800);
    double n = A.norm(2);
    double expected = sqrt(500.0 * 800.0);
    ASSERT_TRUE(abs(n - expected) < 1e-9);
}

TEST(LargeMatrices, Trace_SIMD_Large)
{
    auto I = matrix<double>::eye(500, 500);
    ASSERT_EQ(I.trace(), 500.0);
}

TEST(LargeMatrices, Trace_SIMD_NonIdentity)
{
    matrix<double> A(1000, 1000);
    for (size_t i = 0; i < 1000; ++i)
    {
        A(i, i) = static_cast<double>(i + 1);
    }
    double expected = 1000.0 * 1001.0 / 2.0;
    ASSERT_TRUE(abs(A.trace() - expected) < 1e-9);
}

TEST(LargeMatrices, GaussianElimination_LargeKnownMatrix)
{
    auto A = matrix<double>::eye(100, 100);
    for (size_t i = 0; i < 100; i++)
    {
        for (size_t j = i + 1; j < 100; j++)
        {
            A(i, j) = static_cast<double>(i + j) / 100.0;
        }
    }
    auto G = A.gaussian_elimination();
    ASSERT_TRUE(G.rows() == 100);
    ASSERT_TRUE(G.cols() == 100);
    ASSERT_TRUE(G.is_upper_triangular());
    for (size_t i = 0; i < 100; i++)
    {
        ASSERT_TRUE(abs(G(i, i)) > 1e-10);
    }
}

TEST(LargeMatrices, Inverse_BoundarySize_7x7)
{
    auto A = matrix<double>::eye(7, 7);
    for (size_t i = 0; i < 7; ++i)
    {
        A(i, i) = static_cast<double>(i + 1);
    }
    auto B = A.inverse();
    ASSERT_TRUE(B.rows() == 7);
    auto I = A * B;
    for (size_t i = 0; i < 7; i++)
    {
        ASSERT_TRUE(abs(I(i, i) - 1.0) < 1e-9);
        for (size_t j = 0; j < 7; j++)
        {
            if (i != j)
                ASSERT_TRUE(abs(I(i, j)) < 1e-9);
        }
    }
}

TEST(LargeMatrices, Inverse_BoundarySize_8x8)
{
    auto A = matrix<double>::eye(8, 8);
    for (size_t i = 0; i < 8; ++i)
    {
        A(i, i) = static_cast<double>(i + 1);
    }
    auto B = A.inverse();
    auto I = A * B;
    ASSERT_TRUE(abs(I(0, 0) - 1.0) < 1e-9);
}

TEST(LargeMatrices, Transpose_BoundarySize_50x50)
{
    matrix<double> A(50, 50);
    for (size_t i = 0; i < 50; i++)
    {
        for (size_t j = 0; j < 50; j++)
        {
            A(i, j) = static_cast<double>(i * 50 + j);
        }
    }
    auto B = A.transpose();
    ASSERT_TRUE(B.rows() == 50);
    ASSERT_TRUE(B.cols() == 50);
    ASSERT_EQ(B(0, 0), A(0, 0));
    ASSERT_EQ(B(10, 20), A(20, 10));
    ASSERT_EQ(B(49, 0), A(0, 49));
}

TEST(LargeMatrices, Transpose_100x100_Correctness)
{
    matrix<double> A(100, 100);
    for (size_t i = 0; i < 100; i++)
    {
        for (size_t j = 0; j < 100; j++)
        {
            A(i, j) = static_cast<double>(i * 100 + j);
        }
    }
    auto B = A.transpose();
    ASSERT_TRUE(B.rows() == 100);
    ASSERT_TRUE(B.cols() == 100);
    ASSERT_EQ(B(0, 99), A(99, 0));
    ASSERT_EQ(B(50, 50), A(50, 50));
    ASSERT_EQ(B(99, 0), A(0, 99));
}

TEST(LargeMatrices, Norm_Accuracy_MixedValues)
{
    matrix<double> A = {{1e-10, 1e10}, {1e5, 1e-5}};
    double n = A.norm(2);
    ASSERT_TRUE(n > 0);
}

TEST(LargeMatrices, Trace_Accuracy_SmallValues)
{
    matrix<double> A(100, 100);
    for (size_t i = 0; i < 100; ++i)
    {
        A(i, i) = 1e-10;
    }
    ASSERT_TRUE(abs(A.trace() - 1e-8) < 1e-15);
}

TEST(LargeMatrices, GaussianElimination_SmallPivot)
{
    matrix<double> A = {{1e-10, 1}, {1, 1}};
    auto G = A.gaussian_elimination();
    ASSERT_TRUE(G.rows() == 2);
    ASSERT_TRUE(G.is_upper_triangular());
    ASSERT_TRUE(abs(G(0, 0)) > 0.1);
}

TEST(LargeMatrices, LargeMatrixOperations_NoSegfault)
{
    auto A = matrix<double>::random(200, 200, 0.0, 1.0);
    auto B = A.transpose();
    auto C = A + B;
    ASSERT_TRUE(C.rows() == 200);
}

TEST(LargeMatrices, MultipleOperations_CofactorAndTranspose)
{
    matrix<double> A = {{2, 1, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 2, 1, 0, 0, 0, 0, 0, 0, 0}, {0, 1, 2, 1, 0, 0, 0, 0, 0, 0},
                        {0, 0, 1, 2, 1, 0, 0, 0, 0, 0}, {0, 0, 0, 1, 2, 1, 0, 0, 0, 0}, {0, 0, 0, 0, 1, 2, 1, 0, 0, 0},
                        {0, 0, 0, 0, 0, 1, 2, 1, 0, 0}, {0, 0, 0, 0, 0, 0, 1, 2, 1, 0}, {0, 0, 0, 0, 0, 0, 0, 1, 2, 1},
                        {0, 0, 0, 0, 0, 0, 0, 0, 1, 2}};
    auto C1 = A.cofactor();
    auto T = A.transpose();
    auto C2 = T.cofactor();
    ASSERT_TRUE(C1.rows() == 10);
    ASSERT_TRUE(C2.rows() == 10);
    ASSERT_TRUE((C1 - C1.transpose()).norm(2) < 1e-9);
}

TEST(LargeMatrices, MixedOptimizations)
{
    auto A = matrix<double>::random(50, 50, -10.0, 10.0);

    double t1 = A.trace();
    auto T = A.transpose();
    double t2 = T.trace();
    ASSERT_EQ(t1, t2);

    double n = A.norm(2);
    ASSERT_TRUE(n > 0);

    if (A.determinant() != 0)
    {
        auto inv = A.inverse();
        ASSERT_TRUE(inv.rows() == 50);
    }
}

TEST(LargeMatrices, ParallelOperations_Independent)
{
    auto A = matrix<double>::random(5, 5, 0.0, 1.0);
    auto B = matrix<double>::random(5, 5, 0.0, 1.0);

    auto C1 = A.cofactor();
    auto C2 = B.cofactor();

    ASSERT_TRUE(C1.rows() == 5);
    ASSERT_TRUE(C2.rows() == 5);
}

TEST(LargeMatrices, Transpose_Symmetric)
{
    matrix<double> A = {{1, 2, 3}, {2, 4, 5}, {3, 5, 6}};
    auto T = A.transpose();
    ASSERT_TRUE((A - T).norm(2) < 1e-12);
}

TEST(LargeMatrices, Norm_Diagonal)
{
    auto D = matrix<double>::eye(100, 100);
    for (size_t i = 0; i < 100; ++i)
    {
        D(i, i) = static_cast<double>(i + 1);
    }
    double n = D.norm(2);

    double expected = sqrt(100.0 * 101.0 * 201.0 / 6.0);
    ASSERT_TRUE(abs(n - expected) < 1e-9);
}

TEST(LargeMatrices, Cofactor_Sparse)
{
    auto S = matrix<double>::eye(5, 5);
    S(0, 1) = 1;
    S(1, 0) = 1;
    auto C = S.cofactor();
    ASSERT_TRUE(C.rows() == 5);
    double det = S.determinant();
    double det_from_cofactor = 0;
    for (size_t j = 0; j < 5; j++)
    {
        det_from_cofactor += S(0, j) * C(0, j);
    }
    ASSERT_TRUE(abs(det - det_from_cofactor) < 1e-9);
}
