#include "../include/matrix.hpp"
#include "test_framework.hpp"
#include <limits>
#include <cmath>

using namespace std;

TEST(EdgeCases, Matrix_1x1_Construction)
{
    matrix<double> M = {{5}};
    ASSERT_TRUE(M.rows() == 1);
    ASSERT_TRUE(M.cols() == 1);
    ASSERT_EQ(M(0, 0), 5);
}

TEST(EdgeCases, Matrix_1x1_Operations)
{
    matrix<double> A = {{3}};
    matrix<double> B = {{2}};
    auto C = A + B;
    ASSERT_EQ(C(0, 0), 5);
    auto D = A * B;
    ASSERT_EQ(D(0, 0), 6);
}

TEST(EdgeCases, Matrix_1x1_Transpose)
{
    matrix<double> M = {{5}};
    auto T = M.transpose();
    ASSERT_TRUE(T == M);
}

TEST(EdgeCases, Matrix_1x1_Determinant)
{
    matrix<double> M = {{7}};
    ASSERT_EQ(M.determinant(), 7);
}

TEST(EdgeCases, Matrix_1x1_Inverse)
{
    matrix<double> M = {{4}};
    auto I = M.inverse();
    ASSERT_EQ(I(0, 0), 0.25);
}

TEST(EdgeCases, Matrix_1xN_Operations)
{
    matrix<double> A = {{1, 2, 3, 4}};
    matrix<double> B = {{5, 6, 7, 8}};
    auto C = A + B;
    ASSERT_EQ(C(0, 0), 6);
    ASSERT_EQ(C(0, 3), 12);
}

TEST(EdgeCases, Matrix_Nx1_Operations)
{
    matrix<double> A = {{1}, {2}, {3}, {4}};
    matrix<double> B = {{5}, {6}, {7}, {8}};
    auto C = A + B;
    ASSERT_EQ(C(0, 0), 6);
    ASSERT_EQ(C(3, 0), 12);
}

TEST(EdgeCases, ZeroMatrix_Addition)
{
    auto Z = matrix<double>::zeros(2, 2);
    matrix<double> A = {{1, 2}, {3, 4}};
    auto C = Z + A;
    ASSERT_TRUE(C == A);
}

TEST(EdgeCases, ZeroMatrix_Multiplication)
{
    auto Z = matrix<double>::zeros(2, 2);
    matrix<double> A = {{1, 2}, {3, 4}};
    auto C = Z * A;
    ASSERT_EQ(C(0, 0), 0);
    ASSERT_EQ(C(1, 1), 0);
}

TEST(EdgeCases, ZeroMatrix_Determinant)
{
    auto Z = matrix<double>::zeros(3, 3);
    ASSERT_EQ(Z.determinant(), 0);
}

TEST(EdgeCases, ZeroMatrix_Trace)
{
    auto Z = matrix<double>::zeros(3, 3);
    ASSERT_EQ(Z.trace(), 0);
}

TEST(EdgeCases, ZeroMatrix_Norm)
{
    auto Z = matrix<double>::zeros(3, 3);
    ASSERT_EQ(Z.norm(2), 0);
}

TEST(EdgeCases, ZeroMatrix_Rank)
{
    auto Z = matrix<double>::zeros(3, 3);
    ASSERT_TRUE(Z.rank() == 0);
}

TEST(EdgeCases, ZeroMatrix_IsSingular)
{
    auto Z = matrix<double>::zeros(2, 2);
    ASSERT_TRUE(Z.is_singular());
}

TEST(EdgeCases, SingularMatrix_Determinant)
{
    matrix<double> S = {{1, 2}, {2, 4}};
    ASSERT_TRUE(abs(S.determinant()) < 1e-10);
}

TEST(EdgeCases, SingularMatrix_IsSingular)
{
    matrix<double> S = {{1, 2, 3}, {2, 4, 6}, {3, 6, 9}};
    ASSERT_TRUE(S.is_singular());
}

TEST(EdgeCases, SingularMatrix_Inverse_Throws)
{
    matrix<double> S = {{1, 2}, {2, 4}};
    bool caught = false;
    try
    {
        auto I = S.inverse();
    }
    catch (const std::invalid_argument&)
    {
        caught = true;
    }
    ASSERT_TRUE(caught);
}

TEST(EdgeCases, SingularMatrix_Rank)
{
    matrix<double> S = {{1, 2, 3}, {2, 4, 6}, {3, 6, 9}};
    ASSERT_TRUE(S.rank() == 1);
}

TEST(EdgeCases, SingularMatrix_RowOfZeros)
{
    matrix<double> S = {{1, 2, 3}, {0, 0, 0}, {4, 5, 6}};
    ASSERT_EQ(S.determinant(), 0);
}

TEST(EdgeCases, SingularMatrix_ColumnOfZeros)
{
    matrix<double> S = {{1, 0, 3}, {2, 0, 4}, {3, 0, 5}};
    ASSERT_EQ(S.determinant(), 0);
}

TEST(EdgeCases, NegativeValues_Addition)
{
    matrix<double> A = {{-1, -2}, {-3, -4}};
    matrix<double> B = {{1, 2}, {3, 4}};
    auto C = A + B;
    ASSERT_EQ(C(0, 0), 0);
    ASSERT_EQ(C(1, 1), 0);
}

TEST(EdgeCases, NegativeValues_Multiplication)
{
    matrix<double> A = {{-1, -2}, {-3, -4}};
    matrix<double> B = {{1, 2}, {3, 4}};
    auto C = A * B;
    ASSERT_EQ(C(0, 0), -7);
    ASSERT_EQ(C(1, 1), -22);
}

TEST(EdgeCases, NegativeValues_Determinant)
{
    matrix<double> M = {{-1, -2}, {-3, -4}};
    ASSERT_EQ(M.determinant(), -2);
}

TEST(EdgeCases, NegativeValues_Trace)
{
    matrix<double> M = {{-1, 2}, {3, -4}};
    ASSERT_EQ(M.trace(), -5);
}

TEST(EdgeCases, NegativeValues_ScalarMultiplication)
{
    matrix<double> M = {{1, 2}, {3, 4}};
    auto N = M * (-1.0);
    ASSERT_EQ(N(0, 0), -1);
    ASSERT_EQ(N(1, 1), -4);
}

TEST(EdgeCases, AllNegative_Matrix)
{
    matrix<double> M = {{-1, -2, -3}, {-4, -5, -6}, {-7, -8, -9}};
    ASSERT_TRUE(M.trace() == -15);
}

TEST(EdgeCases, TinyValues_Addition)
{
    matrix<double> A = {{1e-15, 2e-15}, {3e-15, 4e-15}};
    matrix<double> B = {{1e-15, 1e-15}, {1e-15, 1e-15}};
    auto C = A + B;
    ASSERT_TRUE(abs(C(0, 0) - 2e-15) < 1e-20);
}

TEST(EdgeCases, TinyValues_Norm)
{
    matrix<double> M = {{1e-10, 1e-10}, {1e-10, 1e-10}};
    double n = M.norm(2);
    ASSERT_TRUE(n > 0);
    ASSERT_TRUE(n < 1e-9);
}

TEST(EdgeCases, TinyValues_Determinant)
{
    matrix<double> M = {{1e-10, 0}, {0, 1e-10}};
    double det = M.determinant();
    ASSERT_TRUE(abs(det - 1e-20) < 1e-25);
}

TEST(EdgeCases, LargeValues_Addition)
{
    matrix<double> A = {{1e10, 2e10}, {3e10, 4e10}};
    matrix<double> B = {{1e10, 1e10}, {1e10, 1e10}};
    auto C = A + B;
    ASSERT_TRUE(abs(C(0, 0) - 2e10) < 1e5);
}

TEST(EdgeCases, LargeValues_Multiplication)
{
    matrix<double> A = {{1e5, 2e5}, {3e5, 4e5}};
    matrix<double> B = {{1e5, 2e5}, {3e5, 4e5}};
    auto C = A * B;
    ASSERT_TRUE(C(0, 0) > 1e10);
}

TEST(EdgeCases, LargeValues_Norm)
{
    matrix<double> M = {{1e10, 1e10}, {1e10, 1e10}};
    double n = M.norm(2);
    ASSERT_TRUE(n > 1e10);
}

TEST(EdgeCases, MixedMagnitude_SmallAndLarge)
{
    matrix<double> M = {{1e-10, 1e10}, {1e5, 1e-5}};
    double n = M.norm(2);
    ASSERT_TRUE(n > 0);
}

TEST(EdgeCases, MixedMagnitude_Addition)
{
    matrix<double> A = {{1e-10, 1e10}, {1, 1}};
    matrix<double> B = {{1e-10, 1e10}, {1, 1}};
    auto C = A + B;
    ASSERT_TRUE(abs(C(0, 1) - 2e10) < 1e5);
}

TEST(EdgeCases, DivisionByZero_Throws)
{
    matrix<double> M = {{1, 2}, {3, 4}};
    bool caught = false;
    try
    {
        auto N = M / 0.0;
    }
    catch (const std::invalid_argument&)
    {
        caught = true;
    }
    ASSERT_TRUE(caught);
}

TEST(EdgeCases, LargeIndex_Access)
{
    matrix<double> M(100, 100);
    for (size_t i = 0; i < 100; i++)
    {
        for (size_t j = 0; j < 100; j++)
        {
            M(i, j) = static_cast<double>(i * 100 + j);
        }
    }
    ASSERT_EQ(M(0, 0), 0);
    ASSERT_EQ(M(99, 99), 9999);
    ASSERT_EQ(M(50, 75), 5075);
}

TEST(EdgeCases, SubmatrixEdgeIndex_TopLeft)
{
    matrix<double> M = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto sub = M.submatrix(0, 0, 0, 0); // Single element at (0,0)
    ASSERT_EQ(sub(0, 0), 1);
}

TEST(EdgeCases, SubmatrixEdgeIndex_BottomRight)
{
    matrix<double> M = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto sub = M.submatrix(2, 2, 2, 2); // Single element at (2,2)
    ASSERT_EQ(sub(0, 0), 9);
}

TEST(EdgeCases, Identity_1x1)
{
    auto I = matrix<double>::eye(1, 1);
    ASSERT_EQ(I(0, 0), 1);
    ASSERT_EQ(I.determinant(), 1);
}

TEST(EdgeCases, Identity_DeterministicMultiplication)
{
    auto I = matrix<double>::eye(3, 3);
    matrix<double> M = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto C1 = I * M;
    auto C2 = M * I;
    ASSERT_TRUE((M - C1).norm(2) < 1e-10);
    ASSERT_TRUE((M - C2).norm(2) < 1e-10);
    ASSERT_TRUE(M == C1);
    ASSERT_TRUE(M == C2);
}

TEST(EdgeCases, Identity_Inverse)
{
    auto I = matrix<double>::eye(3, 3);
    auto Inv = I.inverse();
    ASSERT_TRUE(I == Inv);
}

TEST(EdgeCases, Identity_Power)
{
    auto I = matrix<double>::eye(3, 3);
    auto P = I.pow(100);
    ASSERT_TRUE(I == P);
}

TEST(EdgeCases, DimensionMismatch_Addition_Throws)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    matrix<double> B = {{1, 2, 3}, {4, 5, 6}};
    bool caught = false;
    try
    {
        auto C = A + B;
    }
    catch (const std::invalid_argument&)
    {
        caught = true;
    }
    ASSERT_TRUE(caught);
}

TEST(EdgeCases, DimensionMismatch_Multiplication_Throws)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    matrix<double> B = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    bool caught = false;
    try
    {
        auto C = A * B;
    }
    catch (const std::invalid_argument&)
    {
        caught = true;
    }
    ASSERT_TRUE(caught);
}

TEST(EdgeCases, DimensionMismatch_Hadamard_Throws)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    matrix<double> B = {{1, 2, 3}};
    bool caught = false;
    try
    {
        auto C = A.hadamard(B);
    }
    catch (const std::invalid_argument&)
    {
        caught = true;
    }
    ASSERT_TRUE(caught);
}

TEST(EdgeCases, NonSquare_Determinant_Throws)
{
    matrix<double> M = {{1, 2, 3}, {4, 5, 6}};
    bool caught = false;
    try
    {
        (void)M.determinant();
    }
    catch (const std::invalid_argument&)
    {
        caught = true;
    }
    ASSERT_TRUE(caught);
}

TEST(EdgeCases, NonSquare_Trace_Throws)
{
    matrix<double> M = {{1, 2, 3}, {4, 5, 6}};
    bool caught = false;
    try
    {
        (void)M.trace();
    }
    catch (const std::invalid_argument&)
    {
        caught = true;
    }
    ASSERT_TRUE(caught);
}

TEST(EdgeCases, NonSquare_Inverse_Throws)
{
    matrix<double> M = {{1, 2, 3}, {4, 5, 6}};
    bool caught = false;
    try
    {
        auto inv = M.inverse();
    }
    catch (const std::invalid_argument&)
    {
        caught = true;
    }
    ASSERT_TRUE(caught);
}

TEST(EdgeCases, ChainedOperations_ArithmeticOnly)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    matrix<double> B = {{5, 6}, {7, 8}};
    auto C = ((A + B) * 2.0) - A;
    ASSERT_EQ(C(0, 0), 11);
}

TEST(EdgeCases, ChainedOperations_WithTranspose)
{
    matrix<double> M = {{1, 2}, {3, 4}};
    auto R = M.transpose().transpose();
    ASSERT_TRUE(M == R);
}

TEST(EdgeCases, ChainedOperations_Complex)
{
    matrix<double> M = {{2, 0}, {0, 2}};
    auto R = (M * M.inverse()) * M;
    ASSERT_TRUE((R - M).norm(2) < 1e-9);
}

TEST(EdgeCases, NearSingular_SmallDeterminant)
{
    matrix<double> M = {{1, 1}, {1, 1.0000001}};
    double det = M.determinant();
    ASSERT_TRUE(abs(det) < 1e-6);
    ASSERT_TRUE(abs(det) > 0);
}

TEST(EdgeCases, NearSingular_LargeConditionNumber)
{
    matrix<double> M = {{1e10, 1}, {1, 1e-10}};
    double det = M.determinant();
    ASSERT_TRUE(abs(det) < 1e5); // Much smaller than diagonal elements
}

TEST(EdgeCases, DiagonalMatrix_AllSameValue)
{
    auto M = matrix<double>::eye(3, 3);
    M = M * 5.0;
    ASSERT_TRUE(M.is_diagonal());
    ASSERT_EQ(M.trace(), 15);
}

TEST(EdgeCases, UpperTriangular_MainDiagonalZero)
{
    matrix<double> M = {{0, 1, 2}, {0, 0, 3}, {0, 0, 0}};
    ASSERT_TRUE(M.is_upper_triangular());
    ASSERT_EQ(M.determinant(), 0);
}

TEST(EdgeCases, LowerTriangular_MainDiagonalZero)
{
    matrix<double> M = {{0, 0, 0}, {1, 0, 0}, {2, 3, 0}};
    ASSERT_TRUE(M.is_lower_triangular());
    ASSERT_EQ(M.determinant(), 0);
}

TEST(EdgeCases, Symmetric_AllSameValue)
{
    matrix<double> M = {{5, 5, 5}, {5, 5, 5}, {5, 5, 5}};
    ASSERT_TRUE(M.is_symmetric());
}

TEST(EdgeCases, FloatingPoint_AccumulationError)
{
    matrix<double> M(100, 100);
    for (size_t i = 0; i < 100; ++i)
    {
        for (size_t j = 0; j < 100; ++j)
        {
            M(i, j) = 0.1;
        }
    }
    double sum = 0;
    for (const auto& val : M)
    {
        sum += val;
    }
    ASSERT_TRUE(abs(sum - 1000.0) < 1e-9);
}

TEST(EdgeCases, IntegerMatrix_NoRounding)
{
    matrix<int> M = {{1, 2}, {3, 4}};
    matrix<int> N = {{5, 6}, {7, 8}};
    auto C = M + N;
    ASSERT_TRUE(C(0, 0) == 6);
    ASSERT_TRUE(C(1, 1) == 12);
}
