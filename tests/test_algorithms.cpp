#include "../include/matrix.hpp"
#include "test_framework.hpp"
#include <cmath>

using namespace std;

TEST(Algorithms, Minor_2x2_TopLeft)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    auto m = A.minor(0, 0);
    ASSERT_TRUE(m.rows() == 1);
    ASSERT_TRUE(m.cols() == 1);
    ASSERT_EQ(m(0, 0), 4);
}

TEST(Algorithms, Minor_2x2_BottomRight)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    auto m = A.minor(1, 1);
    ASSERT_EQ(m(0, 0), 1);
}

TEST(Algorithms, Minor_3x3_Center)
{
    matrix<double> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto m = A.minor(1, 1);
    ASSERT_TRUE(m.rows() == 2);
    ASSERT_TRUE(m.cols() == 2);
    ASSERT_EQ(m(0, 0), 1);
    ASSERT_EQ(m(0, 1), 3);
    ASSERT_EQ(m(1, 0), 7);
    ASSERT_EQ(m(1, 1), 9);
}

TEST(Algorithms, Minor_3x3_TopLeft)
{
    matrix<double> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto m = A.minor(0, 0);
    ASSERT_EQ(m(0, 0), 5);
    ASSERT_EQ(m(1, 1), 9);
}

TEST(Algorithms, Minor_3x3_BottomRight)
{
    matrix<double> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto m = A.minor(2, 2);
    ASSERT_EQ(m(0, 0), 1);
    ASSERT_EQ(m(1, 1), 5);
}

TEST(Algorithms, Minor_4x4)
{
    matrix<double> A = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
    auto m = A.minor(1, 1);
    ASSERT_TRUE(m.rows() == 3);
    ASSERT_TRUE(m.cols() == 3);
    ASSERT_EQ(m(0, 0), 1);
    ASSERT_EQ(m(0, 1), 3);
    ASSERT_EQ(m(0, 2), 4);
    ASSERT_EQ(m(1, 0), 9);
    ASSERT_EQ(m(2, 2), 16);
}

TEST(Algorithms, Cofactor_2x2)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    auto C = A.cofactor();
    ASSERT_EQ(C(0, 0), 4);
    ASSERT_EQ(C(0, 1), -3);
    ASSERT_EQ(C(1, 0), -2);
    ASSERT_EQ(C(1, 1), 1);
}

TEST(Algorithms, Cofactor_3x3)
{
    matrix<double> A = {{1, 2, 3}, {0, 1, 4}, {5, 6, 0}};
    auto C = A.cofactor();
    ASSERT_TRUE(C.rows() == 3);
    ASSERT_TRUE(C.cols() == 3);
    ASSERT_EQ(C(0, 0), -24);
    ASSERT_EQ(C(0, 1), 20);
    ASSERT_EQ(C(0, 2), -5);
}

TEST(Algorithms, Cofactor_Identity)
{
    auto I = matrix<double>::eye(3, 3);
    auto C = I.cofactor();
    ASSERT_TRUE(C == I);
}

TEST(Algorithms, Cofactor_Diagonal)
{
    matrix<double> D = {{2, 0, 0}, {0, 3, 0}, {0, 0, 4}};
    auto C = D.cofactor();
    ASSERT_EQ(C(0, 0), 12);
    ASSERT_EQ(C(1, 1), 8);
    ASSERT_EQ(C(2, 2), 6);
}

TEST(Algorithms, Cofactor_SignPattern)
{
    matrix<double> A = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    auto C = A.cofactor();
    ASSERT_TRUE(C(0, 0) >= 0 || C(0, 0) == 0);
    ASSERT_TRUE(C(0, 1) <= 0 || C(0, 1) == 0);
}

TEST(Algorithms, Adjoint_2x2)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    auto adj = A.adjoint();
    ASSERT_EQ(adj(0, 0), 4);
    ASSERT_EQ(adj(0, 1), -2);
    ASSERT_EQ(adj(1, 0), -3);
    ASSERT_EQ(adj(1, 1), 1);
}

TEST(Algorithms, Adjoint_3x3)
{
    matrix<double> A = {{1, 2, 3}, {0, 1, 4}, {5, 6, 0}};
    auto adj = A.adjoint();
    ASSERT_TRUE(adj.rows() == 3);
    ASSERT_TRUE(adj.cols() == 3);
    // Adjoint is transpose of cofactor matrix
    auto C = A.cofactor();
    auto adjCheck = C.transpose();
    ASSERT_TRUE((adj - adjCheck).norm(2) < 1e-9);
}

TEST(Algorithms, Adjoint_Identity)
{
    auto I = matrix<double>::eye(3, 3);
    auto adj = I.adjoint();
    ASSERT_TRUE(adj == I);
}

TEST(Algorithms, Adjoint_InverseRelation)
{
    matrix<double> A = {{4, 7}, {2, 6}};
    auto adj = A.adjoint();
    double det = A.determinant();
    auto inv = A.inverse();
    auto check = adj / det;
    ASSERT_TRUE((inv - check).norm(2) < 1e-9);
}

TEST(Algorithms, GaussianElimination_Identity)
{
    auto I = matrix<double>::eye(3, 3);
    auto G = I.gaussian_elimination();
    ASSERT_TRUE((I - G).norm(2) < 1e-9);
}

TEST(Algorithms, GaussianElimination_UpperTriangular)
{
    matrix<double> A = {{1, 2, 3}, {0, 4, 5}, {0, 0, 6}};
    auto G = A.gaussian_elimination();
    ASSERT_TRUE((A - G).norm(2) < 1e-9);
}

TEST(Algorithms, GaussianElimination_NeedsPivoting)
{
    matrix<double> A = {{0, 1, 2}, {1, 2, 3}, {3, 4, 5}};
    auto G = A.gaussian_elimination();
    ASSERT_TRUE(G.rows() == 3);
    ASSERT_TRUE(G.is_upper_triangular());
}

TEST(Algorithms, GaussianElimination_Singular)
{
    matrix<double> A = {{1, 2}, {2, 4}};
    auto G = A.gaussian_elimination();
    ASSERT_TRUE(abs(G(1, 0)) < 1e-10);
    ASSERT_TRUE(abs(G(1, 1)) < 1e-10);
}

TEST(Algorithms, GaussianElimination_3x3)
{
    matrix<double> A = {{2, 1, -1}, {-3, -1, 2}, {-2, 1, 2}};
    auto G = A.gaussian_elimination();
    ASSERT_TRUE(G.is_upper_triangular());
}

TEST(Algorithms, GaussianElimination_WithZeroPivot)
{
    matrix<double> A = {{0, 2, 3}, {1, 4, 5}, {2, 6, 7}};
    auto G = A.gaussian_elimination();
    ASSERT_TRUE(G.rows() == 3);
    ASSERT_TRUE(G.is_upper_triangular());
    ASSERT_TRUE(abs(G(0, 0)) > 1e-10);
}

TEST(Algorithms, Power_Identity)
{
    auto I = matrix<double>::eye(3, 3);
    auto P = I.pow(10);
    ASSERT_TRUE(I == P);
}

TEST(Algorithms, Power_Zero)
{
    matrix<double> M = {{1, 2}, {3, 4}};
    auto P = M.pow(0);
    auto I = matrix<double>::eye(2, 2);
    ASSERT_TRUE(P == I);
}

TEST(Algorithms, Power_One)
{
    matrix<double> M = {{1, 2}, {3, 4}};
    auto P = M.pow(1);
    ASSERT_TRUE(P == M);
}

TEST(Algorithms, Power_Two)
{
    matrix<double> M = {{1, 2}, {0, 1}};
    auto P = M.pow(2);
    ASSERT_EQ(P(0, 0), 1);
    ASSERT_EQ(P(0, 1), 4);
    ASSERT_EQ(P(1, 0), 0);
    ASSERT_EQ(P(1, 1), 1);
}

TEST(Algorithms, Power_Three)
{
    matrix<double> M = {{2, 0}, {0, 2}};
    auto P = M.pow(3);
    ASSERT_EQ(P(0, 0), 8);
    ASSERT_EQ(P(1, 1), 8);
}

TEST(Algorithms, Power_Diagonal)
{
    matrix<double> D = {{2, 0}, {0, 3}};
    auto P = D.pow(4);
    ASSERT_EQ(P(0, 0), 16);
    ASSERT_EQ(P(1, 1), 81);
}

TEST(Algorithms, Power_LargeExponent)
{
    matrix<double> M = {{1, 1}, {0, 1}};
    auto P = M.pow(10);
    ASSERT_EQ(P(0, 1), 10);
}

TEST(Algorithms, Apply_Abs)
{
    matrix<double> M = {{-1, -2}, {3, -4}};
    auto A = M.apply(static_cast<double (*)(double)>(std::abs));
    ASSERT_EQ(A(0, 0), 1);
    ASSERT_EQ(A(0, 1), 2);
    ASSERT_EQ(A(1, 0), 3);
    ASSERT_EQ(A(1, 1), 4);
}

TEST(Algorithms, Apply_Square)
{
    matrix<double> M = {{2, 3}, {4, 5}};
    auto S = M.apply([](double x) { return x * x; });
    ASSERT_EQ(S(0, 0), 4);
    ASSERT_EQ(S(0, 1), 9);
    ASSERT_EQ(S(1, 0), 16);
    ASSERT_EQ(S(1, 1), 25);
}

TEST(Algorithms, Apply_Sqrt)
{
    matrix<double> M = {{4, 9}, {16, 25}};
    auto S = M.apply(static_cast<double (*)(double)>(std::sqrt));
    ASSERT_EQ(S(0, 0), 2);
    ASSERT_EQ(S(0, 1), 3);
    ASSERT_EQ(S(1, 0), 4);
    ASSERT_EQ(S(1, 1), 5);
}

TEST(Algorithms, Apply_Ceil)
{
    matrix<double> M = {{1.1, 2.3}, {3.7, 4.9}};
    auto C = M.apply(static_cast<double (*)(double)>(std::ceil));
    ASSERT_EQ(C(0, 0), 2);
    ASSERT_EQ(C(1, 1), 5);
}

TEST(Algorithms, Apply_Floor)
{
    matrix<double> M = {{1.1, 2.3}, {3.7, 4.9}};
    auto F = M.apply(static_cast<double (*)(double)>(std::floor));
    ASSERT_EQ(F(0, 0), 1);
    ASSERT_EQ(F(1, 1), 4);
}

TEST(Algorithms, Apply_Negate)
{
    matrix<double> M = {{1, -2}, {3, -4}};
    auto N = M.apply([](double x) { return -x; });
    ASSERT_EQ(N(0, 0), -1);
    ASSERT_EQ(N(0, 1), 2);
}

TEST(Algorithms, Apply_AddConstant)
{
    matrix<double> M = {{1, 2}, {3, 4}};
    auto A = M.apply([](double x) { return x + 10; });
    ASSERT_EQ(A(0, 0), 11);
    ASSERT_EQ(A(1, 1), 14);
}

TEST(Algorithms, Clamp_BothBounds)
{
    matrix<double> M = {{-5, 0, 5}, {3, 10, -2}};
    auto C = M.clamp(0.0, 5.0);
    ASSERT_EQ(C(0, 0), 0.0);
    ASSERT_EQ(C(0, 1), 0.0);
    ASSERT_EQ(C(0, 2), 5.0);
    ASSERT_EQ(C(1, 0), 3.0);
    ASSERT_EQ(C(1, 1), 5.0);
    ASSERT_EQ(C(1, 2), 0.0);
}

TEST(Algorithms, Clamp_UpperBound)
{
    matrix<double> M = {{1, 5, 10}, {15, 20, 25}};
    auto C = M.clamp(0.0, 10.0);
    ASSERT_EQ(C(0, 2), 10.0);
    ASSERT_EQ(C(1, 0), 10.0);
}

TEST(Algorithms, Clamp_LowerBound)
{
    matrix<double> M = {{-10, -5, 0}, {5, 10, 15}};
    auto C = M.clamp(0.0, 100.0);
    ASSERT_EQ(C(0, 0), 0.0);
    ASSERT_EQ(C(0, 1), 0.0);
    ASSERT_EQ(C(0, 2), 0.0);
}

TEST(Algorithms, Clamp_NoChange)
{
    matrix<double> M = {{1, 2}, {3, 4}};
    auto C = M.clamp(0.0, 10.0);
    ASSERT_TRUE(C == M);
}

TEST(Algorithms, Clamp_AllOutOfRange)
{
    matrix<double> M = {{100, 200}, {300, 400}};
    auto C = M.clamp(0.0, 50.0);
    ASSERT_EQ(C(0, 0), 50.0);
    ASSERT_EQ(C(1, 1), 50.0);
}

TEST(Algorithms, Clamp_NegativeRange)
{
    matrix<double> M = {{-10, -5, 0}, {5, 10, 15}};
    auto C = M.clamp(-8.0, -2.0);
    ASSERT_EQ(C(0, 0), -8.0);
    ASSERT_EQ(C(0, 1), -5.0);
    ASSERT_EQ(C(0, 2), -2.0);
}

TEST(Algorithms, Clamp_EqualBounds)
{
    matrix<double> M = {{1, 2}, {3, 4}};
    auto C = M.clamp(2.5, 2.5);
    ASSERT_EQ(C(0, 0), 2.5);
    ASSERT_EQ(C(1, 1), 2.5);
}

TEST(Algorithms, Cofactor_4x4)
{
    matrix<double> A = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 17}};
    auto C = A.cofactor();
    ASSERT_TRUE(C.rows() == 4);
    ASSERT_TRUE(C.cols() == 4);
    double det = A.determinant();
    double sum = 0;
    for (size_t j = 0; j < 4; j++)
    {
        sum += A(0, j) * C(0, j);
    }
    ASSERT_TRUE(abs(det - sum) < 1e-9);
}

TEST(Algorithms, Cofactor_5x5_KnownMatrix)
{
    matrix<double> A = {{2, 0, 0, 0, 0}, {0, 3, 0, 0, 0}, {0, 0, 4, 0, 0}, {0, 0, 0, 5, 0}, {0, 0, 0, 0, 6}};
    auto C = A.cofactor();
    ASSERT_TRUE(C.rows() == 5);
    ASSERT_TRUE(C.cols() == 5);
    ASSERT_EQ(C(0, 0), 3 * 4 * 5 * 6);
    ASSERT_EQ(C(1, 1), 2 * 4 * 5 * 6);
    ASSERT_EQ(C(4, 4), 2 * 3 * 4 * 5);
}

TEST(Algorithms, Cofactor_6x6_Determinant)
{
    matrix<double> A = {{2, 1, 1, 1, 1, 1}, {0, 3, 1, 1, 1, 1}, {0, 0, 4, 1, 1, 1},
                        {0, 0, 0, 5, 1, 1}, {0, 0, 0, 0, 6, 1}, {0, 0, 0, 0, 0, 7}};
    auto C = A.cofactor();
    ASSERT_TRUE(C.rows() == 6);
    ASSERT_TRUE(C.cols() == 6);
    double det = A.determinant();
    double expected_det = 2 * 3 * 4 * 5 * 6 * 7;
    ASSERT_TRUE(abs(det - expected_det) < 1e-9);
}
