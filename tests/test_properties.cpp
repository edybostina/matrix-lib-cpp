#include "../include/matrix.hpp"
#include "test_framework.hpp"

TEST(Properties, IsSquare)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    matrix<double> B = {{1, 2, 3}, {4, 5, 6}};
    ASSERT_TRUE(A.is_square());
    ASSERT_TRUE(!B.is_square());
}

TEST(Properties, IsSymmetric)
{
    matrix<double> A = {{1, 2}, {2, 1}};
    matrix<double> B = {{1, 2}, {3, 4}};
    ASSERT_TRUE(A.is_symmetric());
    ASSERT_TRUE(!B.is_symmetric());
}

TEST(Properties, IsDiagonal)
{
    auto I = matrix<double>::eye(3, 3);
    ASSERT_TRUE(I.is_diagonal());
}

TEST(Properties, IsUpperTriangular)
{
    matrix<double> U = {{1, 2}, {0, 4}};
    ASSERT_TRUE(U.is_upper_triangular());
}

TEST(Properties, IsLowerTriangular)
{
    matrix<double> L = {{1, 0}, {2, 3}};
    ASSERT_TRUE(L.is_lower_triangular());
}

TEST(Properties, IsOrthogonal)
{
    matrix<double> O = {{1, 0}, {0, 1}};
    ASSERT_TRUE(O.is_orthogonal());
}

TEST(Properties, IsSingular)
{
    matrix<double> S = {{1, 2}, {2, 4}};
    ASSERT_TRUE(S.is_singular());
}

TEST(Properties, IsNilpotent)
{
    matrix<double> N = {{0, 1}, {0, 0}};
    ASSERT_TRUE(N.is_nilpotent(2));
}

TEST(Properties, IsIdempotent)
{
    matrix<double> I = {{1, 0}, {0, 0}};
    ASSERT_TRUE(I.is_idempotent());
}

TEST(Properties, IsInvolutory)
{
    matrix<double> I = {{1, 0}, {0, 1}};
    ASSERT_TRUE(I.is_involutory());
}

TEST(Properties, IsPositiveDefinite)
{
    matrix<double> P = {{2, -1}, {-1, 2}};
    ASSERT_TRUE(P.is_positive_definite());
}