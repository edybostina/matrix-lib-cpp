#define _USE_MATH_DEFINES
#include <cmath>
#include "../include/matrix.hpp"
#include "test_framework.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

TEST(Properties, IsSquare_True)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    ASSERT_TRUE(A.is_square());
}

TEST(Properties, IsSquare_False)
{
    matrix<double> B = {{1, 2, 3}, {4, 5, 6}};
    ASSERT_TRUE(!B.is_square());
}

TEST(Properties, IsSquare_1x1)
{
    matrix<double> C = {{5}};
    ASSERT_TRUE(C.is_square());
}

TEST(Properties, IsSquare_Large)
{
    auto M = matrix<double>::zeros(100, 100);
    ASSERT_TRUE(M.is_square());
}

TEST(Properties, IsSymmetric_True_2x2)
{
    matrix<double> A = {{1, 2}, {2, 1}};
    ASSERT_TRUE(A.is_symmetric());
}

TEST(Properties, IsSymmetric_True_3x3)
{
    matrix<double> A = {{1, 2, 3}, {2, 4, 5}, {3, 5, 6}};
    ASSERT_TRUE(A.is_symmetric());
}

TEST(Properties, IsSymmetric_False)
{
    matrix<double> B = {{1, 2}, {3, 4}};
    ASSERT_TRUE(!B.is_symmetric());
}

TEST(Properties, IsSymmetric_NonSquare)
{
    matrix<double> B = {{1, 2, 3}, {4, 5, 6}};
    ASSERT_TRUE(!B.is_symmetric());
}

TEST(Properties, IsSymmetric_Identity)
{
    auto I = matrix<double>::eye(3, 3);
    ASSERT_TRUE(I.is_symmetric());
}

TEST(Properties, IsSymmetric_Diagonal)
{
    matrix<double> D = {{5, 0, 0}, {0, 3, 0}, {0, 0, 7}};
    ASSERT_TRUE(D.is_symmetric());
}

TEST(Properties, IsSymmetric_AllSameValues)
{
    matrix<double> M = {{5, 5, 5}, {5, 5, 5}, {5, 5, 5}};
    ASSERT_TRUE(M.is_symmetric());
}

TEST(Properties, IsDiagonal_Identity)
{
    auto I = matrix<double>::eye(3, 3);
    ASSERT_TRUE(I.is_diagonal());
}

TEST(Properties, IsDiagonal_True)
{
    matrix<double> D = {{5, 0, 0}, {0, 3, 0}, {0, 0, 7}};
    ASSERT_TRUE(D.is_diagonal());
}

TEST(Properties, IsDiagonal_False)
{
    matrix<double> M = {{1, 2}, {0, 4}};
    ASSERT_TRUE(!M.is_diagonal());
}

TEST(Properties, IsDiagonal_Zeros)
{
    auto Z = matrix<double>::zeros(3, 3);
    ASSERT_TRUE(Z.is_diagonal());
}

TEST(Properties, IsDiagonal_NonSquare)
{
    matrix<double> M = {{1, 0, 0}, {0, 2, 0}};
    ASSERT_TRUE(!M.is_diagonal());
}

TEST(Properties, IsUpperTriangular_True)
{
    matrix<double> U = {{1, 2, 3}, {0, 4, 5}, {0, 0, 6}};
    ASSERT_TRUE(U.is_upper_triangular());
}

TEST(Properties, IsUpperTriangular_WithZeroDiagonal)
{
    matrix<double> U = {{1, 2}, {0, 0}};
    ASSERT_TRUE(U.is_upper_triangular());
}

TEST(Properties, IsUpperTriangular_False)
{
    matrix<double> M = {{1, 2}, {3, 4}};
    ASSERT_TRUE(!M.is_upper_triangular());
}

TEST(Properties, IsUpperTriangular_Identity)
{
    auto I = matrix<double>::eye(3, 3);
    ASSERT_TRUE(I.is_upper_triangular());
}

TEST(Properties, IsUpperTriangular_Diagonal)
{
    matrix<double> D = {{5, 0}, {0, 3}};
    ASSERT_TRUE(D.is_upper_triangular());
}

TEST(Properties, IsLowerTriangular_True)
{
    matrix<double> L = {{1, 0, 0}, {2, 3, 0}, {4, 5, 6}};
    ASSERT_TRUE(L.is_lower_triangular());
}

TEST(Properties, IsLowerTriangular_WithZeroDiagonal)
{
    matrix<double> L = {{0, 0}, {2, 3}};
    ASSERT_TRUE(L.is_lower_triangular());
}

TEST(Properties, IsLowerTriangular_False)
{
    matrix<double> M = {{1, 2}, {3, 4}};
    ASSERT_TRUE(!M.is_lower_triangular());
}

TEST(Properties, IsLowerTriangular_Identity)
{
    auto I = matrix<double>::eye(3, 3);
    ASSERT_TRUE(I.is_lower_triangular());
}

TEST(Properties, IsLowerTriangular_Diagonal)
{
    matrix<double> D = {{5, 0}, {0, 3}};
    ASSERT_TRUE(D.is_lower_triangular());
}

TEST(Properties, IsOrthogonal_Identity)
{
    matrix<double> O = {{1, 0}, {0, 1}};
    ASSERT_TRUE(O.is_orthogonal());
}

TEST(Properties, IsOrthogonal_RotationMatrix)
{
    double angle = M_PI / 4; // 45 degrees
    matrix<double> R = {{cos(angle), -sin(angle)}, {sin(angle), cos(angle)}};
    ASSERT_TRUE(R.is_orthogonal());
}

TEST(Properties, IsOrthogonal_False)
{
    matrix<double> M = {{1, 2}, {3, 4}};
    ASSERT_TRUE(!M.is_orthogonal());
}

TEST(Properties, IsOrthogonal_Reflection)
{
    matrix<double> R = {{1, 0}, {0, -1}};
    ASSERT_TRUE(R.is_orthogonal());
}

TEST(Properties, IsSingular_True)
{
    matrix<double> S = {{1, 2}, {2, 4}};
    ASSERT_TRUE(S.is_singular());
}

TEST(Properties, IsSingular_ZeroRow)
{
    matrix<double> S = {{1, 2, 3}, {0, 0, 0}, {4, 5, 6}};
    ASSERT_TRUE(S.is_singular());
}

TEST(Properties, IsSingular_False)
{
    matrix<double> M = {{1, 2}, {3, 4}};
    ASSERT_TRUE(!M.is_singular());
}

TEST(Properties, IsSingular_Identity)
{
    auto I = matrix<double>::eye(3, 3);
    ASSERT_TRUE(!I.is_singular());
}

TEST(Properties, IsNilpotent_True)
{
    matrix<double> N = {{0, 1, 0}, {0, 0, 1}, {0, 0, 0}};
    ASSERT_TRUE(N.is_nilpotent(3));
}

TEST(Properties, IsNilpotent_Order2)
{
    matrix<double> N = {{0, 1}, {0, 0}};
    ASSERT_TRUE(N.is_nilpotent(2));
}

TEST(Properties, IsNilpotent_False)
{
    auto I = matrix<double>::eye(2, 2);
    ASSERT_TRUE(!I.is_nilpotent(10));
}

TEST(Properties, IsIdempotent_Projection)
{
    matrix<double> P = {{1, 0}, {0, 0}};
    ASSERT_TRUE(P.is_idempotent());
}

TEST(Properties, IsIdempotent_Identity)
{
    auto I = matrix<double>::eye(3, 3);
    ASSERT_TRUE(I.is_idempotent());
}

TEST(Properties, IsIdempotent_False)
{
    matrix<double> M = {{1, 2}, {3, 4}};
    ASSERT_TRUE(!M.is_idempotent());
}

TEST(Properties, IsIdempotent_Zero)
{
    auto Z = matrix<double>::zeros(2, 2);
    ASSERT_TRUE(Z.is_idempotent());
}

TEST(Properties, IsInvolutory_Identity)
{
    matrix<double> I = {{1, 0}, {0, 1}};
    ASSERT_TRUE(I.is_involutory());
}

TEST(Properties, IsInvolutory_Reflection)
{
    matrix<double> R = {{1, 0}, {0, -1}};
    ASSERT_TRUE(R.is_involutory());
}

TEST(Properties, IsInvolutory_False)
{
    matrix<double> M = {{1, 2}, {3, 4}};
    ASSERT_TRUE(!M.is_involutory());
}

TEST(Properties, IsInvolutory_Swap)
{
    matrix<double> S = {{0, 1}, {1, 0}};
    ASSERT_TRUE(S.is_involutory());
}

TEST(Properties, IsPositiveDefinite_True)
{
    matrix<double> P = {{2, -1, 0}, {-1, 2, -1}, {0, -1, 2}};
    ASSERT_TRUE(P.is_positive_definite());
}

TEST(Properties, IsPositiveDefinite_Identity)
{
    auto I = matrix<double>::eye(3, 3);
    ASSERT_TRUE(I.is_positive_definite());
}

TEST(Properties, IsPositiveDefinite_False)
{
    matrix<double> M = {{-1, 0}, {0, -1}};
    ASSERT_TRUE(!M.is_positive_definite());
}

TEST(Properties, IsPositiveDefinite_Singular)
{
    matrix<double> S = {{1, 1}, {1, 1}};
    ASSERT_TRUE(!S.is_positive_definite());
}

TEST(Properties, DiagonalMatrixProperties)
{
    matrix<double> D = {{5, 0, 0}, {0, 3, 0}, {0, 0, 7}};
    ASSERT_TRUE(D.is_diagonal());
    ASSERT_TRUE(D.is_symmetric());
    ASSERT_TRUE(D.is_upper_triangular());
    ASSERT_TRUE(D.is_lower_triangular());
    ASSERT_TRUE(!D.is_singular());
}

TEST(Properties, IdentityMatrixProperties)
{
    auto I = matrix<double>::eye(3, 3);
    ASSERT_TRUE(I.is_diagonal());
    ASSERT_TRUE(I.is_symmetric());
    ASSERT_TRUE(I.is_upper_triangular());
    ASSERT_TRUE(I.is_lower_triangular());
    ASSERT_TRUE(I.is_orthogonal());
    ASSERT_TRUE(!I.is_singular());
    ASSERT_TRUE(I.is_idempotent());
    ASSERT_TRUE(I.is_involutory());
}

TEST(Properties, ZeroMatrixProperties)
{
    auto Z = matrix<double>::zeros(3, 3);
    ASSERT_TRUE(Z.is_diagonal());
    ASSERT_TRUE(Z.is_symmetric());
    ASSERT_TRUE(Z.is_upper_triangular());
    ASSERT_TRUE(Z.is_lower_triangular());
    ASSERT_TRUE(Z.is_singular());
    ASSERT_TRUE(Z.is_idempotent());
}
