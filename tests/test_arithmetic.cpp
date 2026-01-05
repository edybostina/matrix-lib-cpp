#include "../include/matrix.hpp"
#include "test_framework.hpp"

using namespace std;

TEST(Arithmetic, Addition_2x2)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    matrix<double> B = {{5, 6}, {7, 8}};
    auto C = A + B;
    ASSERT_EQ(C(0, 0), 6);
    ASSERT_EQ(C(0, 1), 8);
    ASSERT_EQ(C(1, 0), 10);
    ASSERT_EQ(C(1, 1), 12);
}

TEST(Arithmetic, Addition_3x3)
{
    matrix<double> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    matrix<double> B = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
    auto C = A + B;
    ASSERT_EQ(C(0, 0), 10);
    ASSERT_EQ(C(1, 1), 10);
    ASSERT_EQ(C(2, 2), 10);
}

TEST(Arithmetic, Addition_Rectangular)
{
    matrix<double> A = {{1, 2, 3}, {4, 5, 6}};
    matrix<double> B = {{6, 5, 4}, {3, 2, 1}};
    auto C = A + B;
    ASSERT_TRUE(C.rows() == 2);
    ASSERT_TRUE(C.cols() == 3);
    ASSERT_EQ(C(0, 0), 7);
}

TEST(Arithmetic, Addition_Identity)
{
    auto A = matrix<double>::eye(3, 3);
    auto B = matrix<double>::eye(3, 3);
    auto C = A + B;
    ASSERT_EQ(C(0, 0), 2);
    ASSERT_EQ(C(1, 1), 2);
    ASSERT_EQ(C(2, 2), 2);
    ASSERT_EQ(C(0, 1), 0);
}

TEST(Arithmetic, Addition_Zeros)
{
    auto A = matrix<double>::zeros(2, 2);
    auto B = matrix<double>::ones(2, 2);
    auto C = A + B;
    ASSERT_EQ(C(0, 0), 1);
    ASSERT_EQ(C(1, 1), 1);
}

TEST(Arithmetic, Addition_Negative)
{
    matrix<double> A = {{-1, -2}, {-3, -4}};
    matrix<double> B = {{1, 2}, {3, 4}};
    auto C = A + B;
    ASSERT_EQ(C(0, 0), 0);
    ASSERT_EQ(C(1, 1), 0);
}

TEST(Arithmetic, Addition_Commutative)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    matrix<double> B = {{5, 6}, {7, 8}};
    auto C1 = A + B;
    auto C2 = B + A;
    ASSERT_TRUE(C1 == C2);
}

TEST(Arithmetic, Subtraction_2x2)
{
    matrix<double> A = {{5, 6}, {7, 8}};
    matrix<double> B = {{1, 2}, {3, 4}};
    auto C = A - B;
    ASSERT_EQ(C(0, 0), 4);
    ASSERT_EQ(C(0, 1), 4);
    ASSERT_EQ(C(1, 0), 4);
    ASSERT_EQ(C(1, 1), 4);
}

TEST(Arithmetic, Subtraction_3x3)
{
    matrix<double> A = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
    matrix<double> B = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto C = A - B;
    ASSERT_EQ(C(0, 0), 8);
    ASSERT_EQ(C(2, 2), -8);
}

TEST(Arithmetic, Subtraction_Self)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    auto C = A - A;
    ASSERT_EQ(C(0, 0), 0);
    ASSERT_EQ(C(1, 1), 0);
}

TEST(Arithmetic, Subtraction_Identity)
{
    auto A = matrix<double>::eye(3, 3);
    auto Z = matrix<double>::zeros(3, 3);
    auto C = A - Z;
    ASSERT_TRUE(C == A);
}

TEST(Arithmetic, Multiplication_2x2)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    matrix<double> B = {{2, 0}, {1, 2}};
    auto C = A * B;
    ASSERT_EQ(C(0, 0), 4);
    ASSERT_EQ(C(0, 1), 4);
    ASSERT_EQ(C(1, 0), 10);
    ASSERT_EQ(C(1, 1), 8);
}

TEST(Arithmetic, Multiplication_3x3)
{
    matrix<double> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    matrix<double> B = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
    auto C = A * B;
    ASSERT_EQ(C(0, 0), 30);
    ASSERT_EQ(C(1, 1), 69);
    ASSERT_EQ(C(2, 2), 90);
}

TEST(Arithmetic, Multiplication_Rectangular_2x3_3x2)
{
    matrix<double> A = {{1, 2, 3}, {4, 5, 6}};
    matrix<double> B = {{7, 8}, {9, 10}, {11, 12}};
    auto C = A * B;
    ASSERT_TRUE(C.rows() == 2);
    ASSERT_TRUE(C.cols() == 2);
    ASSERT_EQ(C(0, 0), 58);
    ASSERT_EQ(C(1, 1), 154);
}

TEST(Arithmetic, Multiplication_Rectangular_3x2_2x4)
{
    matrix<double> A = {{1, 2}, {3, 4}, {5, 6}};
    matrix<double> B = {{1, 2, 3, 4}, {5, 6, 7, 8}};
    auto C = A * B;
    ASSERT_TRUE(C.rows() == 3);
    ASSERT_TRUE(C.cols() == 4);
    ASSERT_EQ(C(0, 0), 11);
}

TEST(Arithmetic, Multiplication_Identity)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    auto I = matrix<double>::eye(2, 2);
    auto C1 = A * I;
    auto C2 = I * A;
    ASSERT_TRUE(C1 == A);
    ASSERT_TRUE(C2 == A);
}

TEST(Arithmetic, Multiplication_Associative)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    matrix<double> B = {{2, 0}, {1, 2}};
    matrix<double> C = {{1, 1}, {0, 1}};
    auto D1 = (A * B) * C;
    auto D2 = A * (B * C);
    ASSERT_TRUE((D1 - D2).norm(2) < 1e-9);
}

TEST(Arithmetic, Multiplication_NonCommutative)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    matrix<double> B = {{2, 0}, {1, 2}};
    auto C1 = A * B;
    auto C2 = B * A;
    ASSERT_TRUE(!(C1 == C2));
}

TEST(Arithmetic, ScalarMultiplication_Right)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    auto B = A * 2.0;
    ASSERT_EQ(B(0, 0), 2);
    ASSERT_EQ(B(0, 1), 4);
    ASSERT_EQ(B(1, 0), 6);
    ASSERT_EQ(B(1, 1), 8);
}

TEST(Arithmetic, ScalarMultiplication_Left)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    auto B = 3.0 * A;
    ASSERT_EQ(B(0, 0), 3);
    ASSERT_EQ(B(1, 1), 12);
}

TEST(Arithmetic, ScalarMultiplication_Zero)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    auto B = A * 0.0;
    ASSERT_EQ(B(0, 0), 0);
    ASSERT_EQ(B(1, 1), 0);
}

TEST(Arithmetic, ScalarMultiplication_Negative)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    auto B = A * (-1.0);
    ASSERT_EQ(B(0, 0), -1);
    ASSERT_EQ(B(1, 1), -4);
}

TEST(Arithmetic, ScalarMultiplication_Fraction)
{
    matrix<double> A = {{2, 4}, {6, 8}};
    auto B = A * 0.5;
    ASSERT_EQ(B(0, 0), 1);
    ASSERT_EQ(B(1, 1), 4);
}

TEST(Arithmetic, ScalarDivision)
{
    matrix<double> A = {{2, 4}, {6, 8}};
    auto B = A / 2.0;
    ASSERT_EQ(B(0, 0), 1);
    ASSERT_EQ(B(0, 1), 2);
    ASSERT_EQ(B(1, 0), 3);
    ASSERT_EQ(B(1, 1), 4);
}

TEST(Arithmetic, ScalarDivision_Fraction)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    auto B = A / 0.5;
    ASSERT_EQ(B(0, 0), 2);
    ASSERT_EQ(B(1, 1), 8);
}

TEST(Arithmetic, ScalarAddition)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    auto B = A + 10.0;
    ASSERT_EQ(B(0, 0), 11);
    ASSERT_EQ(B(0, 1), 12);
    ASSERT_EQ(B(1, 0), 13);
    ASSERT_EQ(B(1, 1), 14);
}

TEST(Arithmetic, ScalarSubtraction)
{
    matrix<double> A = {{10, 20}, {30, 40}};
    auto B = A - 5.0;
    ASSERT_EQ(B(0, 0), 5);
    ASSERT_EQ(B(1, 1), 35);
}

TEST(Arithmetic, CompoundAddition_Scalar)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    A += 10.0;
    ASSERT_EQ(A(0, 0), 11);
    ASSERT_EQ(A(1, 1), 14);
}

TEST(Arithmetic, CompoundSubtraction_Scalar)
{
    matrix<double> A = {{10, 20}, {30, 40}};
    A -= 5.0;
    ASSERT_EQ(A(0, 0), 5);
    ASSERT_EQ(A(1, 1), 35);
}

TEST(Arithmetic, CompoundMultiplication_Scalar)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    A *= 2.0;
    ASSERT_EQ(A(0, 0), 2);
    ASSERT_EQ(A(1, 1), 8);
}

TEST(Arithmetic, CompoundDivision_Scalar)
{
    matrix<double> A = {{10, 20}, {30, 40}};
    A /= 10.0;
    ASSERT_EQ(A(0, 0), 1);
    ASSERT_EQ(A(1, 1), 4);
}

TEST(Arithmetic, CompoundAddition_Matrix)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    matrix<double> B = {{5, 6}, {7, 8}};
    A += B;
    ASSERT_EQ(A(0, 0), 6);
    ASSERT_EQ(A(1, 1), 12);
}

TEST(Arithmetic, CompoundSubtraction_Matrix)
{
    matrix<double> A = {{10, 20}, {30, 40}};
    matrix<double> B = {{1, 2}, {3, 4}};
    A -= B;
    ASSERT_EQ(A(0, 0), 9);
    ASSERT_EQ(A(1, 1), 36);
}

TEST(Arithmetic, Hadamard_2x2)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    matrix<double> B = {{2, 3}, {4, 5}};
    auto C = A.hadamard(B);
    ASSERT_EQ(C(0, 0), 2);
    ASSERT_EQ(C(0, 1), 6);
    ASSERT_EQ(C(1, 0), 12);
    ASSERT_EQ(C(1, 1), 20);
}

TEST(Arithmetic, Hadamard_3x3)
{
    matrix<double> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    matrix<double> B = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
    auto C = A.hadamard(B);
    ASSERT_EQ(C(0, 0), 9);
    ASSERT_EQ(C(1, 1), 25);
    ASSERT_EQ(C(2, 2), 9);
}

TEST(Arithmetic, Hadamard_Identity)
{
    auto A = matrix<double>::eye(3, 3);
    auto B = matrix<double>::ones(3, 3);
    auto C = A.hadamard(B);
    ASSERT_TRUE(C == A);
}

TEST(Arithmetic, Hadamard_Commutative)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    matrix<double> B = {{5, 6}, {7, 8}};
    auto C1 = A.hadamard(B);
    auto C2 = B.hadamard(A);
    ASSERT_TRUE(C1 == C2);
}

TEST(Arithmetic, Hadamard_WithZeros)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    auto Z = matrix<double>::zeros(2, 2);
    auto C = A.hadamard(Z);
    ASSERT_EQ(C(0, 0), 0);
    ASSERT_EQ(C(1, 1), 0);
}

TEST(Arithmetic, Equality_Equal)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    matrix<double> B = {{1, 2}, {3, 4}};
    ASSERT_TRUE(A == B);
}

TEST(Arithmetic, Equality_NotEqual)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    matrix<double> B = {{1, 2}, {3, 5}};
    ASSERT_TRUE(!(A == B));
}

TEST(Arithmetic, Equality_DifferentSize)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    matrix<double> B = {{1, 2, 3}, {4, 5, 6}};
    ASSERT_TRUE(!(A == B));
}

TEST(Arithmetic, Inequality_NotEqual)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    matrix<double> B = {{1, 2}, {3, 5}};
    ASSERT_TRUE(A != B);
}

TEST(Arithmetic, Inequality_Equal)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    matrix<double> B = {{1, 2}, {3, 4}};
    ASSERT_TRUE(!(A != B));
}
