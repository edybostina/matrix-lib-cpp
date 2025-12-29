#include "../include/matrix.hpp"
#include "test_framework.hpp"

using namespace std;

TEST(Arithmetic, Matrix_Addition)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    matrix<double> B = {{5, 6}, {7, 8}};
    auto C = A + B;
    ASSERT_EQ(C(0, 0), 6);
    ASSERT_EQ(C(1, 1), 12);
}

TEST(Arithmetic, Matrix_Subtraction)
{
    matrix<double> A = {{5, 6}, {7, 8}};
    matrix<double> B = {{1, 2}, {3, 4}};
    auto C = A - B;
    ASSERT_EQ(C(0, 0), 4);
    ASSERT_EQ(C(1, 1), 4);
}

TEST(Arithmetic, Matrix_Multiplication)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    matrix<double> B = {{2, 0}, {1, 2}};
    auto C = A * B;
    ASSERT_EQ(C(0, 0), 4);  // 1*2 + 2*1
    ASSERT_EQ(C(0, 1), 4);  // 1*0 + 2*2
    ASSERT_EQ(C(1, 0), 10); // 3*2 + 4*1
    ASSERT_EQ(C(1, 1), 8);  // 3*0 + 4*2
}

TEST(Arithmetic, Scalar_Multiplication)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    auto B = A * 2.0;
    ASSERT_EQ(B(0, 0), 2);
    ASSERT_EQ(B(1, 1), 8);
}

TEST(Arithmetic, Scalar_Division)
{
    matrix<double> A = {{2, 4}, {6, 8}};
    auto B = A / 2.0;
    ASSERT_EQ(B(0, 0), 1);
    ASSERT_EQ(B(1, 1), 4);
}

TEST(Arithmetic, Hadamard_Product)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    matrix<double> B = {{2, 3}, {4, 5}};
    auto C = A.hadamard(B);
    ASSERT_EQ(C(0, 0), 2);
    ASSERT_EQ(C(1, 1), 20);
}

TEST(Arithmetic, Compound_Assignment)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    A += 10.0;
    ASSERT_EQ(A(0, 0), 11);
    A *= 2.0;
    ASSERT_EQ(A(0, 0), 22);
}
