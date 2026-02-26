#include "../include/matrix.hpp"
#include "test_framework.hpp"
#include <vector>

using namespace std;

TEST(Basic, Construction_Default)
{
    matrix<double> A(3, 3);
    ASSERT_TRUE(A.rows() == 3);
    ASSERT_TRUE(A.cols() == 3);
    ASSERT_EQ(A(0, 0), 0);
}

TEST(Basic, Construction_InitializerList)
{
    matrix<double> B = {{1, 2}, {3, 4}};
    ASSERT_EQ(B(0, 0), 1);
    ASSERT_EQ(B(0, 1), 2);
    ASSERT_EQ(B(1, 0), 3);
    ASSERT_EQ(B(1, 1), 4);
}

TEST(Basic, Construction_Vector2D)
{
    std::vector<std::vector<double>> vec = {{1, 2}, {3, 4}};
    matrix<double> M(vec);
    ASSERT_EQ(M(0, 0), 1);
    ASSERT_EQ(M(1, 1), 4);
}

TEST(Basic, Construction_Empty_DefaultSize)
{
    matrix<double> M;
    ASSERT_TRUE(M.rows() == 0);
    ASSERT_TRUE(M.cols() == 0);
}

TEST(Basic, Factory_Zeros)
{
    auto Z = matrix<double>::zeros(3, 3);
    for (size_t i = 0; i < 3; i++)
        for (size_t j = 0; j < 3; j++)
            ASSERT_EQ(Z(i, j), 0);
}

TEST(Basic, Factory_Zeros_Rectangular)
{
    auto Z = matrix<double>::zeros(2, 3);
    ASSERT_TRUE(Z.rows() == 2);
    ASSERT_TRUE(Z.cols() == 3);
    ASSERT_EQ(Z(1, 2), 0);
}

TEST(Basic, Factory_Ones)
{
    auto O = matrix<double>::ones(2, 2);
    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 2; j++)
            ASSERT_EQ(O(i, j), 1);
}

TEST(Basic, Factory_Ones_Rectangular)
{
    auto O = matrix<double>::ones(3, 2);
    ASSERT_TRUE(O.rows() == 3);
    ASSERT_TRUE(O.cols() == 2);
    ASSERT_EQ(O(2, 1), 1);
}

TEST(Basic, Factory_Identity)
{
    auto I = matrix<double>::eye(3, 3);
    ASSERT_EQ(I(0, 0), 1);
    ASSERT_EQ(I(1, 1), 1);
    ASSERT_EQ(I(2, 2), 1);
    ASSERT_EQ(I(0, 1), 0);
    ASSERT_EQ(I(1, 0), 0);
}

TEST(Basic, Factory_Identity_Rectangular)
{
    auto I = matrix<double>::eye(2, 3);
    ASSERT_EQ(I(0, 0), 1);
    ASSERT_EQ(I(1, 1), 1);
    ASSERT_EQ(I(0, 2), 0);
}

TEST(Basic, Factory_Random)
{
    auto R = matrix<double>::random(5, 5, 0.0, 10.0);
    ASSERT_TRUE(R.rows() == 5);
    ASSERT_TRUE(R.cols() == 5);
    bool has_variation = false;
    double first_val = R(0, 0);
    for (size_t i = 0; i < R.rows(); i++)
    {
        for (size_t j = 0; j < R.cols(); j++)
        {
            ASSERT_TRUE(R(i, j) >= 0.0 && R(i, j) <= 10.0);
            if (abs(R(i, j) - first_val) > 1e-9)
                has_variation = true;
        }
    }
    ASSERT_TRUE(has_variation);
}

TEST(Basic, Factory_Random_NegativeRange)
{
    auto R = matrix<double>::random(3, 3, -5.0, 5.0);
    bool has_negative = false, has_positive = false;
    for (const auto& val : R)
    {
        ASSERT_TRUE(val >= -5.0 && val <= 5.0);
        if (val < 0)
            has_negative = true;
        if (val > 0)
            has_positive = true;
    }
    ASSERT_TRUE(has_negative || has_positive);
}

TEST(Basic, Access_ReadWrite)
{
    matrix<double> A = {{1, 2}, {3, 4}};
    ASSERT_EQ(A(0, 0), 1);
    A(0, 0) = 10;
    ASSERT_EQ(A(0, 0), 10);
}

TEST(Basic, Access_AllElements)
{
    matrix<double> M = {{1, 2, 3}, {4, 5, 6}};
    ASSERT_EQ(M(0, 0), 1);
    ASSERT_EQ(M(0, 1), 2);
    ASSERT_EQ(M(0, 2), 3);
    ASSERT_EQ(M(1, 0), 4);
    ASSERT_EQ(M(1, 1), 5);
    ASSERT_EQ(M(1, 2), 6);
}

TEST(Basic, Access_GetRow)
{
    matrix<double> A = {{1, 2, 3}, {4, 5, 6}};
    auto row0 = A.row(0);
    ASSERT_TRUE(row0.size() == 3);
    ASSERT_EQ(row0[0], 1);
    ASSERT_EQ(row0[1], 2);
    ASSERT_EQ(row0[2], 3);
}

TEST(Basic, Access_GetRow_LastRow)
{
    matrix<double> A = {{1, 2}, {3, 4}, {5, 6}};
    auto row2 = A.row(2);
    ASSERT_EQ(row2[0], 5);
    ASSERT_EQ(row2[1], 6);
}

TEST(Basic, Access_ConstMatrix)
{
    const matrix<double> M = {{1, 2}, {3, 4}};
    ASSERT_EQ(M(0, 0), 1);
    ASSERT_EQ(M(1, 1), 4);
}

TEST(Basic, Dimensions_Square)
{
    matrix<double> M = {{1, 2}, {3, 4}};
    ASSERT_TRUE(M.rows() == 2);
    ASSERT_TRUE(M.cols() == 2);
    ASSERT_TRUE(M.size() == 4);
    ASSERT_TRUE(M.is_square());
}

TEST(Basic, Dimensions_Rectangular)
{
    matrix<double> M = {{1, 2, 3}, {4, 5, 6}};
    ASSERT_TRUE(M.rows() == 2);
    ASSERT_TRUE(M.cols() == 3);
    ASSERT_TRUE(M.size() == 6);
    ASSERT_TRUE(!M.is_square());
}

TEST(Basic, Dimensions_1x1)
{
    matrix<double> M = {{5}};
    ASSERT_TRUE(M.rows() == 1);
    ASSERT_TRUE(M.cols() == 1);
    ASSERT_TRUE(M.size() == 1);
    ASSERT_TRUE(M.is_square());
}
