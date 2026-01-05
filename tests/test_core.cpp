#include "../include/matrix.hpp"
#include "test_framework.hpp"

using namespace std;

TEST(Core, ExtractRow_FirstRow)
{
    matrix<double> M = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto r = M.row(0).to_matrix();
    ASSERT_TRUE(r.rows() == 1);
    ASSERT_TRUE(r.cols() == 3);
    ASSERT_EQ(r(0, 0), 1);
    ASSERT_EQ(r(0, 2), 3);
}

TEST(Core, ExtractRow_LastRow)
{
    matrix<double> M = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto r = M.row(2).to_matrix();
    ASSERT_EQ(r(0, 0), 7);
    ASSERT_EQ(r(0, 2), 9);
}

TEST(Core, ExtractRow_MiddleRow)
{
    matrix<double> M = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto r = M.row(1).to_matrix();
    ASSERT_EQ(r(0, 0), 4);
    ASSERT_EQ(r(0, 1), 5);
    ASSERT_EQ(r(0, 2), 6);
}

TEST(Core, ExtractColumn_FirstColumn)
{
    matrix<double> M = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto c = M.col(0).to_matrix();
    ASSERT_TRUE(c.rows() == 3);
    ASSERT_TRUE(c.cols() == 1);
    ASSERT_EQ(c(0, 0), 1);
    ASSERT_EQ(c(1, 0), 4);
    ASSERT_EQ(c(2, 0), 7);
}

TEST(Core, ExtractColumn_LastColumn)
{
    matrix<double> M = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto c = M.col(2).to_matrix();
    ASSERT_EQ(c(0, 0), 3);
    ASSERT_EQ(c(1, 0), 6);
    ASSERT_EQ(c(2, 0), 9);
}

TEST(Core, ExtractColumn_MiddleColumn)
{
    matrix<double> M = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto c = M.col(1).to_matrix();
    ASSERT_EQ(c(0, 0), 2);
    ASSERT_EQ(c(1, 0), 5);
    ASSERT_EQ(c(2, 0), 8);
}

TEST(Core, SwapRows_Adjacent)
{
    matrix<double> M = {{1, 2}, {3, 4}, {5, 6}};
    M.swap_rows(0, 1);
    ASSERT_EQ(M(0, 0), 3);
    ASSERT_EQ(M(0, 1), 4);
    ASSERT_EQ(M(1, 0), 1);
    ASSERT_EQ(M(1, 1), 2);
}

TEST(Core, SwapRows_NonAdjacent)
{
    matrix<double> M = {{1, 2}, {3, 4}, {5, 6}};
    M.swap_rows(0, 2);
    ASSERT_EQ(M(0, 0), 5);
    ASSERT_EQ(M(2, 0), 1);
}

TEST(Core, SwapRows_Same)
{
    matrix<double> M = {{1, 2}, {3, 4}};
    auto original = M;
    M.swap_rows(0, 0);
    ASSERT_TRUE(M == original);
}

TEST(Core, SwapColumns_Adjacent)
{
    matrix<double> M = {{1, 2, 3}, {4, 5, 6}};
    M.swap_cols(0, 1);
    ASSERT_EQ(M(0, 0), 2);
    ASSERT_EQ(M(0, 1), 1);
    ASSERT_EQ(M(1, 0), 5);
    ASSERT_EQ(M(1, 1), 4);
}

TEST(Core, SwapColumns_NonAdjacent)
{
    matrix<double> M = {{1, 2, 3}, {4, 5, 6}};
    M.swap_cols(0, 2);
    ASSERT_EQ(M(0, 0), 3);
    ASSERT_EQ(M(0, 2), 1);
}

TEST(Core, SwapColumns_Same)
{
    matrix<double> M = {{1, 2}, {3, 4}};
    auto original = M;
    M.swap_cols(1, 1);
    ASSERT_TRUE(M == original);
}

TEST(Core, Submatrix_TopLeft)
{
    matrix<double> M = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
    auto sub = M.submatrix(0, 0, 1, 1); // From (0,0) to (1,1)
    ASSERT_TRUE(sub.rows() == 2);
    ASSERT_TRUE(sub.cols() == 2);
    ASSERT_EQ(sub(0, 0), 1);
    ASSERT_EQ(sub(1, 1), 6);
}

TEST(Core, Submatrix_Center)
{
    matrix<double> M = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
    auto sub = M.submatrix(1, 1, 2, 2); // From (1,1) to (2,2)
    ASSERT_EQ(sub(0, 0), 6);
    ASSERT_EQ(sub(0, 1), 7);
    ASSERT_EQ(sub(1, 0), 10);
    ASSERT_EQ(sub(1, 1), 11);
}

TEST(Core, Submatrix_BottomRight)
{
    matrix<double> M = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
    auto sub = M.submatrix(2, 2, 3, 3); // From (2,2) to (3,3)
    ASSERT_EQ(sub(0, 0), 11);
    ASSERT_EQ(sub(1, 1), 16);
}

TEST(Core, Submatrix_SingleElement)
{
    matrix<double> M = {{1, 2}, {3, 4}};
    auto sub = M.submatrix(1, 1, 1, 1); // Single element at (1,1)
    ASSERT_TRUE(sub.rows() == 1);
    ASSERT_TRUE(sub.cols() == 1);
    ASSERT_EQ(sub(0, 0), 4);
}

TEST(Core, Submatrix_FullMatrix)
{
    matrix<double> M = {{1, 2}, {3, 4}};
    auto sub = M.submatrix(0, 0, 1, 1); // Full matrix from (0,0) to (1,1)
    ASSERT_TRUE(sub == M);
}

TEST(Core, SetSubmatrix_TopLeft)
{
    matrix<double> M = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    matrix<double> S = {{99, 88}, {77, 66}};
    M.set_submatrix(0, 0, S);
    ASSERT_EQ(M(0, 0), 99);
    ASSERT_EQ(M(0, 1), 88);
    ASSERT_EQ(M(1, 0), 77);
    ASSERT_EQ(M(1, 1), 66);
    ASSERT_EQ(M(2, 2), 9);
}

TEST(Core, SetSubmatrix_Center)
{
    matrix<double> M = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    matrix<double> S = {{99}};
    M.set_submatrix(1, 1, S);
    ASSERT_EQ(M(1, 1), 99);
    ASSERT_EQ(M(0, 0), 1);
}

TEST(Core, SetSubmatrix_BottomRight)
{
    matrix<double> M = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    matrix<double> S = {{99, 88}};
    M.set_submatrix(2, 1, S);
    ASSERT_EQ(M(2, 1), 99);
    ASSERT_EQ(M(2, 2), 88);
}

TEST(Core, Diagonal_Get_Square)
{
    matrix<double> M = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto d = M.diagonal();
    ASSERT_TRUE(d.size() == 3);
    ASSERT_EQ(d[0], 1);
    ASSERT_EQ(d[1], 5);
    ASSERT_EQ(d[2], 9);
}

TEST(Core, Diagonal_Get_Rectangular_Tall)
{
    matrix<double> M = {{1, 2}, {3, 4}, {5, 6}};
    bool caught = false;
    try
    {
        auto d = M.diagonal();
    }
    catch (const std::invalid_argument&)
    {
        caught = true;
    }
    ASSERT_TRUE(caught);
}

TEST(Core, Diagonal_Get_Rectangular_Wide)
{
    matrix<double> M = {{1, 2, 3}, {4, 5, 6}};
    bool caught = false;
    try
    {
        auto d = M.diagonal();
    }
    catch (const std::invalid_argument&)
    {
        caught = true;
    }
    ASSERT_TRUE(caught);
}

TEST(Core, Diagonal_Set_Square)
{
    matrix<double> M = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    M.set_diagonal({1, 2, 3});
    ASSERT_EQ(M(0, 0), 1);
    ASSERT_EQ(M(1, 1), 2);
    ASSERT_EQ(M(2, 2), 3);
    ASSERT_EQ(M(0, 1), 0);
}

TEST(Core, Diagonal_Set_Identity)
{
    matrix<double> M = {{0, 0}, {0, 0}};
    M.set_diagonal({1, 1});
    auto I = matrix<double>::eye(2, 2);
    ASSERT_TRUE(M == I);
}

TEST(Core, Diagonal_Set_OverwriteExisting)
{
    matrix<double> M = {{1, 2}, {3, 4}};
    M.set_diagonal({99, 88});
    ASSERT_EQ(M(0, 0), 99);
    ASSERT_EQ(M(1, 1), 88);
    ASSERT_EQ(M(0, 1), 2);
}

TEST(Core, Resize_Enlarge)
{
    matrix<double> M = {{1, 2}, {3, 4}};
    M.resize(3, 3);
    ASSERT_TRUE(M.rows() == 3);
    ASSERT_TRUE(M.cols() == 3);
    ASSERT_EQ(M(0, 0), 1);
    ASSERT_EQ(M(1, 1), 4);
}

TEST(Core, Resize_Shrink)
{
    matrix<double> M = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    M.resize(2, 2);
    ASSERT_TRUE(M.rows() == 2);
    ASSERT_TRUE(M.cols() == 2);
    ASSERT_EQ(M(0, 0), 1);
    ASSERT_EQ(M(1, 1), 5);
}

TEST(Core, Resize_SameSize)
{
    matrix<double> M = {{1, 2}, {3, 4}};
    auto original = M;
    M.resize(2, 2);
    ASSERT_TRUE(M == original);
}

TEST(Core, Resize_AddRows)
{
    matrix<double> M = {{1, 2}, {3, 4}};
    M.resize(3, 2);
    ASSERT_TRUE(M.rows() == 3);
    ASSERT_TRUE(M.cols() == 2);
}

TEST(Core, Resize_AddColumns)
{
    matrix<double> M = {{1, 2}, {3, 4}};
    M.resize(2, 3);
    ASSERT_TRUE(M.rows() == 2);
    ASSERT_TRUE(M.cols() == 3);
}

TEST(Core, Transpose_Square_2x2)
{
    matrix<double> M = {{1, 2}, {3, 4}};
    auto T = M.transpose();
    ASSERT_EQ(T(0, 0), 1);
    ASSERT_EQ(T(0, 1), 3);
    ASSERT_EQ(T(1, 0), 2);
    ASSERT_EQ(T(1, 1), 4);
}

TEST(Core, Transpose_Square_3x3)
{
    matrix<double> M = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto T = M.transpose();
    ASSERT_EQ(T(0, 0), 1);
    ASSERT_EQ(T(0, 2), 7);
    ASSERT_EQ(T(2, 0), 3);
}

TEST(Core, Transpose_Rectangular_Tall)
{
    matrix<double> M = {{1, 2}, {3, 4}, {5, 6}};
    auto T = M.transpose();
    ASSERT_TRUE(T.rows() == 2);
    ASSERT_TRUE(T.cols() == 3);
    ASSERT_EQ(T(0, 0), 1);
    ASSERT_EQ(T(1, 2), 6);
}

TEST(Core, Transpose_Rectangular_Wide)
{
    matrix<double> M = {{1, 2, 3}, {4, 5, 6}};
    auto T = M.transpose();
    ASSERT_TRUE(T.rows() == 3);
    ASSERT_TRUE(T.cols() == 2);
    ASSERT_EQ(T(0, 0), 1);
    ASSERT_EQ(T(2, 1), 6);
}

TEST(Core, Transpose_SingleRow)
{
    matrix<double> M = {{1, 2, 3, 4}};
    auto T = M.transpose();
    ASSERT_TRUE(T.rows() == 4);
    ASSERT_TRUE(T.cols() == 1);
    ASSERT_EQ(T(0, 0), 1);
    ASSERT_EQ(T(3, 0), 4);
}

TEST(Core, Transpose_SingleColumn)
{
    matrix<double> M = {{1}, {2}, {3}, {4}};
    auto T = M.transpose();
    ASSERT_TRUE(T.rows() == 1);
    ASSERT_TRUE(T.cols() == 4);
    ASSERT_EQ(T(0, 0), 1);
    ASSERT_EQ(T(0, 3), 4);
}

TEST(Core, Transpose_DoubleTranspose)
{
    matrix<double> M = {{1, 2, 3}, {4, 5, 6}};
    auto T = M.transpose().transpose();
    ASSERT_TRUE(M == T);
}

TEST(Core, Transpose_Symmetric)
{
    matrix<double> M = {{1, 2, 3}, {2, 4, 5}, {3, 5, 6}};
    auto T = M.transpose();
    ASSERT_TRUE(M == T);
}

TEST(Core, Transpose_Identity)
{
    auto I = matrix<double>::eye(3, 3);
    auto T = I.transpose();
    ASSERT_TRUE(I == T);
}

TEST(Core, Iterator_RangeLoop)
{
    matrix<double> M = {{1, 2}, {3, 4}};
    double sum = 0;
    for (const auto& val : M)
    {
        sum += val;
    }
    ASSERT_EQ(sum, 10.0);
}

TEST(Core, Iterator_Count)
{
    matrix<double> M = {{1, 2, 3}, {4, 5, 6}};
    size_t count = 0;
    for (const auto& _ : M)
    {
        count++;
    }
    ASSERT_TRUE(count == 6);
}

TEST(Core, Iterator_Modification)
{
    matrix<double> M = {{1, 2}, {3, 4}};
    for (auto& elem : M)
    {
        elem *= 2;
    }
    ASSERT_EQ(M(0, 0), 2);
    ASSERT_EQ(M(1, 1), 8);
}

TEST(Core, Cast_DoubleToInt)
{
    matrix<double> D = {{1.7, 2.3}, {3.9, 4.1}};
    matrix<int> I = static_cast<matrix<int>>(D);
    ASSERT_TRUE(I(0, 0) == 1);
    ASSERT_TRUE(I(0, 1) == 2);
    ASSERT_TRUE(I(1, 0) == 3);
    ASSERT_TRUE(I(1, 1) == 4);
}

TEST(Core, Cast_IntToDouble)
{
    matrix<int> I = {{1, 2}, {3, 4}};
    matrix<double> D = static_cast<matrix<double>>(I);
    ASSERT_EQ(D(0, 0), 1.0);
    ASSERT_EQ(D(1, 1), 4.0);
}

TEST(Core, Cast_FloatToDouble)
{
    matrix<float> F = {{1.5f, 2.5f}, {3.5f, 4.5f}};
    matrix<double> D = static_cast<matrix<double>>(F);
    ASSERT_TRUE(abs(D(0, 0) - 1.5) < 1e-6);
    ASSERT_TRUE(abs(D(1, 1) - 4.5) < 1e-6);
}

TEST(Core, Cast_DoubleToFloat)
{
    matrix<double> D = {{1.5, 2.5}, {3.5, 4.5}};
    matrix<float> F = static_cast<matrix<float>>(D);
    ASSERT_TRUE(abs(F(0, 0) - 1.5f) < 1e-6);
    ASSERT_TRUE(abs(F(1, 1) - 4.5f) < 1e-6);
}

TEST(Core, Size_Square)
{
    matrix<double> M = {{1, 2}, {3, 4}};
    ASSERT_TRUE(M.size() == 4);
    ASSERT_TRUE(M.rows() == 2);
    ASSERT_TRUE(M.cols() == 2);
}

TEST(Core, Size_Rectangular)
{
    matrix<double> M = {{1, 2, 3}, {4, 5, 6}};
    ASSERT_TRUE(M.size() == 6);
    ASSERT_TRUE(M.rows() == 2);
    ASSERT_TRUE(M.cols() == 3);
}

TEST(Core, Size_SingleElement)
{
    matrix<double> M = {{42}};
    ASSERT_TRUE(M.size() == 1);
    ASSERT_TRUE(M.rows() == 1);
    ASSERT_TRUE(M.cols() == 1);
}

TEST(Core, IsSquare_True)
{
    matrix<double> M = {{1, 2}, {3, 4}};
    ASSERT_TRUE(M.is_square());
}

TEST(Core, IsSquare_False_Tall)
{
    matrix<double> M = {{1, 2}, {3, 4}, {5, 6}};
    ASSERT_TRUE(!M.is_square());
}

TEST(Core, IsSquare_False_Wide)
{
    matrix<double> M = {{1, 2, 3}, {4, 5, 6}};
    ASSERT_TRUE(!M.is_square());
}
