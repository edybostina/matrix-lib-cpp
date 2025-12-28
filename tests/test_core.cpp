#include "../include/matrix.hpp"
#include "test_framework.hpp"
#include <vector>

using namespace std;

// ========================================================================
// Construction
// ========================================================================

TEST(Core, Construction_Default) {
    matrix<double> A(3, 3);
    ASSERT_TRUE(A.rows() == 3);
    ASSERT_TRUE(A.cols() == 3);
}

TEST(Core, Construction_InitializerList) {
    matrix<double> B = {{1, 2}, {3, 4}};
    ASSERT_EQ(B(0, 0), 1);
    ASSERT_EQ(B(0, 1), 2);
    ASSERT_EQ(B(1, 0), 3);
    ASSERT_EQ(B(1, 1), 4);
}

TEST(Core, Construction_Zeros) {
    auto Z = matrix<double>::zeros(3, 3);
    for (size_t i = 0; i < 3; i++)
        for (size_t j = 0; j < 3; j++)
            ASSERT_EQ(Z(i, j), 0);
}

TEST(Core, Construction_Ones) {
    auto O = matrix<double>::ones(2, 2);
    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 2; j++)
            ASSERT_EQ(O(i, j), 1);
}

TEST(Core, Construction_Identity) {
    auto I = matrix<double>::eye(3, 3);
    ASSERT_EQ(I(0, 0), 1);
    ASSERT_EQ(I(1, 1), 1);
    ASSERT_EQ(I(2, 2), 1);
    ASSERT_EQ(I(0, 1), 0);
}

TEST(Core, Construction_Random) {
    auto R = matrix<double>::random(5, 5, 0.0, 10.0);
    ASSERT_TRUE(R.rows() == 5);
    ASSERT_TRUE(R.cols() == 5);
}

// ========================================================================
// Accessors & Modification
// ========================================================================

TEST(Core, Access_ReadWrite) {
    matrix<double> A = {{1, 2}, {3, 4}};
    ASSERT_EQ(A(0, 0), 1);
    A(0, 0) = 10;
    ASSERT_EQ(A(0, 0), 10);
}

TEST(Core, Access_GetRow) {
    matrix<double> A = {{1, 2, 3}, {4, 5, 6}};
    auto row0 = A(0);
    ASSERT_TRUE(row0.size() == 3);
    ASSERT_EQ(row0[0], 1);
    ASSERT_EQ(row0[2], 3);
}

TEST(Core, Extraction_Row) {
    matrix<double> M = {{1, 2, 3}, {4, 5, 6}};
    auto r1 = M.row(1);
    ASSERT_TRUE(r1.rows() == 1);
    ASSERT_TRUE(r1.cols() == 3);
    ASSERT_EQ(r1(0, 0), 4);
}

TEST(Core, Extraction_Col) {
    matrix<double> M = {{1, 2, 3}, {4, 5, 6}};
    auto c1 = M.col(1);
    ASSERT_TRUE(c1.rows() == 2);
    ASSERT_TRUE(c1.cols() == 1);
    ASSERT_EQ(c1(0, 0), 2);
    ASSERT_EQ(c1(1, 0), 5);
}

// ========================================================================
// Swap
// ========================================================================

TEST(Core, Swap_Rows) {
    matrix<double> M = {{1, 2}, {3, 4}, {5, 6}};
    M.swap_rows(0, 2);
    ASSERT_EQ(M(0, 0), 5);
    ASSERT_EQ(M(2, 0), 1);
}

TEST(Core, Swap_Cols) {
    matrix<double> M = {{1, 2, 3}, {4, 5, 6}};
    M.swap_cols(0, 2);
    ASSERT_EQ(M(0, 0), 3);
    ASSERT_EQ(M(0, 2), 1);
}

// ========================================================================
// Submatrix
// ========================================================================

TEST(Core, Submatrix_Extract) {
    matrix<double> M = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
    auto sub = M.submatrix(1, 1, 2, 2);
    ASSERT_EQ(sub(0, 0), 6);
    ASSERT_EQ(sub(1, 1), 11);
}

TEST(Core, Submatrix_Set) {
    matrix<double> M = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    matrix<double> rep = {{99, 88}, {77, 66}};
    M.set_submatrix(0, 1, rep);
    ASSERT_EQ(M(0, 1), 99);
    ASSERT_EQ(M(1, 2), 66);
}

// ========================================================================
// Diagonal
// ========================================================================

TEST(Core, Diagonal_Get) {
    matrix<double> M = {{1, 2}, {3, 4}};
    auto d = M.diagonal();
    ASSERT_EQ(d[0], 1);
    ASSERT_EQ(d[1], 4);
}

TEST(Core, Diagonal_Set) {
    matrix<double> M = {{0, 0}, {0, 0}};
    M.set_diagonal({1, 2});
    ASSERT_EQ(M(0, 0), 1);
    ASSERT_EQ(M(1, 1), 2);
}

// ========================================================================
// Iterators
// ========================================================================

TEST(Core, Iterator_RangeLoop) {
    matrix<double> M = {{1, 2}, {3, 4}};
    double sum = 0;
    for (const auto& val : M) {
        sum += val;
    }
    ASSERT_EQ(sum, 10.0);
}

// ========================================================================
// Resize
// ========================================================================

TEST(Core, Resize_Enlarge) {
    matrix<double> M = {{1, 2}, {3, 4}};
    M.resize(3, 3);
    ASSERT_TRUE(M.rows() == 3);
    ASSERT_TRUE(M.cols() == 3);
    ASSERT_EQ(M(0, 0), 1);
}

// ========================================================================
// Casting
// ========================================================================

TEST(Core, Casting_DoubleToInt) {
    matrix<double> D = {{1.7, 2.3}, {3.9, 4.1}};
    matrix<int> I = static_cast<matrix<int>>(D);
    ASSERT_TRUE(I(0, 0) == 1);
    ASSERT_TRUE(I(1, 1) == 4);
}