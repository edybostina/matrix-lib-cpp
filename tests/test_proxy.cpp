#include "../include/matrix.hpp"
#include "test_framework.hpp"
#include <vector>

using namespace std;

TEST(RowProxy, ElementAccess)
{
    matrix<int> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    // Read access
    ASSERT_EQ(A.row(0)[0], 1);
    ASSERT_EQ(A.row(0)[1], 2);
    ASSERT_EQ(A.row(1)[2], 6);

    // Write access
    A.row(0)[0] = 10;
    ASSERT_EQ(A(0, 0), 10);

    A.row(2)[1] = 99;
    ASSERT_EQ(A(2, 1), 99);
}

TEST(RowProxy, ConstAccess)
{
    const matrix<int> A = {{1, 2, 3}, {4, 5, 6}};

    ASSERT_EQ(A.row(0)[0], 1);
    ASSERT_EQ(A.row(1)[2], 6);
    ASSERT_EQ(A.row(0).size(), 3);
}

TEST(RowProxy, AssignFromVector)
{
    matrix<int> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    vector<int> new_row = {10, 20, 30};
    A.row(0) = new_row;

    ASSERT_EQ(A(0, 0), 10);
    ASSERT_EQ(A(0, 1), 20);
    ASSERT_EQ(A(0, 2), 30);
}

TEST(RowProxy, AssignFromAnotherRow)
{
    matrix<int> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    A.row(0) = A.row(2);

    ASSERT_EQ(A(0, 0), 7);
    ASSERT_EQ(A(0, 1), 8);
    ASSERT_EQ(A(0, 2), 9);
}

TEST(RowProxy, InPlaceAddition)
{
    matrix<int> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    A.row(0) += A.row(1);

    ASSERT_EQ(A(0, 0), 5); // 1 + 4
    ASSERT_EQ(A(0, 1), 7); // 2 + 5
    ASSERT_EQ(A(0, 2), 9); // 3 + 6
}

TEST(RowProxy, InPlaceSubtraction)
{
    matrix<int> A = {{10, 20, 30}, {4, 5, 6}, {7, 8, 9}};

    A.row(0) -= A.row(1);

    ASSERT_EQ(A(0, 0), 6);  // 10 - 4
    ASSERT_EQ(A(0, 1), 15); // 20 - 5
    ASSERT_EQ(A(0, 2), 24); // 30 - 6
}

TEST(RowProxy, ScalarMultiplication)
{
    matrix<int> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    A.row(1) *= 2;

    ASSERT_EQ(A(1, 0), 8);  // 4 * 2
    ASSERT_EQ(A(1, 1), 10); // 5 * 2
    ASSERT_EQ(A(1, 2), 12); // 6 * 2
}

TEST(RowProxy, ScalarDivision)
{
    matrix<int> A = {{10, 20, 30}, {4, 5, 6}, {7, 8, 9}};

    A.row(0) /= 10;

    ASSERT_EQ(A(0, 0), 1); // 10 / 10
    ASSERT_EQ(A(0, 1), 2); // 20 / 10
    ASSERT_EQ(A(0, 2), 3); // 30 / 10
}

TEST(RowProxy, ConvertToVector)
{
    matrix<int> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    vector<int> row = A.row(1);

    ASSERT_EQ(row.size(), 3);
    ASSERT_EQ(row[0], 4);
    ASSERT_EQ(row[1], 5);
    ASSERT_EQ(row[2], 6);
}

TEST(RowProxy, ConvertToMatrix)
{
    matrix<int> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    matrix<int> row_mat = A.row(2).to_matrix();

    ASSERT_EQ(row_mat.rows(), 1);
    ASSERT_EQ(row_mat.cols(), 3);
    ASSERT_EQ(row_mat(0, 0), 7);
    ASSERT_EQ(row_mat(0, 1), 8);
    ASSERT_EQ(row_mat(0, 2), 9);
}

TEST(RowProxy, RangeBasedForLoop)
{
    matrix<int> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    // Read
    int sum = 0;
    for (const auto& val : A.row(1))
    {
        sum += val;
    }
    ASSERT_EQ(sum, 15); // 4 + 5 + 6

    // Write
    for (auto& val : A.row(0))
    {
        val *= 2;
    }
    ASSERT_EQ(A(0, 0), 2);
    ASSERT_EQ(A(0, 1), 4);
    ASSERT_EQ(A(0, 2), 6);
}

TEST(ColProxy, ElementAccess)
{
    matrix<int> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    // Read access
    ASSERT_EQ(A.col(0)[0], 1);
    ASSERT_EQ(A.col(1)[1], 5);
    ASSERT_EQ(A.col(2)[2], 9);

    // Write access
    A.col(0)[0] = 10;
    ASSERT_EQ(A(0, 0), 10);

    A.col(1)[2] = 99;
    ASSERT_EQ(A(2, 1), 99);
}

TEST(ColProxy, ConstAccess)
{
    const matrix<int> A = {{1, 2, 3}, {4, 5, 6}};

    ASSERT_EQ(A.col(0)[0], 1);
    ASSERT_EQ(A.col(2)[1], 6);
    ASSERT_EQ(A.col(0).size(), 2);
}

TEST(ColProxy, AssignFromVector)
{
    matrix<int> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    vector<int> new_col = {10, 20, 30};
    A.col(1) = new_col;

    ASSERT_EQ(A(0, 1), 10);
    ASSERT_EQ(A(1, 1), 20);
    ASSERT_EQ(A(2, 1), 30);
}

TEST(ColProxy, AssignFromAnotherColumn)
{
    matrix<int> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    A.col(0) = A.col(2);

    ASSERT_EQ(A(0, 0), 3);
    ASSERT_EQ(A(1, 0), 6);
    ASSERT_EQ(A(2, 0), 9);
}

TEST(ColProxy, InPlaceAddition)
{
    matrix<int> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    A.col(0) += A.col(1);

    ASSERT_EQ(A(0, 0), 3);  // 1 + 2
    ASSERT_EQ(A(1, 0), 9);  // 4 + 5
    ASSERT_EQ(A(2, 0), 15); // 7 + 8
}

TEST(ColProxy, InPlaceSubtraction)
{
    matrix<int> A = {{10, 2, 3}, {20, 5, 6}, {30, 8, 9}};

    A.col(0) -= A.col(1);

    ASSERT_EQ(A(0, 0), 8);  // 10 - 2
    ASSERT_EQ(A(1, 0), 15); // 20 - 5
    ASSERT_EQ(A(2, 0), 22); // 30 - 8
}

TEST(ColProxy, ScalarMultiplication)
{
    matrix<int> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    A.col(2) *= 3;

    ASSERT_EQ(A(0, 2), 9);  // 3 * 3
    ASSERT_EQ(A(1, 2), 18); // 6 * 3
    ASSERT_EQ(A(2, 2), 27); // 9 * 3
}

TEST(ColProxy, ScalarDivision)
{
    matrix<int> A = {{10, 20, 30}, {40, 50, 60}, {70, 80, 90}};

    A.col(0) /= 10;

    ASSERT_EQ(A(0, 0), 1); // 10 / 10
    ASSERT_EQ(A(1, 0), 4); // 40 / 10
    ASSERT_EQ(A(2, 0), 7); // 70 / 10
}

TEST(ColProxy, ConvertToVector)
{
    matrix<int> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    vector<int> col = A.col(1);

    ASSERT_EQ(col.size(), 3);
    ASSERT_EQ(col[0], 2);
    ASSERT_EQ(col[1], 5);
    ASSERT_EQ(col[2], 8);
}

TEST(ColProxy, ConvertToMatrix)
{
    matrix<int> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    matrix<int> col_mat = A.col(0).to_matrix();

    ASSERT_EQ(col_mat.rows(), 3);
    ASSERT_EQ(col_mat.cols(), 1);
    ASSERT_EQ(col_mat(0, 0), 1);
    ASSERT_EQ(col_mat(1, 0), 4);
    ASSERT_EQ(col_mat(2, 0), 7);
}

TEST(ColProxy, RangeBasedForLoop)
{
    matrix<int> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    // Read
    int sum = 0;
    for (const auto& val : A.col(2))
    {
        sum += val;
    }
    ASSERT_EQ(sum, 18); // 3 + 6 + 9

    // Write
    for (auto& val : A.col(1))
    {
        val *= 2;
    }
    ASSERT_EQ(A(0, 1), 4);
    ASSERT_EQ(A(1, 1), 10);
    ASSERT_EQ(A(2, 1), 16);
}

TEST(RowColProxy, SwapRows)
{
    matrix<int> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    auto temp = vector<int>(A.row(0));
    A.row(0) = A.row(2);
    A.row(2) = temp;

    ASSERT_EQ(A(0, 0), 7);
    ASSERT_EQ(A(2, 0), 1);
}

TEST(RowColProxy, SwapColumns)
{
    matrix<int> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    auto temp = vector<int>(A.col(0));
    A.col(0) = A.col(2);
    A.col(2) = temp;

    ASSERT_EQ(A(0, 0), 3);
    ASSERT_EQ(A(0, 2), 1);
}

TEST(RowColProxy, ChainedOperations)
{
    matrix<double> A = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

    for (size_t i = 0; i < A.rows(); ++i)
    {
        double sum = 0.0;
        for (const auto& val : A.row(i))
        {
            sum += val;
        }
        A.row(i) /= sum;
    }

    double row_sum = 0.0;
    for (const auto& val : A.row(0))
    {
        row_sum += val;
    }
    ASSERT_TRUE(std::abs(row_sum - 1.0) < 1e-10);
}

TEST(RowColProxy, MatrixOperations)
{
    matrix<int> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    for (size_t i = 1; i < A.rows(); ++i)
    {
        A.row(i) += A.row(0);
    }

    ASSERT_EQ(A(1, 0), 5); // 4 + 1
    ASSERT_EQ(A(1, 1), 7); // 5 + 2
    ASSERT_EQ(A(2, 0), 8); // 7 + 1
}

TEST(RowColProxy, FloatingPointOperations)
{
    matrix<double> A = {{1.5, 2.5, 3.5}, {4.5, 5.5, 6.5}};

    A.row(0) *= 2.0;
    ASSERT_TRUE(std::abs(A(0, 0) - 3.0) < 1e-10);
    ASSERT_TRUE(std::abs(A(0, 1) - 5.0) < 1e-10);

    A.col(1) /= 2.5;
    ASSERT_TRUE(std::abs(A(0, 1) - 2.0) < 1e-10);
    ASSERT_TRUE(std::abs(A(1, 1) - 2.2) < 1e-10);
}

TEST(RowColProxy, AddVectorToRow)
{
    matrix<int> A = {{1, 2, 3}, {4, 5, 6}};

    vector<int> vec = {10, 20, 30};
    A.row(0) += vec;

    ASSERT_EQ(A(0, 0), 11);
    ASSERT_EQ(A(0, 1), 22);
    ASSERT_EQ(A(0, 2), 33);
}

TEST(RowColProxy, SubtractVectorFromColumn)
{
    matrix<int> A = {{10, 20}, {30, 40}, {50, 60}};

    vector<int> vec = {5, 10, 15};
    A.col(0) -= vec;

    ASSERT_EQ(A(0, 0), 5);
    ASSERT_EQ(A(1, 0), 20);
    ASSERT_EQ(A(2, 0), 35);
}

TEST(RowProxy, Fill)
{
    matrix<int> A = {{1, 2, 3}, {4, 5, 6}};
    A.row(0).fill(99);

    ASSERT_EQ(A(0, 0), 99);
    ASSERT_EQ(A(0, 1), 99);
    ASSERT_EQ(A(0, 2), 99);
    ASSERT_EQ(A(1, 0), 4); // Other row unchanged
}

TEST(RowProxy, Swap)
{
    matrix<int> A = {{1, 2, 3}, {4, 5, 6}};
    auto r0 = A.row(0);
    auto r1 = A.row(1);
    r0.swap(r1);

    ASSERT_EQ(A(0, 0), 4);
    ASSERT_EQ(A(0, 1), 5);
    ASSERT_EQ(A(0, 2), 6);
    ASSERT_EQ(A(1, 0), 1);
    ASSERT_EQ(A(1, 1), 2);
    ASSERT_EQ(A(1, 2), 3);
}

TEST(RowProxy, Sum)
{
    matrix<int> A = {{1, 2, 3}, {4, 5, 6}};
    ASSERT_EQ(A.row(0).sum(), 6);
    ASSERT_EQ(A.row(1).sum(), 15);

    matrix<double> B = {{1.5, 2.5, 3.0}};
    ASSERT_TRUE(std::abs(B.row(0).sum() - 7.0) < 1e-10);
}

TEST(RowProxy, Product)
{
    matrix<int> A = {{1, 2, 3}, {2, 3, 4}};
    ASSERT_EQ(A.row(0).product(), 6);
    ASSERT_EQ(A.row(1).product(), 24);

    matrix<double> B = {{2.0, 0.5, 4.0}};
    ASSERT_TRUE(std::abs(B.row(0).product() - 4.0) < 1e-10);
}

TEST(RowProxy, MinMax)
{
    matrix<int> A = {{3, 1, 5, 2}, {-5, 10, 0, 3}};
    ASSERT_EQ(A.row(0).min(), 1);
    ASSERT_EQ(A.row(0).max(), 5);
    ASSERT_EQ(A.row(1).min(), -5);
    ASSERT_EQ(A.row(1).max(), 10);
}

TEST(RowProxy, DotProduct)
{
    matrix<int> A = {{1, 2, 3}, {4, 5, 6}};
    ASSERT_EQ(A.row(0).dot(A.row(1)), 32); // 1*4 + 2*5 + 3*6 = 32

    vector<int> vec = {2, 2, 2};
    ASSERT_EQ(A.row(0).dot(vec), 12); // 1*2 + 2*2 + 3*2 = 12
}

TEST(ColProxy, Fill)
{
    matrix<int> A = {{1, 2, 3}, {4, 5, 6}};
    A.col(1).fill(77);

    ASSERT_EQ(A(0, 1), 77);
    ASSERT_EQ(A(1, 1), 77);
    ASSERT_EQ(A(0, 0), 1); // Other column unchanged
}

TEST(ColProxy, Swap)
{
    matrix<int> A = {{1, 2, 3}, {4, 5, 6}};
    auto c0 = A.col(0);
    auto c2 = A.col(2);
    c0.swap(c2);

    ASSERT_EQ(A(0, 0), 3);
    ASSERT_EQ(A(1, 0), 6);
    ASSERT_EQ(A(0, 2), 1);
    ASSERT_EQ(A(1, 2), 4);
}

TEST(ColProxy, Sum)
{
    matrix<int> A = {{1, 2}, {3, 4}, {5, 6}};
    ASSERT_EQ(A.col(0).sum(), 9);  // 1 + 3 + 5
    ASSERT_EQ(A.col(1).sum(), 12); // 2 + 4 + 6
}

TEST(ColProxy, Product)
{
    matrix<int> A = {{1, 2}, {3, 4}, {5, 6}};
    ASSERT_EQ(A.col(0).product(), 15); // 1 * 3 * 5
    ASSERT_EQ(A.col(1).product(), 48); // 2 * 4 * 6
}

TEST(ColProxy, MinMax)
{
    matrix<int> A = {{10, -5}, {3, 20}, {7, 0}};
    ASSERT_EQ(A.col(0).min(), 3);
    ASSERT_EQ(A.col(0).max(), 10);
    ASSERT_EQ(A.col(1).min(), -5);
    ASSERT_EQ(A.col(1).max(), 20);
}

TEST(ColProxy, DotProduct)
{
    matrix<int> A = {{1, 2}, {3, 4}, {5, 6}};
    ASSERT_EQ(A.col(0).dot(A.col(1)), 44); // 1*2 + 3*4 + 5*6 = 44

    vector<int> vec = {2, 2, 2};
    ASSERT_EQ(A.col(0).dot(vec), 18); // 1*2 + 3*2 + 5*2 = 18
}
