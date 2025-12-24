#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "../include/matrix.hpp"

using namespace std;

// Test counter
int tests_passed = 0;
int tests_failed = 0;

#define TEST(name)                        \
    cout << "Testing " << name << "... "; \
    try                                   \
    {
#define END_TEST                              \
    tests_passed++;                           \
    cout << "PASS" << endl;                   \
    }                                         \
    catch (const exception& e)                \
    {                                         \
        tests_failed++;                       \
        cout << "FAIL: " << e.what() << endl; \
    }

#define ASSERT_EQ(a, b)                                                               \
    if (abs((a) - (b)) > 1e-9)                                                        \
    {                                                                                 \
        throw runtime_error("Expected " + to_string(a) + " but got " + to_string(b)); \
    }

#define ASSERT_TRUE(expr)                                 \
    if (!(expr))                                          \
    {                                                     \
        throw runtime_error("Expression failed: " #expr); \
    }

int main()
{
    // ========================================================================
    // Construction Tests
    // ========================================================================
    cout << "=== Construction Tests ===" << endl;

    TEST("Matrix construction")
    matrix<double> A(3, 3);
    ASSERT_TRUE(A.rows() == 3);
    ASSERT_TRUE(A.cols() == 3);
    END_TEST

    TEST("Matrix from initializer list")
    matrix<double> B = {{1, 2}, {3, 4}};
    ASSERT_EQ(B(0, 0), 1);
    ASSERT_EQ(B(0, 1), 2);
    ASSERT_EQ(B(1, 0), 3);
    ASSERT_EQ(B(1, 1), 4);
    END_TEST

    TEST("Zero matrix")
    auto Z = matrix<double>::zeros(3, 3);
    for (size_t i = 0; i < 3; i++)
        for (size_t j = 0; j < 3; j++)
            ASSERT_EQ(Z(i, j), 0);
    END_TEST

    TEST("Ones matrix")
    auto O = matrix<double>::ones(2, 2);
    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 2; j++)
            ASSERT_EQ(O(i, j), 1);
    END_TEST

    TEST("Identity matrix")
    auto I = matrix<double>::eye(3, 3);
    ASSERT_EQ(I(0, 0), 1);
    ASSERT_EQ(I(1, 1), 1);
    ASSERT_EQ(I(2, 2), 1);
    ASSERT_EQ(I(0, 1), 0);
    ASSERT_EQ(I(1, 0), 0);
    END_TEST

    TEST("Random matrix")
    auto R = matrix<double>::random(5, 5, 0.0, 10.0);
    ASSERT_TRUE(R.rows() == 5);
    ASSERT_TRUE(R.cols() == 5);
    END_TEST

    // ========================================================================
    // Accessor Tests
    // ========================================================================
    cout << "\n=== Accessor Tests ===" << endl;
    TEST("Element access and modification")
    matrix<double> A0 = {{1, 2}, {3, 4}};
    ASSERT_EQ(A0(0, 0), 1);
    ASSERT_EQ(A0(1, 1), 4);
    A0(0, 0) = 10;
    ASSERT_EQ(A0(0, 0), 10);
    END_TEST

    cout << "\n=== Row and Column Operations ===" << endl;
    TEST("Get row as vector")
    matrix<double> A_row = {{1, 2, 3}, {4, 5, 6}};
    auto row0 = A_row(0);
    ASSERT_TRUE(row0.size() == 3);
    ASSERT_EQ(row0[0], 1);
    ASSERT_EQ(row0[1], 2);
    ASSERT_EQ(row0[2], 3);
    END_TEST

    // ========================================================================
    // Arithmetic Tests
    // ========================================================================
    cout << "\n=== Arithmetic Operations ===" << endl;

    TEST("Matrix addition")
    matrix<double> A1 = {{1, 2}, {3, 4}};
    matrix<double> B1 = {{5, 6}, {7, 8}};
    auto C1 = A1 + B1;
    ASSERT_EQ(C1(0, 0), 6);
    ASSERT_EQ(C1(0, 1), 8);
    ASSERT_EQ(C1(1, 0), 10);
    ASSERT_EQ(C1(1, 1), 12);
    C1 = A1;
    C1 += B1;
    ASSERT_EQ(C1(0, 0), 6);
    ASSERT_EQ(C1(0, 1), 8);
    ASSERT_EQ(C1(1, 0), 10);
    ASSERT_EQ(C1(1, 1), 12);
    END_TEST

    TEST("Matrix subtraction")
    matrix<double> A2 = {{5, 6}, {7, 8}};
    matrix<double> B2 = {{1, 2}, {3, 4}};
    auto C2 = A2 - B2;
    ASSERT_EQ(C2(0, 0), 4);
    ASSERT_EQ(C2(0, 1), 4);
    ASSERT_EQ(C2(1, 0), 4);
    ASSERT_EQ(C2(1, 1), 4);
    C2 = A2;
    C2 -= B2;
    ASSERT_EQ(C2(0, 0), 4);
    ASSERT_EQ(C2(0, 1), 4);
    ASSERT_EQ(C2(1, 0), 4);
    ASSERT_EQ(C2(1, 1), 4);
    END_TEST

    TEST("Matrix multiplication")
    matrix<double> A3 = {{1, 2}, {3, 4}};
    matrix<double> B3 = {{2, 0}, {1, 2}};
    auto C3 = A3 * B3;
    ASSERT_EQ(C3(0, 0), 4);
    ASSERT_EQ(C3(0, 1), 4);
    ASSERT_EQ(C3(1, 0), 10);
    ASSERT_EQ(C3(1, 1), 8);
    C3 = A3;
    C3 *= B3;
    ASSERT_EQ(C3(0, 0), 4);
    ASSERT_EQ(C3(0, 1), 4);
    ASSERT_EQ(C3(1, 0), 10);
    ASSERT_EQ(C3(1, 1), 8);
    END_TEST

    TEST("Scalar multiplication")
    matrix<double> A4 = {{1, 2}, {3, 4}};
    auto B4 = A4 * 2.0;
    ASSERT_EQ(B4(0, 0), 2);
    ASSERT_EQ(B4(0, 1), 4);
    ASSERT_EQ(B4(1, 0), 6);
    ASSERT_EQ(B4(1, 1), 8);
    END_TEST

    TEST("Scalar division")
    matrix<double> A5 = {{2, 4}, {6, 8}};
    auto B5 = A5 / 2.0;
    ASSERT_EQ(B5(0, 0), 1);
    ASSERT_EQ(B5(0, 1), 2);
    ASSERT_EQ(B5(1, 0), 3);
    ASSERT_EQ(B5(1, 1), 4);
    END_TEST

    TEST("Hadamard product")
    matrix<double> A6 = {{1, 2}, {3, 4}};
    matrix<double> B6 = {{2, 3}, {4, 5}};
    auto C6 = A6.hadamard(B6);
    ASSERT_EQ(C6(0, 0), 2);
    ASSERT_EQ(C6(0, 1), 6);
    ASSERT_EQ(C6(1, 0), 12);
    ASSERT_EQ(C6(1, 1), 20);
    END_TEST

    // ========================================================================
    // Matrix Operations Tests
    // ========================================================================
    cout << "\n=== Matrix Operations ===" << endl;

    TEST("Transpose")
    matrix<double> A7 = {{1, 2, 3}, {4, 5, 6}};
    auto B7 = A7.transpose();
    ASSERT_TRUE(B7.rows() == 3);
    ASSERT_TRUE(B7.cols() == 2);
    ASSERT_EQ(B7(0, 0), 1);
    ASSERT_EQ(B7(1, 0), 2);
    ASSERT_EQ(B7(0, 1), 4);
    ASSERT_EQ(B7(2, 1), 6);
    END_TEST

    TEST("Trace")
    matrix<double> A8 = {{1, 2}, {3, 4}};
    double trace = A8.trace();
    ASSERT_EQ(trace, 5);
    END_TEST

    TEST("Determinant 2x2")
    matrix<double> A9 = {{3, 8}, {4, 6}};
    double det = A9.determinant();
    ASSERT_EQ(det, -14);
    END_TEST

    TEST("Determinant 3x3")
    matrix<double> A10 = {{6, 1, 1}, {4, -2, 5}, {2, 8, 7}};
    double det = A10.determinant();
    ASSERT_EQ(det, -306);
    END_TEST

    TEST("Inverse 2x2")
    matrix<double> A11 = {{4, 7}, {2, 6}};
    auto B11 = A11.inverse();
    auto I11 = A11 * B11;
    ASSERT_EQ(I11(0, 0), 1);
    ASSERT_EQ(I11(1, 1), 1);
    ASSERT_TRUE(abs(I11(0, 1)) < 1e-9);
    ASSERT_TRUE(abs(I11(1, 0)) < 1e-9);
    END_TEST

    TEST("Rank")
    matrix<double> A12 = {{1, 2, 3}, {2, 4, 6}, {3, 6, 9}};
    size_t rank_val = A12.rank();
    ASSERT_TRUE(rank_val == 1);
    END_TEST

    TEST("Solve linear system")
    matrix<double> A21 = {{3, 2}, {1, 2}};
    matrix<double> b21 = {{5}, {4}};
    auto x21 = A21.solve(b21);
    ASSERT_TRUE(abs(x21(0, 0) - 0.5) < 1e-9);
    ASSERT_TRUE(abs(x21(1, 0) - 1.75) < 1e-9);
    END_TEST

    TEST("Apply function element-wise")
    matrix<double> A22 = {{-1.0, -2.0}, {3.0, -4.0}};
    auto B22 = A22.apply(static_cast<double(*)(double)>(std::abs));
    ASSERT_EQ(B22(0, 0), 1.0);
    ASSERT_EQ(B22(0, 1), 2.0);
    ASSERT_EQ(B22(1, 0), 3.0);
    ASSERT_EQ(B22(1, 1), 4.0);
    END_TEST

    TEST("Clamp elements")
    matrix<double> A23 = {{-5.0, 0.5}, {3.0, 10.0}};
    auto B23 = A23.clamp(0.0, 5.0);
    ASSERT_EQ(B23(0, 0), 0.0);
    ASSERT_EQ(B23(0, 1), 0.5);
    ASSERT_EQ(B23(1, 0), 3.0);
    ASSERT_EQ(B23(1, 1), 5.0);
    END_TEST

    // ========================================================================
    // Property Tests
    // ========================================================================
    cout << "\n=== Property Tests ===" << endl;

    TEST("is_square")
    matrix<double> A13 = {{1, 2}, {3, 4}};
    matrix<double> B13 = {{1, 2, 3}, {4, 5, 6}};
    ASSERT_TRUE(A13.is_square());
    ASSERT_TRUE(!B13.is_square());
    END_TEST

    TEST("is_symmetric")
    matrix<double> A14 = {{1, 2, 3}, {2, 4, 5}, {3, 5, 6}};
    matrix<double> B14 = {{1, 2}, {3, 4}};
    ASSERT_TRUE(A14.is_symmetric());
    ASSERT_TRUE(!B14.is_symmetric());
    END_TEST

    TEST("is_diagonal")
    auto I = matrix<double>::eye(3, 3);
    matrix<double> B15 = {{1, 2}, {3, 4}};
    ASSERT_TRUE(I.is_diagonal());
    ASSERT_TRUE(!B15.is_diagonal());
    END_TEST

    // ========================================================================
    // Type Tests
    // ========================================================================
    cout << "\n=== Type Tests ===" << endl;

    TEST("Integer matrices")
    matrix<int> A16 = {{1, 2}, {3, 4}};
    matrix<int> B16 = {{5, 6}, {7, 8}};
    auto C16 = A16 + B16;
    ASSERT_TRUE(C16(0, 0) == 6);
    ASSERT_TRUE(C16(1, 1) == 12);
    END_TEST

    TEST("Float matrices")
    matrix<float> A17 = {{1.5f, 2.5f}, {3.5f, 4.5f}};
    matrix<float> B17 = {{0.5f, 0.5f}, {0.5f, 0.5f}};
    auto C17 = A17 + B17;
    ASSERT_TRUE(abs(C17(0, 0) - 2.0f) < 1e-6);
    ASSERT_TRUE(abs(C17(1, 1) - 5.0f) < 1e-6);
    END_TEST

    // ========================================================================
    // Large Matrix Tests (SIMD & Threading)
    // ========================================================================
    cout << "\n=== Performance Tests (SIMD & Threading) ===" << endl;

    TEST("Large matrix addition (1000x1000)")
    auto L1 = matrix<double>::random(1000, 1000, 0.0, 1.0);
    auto L2 = matrix<double>::random(1000, 1000, 0.0, 1.0);
    auto L3 = L1 + L2;
    ASSERT_TRUE(L3.rows() == 1000);
    ASSERT_TRUE(L3.cols() == 1000);
    END_TEST

    TEST("Large matrix multiplication (256x256)")
    auto M1 = matrix<double>::random(256, 256, 0.0, 1.0);
    auto M2 = matrix<double>::random(256, 256, 0.0, 1.0);
    auto M3 = M1 * M2;
    ASSERT_TRUE(M3.rows() == 256);
    ASSERT_TRUE(M3.cols() == 256);
    END_TEST

    TEST("Large matrix subtraction (1000x1000)")
    auto S1 = matrix<double>::random(1000, 1000, 0.0, 1.0);
    auto S2 = matrix<double>::random(1000, 1000, 0.0, 1.0);
    auto S3 = S1 - S2;
    ASSERT_TRUE(S3.rows() == 1000);
    ASSERT_TRUE(S3.cols() == 1000);
    END_TEST

    // ========================================================================
    // Edge Case Tests
    // ========================================================================
    cout << "\n=== Edge Case Tests ===" << endl;

    TEST("1x1 matrix operations")
    matrix<double> A18 = {{5}};
    matrix<double> B18 = {{3}};
    auto C18 = A18 + B18;
    ASSERT_EQ(C18(0, 0), 8);
    END_TEST

    TEST("Matrix equality")
    matrix<double> A19 = {{1, 2}, {3, 4}};
    matrix<double> B19 = {{1, 2}, {3, 4}};
    matrix<double> C19 = {{1, 2}, {3, 5}};
    ASSERT_TRUE(A19 == B19);
    ASSERT_TRUE(A19 != C19);
    END_TEST

    TEST("Assignment operators")
    matrix<double> A20 = {{1, 2}, {3, 4}};
    matrix<double> B20 = {{5, 6}, {7, 8}};
    A20 += B20;
    ASSERT_EQ(A20(0, 0), 6);
    ASSERT_EQ(A20(1, 1), 12);
    END_TEST

    // ========================================================================
    // Scalar Operations Tests
    // ========================================================================
    cout << "\n=== Scalar Operations ===" << endl;

    TEST("Scalar addition")
    matrix<double> S1 = {{1, 2}, {3, 4}};
    auto S2 = S1 + 5.0;
    ASSERT_EQ(S2(0, 0), 6);
    ASSERT_EQ(S2(1, 1), 9);
    END_TEST

    TEST("Scalar subtraction")
    matrix<double> S3 = {{10, 20}, {30, 40}};
    auto S4 = S3 - 5.0;
    ASSERT_EQ(S4(0, 0), 5);
    ASSERT_EQ(S4(1, 1), 35);
    END_TEST

    TEST("Scalar addition assignment")
    matrix<double> S5 = {{1, 2}, {3, 4}};
    S5 += 10.0;
    ASSERT_EQ(S5(0, 0), 11);
    ASSERT_EQ(S5(1, 1), 14);
    END_TEST

    TEST("Scalar subtraction assignment")
    matrix<double> S6 = {{10, 20}, {30, 40}};
    S6 -= 5.0;
    ASSERT_EQ(S6(0, 0), 5);
    ASSERT_EQ(S6(1, 1), 35);
    END_TEST

    TEST("Scalar multiplication assignment")
    matrix<double> S7 = {{1, 2}, {3, 4}};
    S7 *= 3.0;
    ASSERT_EQ(S7(0, 0), 3);
    ASSERT_EQ(S7(1, 1), 12);
    END_TEST

    TEST("Scalar division assignment")
    matrix<double> S8 = {{10, 20}, {30, 40}};
    S8 /= 10.0;
    ASSERT_EQ(S8(0, 0), 1);
    ASSERT_EQ(S8(1, 1), 4);
    END_TEST

    // ========================================================================
    // Row/Column Extraction Tests
    // ========================================================================
    cout << "\n=== Row/Column Extraction ===" << endl;

    TEST("Row extraction")
    matrix<double> RC1 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto row1 = RC1.row(1);
    ASSERT_TRUE(row1.rows() == 1);
    ASSERT_TRUE(row1.cols() == 3);
    ASSERT_EQ(row1(0, 0), 4);
    ASSERT_EQ(row1(0, 1), 5);
    ASSERT_EQ(row1(0, 2), 6);
    END_TEST

    TEST("Column extraction")
    matrix<double> RC2 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto col1 = RC2.col(1);
    ASSERT_TRUE(col1.rows() == 3);
    ASSERT_TRUE(col1.cols() == 1);
    ASSERT_EQ(col1(0, 0), 2);
    ASSERT_EQ(col1(1, 0), 5);
    ASSERT_EQ(col1(2, 0), 8);
    END_TEST

    // ========================================================================
    // Swap Operations Tests
    // ========================================================================
    cout << "\n=== Swap Operations ===" << endl;

    TEST("Swap rows")
    matrix<double> SW1 = {{1, 2}, {3, 4}, {5, 6}};
    SW1.swap_rows(0, 2);
    ASSERT_EQ(SW1(0, 0), 5);
    ASSERT_EQ(SW1(0, 1), 6);
    ASSERT_EQ(SW1(2, 0), 1);
    ASSERT_EQ(SW1(2, 1), 2);
    END_TEST

    TEST("Swap columns")
    matrix<double> SW2 = {{1, 2, 3}, {4, 5, 6}};
    SW2.swap_cols(0, 2);
    ASSERT_EQ(SW2(0, 0), 3);
    ASSERT_EQ(SW2(0, 2), 1);
    ASSERT_EQ(SW2(1, 0), 6);
    ASSERT_EQ(SW2(1, 2), 4);
    END_TEST

    // ========================================================================
    // Submatrix Operations Tests
    // ========================================================================
    cout << "\n=== Submatrix Operations ===" << endl;

    TEST("Extract submatrix")
    matrix<double> SUB1 = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
    auto sub = SUB1.submatrix(1, 1, 2, 2);
    ASSERT_TRUE(sub.rows() == 2);
    ASSERT_TRUE(sub.cols() == 2);
    ASSERT_EQ(sub(0, 0), 6);
    ASSERT_EQ(sub(0, 1), 7);
    ASSERT_EQ(sub(1, 0), 10);
    ASSERT_EQ(sub(1, 1), 11);
    END_TEST

    TEST("Set submatrix")
    matrix<double> SUB2 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    matrix<double> replacement = {{99, 88}, {77, 66}};
    SUB2.set_submatrix(0, 1, replacement);
    ASSERT_EQ(SUB2(0, 1), 99);
    ASSERT_EQ(SUB2(0, 2), 88);
    ASSERT_EQ(SUB2(1, 1), 77);
    ASSERT_EQ(SUB2(1, 2), 66);
    END_TEST

    // ========================================================================
    // Diagonal Operations Tests
    // ========================================================================
    cout << "\n=== Diagonal Operations ===" << endl;

    TEST("Get main diagonal")
    matrix<double> D1 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto diag = D1.diagonal();
    ASSERT_TRUE(diag.size() == 3);
    ASSERT_EQ(diag[0], 1);
    ASSERT_EQ(diag[1], 5);
    ASSERT_EQ(diag[2], 9);
    END_TEST

    TEST("Get anti-diagonal")
    matrix<double> D2 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto anti_diag = D2.anti_diagonal();
    ASSERT_TRUE(anti_diag.size() == 3);
    ASSERT_EQ(anti_diag[0], 3);
    ASSERT_EQ(anti_diag[1], 5);
    ASSERT_EQ(anti_diag[2], 7);
    END_TEST

    TEST("Set diagonal")
    matrix<double> D3 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    std::vector<double> new_diag = {10, 20, 30};
    D3.set_diagonal(new_diag);
    ASSERT_EQ(D3(0, 0), 10);
    ASSERT_EQ(D3(1, 1), 20);
    ASSERT_EQ(D3(2, 2), 30);
    END_TEST

    TEST("Set anti-diagonal")
    matrix<double> D4 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    std::vector<double> new_anti_diag = {10, 20, 30};
    D4.set_anti_diagonal(new_anti_diag);
    ASSERT_EQ(D4(0, 2), 10);
    ASSERT_EQ(D4(1, 1), 20);
    ASSERT_EQ(D4(2, 0), 30);
    END_TEST

    // ========================================================================
    // Additional Property Tests
    // ========================================================================
    cout << "\n=== Additional Property Tests ===" << endl;

    TEST("is_upper_triangular")
    matrix<double> UT1 = {{1, 2, 3}, {0, 4, 5}, {0, 0, 6}};
    matrix<double> UT2 = {{1, 2}, {3, 4}};
    ASSERT_TRUE(UT1.is_upper_triangular());
    ASSERT_TRUE(!UT2.is_upper_triangular());
    END_TEST

    TEST("is_lower_triangular")
    matrix<double> LT1 = {{1, 0, 0}, {2, 3, 0}, {4, 5, 6}};
    matrix<double> LT2 = {{1, 2}, {3, 4}};
    ASSERT_TRUE(LT1.is_lower_triangular());
    ASSERT_TRUE(!LT2.is_lower_triangular());
    END_TEST

    TEST("is_orthogonal")
    matrix<double> ORT1 = {{1, 0}, {0, 1}};
    matrix<double> ORT2 = {{1, 2}, {3, 4}};
    ASSERT_TRUE(ORT1.is_orthogonal());
    ASSERT_TRUE(!ORT2.is_orthogonal());
    END_TEST

    TEST("is_singular")
    matrix<double> SING1 = {{1, 2}, {2, 4}};
    matrix<double> SING2 = {{1, 2}, {3, 4}};
    ASSERT_TRUE(SING1.is_singular());
    ASSERT_TRUE(!SING2.is_singular());
    END_TEST

    TEST("is_nilpotent")
    matrix<double> NILP1 = {{0, 1}, {0, 0}};
    matrix<double> NILP2 = {{1, 2}, {3, 4}};
    ASSERT_TRUE(NILP1.is_nilpotent(2));
    ASSERT_TRUE(!NILP2.is_nilpotent(2));
    END_TEST

    TEST("is_idempotent")
    matrix<double> IDEMP1 = {{1, 0}, {0, 0}};
    matrix<double> IDEMP2 = {{1, 2}, {3, 4}};
    ASSERT_TRUE(IDEMP1.is_idempotent());
    ASSERT_TRUE(!IDEMP2.is_idempotent());
    END_TEST

    TEST("is_involutory")
    matrix<double> INV1 = {{1, 0}, {0, 1}};
    matrix<double> INV2 = {{1, 2}, {3, 4}};
    ASSERT_TRUE(INV1.is_involutory());
    ASSERT_TRUE(!INV2.is_involutory());
    END_TEST

    TEST("is_positive_definite")
    matrix<double> PD1 = {{2, -1}, {-1, 2}};
    matrix<double> PD2 = {{1, 2}, {2, 1}};
    ASSERT_TRUE(PD1.is_positive_definite());
    ASSERT_TRUE(!PD2.is_positive_definite());
    END_TEST

    // ========================================================================
    // Advanced Matrix Operations Tests
    // ========================================================================
    cout << "\n=== Advanced Matrix Operations ===" << endl;

    TEST("Minor matrix")
    matrix<double> MIN1 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto minor_mat = MIN1.minor(1, 1);
    ASSERT_TRUE(minor_mat.rows() == 2);
    ASSERT_TRUE(minor_mat.cols() == 2);
    ASSERT_EQ(minor_mat(0, 0), 1);
    ASSERT_EQ(minor_mat(0, 1), 3);
    ASSERT_EQ(minor_mat(1, 0), 7);
    ASSERT_EQ(minor_mat(1, 1), 9);
    END_TEST

    TEST("Cofactor matrix")
    matrix<double> COF1 = {{1, 2, 3}, {0, 4, 5}, {1, 0, 6}};
    auto cof = COF1.cofactor();
    ASSERT_TRUE(cof.rows() == 3);
    ASSERT_TRUE(cof.cols() == 3);
    END_TEST

    TEST("Adjoint matrix")
    matrix<double> ADJ1 = {{1, 2}, {3, 4}};
    auto adj = ADJ1.adjoint();
    ASSERT_EQ(adj(0, 0), 4);
    ASSERT_EQ(adj(0, 1), -2);
    ASSERT_EQ(adj(1, 0), -3);
    ASSERT_EQ(adj(1, 1), 1);
    END_TEST

    TEST("Gaussian elimination")
    matrix<double> GAUSS1 = {{2, 1, -1}, {-3, -1, 2}, {-2, 1, 2}};
    auto gauss = GAUSS1.gaussian_elimination();
    ASSERT_TRUE(gauss.rows() == 3);
    ASSERT_TRUE(gauss.cols() == 3);
    END_TEST

    TEST("Matrix norm (L2)")
    matrix<double> NORM1 = {{3, 4}};
    double norm = NORM1.norm(2);
    ASSERT_EQ(norm, 5.0);
    END_TEST

    TEST("Matrix norm (L1)")
    matrix<double> NORM2 = {{1, 2}, {3, 4}};
    double norm_l1 = NORM2.norm(1);
    ASSERT_EQ(norm_l1, 10.0);
    END_TEST

    TEST("Matrix power")
    matrix<double> POW1 = {{1, 2}, {0, 1}};
    auto pow2 = POW1.pow(2);
    ASSERT_EQ(pow2(0, 0), 1);
    ASSERT_EQ(pow2(0, 1), 4);
    ASSERT_EQ(pow2(1, 0), 0);
    ASSERT_EQ(pow2(1, 1), 1);
    END_TEST

    TEST("Matrix power (identity)")
    matrix<double> POW2 = {{2, 1}, {1, 2}};
    auto pow0 = POW2.pow(0);
    ASSERT_EQ(pow0(0, 0), 1);
    ASSERT_EQ(pow0(0, 1), 0);
    ASSERT_EQ(pow0(1, 0), 0);
    ASSERT_EQ(pow0(1, 1), 1);
    END_TEST

    // ========================================================================
    // Decomposition Tests
    // ========================================================================
    cout << "\n=== Matrix Decomposition ===" << endl;

    TEST("LU decomposition")
    matrix<double> LU1 = {{4, 3}, {6, 3}};
    auto [L, U] = LU1.LU_decomposition();
    auto product = L * U;
    ASSERT_TRUE(abs(product(0, 0) - LU1(0, 0)) < 1e-6);
    ASSERT_TRUE(abs(product(0, 1) - LU1(0, 1)) < 1e-6);
    ASSERT_TRUE(abs(product(1, 0) - LU1(1, 0)) < 1e-6);
    ASSERT_TRUE(abs(product(1, 1) - LU1(1, 1)) < 1e-6);
    END_TEST

    TEST("QR decomposition")
    matrix<double> QR1 = {{12, -51, 4}, {6, 167, -68}, {-4, 24, -41}};
    auto [Q, R] = QR1.QR_decomposition();
    ASSERT_TRUE(Q.rows() == 3);
    ASSERT_TRUE(R.rows() == 3);
    END_TEST

    TEST("Eigenvalues computation")
    matrix<double> EIG1 = {{2, -4}, {-1, -1}};
    auto eigenvals = EIG1.eigenvalues(50);
    ASSERT_TRUE(eigenvals.size() == 2);
    ASSERT_TRUE((abs(eigenvals(0, 0) - 3.0) < 1e-6));
    ASSERT_TRUE((abs(eigenvals(1, 0) + 2.0) < 1e-6));
    END_TEST

    TEST("Eigenvectors computation")
    matrix<double> EIG2 = {{2, 1}, {1, 2}};
    auto eigenvecs = EIG2.eigenvectors(50);
    ASSERT_TRUE(eigenvecs.rows() == 2);
    ASSERT_TRUE(eigenvecs.cols() == 2);
    END_TEST

    TEST("Singular Value Decomposition (SVD)")
    matrix<double> SVD1 = {{3, 2, 2}, {2, 3, -2}};
    auto [U_svd, S_svd, V_svd] = SVD1.SVD();
    matrix<double> recomposed = U_svd * S_svd * V_svd;
    for (size_t i = 0; i < S_svd.rows(); i++)
        for (size_t j = 0; j < S_svd.cols(); j++)
            ASSERT_TRUE(abs(recomposed(i, j) - SVD1(i, j)) < 1e-6);
    END_TEST

    // ========================================================================
    // Iterator Tests
    // ========================================================================
    cout << "\n=== Iterator Tests ===" << endl;

    TEST("Iterator begin/end")
    matrix<double> IT1 = {{1, 2, 3}, {4, 5, 6}};
    int count = 0;
    for (auto it = IT1.begin(); it != IT1.end(); ++it)
    {
        count++;
    }
    ASSERT_TRUE(count == 6);
    END_TEST

    TEST("Range-based for loop")
    matrix<double> IT2 = {{1, 2}, {3, 4}};
    double sum = 0;
    for (const auto& val : IT2)
    {
        sum += val;
    }
    ASSERT_EQ(sum, 10.0);
    END_TEST

    TEST("Const iterator")
    const matrix<double> IT3 = {{5, 10, 15}};
    double product = 1;
    for (auto it = IT3.cbegin(); it != IT3.cend(); ++it)
    {
        product *= (*it);
    }
    ASSERT_EQ(product, 750.0);
    END_TEST

    // ========================================================================
    // Resize Tests
    // ========================================================================
    cout << "\n=== Resize Tests ===" << endl;

    TEST("Resize matrix (enlarge)")
    matrix<double> RES1 = {{1, 2}, {3, 4}};
    RES1.resize(3, 3);
    ASSERT_TRUE(RES1.rows() == 3);
    ASSERT_TRUE(RES1.cols() == 3);
    ASSERT_EQ(RES1(0, 0), 1);
    ASSERT_EQ(RES1(1, 1), 4);
    END_TEST

    TEST("Resize matrix (shrink)")
    matrix<double> RES2 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    RES2.resize(2, 2);
    ASSERT_TRUE(RES2.rows() == 2);
    ASSERT_TRUE(RES2.cols() == 2);
    END_TEST

    // ========================================================================
    // Type Casting Tests
    // ========================================================================
    cout << "\n=== Type Casting Tests ===" << endl;

    TEST("Cast double to int")
    matrix<double> CAST1 = {{1.7, 2.3}, {3.9, 4.1}};
    matrix<int> cast_result = static_cast<matrix<int>>(CAST1);
    ASSERT_TRUE(cast_result(0, 0) == 1);
    ASSERT_TRUE(cast_result(0, 1) == 2);
    ASSERT_TRUE(cast_result(1, 0) == 3);
    ASSERT_TRUE(cast_result(1, 1) == 4);
    END_TEST

    TEST("Cast int to double")
    matrix<int> CAST2 = {{1, 2}, {3, 4}};
    matrix<double> cast_double = static_cast<matrix<double>>(CAST2);
    ASSERT_EQ(cast_double(0, 0), 1.0);
    ASSERT_EQ(cast_double(1, 1), 4.0);
    END_TEST

    // ========================================================================
    // Additional Edge Cases
    // ========================================================================
    cout << "\n=== Additional Edge Cases ===" << endl;

    TEST("Division by zero error")
    matrix<double> DIV1 = {{1, 2}, {3, 4}};
    bool caught = false;
    try
    {
        auto result = DIV1 / 0.0;
    }
    catch (const std::invalid_argument&)
    {
        caught = true;
    }
    ASSERT_TRUE(caught);
    END_TEST

    TEST("Incompatible dimensions for addition")
    matrix<double> DIM1 = {{1, 2}};
    matrix<double> DIM2 = {{1}, {2}};
    bool caught = false;
    try
    {
        auto result = DIM1 + DIM2;
    }
    catch (const std::invalid_argument&)
    {
        caught = true;
    }
    ASSERT_TRUE(caught);
    END_TEST

    TEST("Incompatible dimensions for multiplication")
    matrix<double> DIM3 = {{1, 2, 3}};
    matrix<double> DIM4 = {{1, 2}, {3, 4}};
    bool caught = false;
    try
    {
        auto result = DIM3 * DIM4;
    }
    catch (const std::invalid_argument&)
    {
        caught = true;
    }
    ASSERT_TRUE(caught);
    END_TEST

    TEST("Non-square matrix trace error")
    matrix<double> NSQ1 = {{1, 2, 3}, {4, 5, 6}};
    bool caught = false;
    try
    {
        auto _ = NSQ1.trace();
    }
    catch (const std::invalid_argument&)
    {
        caught = true;
    }
    ASSERT_TRUE(caught);
    END_TEST

    TEST("Non-square matrix determinant error")
    matrix<double> NSQ2 = {{1, 2, 3}, {4, 5, 6}};
    bool caught = false;
    try
    {
        auto _ = NSQ2.determinant();
    }
    catch (const std::invalid_argument&)
    {
        caught = true;
    }
    ASSERT_TRUE(caught);
    END_TEST

    TEST("Empty matrix size")
    matrix<double> EMPTY1;
    ASSERT_TRUE(EMPTY1.rows() == 0);
    ASSERT_TRUE(EMPTY1.cols() == 0);
    ASSERT_TRUE(EMPTY1.size() == 0);
    END_TEST

    // ========================================================================
    // Results
    // ========================================================================
    cout << "\n╔═══════════════════════════════════════════════════════╗" << endl;
    cout << "║                   Test Results                        ║" << endl;
    cout << "╚═══════════════════════════════════════════════════════╝" << endl;
    cout << "Total tests:  " << (tests_passed + tests_failed) << endl;
    cout << "Passed:       " << tests_passed << endl;
    cout << "Failed:       " << tests_failed << endl;
    cout << "Success rate: " << fixed << setprecision(1) << (100.0 * tests_passed / (tests_passed + tests_failed))
         << "%" << endl;

    if (tests_failed == 0)
    {
        cout << "\n[SUCCESS] All tests passed!" << endl;
        return 0;
    }
    else
    {
        cout << "\n[FAILURE] Some tests failed!" << endl;
        return 1;
    }
}
