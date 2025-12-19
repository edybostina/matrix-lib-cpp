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
    END_TEST

    TEST("Matrix subtraction")
    matrix<double> A2 = {{5, 6}, {7, 8}};
    matrix<double> B2 = {{1, 2}, {3, 4}};
    auto C2 = A2 - B2;
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