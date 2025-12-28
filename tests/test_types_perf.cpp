#include "../include/matrix.hpp"
#include "test_framework.hpp"
#include <vector>

using namespace std;

// ========================================================================
// Type Tests
// ========================================================================

TEST(Types, IntMatrix) {
    matrix<int> A = {{1, 2}, {3, 4}};
    matrix<int> B = {{5, 6}, {7, 8}};
    auto C = A + B;
    ASSERT_EQ(C(0, 0), 6);
    ASSERT_EQ(C(1, 1), 12);
}

TEST(Types, FloatMatrix) {
    matrix<float> A = {{1.5f, 2.5f}, {3.5f, 4.5f}};
    matrix<float> B = {{0.5f, 0.5f}, {0.5f, 0.5f}};
    auto C = A + B;
    ASSERT_TRUE(abs(C(0, 0) - 2.0f) < 1e-6);
}

// ========================================================================
// Performance Tests
// ========================================================================

TEST(Performance, LargeMatrix_Add) {
    auto L1 = matrix<double>::random(100, 100, 0.0, 1.0);
    auto L2 = matrix<double>::random(100, 100, 0.0, 1.0);
    auto L3 = L1 + L2;
    ASSERT_TRUE(L3.rows() == 100);
}

TEST(Performance, LargeMatrix_Mult) {
    auto M1 = matrix<double>::random(50, 50, 0.0, 1.0);
    auto M2 = matrix<double>::random(50, 50, 0.0, 1.0);
    auto M3 = M1 * M2;
    ASSERT_TRUE(M3.rows() == 50);
}

// ========================================================================
// Edge Cases
// ========================================================================

TEST(EdgeCases, SmallMatrix_1x1) {
    matrix<double> A = {{5}};
    matrix<double> B = {{3}};
    auto C = A + B;
    ASSERT_EQ(C(0, 0), 8);
}

TEST(EdgeCases, Equality) {
    matrix<double> A = {{1, 2}, {3, 4}};
    matrix<double> B = {{1, 2}, {3, 4}};
    matrix<double> C = {{1, 2}, {3, 5}};
    ASSERT_TRUE(A == B);
    ASSERT_TRUE(A != C);
}

TEST(EdgeCases, EmptyMatrix) {
    matrix<double> E;
    ASSERT_TRUE(E.rows() == 0);
    ASSERT_TRUE(E.size() == 0);
}

TEST(EdgeCases, Incompatible_Add) {
    matrix<double> A = {{1, 2}};
    matrix<double> B = {{1}, {2}};
    bool caught = false;
    try {
        auto C = A + B;
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    ASSERT_TRUE(caught);
}

TEST(EdgeCases, DivisionByZero) {
    matrix<double> A = {{1, 2}};
    bool caught = false;
    try {
        auto B = A / 0.0;
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    ASSERT_TRUE(caught);
}
