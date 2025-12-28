#include "../include/matrix.hpp"
#include "test_framework.hpp"
#include <vector>

using namespace std;

TEST(LinAlg, Transpose) {
    matrix<double> A = {{1, 2, 3}, {4, 5, 6}};
    auto B = A.transpose();
    ASSERT_TRUE(B.rows() == 3);
    ASSERT_TRUE(B.cols() == 2);
    ASSERT_EQ(B(0, 0), 1);
    ASSERT_EQ(B(2, 1), 6);
}

TEST(LinAlg, Trace) {
    matrix<double> A = {{1, 2}, {3, 4}};
    ASSERT_EQ(A.trace(), 5);
}

TEST(LinAlg, Determinant_2x2) {
    matrix<double> A = {{3, 8}, {4, 6}};
    ASSERT_EQ(A.determinant(), -14);
}

TEST(LinAlg, Determinant_3x3) {
    matrix<double> A = {{6, 1, 1}, {4, -2, 5}, {2, 8, 7}};
    ASSERT_EQ(A.determinant(), -306);
}

TEST(LinAlg, Inverse_2x2) {
    matrix<double> A = {{4, 7}, {2, 6}};
    auto B = A.inverse();
    auto I = A * B;
    ASSERT_EQ(I(0, 0), 1);
    ASSERT_EQ(I(1, 1), 1);
    ASSERT_TRUE(abs(I(0, 1)) < 1e-9);
}

TEST(LinAlg, Rank) {
    matrix<double> A = {{1, 2, 3}, {2, 4, 6}, {3, 6, 9}};
    ASSERT_TRUE(A.rank() == 1);
}

TEST(LinAlg, Solve_System) {
    matrix<double> A = {{3, 2}, {1, 2}};
    matrix<double> b = {{5}, {4}};
    auto x = A.solve(b);
    ASSERT_TRUE(abs(x(0, 0) - 0.5) < 1e-9);
    ASSERT_TRUE(abs(x(1, 0) - 1.75) < 1e-9);
}

TEST(LinAlg, Apply_Function) {
    matrix<double> A = {{-1, -2}, {3, -4}};
    auto B = A.apply(static_cast<double(*)(double)>(std::abs));
    ASSERT_EQ(B(0, 0), 1);
    ASSERT_EQ(B(1, 1), 4);
}

TEST(LinAlg, Clamp) {
    matrix<double> A = {{-5, 0.5}, {3, 10}};
    auto B = A.clamp(0.0, 5.0);
    ASSERT_EQ(B(0, 0), 0.0);
    ASSERT_EQ(B(1, 1), 5.0);
}

// ========================================================================
// Advanced Operations
// ========================================================================

TEST(LinAlg, Minor) {
    matrix<double> M = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto min = M.minor(1, 1);
    ASSERT_EQ(min(0, 0), 1);
    ASSERT_EQ(min(1, 1), 9);
}

TEST(LinAlg, Adjoint) {
    matrix<double> A = {{1, 2}, {3, 4}};
    auto adj = A.adjoint();
    ASSERT_EQ(adj(0, 0), 4);
    ASSERT_EQ(adj(0, 1), -2);
}

TEST(LinAlg, Norm_L2) {
    matrix<double> A = {{3, 4}};
    ASSERT_EQ(A.norm(2), 5.0);
}

TEST(LinAlg, Power) {
    matrix<double> A = {{1, 2}, {0, 1}};
    auto P = A.pow(2);
    ASSERT_EQ(P(0, 1), 4);
}

// ========================================================================
// Decompositions
// ========================================================================

TEST(LinAlg, LU_Decomposition) {
    matrix<double> A = {{4, 3}, {6, 3}};
    auto [L, U] = A.LU_decomposition();
    auto P = L * U;
    ASSERT_TRUE(abs(P(0, 0) - A(0, 0)) < 1e-6);
}

TEST(LinAlg, QR_Decomposition) {
    matrix<double> A = {{12, -51, 4}, {6, 167, -68}, {-4, 24, -41}};
    auto [Q, R] = A.QR_decomposition();
    ASSERT_TRUE(Q.rows() == 3);
    ASSERT_TRUE(R.rows() == 3);
}

TEST(LinAlg, Eigenvalues) {
    matrix<double> A = {{2, -4}, {-1, -1}};
    auto ev = A.eigenvalues(50);
    ASSERT_TRUE(ev.size() == 2);
    ASSERT_TRUE(abs(ev(0, 0) - 3.0) < 1e-6 || abs(ev(1, 0) - 3.0) < 1e-6);
}

TEST(LinAlg, SVD) {
    matrix<double> A = {{3, 2, 2}, {2, 3, -2}};
    auto [U, S, V] = A.SVD();
    auto R = U * S * V;
    ASSERT_TRUE(abs(R(0, 0) - A(0, 0)) < 1e-6);
}