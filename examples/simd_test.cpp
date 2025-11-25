#include <iostream>
#include "../include/matrix.hpp"
#include "../include/matrix_operators_simd.hpp"

using namespace std;

int main()
{
    cout << "Testing SIMD Integration\n";
    cout << "========================\n\n";

#if HAS_XSIMD
    cout << "✓ XSIMD support: ENABLED\n\n";
#else
    cout << "✗ XSIMD support: DISABLED\n\n";
#endif

    // Create test matrices
    matrix<double> A = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}};

    matrix<double> B = {
        {9.0, 8.0, 7.0},
        {6.0, 5.0, 4.0},
        {3.0, 2.0, 1.0}};

    cout << "Matrix A:\n"
         << A << "\n";
    cout << "Matrix B:\n"
         << B << "\n";

    // Test SIMD addition
    cout << "A + B (SIMD):\n";
    matrix<double> C = matrix_add_simd(A, B);
    cout << C << "\n";

    // Test SIMD subtraction
    cout << "A - B (SIMD):\n";
    matrix<double> D = matrix_sub_simd(A, B);
    cout << D << "\n";

    // Test SIMD Hadamard product
    cout << "A ⊙ B (Hadamard, SIMD):\n";
    matrix<double> E = matrix_hadamard_simd(A, B);
    cout << E << "\n";

    // Test SIMD scalar multiplication
    cout << "A * 2.5 (SIMD):\n";
    matrix<double> F = matrix_mul_scalar_simd(A, 2.5);
    cout << F << "\n";

    // Test SIMD matrix multiplication
    cout << "A * B (SIMD):\n";
    matrix<double> G = matrix_mul_simd(A, B);
    cout << G << "\n";

    // Verify correctness with regular operations
    cout << "Verification:\n";
    cout << "Regular A + B:\n"
         << (A + B) << "\n";

    bool add_correct = (matrix_add_simd(A, B) == (A + B));
    bool sub_correct = (matrix_sub_simd(A, B) == (A - B));
    bool had_correct = (matrix_hadamard_simd(A, B) == A.hadamard(B));
    bool scalar_correct = (matrix_mul_scalar_simd(A, 2.5) == (A * 2.5));
    bool mul_correct = (matrix_mul_simd(A, B) == (A * B));

    cout << "\nCorrectness checks:\n";
    cout << "Addition: " << (add_correct ? "✓ PASS" : "✗ FAIL") << "\n";
    cout << "Subtraction: " << (sub_correct ? "✓ PASS" : "✗ FAIL") << "\n";
    cout << "Hadamard: " << (had_correct ? "✓ PASS" : "✗ FAIL") << "\n";
    cout << "Scalar: " << (scalar_correct ? "✓ PASS" : "✗ FAIL") << "\n";
    cout << "Multiplication: " << (mul_correct ? "✓ PASS" : "✗ FAIL") << "\n";

    if (add_correct && sub_correct && had_correct && scalar_correct && mul_correct)
    {
        cout << "\n✓ All SIMD operations produce correct results!\n";
        return 0;
    }
    else
    {
        cout << "\n✗ Some SIMD operations failed verification!\n";
        return 1;
    }
}
