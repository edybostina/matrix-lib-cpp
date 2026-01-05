#include <iostream>
#include <iomanip>
#include "../include/matrix.hpp"

using namespace std;

void demo_section(const string& title)
{
    cout << "\n" << title << "\n" << string(title.length(), '-') << "\n";
}

int main()
{
    cout << "\nMatrix Library Demo (v" << MATRIX_LIB_VERSION << ")\n";
    cout << "══════════════════════════════════════\n";

    // Basic operations
    demo_section("1. Matrix Creation");
    
    matrix<int> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    cout << "Matrix A:\n" << A;
    
    auto I = matrix<double>::eye(3, 3);
    cout << "\nIdentity 3x3:\n" << I;

    // Arithmetic
    demo_section("2. Arithmetic");
    
    matrix<int> B = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
    cout << "A + B:\n" << A + B;
    cout << "\nA - B:\n" << A - B;
    cout << "\nA * 2:\n" << A * 2;
    cout << "\n-A:\n" << -A;

    // Matrix multiplication
    demo_section("3. Matrix Multiplication");
    
    matrix<int> C = {{1, 2}, {3, 4}};
    matrix<int> D = {{5, 6}, {7, 8}};
    cout << "C:\n" << C << "\nD:\n" << D;
    cout << "\nC × D:\n" << C * D;

    // Linear algebra
    demo_section("4. Linear Algebra");
    
    matrix<double> M = {{4, 3}, {6, 3}};
    cout << "Matrix M:\n" << M;
    cout << "\nTranspose:\n" << M.transpose();
    cout << "\nDeterminant: " << M.determinant() << "\n";
    cout << "\nInverse:\n" << M.inverse();

    // Decompositions
    demo_section("5. Decompositions");
    
    matrix<double> E = {{4, 2, 1}, {2, 5, 3}, {1, 3, 6}};
    cout << "Matrix E:\n" << E;
    
    auto [L, U] = E.LU_decomposition();
    cout << "\nLU Decomposition:\nL:\n" << L << "\nU:\n" << U;

    // Row/Column operations
    demo_section("6. Row/Column Proxies");
    
    matrix<int> F = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    cout << "Matrix F:\n" << F;
    
    F.row(0) *= 2;
    cout << "\nAfter F.row(0) *= 2:\n" << F;
    
    F.col(1) += F.col(2);
    cout << "\nAfter F.col(1) += F.col(2):\n" << F;
    
    cout << "\nRow 0 sum: " << F.row(0).sum() << "\n";
    cout << "Column 2 max: " << F.col(2).max() << "\n";

    // Matrix properties
    demo_section("7. Matrix Properties");
    
    matrix<double> sym = {{1, 2, 3}, {2, 4, 5}, {3, 5, 6}};
    cout << "Symmetric matrix:\n" << sym;
    cout << "\nIs symmetric? " << (sym.is_symmetric() ? "Yes" : "No");
    cout << "\nIs diagonal? " << (sym.is_diagonal() ? "Yes" : "No");
    cout << "\nTrace: " << sym.trace();
    cout << "\nNorm: " << sym.norm() << "\n";

    // Submatrix operations
    demo_section("8. Submatrix Operations");
    
    matrix<int> G = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
    cout << "Matrix G:\n" << G;
    cout << "\nSubmatrix [0:1, 1:3]:\n" << G.submatrix(0, 1, 1, 3);

    cout << "\n\nDemo complete!\n\n";
    return 0;
}
