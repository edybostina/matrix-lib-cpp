#include <iostream>
#include <iomanip>

#include "../include/matrix.hpp"

using namespace std;

// Helper function to print section headers
void print_section(const string& title)
{
    cout << "\n╔════════════════════════════════════════════════════════════════════╗\n";
    cout << "║ " << setw(66) << left << title << " ║\n";
    cout << "╚════════════════════════════════════════════════════════════════════╝\n\n";
}

void print_subsection(const string& title)
{
    cout << "\n▸ " << title << "\n";
    cout << string(70, '-') << "\n";
}

int main()
{
    cout << "\n";
    cout << "╔════════════════════════════════════════════════════════════════════╗\n";
    cout << "║               MATRIX LIBRARY C++ - FEATURE DEMONSTRATION           ║\n";
    cout << "║                         Version " << MATRIX_LIB_VERSION << "                              ║\n";
    cout << "╚════════════════════════════════════════════════════════════════════╝\n";

    try
    {
        // ================================================================
        // SECTION 1: Matrix Creation
        // ================================================================
        print_section("1. Matrix Creation");

        print_subsection("From Initializer List");
        matrix<int> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        cout << "Matrix A (3x3):\n" << A << endl;

        print_subsection("Factory Methods");
        auto I = Matrixd::eye(3, 3);
        cout << "Identity Matrix I (3x3):\n" << I << endl;

        auto zeros = Matrixi::zeros(2, 3);
        cout << "Zero Matrix (2x3):\n" << zeros << endl;

        auto ones = Matrixi::ones(2, 2);
        cout << "Ones Matrix (2x2):\n" << ones << endl;

        auto rand = Matrixd::random(3, 3, -5.0, 5.0);
        cout << "Random Matrix (3x3, range [-5, 5]):\n" << rand << endl;

        // ================================================================
        // SECTION 2: Basic Arithmetic Operations
        // ================================================================
        print_section("2. Basic Arithmetic Operations");

        matrix<int> B = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
        cout << "Matrix B:\n" << B << endl;

        print_subsection("Addition");
        auto C = A + B;
        cout << "A + B =\n" << C << endl;

        print_subsection("Subtraction");
        auto D = A - B;
        cout << "A - B =\n" << D << endl;

        print_subsection("Scalar Multiplication");
        auto E = A * 2;
        cout << "A * 2 =\n" << E << endl;

        print_subsection("Matrix Multiplication");
        matrix<int> M1 = {{1, 2}, {3, 4}};
        matrix<int> M2 = {{5, 6}, {7, 8}};
        cout << "M1:\n" << M1 << endl;
        cout << "M2:\n" << M2 << endl;
        auto M3 = M1 * M2;
        cout << "M1 x M2 =\n" << M3 << endl;

        print_subsection("Hadamard Product (Element-wise)");
        auto H = M1.hadamard(M2);
        cout << "M1 ⊙ M2 =\n" << H << endl;

        // ================================================================
        // SECTION 3: Matrix Properties
        // ================================================================
        print_section("3. Matrix Properties & Analysis");

        matrix<double> test = {{2, 1, 3, 4}, {1, 3, 2, 1}, {3, 2, 5, 2}, {4, 1, 2, 6}};
        cout << "Test Matrix (4x4):\n" << test << endl;

        print_subsection("Basic Properties");
        cout << "Dimensions: " << test.rows() << "x" << test.cols() << "\n";
        cout << "Total elements: " << test.size() << "\n";
        cout << "Trace: " << test.trace() << "\n";
        cout << "Determinant: " << test.determinant() << "\n";
        cout << "Rank: " << test.rank() << "\n";
        cout << "2-Norm: " << fixed << setprecision(4) << test.norm(2) << "\n";

        print_subsection("Property Checks");
        matrix<double> sym = {{1, 2, 3}, {2, 4, 5}, {3, 5, 6}};
        cout << "Symmetric matrix:\n" << sym << endl;
        cout << "Is symmetric? " << (sym.is_symmetric() ? "Yes" : "No") << "\n";
        cout << "Is diagonal? " << (sym.is_diagonal() ? "Yes" : "No") << "\n\n";

        matrix<double> diag = {{1, 0, 0}, {0, 2, 0}, {0, 0, 3}};
        cout << "Diagonal matrix:\n" << diag << endl;
        cout << "Is diagonal? " << (diag.is_diagonal() ? "Yes" : "No") << "\n";
        cout << "Is lower triangular? " << (diag.is_lower_triangular() ? "Yes" : "No") << "\n";
        cout << "Is upper triangular? " << (diag.is_upper_triangular() ? "Yes" : "No") << "\n";

        // ================================================================
        // SECTION 4: Linear Algebra Operations
        // ================================================================
        print_section("4. Linear Algebra Operations");

        matrix<double> L = {{4, 3}, {6, 3}};
        cout << "Matrix L:\n" << L << endl;

        print_subsection("Transpose");
        cout << "L^T =\n" << L.transpose() << endl;

        print_subsection("Inverse");
        auto L_inv = L.inverse();
        cout << "L^(-1) =\n" << L_inv << endl;
        cout << "Verification: L x L^(-1) =\n" << L * L_inv << endl;

        print_subsection("Adjoint (Adjugate)");
        cout << "adj(L) =\n" << L.adjoint() << endl;

        print_subsection("Cofactor Matrix");
        cout << "Cofactor matrix:\n" << L.cofactor() << endl;

        // ================================================================
        // SECTION 5: Matrix Decompositions
        // ================================================================
        print_section("5. Matrix Decompositions");

        matrix<double> decomp = {{4, 3, 2}, {6, 8, 5}, {2, 5, 7}};
        cout << "Original Matrix:\n" << decomp << endl;

        print_subsection("LU Decomposition");
        auto [L_lu, U_lu] = decomp.LU_decomposition();
        cout << "L (Lower):\n" << L_lu << endl;
        cout << "U (Upper):\n" << U_lu << endl;
        cout << "Verification: L x U =\n" << L_lu * U_lu << endl;

        print_subsection("QR Decomposition");
        auto [Q, R] = decomp.QR_decomposition();
        cout << "Q (Orthogonal):\n" << Q << endl;
        cout << "R (Upper triangular):\n" << R << endl;
        cout << "Verification: Q x R =\n" << Q * R << endl;

        // ================================================================
        // SECTION 6: Eigenanalysis
        // ================================================================
        print_section("6. Eigenvalues & Eigenvectors");

        matrix<double> eigen_mat = {{4, 1, 0}, {1, 3, 1}, {0, 1, 2}};
        cout << "Symmetric Matrix:\n" << eigen_mat << endl;

        print_subsection("Eigenvalues");
        auto eigenvals = eigen_mat.eigenvalues(100);
        cout << "Eigenvalues:\n" << eigenvals << endl;

        print_subsection("Eigenvectors");
        auto eigenvecs = eigen_mat.eigenvectors(100);
        cout << "Eigenvectors (as columns):\n" << eigenvecs << endl;

        // ================================================================
        // SECTION 7: Advanced Operations
        // ================================================================
        print_section("7. Advanced Operations");

        matrix<double> P = {{2, 1}, {1, 2}};
        cout << "Matrix P:\n" << P << endl;

        print_subsection("Matrix Power");
        auto P2 = P.pow(2);
        cout << "P^2 =\n" << P2 << endl;
        auto P3 = P.pow(3);
        cout << "P^3 =\n" << P3 << endl;

        print_subsection("Matrix Exponential");
        auto exp_P = P.exponential_pow(20);
        cout << "e^P (Taylor series, 20 terms) =\n" << exp_P << endl;

        print_subsection("Gaussian Elimination");
        matrix<double> gauss = {{2, 1, -1}, {-3, -1, 2}, {-2, 1, 2}};
        cout << "Original:\n" << gauss << endl;
        cout << "Row echelon form:\n" << gauss.gaussian_elimination() << endl;

        // ================================================================
        // SECTION 8: Matrix Manipulation
        // ================================================================
        print_section("8. Matrix Manipulation");

        matrix<int> manip = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        cout << "Original Matrix:\n" << manip << endl;

        print_subsection("Row & Column Extraction");
        cout << "Row 1:\n" << manip.row(1) << endl;
        cout << "Column 2:\n" << manip.col(2) << endl;

        print_subsection("Submatrix Extraction");
        auto sub = manip.submatrix(0, 0, 1, 1);
        cout << "Submatrix [0:1, 0:1]:\n" << sub << endl;

        print_subsection("Row/Column Swapping");
        matrix<int> swap_test = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        swap_test.swapRows(0, 2);
        cout << "After swapping rows 0 and 2:\n" << swap_test << endl;

        print_subsection("Diagonal Manipulation");
        matrix<int> diag_test = Matrixi::zeros(4, 4);
        diag_test.set_diagonal({1, 2, 3, 4});
        cout << "After setting diagonal {1,2,3,4}:\n" << diag_test << endl;

        // ================================================================
        // SECTION 9: Type Aliases & Element Access
        // ================================================================
        print_section("9. Type Aliases & Element Access");

        print_subsection("Type Aliases");
        Matrixi int_mat = {{1, 2}, {3, 4}};
        Matrixf float_mat = {{1.5f, 2.5f}, {3.5f, 4.5f}};
        Matrixd double_mat = {{1.1, 2.2}, {3.3, 4.4}};

        cout << "Matrixi (int):\n" << int_mat << endl;
        cout << "Matrixf (float):\n" << float_mat << endl;
        cout << "Matrixd (double):\n" << double_mat << endl;

        print_subsection("Element Access & Modification");
        matrix<int> access = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        cout << "Original:\n" << access << endl;
        cout << "Element at (1, 2): " << access(1, 2) << "\n";
        access(1, 2) = 99;
        cout << "After setting (1, 2) = 99:\n" << access << endl;

        // ================================================================
        // SECTION 10: Error Handling
        // ================================================================
        print_section("10. Error Handling");

        cout << "Attempting invalid operations...\n\n";

        try
        {
            matrix<int> m1(2, 3);
            matrix<int> m2(4, 5);
            auto invalid = m1 + m2;
        }
        catch (const exception& e)
        {
            cout << "✓ Caught expected error: " << e.what() << "\n\n";
        }

        try
        {
            matrix<int> singular = {{1, 2}, {2, 4}};
            auto inv = singular.inverse();
        }
        catch (const exception& e)
        {
            cout << "✓ Caught expected error: " << e.what() << "\n\n";
        }

        try
        {
            matrix<int> rect(2, 3);
            auto _ = rect.determinant();
        }
        catch (const exception& e)
        {
            cout << "✓ Caught expected error: " << e.what() << "\n";
        }

        // ================================================================
        // Final Message
        // ================================================================
        cout << "\n";
        cout << "╔════════════════════════════════════════════════════════════════════╗\n";
        cout << "║                  DEMONSTRATION COMPLETED SUCCESSFULLY!             ║\n";
        cout << "╚════════════════════════════════════════════════════════════════════╝\n\n";
    }
    catch (const exception& e)
    {
        cerr << "\nError: " << e.what() << "\n";
        return 1;
    }

    return 0;
}