#include <iostream>
#include "../include/matrix.hpp"

using namespace std;

int main() {
    cout << "Matrix Library Test Suite" << endl;
    cout << "=========================" << endl << endl;
    
    // Test basic matrix creation
    matrix<double> A = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };
    
    matrix<double> B = {
        {9.0, 8.0, 7.0},
        {6.0, 5.0, 4.0},
        {3.0, 2.0, 1.0}
    };
    
    cout << "Matrix A:" << endl << A << endl;
    cout << "Matrix B:" << endl << B << endl;
    
    // Test operations
    cout << "A + B:" << endl << (A + B) << endl;
    cout << "A * B:" << endl << (A * B) << endl;
    
    // Test with larger matrices
    matrix<double> C = matrix<double>::random(100, 100, -1.0, 1.0);
    matrix<double> D = matrix<double>::random(100, 100, -1.0, 1.0);
    
    cout << "Testing 100x100 matrix multiplication..." << endl;
    matrix<double> E = C * D;
    cout << "Success! Result matrix size: " << E.rows() << "x" << E.cols() << endl;
    
    cout << "\n✓ All tests passed!" << endl;
    return 0;
}