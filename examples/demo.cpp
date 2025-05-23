#include <iostream>
#include "../include/matrix.hpp"

using namespace std;

int main()
{
    // Test matrix creation
    matrix<int> mat1 = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}};

    matrix<int> mat2 = matrix<int>::eye(3, 3);
    matrix<int> mat3 = matrix<int>::ones(3, 3);
    matrix<int> mat4 = matrix<int>::zeros(3, 3);
    matrix<double> mat_random = matrix<double>::random(3, 3, -5.0, 5.0);

    cout << "mat1:" << endl
         << mat1 << endl;
    cout << "mat2:" << endl
         << mat2 << endl;
    cout << "mat3:" << endl
         << mat3 << endl;
    cout << "mat4:" << endl
         << mat4 << endl;
    cout << "mat_random:" << endl
         << mat_random << endl;

    // Test arithmetic operators
    matrix<int> mat5 = mat1 + mat2;
    cout << "mat1 + mat2:" << endl
         << mat5 << endl;

    // Test casting, matrix multiplication and inverse
    // might have precision issues, very small numbers close to 0 may appear
    cout << mat5.inverse() * (matrix<double>)mat5 << endl;

    matrix<int> test{
        {2, 1, 3, 4, 5},
        {6, 7, 2, 1, 2},
        {4, 5, 9, 2, 0},
        {3, 1, 0, 8, 6},
        {7, 2, 4, 5, 1}};
    cout << test.determinant() << endl;                                             // should be -5296
    cout << test.inverse() * (matrix<double>)test << endl;                          // should be identity
    cout << mat5 * mat5.adjoint() * 1 / mat5.determinant() << endl;                 // should be identity
    cout << test.LU_decomposition().first * test.LU_decomposition().second << endl; // should be test
    cout << test.QR_decomposition().first * test.QR_decomposition().second << endl; // should be test
    cout << test.eigenvalues() << endl;                                             // should be eigenvalues
    cout << test.eigenvectors() << endl;                                            // should be eigenvectors
}