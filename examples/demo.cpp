#include <iostream>
#include "../include/matrix.hpp"

using namespace std;

int main() {
    // Test matrix creation
    matrix<int> mat1(3, 3);
    matrix<int> mat2 = matrix<int>::eye(3, 3);
    matrix<int> mat3 = matrix<int>::ones(3, 3);
    matrix<int> mat4 = matrix<int>::zeros(3, 3);
    //matrix<int> mat5 = matrix<int>::rand(3, 3); soon

    // Test access operators
    mat1(0, 0) = 1;
    mat1(1, 1) = 2;
    mat1(2, 2) = 3;

    cout << "mat1:" << endl << mat1 << endl;
    cout << "mat2:" << endl << mat2 << endl;
    cout << "mat3:" << endl << mat3 << endl;
    cout << "mat4:" << endl << mat4 << endl;

    // Test arithmetic operators
    matrix<int> mat5 = mat1 + mat2;
    cout << "mat1 + mat2:" << endl << mat5 << endl;

    matrix<int> mat6 = mat1 - mat2;
    cout << "mat1 - mat2:" << endl << mat6 << endl;
  
    matrix<int> mat7 = mat1;
    mat7 *= mat3;
    cout << "mat1 * mat2:" << endl << mat7 << endl;
  

}