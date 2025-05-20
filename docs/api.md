# Matrix Library C++ API

This document describes the public API for the `matrix<T>` class.

---

## Table of Contents

- [Class Template](#class-template)
- [Constructors](#constructors)
- [Element Access](#element-access)
- [Static Initialization Methods](#static-initialization-methods)
- [File I/O](#file-io)
- [Arithmetic Operators](#arithmetic-operators)
- [Matrix Functions](#matrix-functions)
- [Exceptions](#exceptions)
- [Some Examples](#some-examples)


---

## Class Template

```cpp
template <typename T>
class matrix;
```
- The `matrix<T>` class is a template class that represents a 2D matrix of type `T`. It supports various operations and functions for matrix manipulation. 
- The class is defined in `include/matrix.hpp` and is designed to be used with any numeric type, including `int`, `float`, and `double`
---

## Constructors
```cpp
matrix(int rows, int cols);
matrix(const vector<std::vector<T>>& data);
```
- **matrix(int rows, int cols)**: Create a matrix with given dimensions, uninitialized.
- **matrix(const vector<std::vector<T>>& data)**: Create a matrix from a 2D vector.
---
## Element Access
```cpp
T& operator()(int row, int col);
const T& operator()(int row, int col) const;
vector<T>& operator()(int row);
const vector<T>& operator()(int row) const;
```
- **T& operator()(int row, int col)**: Access element at (row, col) for read/write.
- **const T& operator()(int row, int col) const**: Access element at (row, col) for read-only.
- **vector<T>& operator()(int row)**: Access row for read/write.
- **const vector<T>& operator()(int row) const**: Access row for read-only.
---
## Static Initialization Methods
```cpp
static matrix<T> zeros(int rows, int cols);
static matrix<T> ones(int rows, int cols);
static matrix<T> identity(int size);
static matrix<T> random(int rows, int cols, T min, T max);
```
- **static matrix<T> zeros(int rows, int cols)**: Create a matrix filled with zeros.
- **static matrix<T> ones(int rows, int cols)**: Create a matrix filled with ones.
- **static matrix<T> identity(int size)**: Create an identity matrix of given size.
- **static matrix<T> random(int rows, int cols, T min, T max)**: Create a matrix with random values in the range [min, max].
---
## File I/O
```cpp
friend ostream& operator<<(ostream& os, const matrix<U>& m); // cout << m
friend istream& operator>>(istream& is, matrix<U>& m); // cin >> m
```
- **friend ostream& operator<<(ostream& os, const matrix<U>& m)**: Output the matrix to a stream.
- **friend istream& operator>>(istream& is, matrix<U>& m)**: Input the matrix from a stream.
---
## Casting
```cpp
template <typename U>
explicit operator matrix<U>() const;
```
- **template <typename U> explicit operator matrix<U>() const**: Cast the matrix to another type.
### Example
```cpp
matrix<int> A(2, 2);
A(0, 0) = 1; A(0, 1) = 2;
A(1, 0) = 3; A(1, 1) = 4;
matrix<double> B = (matrix<double>)A;
// or
matrix<double> B = static_cast<matrix<double>>(A);
```
---
## Arithmetic Operators
Matrix-matrix and matrix-scalar operations are supported. The following operators are overloaded:
```cpp
matrix<T> operator+(const matrix<T>& other) const;
matrix<T> operator-(const matrix<T>& other) const;
matrix<T> operator*(const matrix<T>& other) const;
matrix<T> operator+=(const matrix<T>& other);
matrix<T> operator-=(const matrix<T>& other);
matrix<T> operator*=(const matrix<T>& other);
bool operator==(const matrix<T>& other) const;
bool operator!=(const matrix<T>& other) const;
```
- **matrix<T> operator+(const matrix<T>& other) const**: Add two matrices.
- **matrix<T> operator-(const matrix<T>& other) const**: Subtract two matrices.
- **matrix<T> operator*(const matrix<T>& other) const**: Multiply two matrices.
- **matrix<T> operator+=(const matrix<T>& other)**: Add and assign.
- **matrix<T> operator-=(const matrix<T>& other)**: Subtract and assign.
- **matrix<T> operator*=(const matrix<T>& other)**: Multiply and assign.
- **bool operator==(const matrix<T>& other) const**: Check if two matrices are equal.
- **bool operator!=(const matrix<T>& other) const**: Check if two matrices are not equal.
---
Matrix-scalar operations:
```cpp
matrix<T> operator+(const T& scalar) const;
matrix<T> operator-(const T& scalar) const;
matrix<T> operator*(const T& scalar) const;
matrix<T> operator/(const T& scalar) const;
matrix<T> operator+=(const T& scalar);
matrix<T> operator-=(const T& scalar);
matrix<T> operator*=(const T& scalar);
matrix<T> operator/=(const T& scalar);
```
- **matrix<T> operator+(const T& scalar) const**: Add a scalar to the matrix.
- **matrix<T> operator-(const T& scalar) const**: Subtract a scalar from the matrix.
- **matrix<T> operator*(const T& scalar) const**: Multiply the matrix by a scalar.
- **matrix<T> operator/(const T& scalar) const**: Divide the matrix by a scalar.
- **matrix<T> operator+=(const T& scalar)**: Add a scalar and assign.
- **matrix<T> operator-=(const T& scalar)**: Subtract a scalar and assign.
- **matrix<T> operator*=(const T& scalar)**: Multiply by a scalar and assign.
- **matrix<T> operator/=(const T& scalar)**: Divide by a scalar and assign.
---
## Matrix Functions
```cpp
T determinant() const;
    T trace() const;
    matrix<T> transpose() const;
    matrix<T> cofactor() const;
    matrix<T> minor(int row, int col) const;
    matrix<T> adjoint() const;
    matrix<double> inverse() const;

    void swapRows(int row1, int row2);
    void swapCols(int col1, int col2);

    void resize(int rows, int cols);
```
- **T determinant() const**: Calculate the determinant of the matrix.
- **T trace() const**: Calculate the trace of the matrix.
- **matrix<T> transpose() const**: Transpose the matrix.
- **matrix<T> cofactor(int row, int col) const**: Calculate the cofactor of the matrix.
- **matrix<T> minor(int row, int col) const**: Calculate the minor of the matrix.
- **matrix<T> adjoint() const**: Calculate the adjoint of the matrix.
- **matrix<T> inverse() const**: Calculate the inverse of the matrix.
- **void swapRows(int row1, int row2)**: Swap two rows of the matrix.
- **void swapCols(int col1, int col2)**: Swap two columns of the matrix.
- **void resize(int rows, int cols)**: Resize the matrix to the given dimensions.
---
## Exceptions
The library uses `out_of_range` and `invalid_argument` exceptions for error handling. These exceptions are thrown in the following cases:
- Accessing an element outside the matrix bounds.
- Performing operations on matrices of incompatible sizes.
- Attempting to calculate the inverse of a non-square matrix.
- Attempting to calculate the determinant of a non-square matrix.
- Attempting to calculate the inverse of a singular matrix.
- Attempting to calculate the determinant of a singular matrix.
- Attempting to calculate the adjoint of a singular matrix.
- Attempting to calculate the cofactor of a singular matrix.
- And many more.

---
## Some Examples
```cpp
#include "include/matrix.hpp"
#include <iostream>
using namespace std;
int main() {
    matrix<double> A(2, 2);
    A(0, 0) = 1; A(0, 1) = 2;
    A(1, 0) = 3; A(1, 1) = 4;

    matrix<double> B = A.transpose();
    matrix<double> C = A + B;
    matrix<double> D = A * 2.0;
    double det = A.determinant();

    cout << "Matrix A:\n" << A << endl;
    cout << "Matrix B (transpose of A):\n" << B << endl;
    cout << "Matrix C (A + B):\n" << C << endl;
    cout << "Matrix D (A * 2.0):\n" << D << endl;
    cout << "Determinant of A: " << det << endl;

    return 0;
}
```
