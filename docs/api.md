# Matrix Library API Reference

Quick reference for the `matrix<T>` template class.

## Class Definition

```cpp
template <typename T>
class matrix;

// Type aliases
using Matrixi = matrix<int>;
using Matrixf = matrix<float>;
using Matrixd = matrix<double>;
```

## Construction & Initialization

```cpp
matrix<T>(int rows, int cols);                          // Create uninitialized matrix
matrix<T>(const vector<vector<T>>& data);               // Create from 2D vector

static matrix<T> zeros(int rows, int cols);             // Fill with zeros
static matrix<T> ones(int rows, int cols);              // Fill with ones
static matrix<T> identity(int size);                    // Identity matrix
static matrix<T> random(int rows, int cols, T min, T max); // Random values
```

**Example:**

```cpp
matrix<double> A(3, 3);                     // 3×3 uninitialized
matrix<double> B = matrix<double>::zeros(3, 3);
matrix<double> I = matrix<double>::identity(3);
```

## Element Access

```cpp
T& operator()(int row, int col);                // Read/write element
const T& operator()(int row, int col) const;    // Read-only element
```

**Example:**

```cpp
A(0, 0) = 5.0;      // Set element
double val = A(1, 2); // Get element
```

## Arithmetic Operators

### Matrix-Matrix Operations

```cpp
matrix<T> operator+(const matrix<T>& other) const;  // Addition
matrix<T> operator-(const matrix<T>& other) const;  // Subtraction
matrix<T> operator*(const matrix<T>& other) const;  // Multiplication
matrix<T> hadamard(const matrix<T>& other) const;   // Element-wise product

matrix<T>& operator+=(const matrix<T>& other);      // Add-assign
matrix<T>& operator-=(const matrix<T>& other);      // Subtract-assign
matrix<T>& operator*=(const matrix<T>& other);      // Multiply-assign

bool operator==(const matrix<T>& other) const;      // Equality
bool operator!=(const matrix<T>& other) const;      // Inequality
```

### Matrix-Scalar Operations

```cpp
matrix<T> operator+(const T& scalar) const;         // Add scalar
matrix<T> operator-(const T& scalar) const;         // Subtract scalar
matrix<T> operator*(const T& scalar) const;         // Multiply by scalar
matrix<T> operator/(const T& scalar) const;         // Divide by scalar

matrix<T>& operator+=(const T& scalar);             // Add-assign
matrix<T>& operator-=(const T& scalar);             // Subtract-assign
matrix<T>& operator*=(const T& scalar);             // Multiply-assign
matrix<T>& operator/=(const T& scalar);             // Divide-assign
```

## Matrix Operations

### Basic Operations

```cpp
matrix<T> transpose() const;                        // Transpose
T trace() const;                                    // Sum of diagonal
double determinant() const;                         // Determinant
matrix<double> inverse() const;                     // Inverse
int rank() const;                                   // Rank
```

### Norms & Properties

```cpp
double norm(int p = 2) const;                       // p-norm
bool is_square() const;                             // Square matrix?
bool is_symmetric() const;                          // Symmetric?
bool is_diagonal() const;                           // Diagonal?
bool is_upper_triangular() const;                   // Upper triangular?
bool is_lower_triangular() const;                   // Lower triangular?
```

### Matrix Manipulation

```cpp
void resize(int rows, int cols);                    // Resize
void swapRows(int row1, int row2);                  // Swap rows
void swapCols(int col1, int col2);                  // Swap columns
matrix<T> submatrix(int x1, int y1, int x2, int y2) const;  // Extract submatrix
void set_submatrix(int x, int y, const matrix<T>& sub);     // Set submatrix
```

### Diagonal Operations

```cpp
vector<T> diagonal(int k = 0) const;                // Get diagonal
vector<T> anti_diagonal(int k = 0) const;           // Get anti-diagonal
void set_diagonal(const vector<T>& diag, int k = 0);        // Set diagonal
void set_anti_diagonal(const vector<T>& anti_diag, int k = 0); // Set anti-diagonal
```

### Advanced Operations

```cpp
matrix<T> power(int n) const;                       // Matrix power A^n
matrix<double> exponential_pow(int max_iter = 30) const;  // Matrix exponential
matrix<T> cofactor() const;                         // Cofactor matrix
matrix<T> adjoint() const;                          // Adjoint matrix
matrix<T> minor(int row, int col) const;            // Minor matrix
```

## Decompositions

```cpp
matrix<double> LU_decomposition() const;            // LU decomposition
matrix<double> QR_decomposition() const;            // QR decomposition
matrix<double> gaussian_elimination() const;        // Row echelon form
matrix<double> eigenvalues() const;                 // Eigenvalues
matrix<double> eigenvectors() const;                // Eigenvectors
```

## I/O Operations

```cpp
friend ostream& operator<<(ostream& os, const matrix<T>& m);  // Print matrix
friend istream& operator>>(istream& is, matrix<T>& m);        // Read matrix
```

**Example:**

```cpp
cout << A << endl;      // Print matrix
cin >> A;               // Read matrix from stdin
```

## Type Casting

```cpp
template <typename U>
explicit operator matrix<U>() const;                // Cast to different type
```

**Example:**

```cpp
matrix<int> A(2, 2);
matrix<double> B = static_cast<matrix<double>>(A);
```

## Example Usage

```cpp
#include "matrix.hpp"
#include <iostream>

int main() {
    // Create matrices
    auto A = matrix<double>::random(3, 3, 0.0, 10.0);
    auto B = matrix<double>::identity(3);

    // Arithmetic
    auto C = A + B;
    auto D = A * B;
    auto E = A * 2.0;

    // Properties
    double det = A.determinant();
    int rank = A.rank();
    bool symmetric = A.is_symmetric();

    // Advanced operations
    auto At = A.transpose();
    auto inv = A.inverse();

    // Output
    std::cout << "Matrix A:\n" << A << std::endl;
    std::cout << "Determinant: " << det << std::endl;

    return 0;
}
```

## Exception Handling

The library throws standard exceptions:

- `std::out_of_range` - Invalid indices
- `std::invalid_argument` - Incompatible dimensions, singular matrices, non-square operations

**Example:**

```cpp
try {
    matrix<double> A(2, 3);
    double det = A.determinant();  // Throws: non-square matrix
} catch (const std::invalid_argument& e) {
    std::cerr << "Error: " << e.what() << std::endl;
}
```

See [BUILD_GUIDE.md](BUILD_GUIDE.md) for optimization options.
