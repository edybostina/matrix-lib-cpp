# Matrix Library API Reference

Complete API documentation for the `matrix<T>` template class.

## Table of Contents

- [Overview](#overview)
- [Class Definition](#class-definition)
- [Construction & Initialization](#construction--initialization)
- [Element Access](#element-access)
- [Arithmetic Operators](#arithmetic-operators)
- [Matrix Operations](#matrix-operations)
- [Decompositions](#decompositions)
- [Manipulation](#manipulation)
- [I/O Operations](#io-operations)
- [Type Casting](#type-casting)
- [Exception Handling](#exception-handling)

## Overview

The `matrix<T>` class provides a generic, high-performance matrix implementation for C++17. It supports all common linear algebra operations, advanced decompositions, and optimizations through BLAS and SIMD.

**Key Features:**

- Template-based design for any numeric type
- Intuitive operator overloading
- Comprehensive linear algebra operations
- Exception-safe with clear error messages
- Header-only or compiled library modes

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
// Default constructor
matrix<T>();

// Create uninitialized matrix of size rows x cols
matrix<T>(size_t rows, size_t cols);

// Create from 2D vector
matrix<T>(const std::vector<std::vector<T>>& data);

// Create from initializer list
matrix<T>(std::initializer_list<std::initializer_list<T>> init);

// Static Factory Methods
static matrix<T> zeros(size_t rows, size_t cols);             // Fill with zeros
static matrix<T> ones(size_t rows, size_t cols);              // Fill with ones
static matrix<T> eye(size_t rows, size_t cols);               // Identity matrix
static matrix<T> random(size_t rows, size_t cols, T min, T max); // Random values
```

**Example:**

```cpp
matrix<double> A(3, 3);                     // 3x3 uninitialized
matrix<double> B = matrix<double>::zeros(3, 3);
matrix<double> I = matrix<double>::eye(3, 3);
matrix<int> C = {{1, 2}, {3, 4}};           // Initializer list
```

## Element Access

```cpp
// 2D Access
T& operator()(size_t row, size_t col);                // Read/write element
const T& operator()(size_t row, size_t col) const;    // Read-only element

// 1D Access (Row)
std::vector<T> operator()(size_t index);               // Get row as vector
std::vector<T> operator()(size_t index) const;         // Get row as vector

// Raw Data Access
T* data_ptr();
const T* data_ptr() const;

// Dimensions
size_t rows() const;
size_t cols() const;
size_t size() const;
```

**Example:**

```cpp
A(0, 0) = 5.0;      // Set element at (0,0)
double val = A(1, 2); // Get element at (1,2)
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

// In-place versions
matrix<T>& operator+=(const T& scalar);
matrix<T>& operator-=(const T& scalar);
matrix<T>& operator*=(const T& scalar);
matrix<T>& operator/=(const T& scalar);
```

## Matrix Operations

```cpp
// Basic Properties
bool is_square() const;
bool is_symmetric() const;
bool is_diagonal() const;
bool is_upper_triangular() const;
bool is_lower_triangular() const;
bool is_orthogonal(double tolerance = 1e-10) const;
bool is_singular(double tolerance = 1e-10) const;
bool is_idempotent(double tolerance = 1e-10) const;
bool is_nilpotent(size_t k, double tolerance = 1e-10) const;
bool is_involutory(double tolerance = 1e-10) const;
bool is_positive_definite() const;
bool is_negative_definite() const;

// Linear Algebra
T trace() const;
matrix<T> transpose() const;
double determinant() const;
matrix<double> inverse() const;
size_t rank() const;
double norm(int p = 2) const; // p-norm (default: Euclidean)
matrix<double> solve(const matrix<double>& b) const; // Solve Ax = b

// Advanced
matrix<T> cofactor() const;
matrix<T> adjoint() const;
matrix<T> minor(size_t row, size_t col) const;
matrix<double> gaussian_elimination() const;

// Power Functions
matrix<T> pow(const int& power) const;
matrix<double> exponential_pow(int max_iter = 30) const; // Matrix exponential e^A
```

## Decompositions

```cpp
// LU Decomposition
// Returns pair {L, U} where A = L * U
std::pair<matrix<double>, matrix<double>> LU_decomposition() const;

// QR Decomposition
// Returns pair {Q, R} where A = Q * R
std::pair<matrix<double>, matrix<double>> QR_decomposition() const;

// Eigenvalues & Eigenvectors
matrix<double> eigenvalues(int max_iter = 100) const;
matrix<double> eigenvectors(int max_iter = 100) const;

// Singular Value Decomposition
// Returns tuple {U, Σ, V^T} where A = U * Σ * V^T
std::tuple<matrix<double>, matrix<double>, matrix<double>> SVD() const;
```

## Manipulation

```cpp
// Resizing
void resize(size_t rows, size_t cols);

// Swapping
void swap_rows(size_t row1, size_t row2);
void swap_cols(size_t col1, size_t col2);

// Submatrices
matrix<T> submatrix(size_t top_corner_x, size_t top_corner_y,
                    size_t bottom_corner_x, size_t bottom_corner_y) const;
void set_submatrix(size_t top_corner_x, size_t top_corner_y, const matrix<T>& submatrix);

// Diagonals
std::vector<T> diagonal(int k = 0) const;
std::vector<T> anti_diagonal(int k = 0) const;
void set_diagonal(const std::vector<T>& diag, int k = 0);
void set_anti_diagonal(const std::vector<T>& anti_diag, int k = 0);

// Row/Column Extraction
matrix<T> row(size_t index) const;
matrix<T> col(size_t index) const;

// Set Row/Column
void set_row(size_t index, const std::vector<T>& values);
void set_col(size_t index, const std::vector<T>& values);

// Clamping
matrix<T> clamp(const T& min, const T& max) const;

// Element-wise Function Application
matrix<T> apply(T (*func)(T)) const;
```

## I/O Operations

```cpp
// Output stream operator
template <typename U>
friend std::ostream& operator<<(std::ostream& os, const matrix<U>& m);

// Input stream operator
template <typename U>
friend std::istream& operator>>(std::istream& is, matrix<U>& m);
```

## Type Casting

```cpp
// Explicit cast to matrix of another type
template <typename U>
explicit operator matrix<U>() const;
```

**Example:**

```cpp
matrix<int> A = {{1, 2}, {3, 4}};
matrix<double> B = (matrix<double>)A; // Cast to double
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
