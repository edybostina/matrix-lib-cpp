#pragma once

#include "matrix_fwd.hpp"
#include "matrix_proxy.hpp"

template <typename T>
class matrix
{
public:
    // ========================================================================
    // TYPE DEFINITIONS
    // ========================================================================

    using value_type = T;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;
    using row_proxy_type = row_proxy<T>;
    using const_row_proxy_type = const_row_proxy<T>;
    using col_proxy_type = col_proxy<T>;
    using const_col_proxy_type = const_col_proxy<T>;

    // ========================================================================
    // CONSTRUCTORS & DESTRUCTOR
    // ========================================================================

    /// Default constructor - creates empty matrix
    constexpr matrix() noexcept = default;

    /// Construct matrix with given dimensions, zero-initialized
    explicit matrix(size_t rows, size_t cols);

    /// Construct from 2D vector
    matrix(const std::vector<std::vector<T>>& init);

    /// Construct from initializer list
    matrix(std::initializer_list<std::initializer_list<T>> init);

    /// Copy constructor
    matrix(const matrix&) = default;

    /// Move constructor
    matrix(matrix&&) noexcept = default;

    /// Copy assignment operator
    matrix& operator=(const matrix&) = default;

    /// Move assignment operator
    matrix& operator=(matrix&&) noexcept = default;

    /// Destructor
    ~matrix() = default;

    // ========================================================================
    // FACTORY METHODS
    // ========================================================================

    /// Create matrix filled with zeros
    [[nodiscard]] static matrix<T> zeros(size_t rows, size_t cols);

    /// Create matrix filled with ones
    [[nodiscard]] static matrix<T> ones(size_t rows, size_t cols);

    /// Create identity matrix
    [[nodiscard]] static matrix<T> eye(size_t rows, size_t cols);

    /// Create matrix with random values in [min, max]
    [[nodiscard]] static matrix<T> random(size_t rows, size_t cols, T min, T max);

    // ========================================================================
    // DIMENSION QUERIES
    // ========================================================================

    /// Get number of rows
    [[nodiscard]] constexpr size_t rows() const noexcept
    {
        return _rows;
    }

    /// Get number of columns
    [[nodiscard]] constexpr size_t cols() const noexcept
    {
        return _cols;
    }

    /// Get total number of elements
    [[nodiscard]] constexpr size_t size() const noexcept
    {
        return _rows * _cols;
    }

    /// Check if matrix is square (rows == cols)
    [[nodiscard]] bool is_square() const noexcept
    {
        return _rows == _cols;
    }

    // ========================================================================
    // ELEMENT ACCESS
    // ========================================================================

    /// Access element at (row, col)
    T& operator()(size_t row, size_t col);
    const T& operator()(size_t row, size_t col) const;

    /// Get row proxy for reading and writing
    [[nodiscard]] row_proxy<T> row(size_t index);
    [[nodiscard]] const_row_proxy<T> row(size_t index) const;

    /// Get column proxy for reading and writing
    [[nodiscard]] col_proxy<T> col(size_t index);
    [[nodiscard]] const_col_proxy<T> col(size_t index) const;

    /// Legacy: Access row by index (returns vector) - deprecated, use row() proxy instead
    [[deprecated("Use row() proxy instead")]]
    std::vector<T> get_row(size_t index) const;

    /// Legacy: Access column by index (returns vector) - deprecated, use col() proxy instead
    [[deprecated("Use col() proxy instead")]]
    std::vector<T> get_col(size_t index) const;

    /// Legacy: Set row from vector - deprecated, use row() = vec instead
    [[deprecated("Use row() = vec instead")]]
    void set_row(size_t index, const std::vector<T>& values);

    /// Legacy: Set column from vector - deprecated, use col() = vec instead
    [[deprecated("Use col() = vec instead")]]
    void set_col(size_t index, const std::vector<T>& values);

    /// Get raw data pointer (const)
    [[nodiscard]] const T* data_ptr() const
    {
        return _data.data();
    }

    /// Get raw data pointer (mutable)
    [[nodiscard]] T* data_ptr()
    {
        return _data.data();
    }

    // ========================================================================
    // ITERATORS (STL-compatible)
    // ========================================================================

    [[nodiscard]] iterator begin() noexcept
    {
        return _data.begin();
    }
    [[nodiscard]] iterator end() noexcept
    {
        return _data.end();
    }
    [[nodiscard]] const_iterator begin() const noexcept
    {
        return _data.begin();
    }
    [[nodiscard]] const_iterator end() const noexcept
    {
        return _data.end();
    }
    [[nodiscard]] const_iterator cbegin() const noexcept
    {
        return _data.cbegin();
    }
    [[nodiscard]] const_iterator cend() const noexcept
    {
        return _data.cend();
    }

    // ========================================================================
    // COMPARISON OPERATORS
    // ========================================================================

    [[nodiscard]] bool operator==(const matrix<T>& other) const noexcept;
    [[nodiscard]] bool operator!=(const matrix<T>& other) const noexcept;

    // ========================================================================
    // UNARY OPERATORS
    // ========================================================================

    /// Unary negation: -A
    [[nodiscard]] matrix<T> operator-() const;

    /// Unary plus: +A (identity operation)
    [[nodiscard]] matrix<T> operator+() const;

    // ========================================================================
    // ARITHMETIC OPERATORS - Matrix-Matrix
    // ========================================================================

    [[nodiscard]] matrix<T> operator+(const matrix<T>& other) const;
    [[nodiscard]] matrix<T> operator-(const matrix<T>& other) const;
    [[nodiscard]] matrix<T> operator*(const matrix<T>& other) const;

    matrix<T> operator+=(const matrix<T>& other);
    matrix<T> operator-=(const matrix<T>& other);
    matrix<T> operator*=(const matrix<T>& other);

    /// Element-wise (Hadamard) product
    [[nodiscard]] matrix<T> hadamard(const matrix<T>& other) const;

    // ========================================================================
    // ARITHMETIC OPERATORS - Matrix-Scalar
    // ========================================================================

    [[nodiscard]] matrix<T> operator+(const T& scalar) const;
    [[nodiscard]] matrix<T> operator-(const T& scalar) const;
    [[nodiscard]] matrix<T> operator*(const T& scalar) const;
    [[nodiscard]] matrix<T> operator/(const T& scalar) const;

    matrix<T> operator+=(const T& scalar);
    matrix<T> operator-=(const T& scalar);
    matrix<T> operator*=(const T& scalar);
    matrix<T> operator/=(const T& scalar);

    // ========================================================================
    // ARITHMETIC OPERATORS - Scalar-Matrix (friend functions)
    // ========================================================================

    [[nodiscard]] friend matrix<T> operator+(const T& scalar, const matrix<T>& m)
    {
        return m + scalar;
    }

    [[nodiscard]] friend matrix<T> operator-(const T& scalar, const matrix<T>& m)
    {
        return m - scalar;
    }

    [[nodiscard]] friend matrix<T> operator*(const T& scalar, const matrix<T>& m)
    {
        return m * scalar;
    }

    [[nodiscard]] friend matrix<T> operator/(const T& scalar, const matrix<T>& m)
    {
        return m / scalar;
    }

    // ========================================================================
    // BASIC MATRIX OPERATIONS
    // ========================================================================

    /// Compute matrix transpose
    [[nodiscard]] matrix<T> transpose() const;

    /// Compute matrix trace (sum of diagonal elements)
    [[nodiscard]] T trace() const;

    /// Compute matrix determinant
    [[nodiscard]] double determinant() const;

    /// Compute matrix inverse
    [[nodiscard]] matrix<double> inverse() const;

    /// Compute matrix raised to integer power
    [[nodiscard]] matrix<T> pow(const int& power) const;

    /// Compute matrix exponential using Taylor series
    [[nodiscard]] matrix<double> exponential_pow(int max_iter = 30) const;

    /// Apply function to each element
    [[nodiscard]] matrix<T> apply(T (*func)(T)) const;

    /// Clamp elements to [min, max]
    [[nodiscard]] matrix<T> clamp(const T& min, const T& max) const;

    // ========================================================================
    // ADVANCED MATRIX OPERATIONS
    // ========================================================================

    /// Compute cofactor matrix
    [[nodiscard]] matrix<T> cofactor() const;

    /// Compute minor by removing specified row and column
    [[nodiscard]] matrix<T> minor(size_t row, size_t col) const;

    /// Compute adjoint (adjugate) matrix
    [[nodiscard]] matrix<T> adjoint() const;

    /// Perform Gaussian elimination to row echelon form
    [[nodiscard]] matrix<double> gaussian_elimination() const;

    /// Compute p-norm of the matrix
    [[nodiscard]] double norm(int p = 2) const;

    /// Compute matrix rank
    [[nodiscard]] size_t rank() const;

    /// Solve linear system Ax = b using Gaussian elimination
    [[nodiscard]] matrix<double> solve(const matrix<double>& b) const;

    // ========================================================================
    // DECOMPOSITIONS
    // ========================================================================

    /// Compute LU decomposition: A = LU
    [[nodiscard]] std::pair<matrix<double>, matrix<double>> LU_decomposition() const;

    /// Compute QR decomposition: A = QR (Gram-Schmidt)
    [[nodiscard]] std::pair<matrix<double>, matrix<double>> QR_decomposition() const;

    /// Compute Singular Value Decomposition: A = UΣV^T
    [[nodiscard]] std::tuple<matrix<double>, matrix<double>, matrix<double>> SVD() const;

    /// Compute eigenvalues using QR algorithm
    [[nodiscard]] matrix<double> eigenvalues(int max_iter = 100) const;

    /// Compute eigenvectors using QR algorithm
    [[nodiscard]] matrix<double> eigenvectors(int max_iter = 100) const;

    // ========================================================================
    // MATRIX PROPERTIES
    // ========================================================================

    /// Check if matrix is symmetric (A == A^T)
    [[nodiscard]] bool is_symmetric() const;

    /// Check if matrix is diagonal
    [[nodiscard]] bool is_diagonal() const;

    /// Check if matrix is upper triangular
    [[nodiscard]] bool is_upper_triangular() const;

    /// Check if matrix is lower triangular
    [[nodiscard]] bool is_lower_triangular() const;

    /// Check if matrix is orthogonal (A^T * A == I)
    [[nodiscard]] bool is_orthogonal(double tolerance = 1e-10) const;

    /// Check if matrix is singular (non-invertible)
    [[nodiscard]] bool is_singular(double tolerance = 1e-10) const;

    /// Check if matrix is idempotent (A^2 == A)
    [[nodiscard]] bool is_idempotent(double tolerance = 1e-10) const;

    /// Check if matrix is nilpotent (A^k == 0 for some k)
    [[nodiscard]] bool is_nilpotent(size_t k, double tolerance = 1e-10) const;

    /// Check if matrix is involutory (A^2 == I)
    [[nodiscard]] bool is_involutory(double tolerance = 1e-10) const;

    /// Check if matrix is positive definite
    [[nodiscard]] bool is_positive_definite() const;

    /// Check if matrix is negative definite
    [[nodiscard]] bool is_negative_definite() const;

    // ========================================================================
    // SUBMATRIX & DIAGONAL OPERATIONS
    // ========================================================================

    /// Extract submatrix from (x1,y1) to (x2,y2)
    [[nodiscard]] matrix<T> submatrix(size_t top_corner_x, size_t top_corner_y, size_t bottom_corner_x,
                                      size_t bottom_corner_y) const;

    /// Set values of submatrix starting at (x,y)
    void set_submatrix(size_t top_corner_x, size_t top_corner_y, const matrix<T>& submatrix);

    /// Get k-th diagonal as vector (k=0 is main diagonal)
    [[nodiscard]] std::vector<T> diagonal(int k = 0) const;

    /// Get k-th anti-diagonal as vector
    [[nodiscard]] std::vector<T> anti_diagonal(int k = 0) const;

    /// Set k-th diagonal from vector
    void set_diagonal(const std::vector<T>& diag, int k = 0);

    /// Set k-th anti-diagonal from vector
    void set_anti_diagonal(const std::vector<T>& anti_diag, int k = 0);

    // ========================================================================
    // MATRIX MANIPULATION
    // ========================================================================

    /// Swap two rows in-place
    void swap_rows(size_t row1, size_t row2);

    /// Swap two columns in-place
    void swap_cols(size_t col1, size_t col2);

    /// Resize matrix, preserving existing elements where possible
    void resize(size_t rows, size_t cols);

    // ========================================================================
    // TYPE CONVERSION
    // ========================================================================

    /// Cast matrix to different element type
    template <typename U>
    [[nodiscard]] explicit operator matrix<U>() const
    {
        matrix<U> result(_rows, _cols);
        for (size_t i = 0; i < _rows; ++i)
        {
            for (size_t j = 0; j < _cols; ++j)
            {
                result(i, j) = static_cast<U>((*this)(i, j));
            }
        }
        return result;
    }

    // ========================================================================
    // I/O OPERATIONS
    // ========================================================================

    /// Output matrix to stream
    template <typename U>
    friend std::ostream& operator<<(std::ostream& os, const matrix<U>& m);

    /// Input matrix from stream
    template <typename U>
    friend std::istream& operator>>(std::istream& is, matrix<U>& m);

private:
    // ========================================================================
    // PRIVATE MEMBERS
    // ========================================================================

    size_t _rows = 0;
    size_t _cols = 0;
    std::vector<T> _data;

    // ========================================================================
    // PRIVATE HELPER METHODS
    // ========================================================================

    /// Convert 2D index to 1D index in row-major order
    [[nodiscard]] size_t _index(size_t row, size_t col) const noexcept
    {
        return row * _cols + col;
    }

    /// Flatten 2D vector to 1D vector
    std::vector<T> _to_row_vector(const std::vector<std::vector<T>>& init) const
    {
        std::vector<T> row_vector;
        for (const auto& row : init)
        {
            row_vector.insert(row_vector.end(), row.begin(), row.end());
        }
        return row_vector;
    }
};

// Include implementations
#include "impl/matrix_core.tpp"
