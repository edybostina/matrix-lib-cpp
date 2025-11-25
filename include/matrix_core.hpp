#pragma once

#include "matrix_fwd.hpp"

template <typename T>
class matrix
{
private:
    int _rows = 0;
    int _cols = 0;
    std::vector<T> _data;

    int _index(int row, int col) const { return row * _cols + col; }

    std::vector<T> _to_row_vector(const std::vector<std::vector<T>> &init) const
    {
        std::vector<T> row_vector;
        for (const auto &row : init)
        {
            row_vector.insert(row_vector.end(), row.begin(), row.end());
        }
        return row_vector;
    }

public:
    constexpr matrix() noexcept = default;

    explicit matrix(int rows, int cols)
        : _rows(rows), _cols(cols), _data(rows * cols, T(0)) {}

    matrix(const std::vector<std::vector<T>> &init)
        : _rows(init.size()), _cols(init.empty() ? 0 : init[0].size()), _data(_to_row_vector(init)) {}

    matrix(std::initializer_list<std::initializer_list<T>> init)
    {
        _rows = init.size();
        _cols = init.begin()->size();
        _data.reserve(_rows * _cols);
        for (const auto &row : init)
        {
            if (row.size() != (size_t)_cols)
            {
                throw std::invalid_argument("All rows must have the same number of columns");
            }
            _data.insert(_data.end(), row.begin(), row.end());
        }
    }

    // Rule of Five
    matrix(const matrix &) = default;
    matrix(matrix &&) noexcept = default;
    matrix &operator=(const matrix &) = default;
    matrix &operator=(matrix &&) noexcept = default;
    ~matrix() = default;

    [[nodiscard]] constexpr int rows() const noexcept { return _rows; }
    [[nodiscard]] constexpr int cols() const noexcept { return _cols; }
    [[nodiscard]] constexpr int size() const noexcept { return _rows * _cols; }

    // Access to raw data for SIMD operations
    [[nodiscard]] const T *data_ptr() const { return _data.data(); }
    [[nodiscard]] T *data_ptr() { return _data.data(); }

    // Access operator

    std::vector<T> &operator()(int index);
    const std::vector<T> &operator()(int index) const;

    T &operator()(int row, int col);
    const T &operator()(int row, int col) const;

    auto begin() { return _data.begin(); }
    auto end() { return _data.end(); }
    auto begin() const { return _data.begin(); }
    auto end() const { return _data.end(); }

    // Access operator for row and column

    [[nodiscard]] matrix<T> row(int index) const;
    [[nodiscard]] matrix<T> col(int index) const;

    // Casting

    template <typename U>
    [[nodiscard]] explicit operator matrix<U>() const
    {
        matrix<U> result(_rows, _cols);
        for (int i = 0; i < _rows; ++i)
        {
            for (int j = 0; j < _cols; ++j)
            {
                result(i, j) = static_cast<U>((*this)(i, j));
            }
        }
        return result;
    }

    // Factory methods

    [[nodiscard]] static matrix<T> zeros(int rows, int cols);
    [[nodiscard]] static matrix<T> ones(int rows, int cols);
    [[nodiscard]] static matrix<T> eye(int rows, int cols);
    [[nodiscard]] static matrix<T> random(int rows, int cols, T min, T max);

    // File I/O

    template <typename U>
    friend std::ostream &operator<<(std::ostream &os, const matrix<U> &m);

    template <typename U>
    friend std::istream &operator>>(std::istream &is, matrix<U> &m);

    // Arithmetic operators
    // Matrix x Matrix

    [[nodiscard]] matrix<T> operator+(const matrix<T> &other) const;
    [[nodiscard]] matrix<T> operator-(const matrix<T> &other) const;
    [[nodiscard]] matrix<T> operator*(const matrix<T> &other) const;

    matrix<T> operator+=(const matrix<T> &other);
    matrix<T> operator-=(const matrix<T> &other);
    matrix<T> operator*=(const matrix<T> &other);

    [[nodiscard]] bool operator==(const matrix<T> &other) const noexcept;
    [[nodiscard]] bool operator!=(const matrix<T> &other) const noexcept;

    [[nodiscard]] matrix<T> hadamard(const matrix<T> &other) const;

    // Matrix x Scalar

    [[nodiscard]] matrix<T> operator+(const T &scalar) const;
    [[nodiscard]] matrix<T> operator-(const T &scalar) const;
    [[nodiscard]] matrix<T> operator*(const T &scalar) const;
    [[nodiscard]] matrix<T> operator/(const T &scalar) const;

    matrix<T> operator+=(const T &scalar);
    matrix<T> operator-=(const T &scalar);
    matrix<T> operator*=(const T &scalar);
    matrix<T> operator/=(const T &scalar);

    // Scalar x Matrix
    [[nodiscard]] friend matrix<T> operator+(const T &scalar, const matrix<T> &m)
    {
        return m + scalar;
    }
    [[nodiscard]] friend matrix<T> operator-(const T &scalar, const matrix<T> &m)
    {
        return m - scalar;
    }
    [[nodiscard]] friend matrix<T> operator*(const T &scalar, const matrix<T> &m)
    {
        return m * scalar;
    }
    [[nodiscard]] friend matrix<T> operator/(const T &scalar, const matrix<T> &m)
    {
        return m / scalar;
    }
    friend matrix<T> operator+=(const T &scalar, matrix<T> &m)
    {
        return m += scalar;
    }
    friend matrix<T> operator-=(const T &scalar, matrix<T> &m)
    {
        return m -= scalar;
    }
    friend matrix<T> operator*=(const T &scalar, matrix<T> &m)
    {
        return m *= scalar;
    }
    friend matrix<T> operator/=(const T &scalar, matrix<T> &m)
    {
        return m /= scalar;
    }

    // Matrix functions

    [[nodiscard]] T trace() const;
    [[nodiscard]] matrix<T> transpose() const;
    [[nodiscard]] matrix<T> cofactor() const;
    [[nodiscard]] matrix<T> minor(int row, int col) const;
    [[nodiscard]] matrix<T> adjoint() const;
    [[nodiscard]] matrix<double> inverse() const;
    [[nodiscard]] matrix<double> gaussian_elimination() const;

    [[nodiscard]] double determinant() const;
    [[nodiscard]] double norm(int p) const;
    [[nodiscard]] int rank() const;

    [[nodiscard]] bool is_square() const noexcept { return _rows == _cols; }
    [[nodiscard]] bool is_symmetric() const;
    [[nodiscard]] bool is_diagonal() const;
    [[nodiscard]] bool is_upper_triangular() const;
    [[nodiscard]] bool is_lower_triangular() const;

    [[nodiscard]] matrix<T> pow(const int &power) const;
    [[nodiscard]] matrix<double> exponential_pow(int max_iter = 30) const;

    void swapRows(int row1, int row2);
    void swapCols(int col1, int col2);
    void resize(int rows, int cols);

    [[nodiscard]] matrix<T> submatrix(int top_corner_x, int top_corner_y, int bottom_corner_x, int bottom_corner_y) const;
    void set_submatrix(int top_corner_x, int top_corner_y, const matrix<T> &submatrix);

    [[nodiscard]] std::vector<T> diagonal(int k = 0) const;
    [[nodiscard]] std::vector<T> anti_diagonal(int k = 0) const;
    void set_diagonal(const std::vector<T> &diag, int k = 0);
    void set_anti_diagonal(const std::vector<T> &anti_diag, int k = 0);

    // Numeric methods

    [[nodiscard]] std::pair<matrix<double>, matrix<double>> LU_decomposition() const;
    [[nodiscard]] std::pair<matrix<double>, matrix<double>> QR_decomposition() const;
    [[nodiscard]] matrix<double> eigenvalues(int max_iter = 100) const;
    [[nodiscard]] matrix<double> eigenvectors(int max_iter = 100) const;
};

// matrix(index) returns the i-th row of the matrix
template <typename T>
std::vector<T> &matrix<T>::operator()(int index)
{
    if (index < 0 || index >= _rows)
    {
        throw std::out_of_range("Index out of range");
    }
    std::vector<T> &row = _data;
    row.resize(_cols);
    std::copy(_data.begin() + index * _cols, _data.begin() + (index + 1) * _cols, row.begin());
    return row;
}
// const matrix(index)
template <typename T>
const std::vector<T> &matrix<T>::operator()(int index) const
{
    if (index < 0 || index >= _rows)
    {
        throw std::out_of_range("Index out of range");
    }
    return std::vector<T>(_data.begin() + index * _cols, _data.begin() + (index + 1) * _cols);
}

// matrix(row, col) returns the element at (row, col)
template <typename T>
T &matrix<T>::operator()(int row, int col)
{
    if (row < 0 || row >= _rows || col < 0 || col >= _cols)
    {
        throw std::out_of_range("Index out of range");
    }
    return _data[_index(row, col)];
}
// const matrix(row, col)
template <typename T>
const T &matrix<T>::operator()(int row, int col) const
{
    if (row < 0 || row >= _rows || col < 0 || col >= _cols)
    {
        throw std::out_of_range("Index out of range");
    }
    return _data[_index(row, col)];
}

// matrix.row(i) returns the i-th row of the matrix
template <typename T>
matrix<T> matrix<T>::row(int index) const
{
    if (index < 0 || index >= _rows)
    {
        throw std::out_of_range("Index out of range");
    }
    matrix<T> result(1, _cols);
    for (int j = 0; j < _cols; ++j)
    {
        result(0, j) = _data[_index(index, j)];
    }
    return result;
}
// matrix.col(j) returns the j-th column of the matrix
template <typename T>
matrix<T> matrix<T>::col(int index) const
{
    if (index < 0 || index >= _cols)
    {
        throw std::out_of_range("Index out of range");
    }
    matrix<T> result(_rows, 1);
    for (int i = 0; i < _rows; ++i)
    {
        result(i, 0) = _data[_index(i, index)];
    }
    return result;
}

// Matrix full of zeros
template <typename T>
matrix<T> matrix<T>::zeros(int rows, int cols)
{
    if (rows <= 0 || cols <= 0)
    {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    return matrix<T>(std::vector<std::vector<T>>(rows, std::vector<T>(cols, 0)));
}
// Matrix full of ones
template <typename T>
matrix<T> matrix<T>::ones(int rows, int cols)
{
    if (rows <= 0 || cols <= 0)
    {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    return matrix<T>(std::vector<std::vector<T>>(rows, std::vector<T>(cols, 1)));
}
// Identity matrix
template <typename T>
matrix<T> matrix<T>::eye(int rows, int cols)
{
    if (rows <= 0 || cols <= 0)
    {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }

    matrix<T> result(rows, cols);
    for (int i = 0; i < std::min(rows, cols); ++i)
    {
        result(i, i) = 1;
    }
    return result;
}

// Random initialization
// This function uses the <random> lib to generate random numbers
// between 0 and 1
template <typename T>
matrix<T> matrix<T>::random(int rows, int cols, T min, T max)
{
    if (rows <= 0 || cols <= 0)
    {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    if (min >= max)
    {
        throw std::invalid_argument("Invalid range for random numbers");
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(min, max);
    matrix<T> result(rows, cols);
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            result(i, j) = dis(gen);
        }
    }
    return result;
}
