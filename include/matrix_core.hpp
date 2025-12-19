#pragma once

#include "matrix_fwd.hpp"

template <typename T>
class matrix
{
private:
    size_t _rows = 0;
    size_t _cols = 0;
    std::vector<T> _data;

    [[nodiscard]] size_t _index(size_t row, size_t col) const noexcept
    {
        return row * _cols + col;
    }

    std::vector<T> _to_row_vector(const std::vector<std::vector<T>>& init) const
    {
        std::vector<T> row_vector;
        for (const auto& row : init)
        {
            row_vector.insert(row_vector.end(), row.begin(), row.end());
        }
        return row_vector;
    }

public:
    constexpr matrix() noexcept = default;

    explicit matrix(size_t rows, size_t cols) : _rows(rows), _cols(cols), _data(rows * cols, T(0))
    {
    }

    matrix(const std::vector<std::vector<T>>& init)
        : _rows(init.size()), _cols(init.empty() ? 0 : init[0].size()), _data(_to_row_vector(init))
    {
    }

    matrix(std::initializer_list<std::initializer_list<T>> init)
    {
        _rows = init.size();
        _cols = init.begin()->size();
        _data.reserve(_rows * _cols);
        for (const auto& row : init)
        {
            if (row.size() != (size_t)_cols)
            {
                throw std::invalid_argument("All rows must have the same number of columns");
            }
            _data.insert(_data.end(), row.begin(), row.end());
        }
    }

    // Rule of Five
    matrix(const matrix&) = default;
    matrix(matrix&&) noexcept = default;
    matrix& operator=(const matrix&) = default;
    matrix& operator=(matrix&&) noexcept = default;
    ~matrix() = default;

    [[nodiscard]] constexpr size_t rows() const noexcept
    {
        return _rows;
    }
    [[nodiscard]] constexpr size_t cols() const noexcept
    {
        return _cols;
    }
    [[nodiscard]] constexpr size_t size() const noexcept
    {
        return _rows * _cols;
    }

    // Access to raw data pointer
    [[nodiscard]] const T* data_ptr() const
    {
        return _data.data();
    }
    [[nodiscard]] T* data_ptr()
    {
        return _data.data();
    }

    // Access operator

    std::vector<T>& operator()(size_t index);
    const std::vector<T>& operator()(size_t index) const;

    T& operator()(size_t row, size_t col);
    const T& operator()(size_t row, size_t col) const;

    // STL-compatible iterators
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;
    using value_type = T;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;

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

    // Access operator for row and column

    [[nodiscard]] matrix<T> row(size_t index) const;
    [[nodiscard]] matrix<T> col(size_t index) const;

    // Casting

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

    // Factory methods

    [[nodiscard]] static matrix<T> zeros(size_t rows, size_t cols);
    [[nodiscard]] static matrix<T> ones(size_t rows, size_t cols);
    [[nodiscard]] static matrix<T> eye(size_t rows, size_t cols);
    [[nodiscard]] static matrix<T> random(size_t rows, size_t cols, T min, T max);

    // File I/O

    template <typename U>
    friend std::ostream& operator<<(std::ostream& os, const matrix<U>& m);

    template <typename U>
    friend std::istream& operator>>(std::istream& is, matrix<U>& m);

    // Arithmetic operators
    // Matrix x Matrix

    [[nodiscard]] matrix<T> operator+(const matrix<T>& other) const;
    [[nodiscard]] matrix<T> operator-(const matrix<T>& other) const;
    [[nodiscard]] matrix<T> operator*(const matrix<T>& other) const;

    matrix<T> operator+=(const matrix<T>& other);
    matrix<T> operator-=(const matrix<T>& other);
    matrix<T> operator*=(const matrix<T>& other);

    [[nodiscard]] bool operator==(const matrix<T>& other) const noexcept;
    [[nodiscard]] bool operator!=(const matrix<T>& other) const noexcept;

    [[nodiscard]] matrix<T> hadamard(const matrix<T>& other) const;

    // Matrix x Scalar

    [[nodiscard]] matrix<T> operator+(const T& scalar) const;
    [[nodiscard]] matrix<T> operator-(const T& scalar) const;
    [[nodiscard]] matrix<T> operator*(const T& scalar) const;
    [[nodiscard]] matrix<T> operator/(const T& scalar) const;

    matrix<T> operator+=(const T& scalar);
    matrix<T> operator-=(const T& scalar);
    matrix<T> operator*=(const T& scalar);
    matrix<T> operator/=(const T& scalar);

    // Scalar x Matrix
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
    friend matrix<T> operator+=(const T& scalar, matrix<T>& m)
    {
        return m += scalar;
    }
    friend matrix<T> operator-=(const T& scalar, matrix<T>& m)
    {
        return m -= scalar;
    }
    friend matrix<T> operator*=(const T& scalar, matrix<T>& m)
    {
        return m *= scalar;
    }
    friend matrix<T> operator/=(const T& scalar, matrix<T>& m)
    {
        return m /= scalar;
    }

    // Matrix functions

    [[nodiscard]] T trace() const;
    [[nodiscard]] matrix<T> transpose() const;
    [[nodiscard]] matrix<T> cofactor() const;
    [[nodiscard]] matrix<T> minor(size_t row, size_t col) const;
    [[nodiscard]] matrix<T> adjoint() const;
    [[nodiscard]] matrix<double> inverse() const;
    [[nodiscard]] matrix<double> gaussian_elimination() const;

    [[nodiscard]] double determinant() const;
    [[nodiscard]] double norm(int p = 2) const;
    [[nodiscard]] size_t rank() const;

    [[nodiscard]] bool is_square() const noexcept
    {
        return _rows == _cols;
    }
    [[nodiscard]] bool is_symmetric() const;
    [[nodiscard]] bool is_diagonal() const;
    [[nodiscard]] bool is_upper_triangular() const;
    [[nodiscard]] bool is_lower_triangular() const;

    [[nodiscard]] matrix<T> pow(const int& power) const;
    [[nodiscard]] matrix<double> exponential_pow(int max_iter = 30) const;

    void swapRows(size_t row1, size_t row2);
    void swapCols(size_t col1, size_t col2);
    void resize(size_t rows, size_t cols);

    [[nodiscard]] matrix<T> submatrix(size_t top_corner_x, size_t top_corner_y, size_t bottom_corner_x,
                                      size_t bottom_corner_y) const;
    void set_submatrix(size_t top_corner_x, size_t top_corner_y, const matrix<T>& submatrix);

    [[nodiscard]] std::vector<T> diagonal(int k = 0) const;
    [[nodiscard]] std::vector<T> anti_diagonal(int k = 0) const;
    void set_diagonal(const std::vector<T>& diag, int k = 0);
    void set_anti_diagonal(const std::vector<T>& anti_diag, int k = 0);

    // Numeric methods

    [[nodiscard]] std::pair<matrix<double>, matrix<double>> LU_decomposition() const;
    [[nodiscard]] std::pair<matrix<double>, matrix<double>> QR_decomposition() const;
    [[nodiscard]] matrix<double> eigenvalues(int max_iter = 100) const;
    [[nodiscard]] matrix<double> eigenvectors(int max_iter = 100) const;
};

// For header-only mode, include implementations
#ifndef MATRIX_EXPLICIT_INSTANTIATION
#include "impl/matrix_core.tpp"
#endif
