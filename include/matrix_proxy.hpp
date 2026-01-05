#pragma once

#include "matrix_fwd.hpp"

// Forward declarations
template <typename T>
class matrix;

/**
 * @brief Proxy class for row access and manipulation
 *
 * Provides zero-copy row access with support for:
 * - Element access via operator[]
 * - In-place modifications (+=, -=, *=, /=)
 * - Assignment from vectors and other rows
 * - Iterator support for range-based loops
 * - Conversion to vector or matrix
 * - Utility methods (fill, swap, sum, product, min, max, dot)
 */
template <typename T>
class row_proxy
{
public:
    // Constructors
    row_proxy(matrix<T>& mat, size_t row_idx);

    // Element access
    T& operator[](size_t col);
    const T& operator[](size_t col) const;
    [[nodiscard]] size_t size() const noexcept;

    // Assignment operators
    row_proxy& operator=(const std::vector<T>& vec);
    row_proxy& operator=(const row_proxy& other);

    // Arithmetic operators
    row_proxy& operator+=(const row_proxy& other);
    row_proxy& operator+=(const std::vector<T>& vec);
    row_proxy& operator-=(const row_proxy& other);
    row_proxy& operator-=(const std::vector<T>& vec);
    row_proxy& operator*=(const T& scalar);
    row_proxy& operator/=(const T& scalar);

    // Conversion operators
    operator std::vector<T>() const;
    [[nodiscard]] matrix<T> to_matrix() const;

    // Utility methods
    void fill(const T& value);
    void swap(row_proxy& other);
    [[nodiscard]] T sum() const;
    [[nodiscard]] T product() const;
    [[nodiscard]] T min() const;
    [[nodiscard]] T max() const;
    [[nodiscard]] T dot(const row_proxy& other) const;
    [[nodiscard]] T dot(const std::vector<T>& vec) const;

    // Iterator support
    class iterator
    {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using reference = T&;

        iterator(row_proxy& row, size_t pos) : _row(row), _pos(pos)
        {
        }

        reference operator*()
        {
            return _row[_pos];
        }
        pointer operator->()
        {
            return &_row[_pos];
        }

        iterator& operator++()
        {
            ++_pos;
            return *this;
        }
        iterator operator++(int)
        {
            iterator tmp = *this;
            ++_pos;
            return tmp;
        }
        iterator& operator--()
        {
            --_pos;
            return *this;
        }
        iterator operator--(int)
        {
            iterator tmp = *this;
            --_pos;
            return tmp;
        }

        iterator& operator+=(difference_type n)
        {
            _pos += n;
            return *this;
        }
        iterator& operator-=(difference_type n)
        {
            _pos -= n;
            return *this;
        }

        iterator operator+(difference_type n) const
        {
            return iterator(_row, _pos + n);
        }
        iterator operator-(difference_type n) const
        {
            return iterator(_row, _pos - n);
        }
        difference_type operator-(const iterator& other) const
        {
            return _pos - other._pos;
        }

        bool operator==(const iterator& other) const
        {
            return _pos == other._pos;
        }
        bool operator!=(const iterator& other) const
        {
            return _pos != other._pos;
        }
        bool operator<(const iterator& other) const
        {
            return _pos < other._pos;
        }
        bool operator>(const iterator& other) const
        {
            return _pos > other._pos;
        }
        bool operator<=(const iterator& other) const
        {
            return _pos <= other._pos;
        }
        bool operator>=(const iterator& other) const
        {
            return _pos >= other._pos;
        }

    private:
        row_proxy& _row;
        size_t _pos;
    };

    class const_iterator
    {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = const T*;
        using reference = const T&;

        const_iterator(const row_proxy& row, size_t pos) : _row(row), _pos(pos)
        {
        }

        reference operator*() const
        {
            return _row[_pos];
        }
        pointer operator->() const
        {
            return &_row[_pos];
        }

        const_iterator& operator++()
        {
            ++_pos;
            return *this;
        }
        const_iterator operator++(int)
        {
            const_iterator tmp = *this;
            ++_pos;
            return tmp;
        }

        bool operator==(const const_iterator& other) const
        {
            return _pos == other._pos;
        }
        bool operator!=(const const_iterator& other) const
        {
            return _pos != other._pos;
        }

    private:
        const row_proxy& _row;
        size_t _pos;
    };

    iterator begin()
    {
        return iterator(*this, 0);
    }
    iterator end()
    {
        return iterator(*this, size());
    }
    const_iterator begin() const
    {
        return const_iterator(*this, 0);
    }
    const_iterator end() const
    {
        return const_iterator(*this, size());
    }
    const_iterator cbegin() const
    {
        return const_iterator(*this, 0);
    }
    const_iterator cend() const
    {
        return const_iterator(*this, size());
    }

private:
    matrix<T>& _mat;
    size_t _row_idx;
};

/**
 * @brief Const proxy class for read-only row access
 */
template <typename T>
class const_row_proxy
{
public:
    const_row_proxy(const matrix<T>& mat, size_t row_idx);

    const T& operator[](size_t col) const;
    [[nodiscard]] size_t size() const noexcept;

    operator std::vector<T>() const;
    [[nodiscard]] matrix<T> to_matrix() const;

    class const_iterator
    {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = const T*;
        using reference = const T&;

        const_iterator(const const_row_proxy& row, size_t pos) : _row(row), _pos(pos)
        {
        }

        reference operator*() const
        {
            return _row[_pos];
        }
        pointer operator->() const
        {
            return &_row[_pos];
        }

        const_iterator& operator++()
        {
            ++_pos;
            return *this;
        }
        const_iterator operator++(int)
        {
            const_iterator tmp = *this;
            ++_pos;
            return tmp;
        }

        bool operator==(const const_iterator& other) const
        {
            return _pos == other._pos;
        }
        bool operator!=(const const_iterator& other) const
        {
            return _pos != other._pos;
        }

    private:
        const const_row_proxy& _row;
        size_t _pos;
    };

    const_iterator begin() const
    {
        return const_iterator(*this, 0);
    }
    const_iterator end() const
    {
        return const_iterator(*this, size());
    }
    const_iterator cbegin() const
    {
        return const_iterator(*this, 0);
    }
    const_iterator cend() const
    {
        return const_iterator(*this, size());
    }

private:
    const matrix<T>& _mat;
    size_t _row_idx;
};

/**
 * @brief Proxy class for column access and manipulation
 */
template <typename T>
class col_proxy
{
public:
    col_proxy(matrix<T>& mat, size_t col_idx);

    T& operator[](size_t row);
    const T& operator[](size_t row) const;
    [[nodiscard]] size_t size() const noexcept;

    col_proxy& operator=(const std::vector<T>& vec);
    col_proxy& operator=(const col_proxy& other);

    col_proxy& operator+=(const col_proxy& other);
    col_proxy& operator+=(const std::vector<T>& vec);
    col_proxy& operator-=(const col_proxy& other);
    col_proxy& operator-=(const std::vector<T>& vec);
    col_proxy& operator*=(const T& scalar);
    col_proxy& operator/=(const T& scalar);

    operator std::vector<T>() const;
    [[nodiscard]] matrix<T> to_matrix() const;

    // Utility methods
    void fill(const T& value);
    void swap(col_proxy& other);
    [[nodiscard]] T sum() const;
    [[nodiscard]] T product() const;
    [[nodiscard]] T min() const;
    [[nodiscard]] T max() const;
    [[nodiscard]] T dot(const col_proxy& other) const;
    [[nodiscard]] T dot(const std::vector<T>& vec) const;

    class iterator
    {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using reference = T&;

        iterator(col_proxy& col, size_t pos) : _col(col), _pos(pos)
        {
        }

        reference operator*()
        {
            return _col[_pos];
        }
        pointer operator->()
        {
            return &_col[_pos];
        }

        iterator& operator++()
        {
            ++_pos;
            return *this;
        }
        iterator operator++(int)
        {
            iterator tmp = *this;
            ++_pos;
            return tmp;
        }

        bool operator==(const iterator& other) const
        {
            return _pos == other._pos;
        }
        bool operator!=(const iterator& other) const
        {
            return _pos != other._pos;
        }

    private:
        col_proxy& _col;
        size_t _pos;
    };

    class const_iterator
    {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = const T*;
        using reference = const T&;

        const_iterator(const col_proxy& col, size_t pos) : _col(col), _pos(pos)
        {
        }

        reference operator*() const
        {
            return _col[_pos];
        }
        pointer operator->() const
        {
            return &_col[_pos];
        }

        const_iterator& operator++()
        {
            ++_pos;
            return *this;
        }
        const_iterator operator++(int)
        {
            const_iterator tmp = *this;
            ++_pos;
            return tmp;
        }

        bool operator==(const const_iterator& other) const
        {
            return _pos == other._pos;
        }
        bool operator!=(const const_iterator& other) const
        {
            return _pos != other._pos;
        }

    private:
        const col_proxy& _col;
        size_t _pos;
    };

    iterator begin()
    {
        return iterator(*this, 0);
    }
    iterator end()
    {
        return iterator(*this, size());
    }
    const_iterator begin() const
    {
        return const_iterator(*this, 0);
    }
    const_iterator end() const
    {
        return const_iterator(*this, size());
    }
    const_iterator cbegin() const
    {
        return const_iterator(*this, 0);
    }
    const_iterator cend() const
    {
        return const_iterator(*this, size());
    }

private:
    matrix<T>& _mat;
    size_t _col_idx;
};

/**
 * @brief Const proxy class for read-only column access
 */
template <typename T>
class const_col_proxy
{
public:
    const_col_proxy(const matrix<T>& mat, size_t col_idx);

    const T& operator[](size_t row) const;
    [[nodiscard]] size_t size() const noexcept;

    operator std::vector<T>() const;
    [[nodiscard]] matrix<T> to_matrix() const;

    class const_iterator
    {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = const T*;
        using reference = const T&;

        const_iterator(const const_col_proxy& col, size_t pos) : _col(col), _pos(pos)
        {
        }

        reference operator*() const
        {
            return _col[_pos];
        }
        pointer operator->() const
        {
            return &_col[_pos];
        }

        const_iterator& operator++()
        {
            ++_pos;
            return *this;
        }
        const_iterator operator++(int)
        {
            const_iterator tmp = *this;
            ++_pos;
            return tmp;
        }

        bool operator==(const const_iterator& other) const
        {
            return _pos == other._pos;
        }
        bool operator!=(const const_iterator& other) const
        {
            return _pos != other._pos;
        }

    private:
        const const_col_proxy& _col;
        size_t _pos;
    };

    const_iterator begin() const
    {
        return const_iterator(*this, 0);
    }
    const_iterator end() const
    {
        return const_iterator(*this, size());
    }
    const_iterator cbegin() const
    {
        return const_iterator(*this, 0);
    }
    const_iterator cend() const
    {
        return const_iterator(*this, size());
    }

private:
    const matrix<T>& _mat;
    size_t _col_idx;
};

// Include template implementations
#include "impl/matrix_proxy.tpp"
