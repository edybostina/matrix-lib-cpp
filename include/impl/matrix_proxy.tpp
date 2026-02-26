#pragma once

// ============================================================================
// ROW_PROXY
// ============================================================================

template <typename T>
inline row_proxy<T>::row_proxy(matrix<T>& mat, size_t row_idx) : _mat(mat), _row_idx(row_idx)
{
}

template <typename T>
inline T& row_proxy<T>::operator[](size_t col)
{
    return _mat(_row_idx, col);
}

template <typename T>
inline const T& row_proxy<T>::operator[](size_t col) const
{
    return _mat(_row_idx, col);
}

template <typename T>
inline size_t row_proxy<T>::size() const noexcept
{
    return _mat.cols();
}

template <typename T>
inline row_proxy<T>& row_proxy<T>::operator=(const std::vector<T>& vec)
{
    if (vec.size() != _mat.cols())
    {
        throw std::invalid_argument("Vector size doesn't match row length");
    }
    for (size_t j = 0; j < vec.size(); ++j)
    {
        _mat(_row_idx, j) = vec[j];
    }
    return *this;
}

template <typename T>
inline row_proxy<T>& row_proxy<T>::operator=(const row_proxy& other)
{
    if (this == &other)
        return *this;

    if (size() != other.size())
    {
        throw std::invalid_argument("Row sizes don't match");
    }
    for (size_t j = 0; j < size(); ++j)
    {
        (*this)[j] = other[j];
    }
    return *this;
}

template <typename T>
inline row_proxy<T>& row_proxy<T>::operator+=(const row_proxy& other)
{
    if (size() != other.size())
    {
        throw std::invalid_argument("Row sizes don't match");
    }
    for (size_t j = 0; j < size(); ++j)
    {
        (*this)[j] += other[j];
    }
    return *this;
}

template <typename T>
inline row_proxy<T>& row_proxy<T>::operator+=(const std::vector<T>& vec)
{
    if (vec.size() != size())
    {
        throw std::invalid_argument("Vector size doesn't match row length");
    }
    for (size_t j = 0; j < size(); ++j)
    {
        (*this)[j] += vec[j];
    }
    return *this;
}

template <typename T>
inline row_proxy<T>& row_proxy<T>::operator-=(const row_proxy& other)
{
    if (size() != other.size())
    {
        throw std::invalid_argument("Row sizes don't match");
    }
    for (size_t j = 0; j < size(); ++j)
    {
        (*this)[j] -= other[j];
    }
    return *this;
}

template <typename T>
inline row_proxy<T>& row_proxy<T>::operator-=(const std::vector<T>& vec)
{
    if (vec.size() != size())
    {
        throw std::invalid_argument("Vector size doesn't match row length");
    }
    for (size_t j = 0; j < size(); ++j)
    {
        (*this)[j] -= vec[j];
    }
    return *this;
}

template <typename T>
inline row_proxy<T>& row_proxy<T>::operator*=(const T& scalar)
{
    for (size_t j = 0; j < size(); ++j)
    {
        (*this)[j] *= scalar;
    }
    return *this;
}

template <typename T>
inline row_proxy<T>& row_proxy<T>::operator/=(const T& scalar)
{
    if (scalar == T(0))
    {
        throw std::invalid_argument("Division by zero");
    }
    for (size_t j = 0; j < size(); ++j)
    {
        (*this)[j] /= scalar;
    }
    return *this;
}

template <typename T>
inline row_proxy<T>::operator std::vector<T>() const
{
    std::vector<T> result(size());
    for (size_t j = 0; j < size(); ++j)
    {
        result[j] = (*this)[j];
    }
    return result;
}

template <typename T>
inline matrix<T> row_proxy<T>::to_matrix() const
{
    matrix<T> result(1, size());
    for (size_t j = 0; j < size(); ++j)
    {
        result(0, j) = (*this)[j];
    }
    return result;
}

template <typename T>
inline void row_proxy<T>::fill(const T& value)
{
    for (size_t j = 0; j < size(); ++j)
    {
        (*this)[j] = value;
    }
}

template <typename T>
inline void row_proxy<T>::swap(row_proxy& other)
{
    if (size() != other.size())
    {
        throw std::invalid_argument("Row sizes don't match");
    }
    for (size_t j = 0; j < size(); ++j)
    {
        std::swap((*this)[j], other[j]);
    }
}

template <typename T>
inline T row_proxy<T>::sum() const
{
    T result = T(0);
    for (size_t j = 0; j < size(); ++j)
    {
        result += (*this)[j];
    }
    return result;
}

template <typename T>
inline T row_proxy<T>::product() const
{
    T result = T(1);
    for (size_t j = 0; j < size(); ++j)
    {
        result *= (*this)[j];
    }
    return result;
}

template <typename T>
inline T row_proxy<T>::min() const
{
    if (size() == 0)
    {
        throw std::runtime_error("Cannot find min of empty row");
    }
    T result = (*this)[0];
    for (size_t j = 1; j < size(); ++j)
    {
        if ((*this)[j] < result)
        {
            result = (*this)[j];
        }
    }
    return result;
}

template <typename T>
inline T row_proxy<T>::max() const
{
    if (size() == 0)
    {
        throw std::runtime_error("Cannot find max of empty row");
    }
    T result = (*this)[0];
    for (size_t j = 1; j < size(); ++j)
    {
        if ((*this)[j] > result)
        {
            result = (*this)[j];
        }
    }
    return result;
}

template <typename T>
inline T row_proxy<T>::dot(const row_proxy& other) const
{
    if (size() != other.size())
    {
        throw std::invalid_argument("Row sizes don't match for dot product");
    }
    T result = T(0);
    for (size_t j = 0; j < size(); ++j)
    {
        result += (*this)[j] * other[j];
    }
    return result;
}

template <typename T>
inline T row_proxy<T>::dot(const std::vector<T>& vec) const
{
    if (size() != vec.size())
    {
        throw std::invalid_argument("Sizes don't match for dot product");
    }
    T result = T(0);
    for (size_t j = 0; j < size(); ++j)
    {
        result += (*this)[j] * vec[j];
    }
    return result;
}

// ============================================================================
// CONST_ROW_PROXY
// ============================================================================

template <typename T>
inline const_row_proxy<T>::const_row_proxy(const matrix<T>& mat, size_t row_idx) : _mat(mat), _row_idx(row_idx)
{
}

template <typename T>
inline const T& const_row_proxy<T>::operator[](size_t col) const
{
    return _mat(_row_idx, col);
}

template <typename T>
inline size_t const_row_proxy<T>::size() const noexcept
{
    return _mat.cols();
}

template <typename T>
inline const_row_proxy<T>::operator std::vector<T>() const
{
    std::vector<T> result(size());
    for (size_t j = 0; j < size(); ++j)
    {
        result[j] = (*this)[j];
    }
    return result;
}

template <typename T>
inline matrix<T> const_row_proxy<T>::to_matrix() const
{
    matrix<T> result(1, size());
    for (size_t j = 0; j < size(); ++j)
    {
        result(0, j) = (*this)[j];
    }
    return result;
}

// ============================================================================
// COL_PROXY
// ============================================================================

template <typename T>
inline col_proxy<T>::col_proxy(matrix<T>& mat, size_t col_idx) : _mat(mat), _col_idx(col_idx)
{
}

template <typename T>
inline T& col_proxy<T>::operator[](size_t row)
{
    return _mat(row, _col_idx);
}

template <typename T>
inline const T& col_proxy<T>::operator[](size_t row) const
{
    return _mat(row, _col_idx);
}

template <typename T>
inline size_t col_proxy<T>::size() const noexcept
{
    return _mat.rows();
}

template <typename T>
inline col_proxy<T>& col_proxy<T>::operator=(const std::vector<T>& vec)
{
    if (vec.size() != _mat.rows())
    {
        throw std::invalid_argument("Vector size doesn't match column length");
    }
    for (size_t i = 0; i < vec.size(); ++i)
    {
        _mat(i, _col_idx) = vec[i];
    }
    return *this;
}

template <typename T>
inline col_proxy<T>& col_proxy<T>::operator=(const col_proxy& other)
{
    if (this == &other)
        return *this;

    if (size() != other.size())
    {
        throw std::invalid_argument("Column sizes don't match");
    }
    for (size_t i = 0; i < size(); ++i)
    {
        (*this)[i] = other[i];
    }
    return *this;
}

template <typename T>
inline col_proxy<T>& col_proxy<T>::operator+=(const col_proxy& other)
{
    if (size() != other.size())
    {
        throw std::invalid_argument("Column sizes don't match");
    }
    for (size_t i = 0; i < size(); ++i)
    {
        (*this)[i] += other[i];
    }
    return *this;
}

template <typename T>
inline col_proxy<T>& col_proxy<T>::operator+=(const std::vector<T>& vec)
{
    if (vec.size() != size())
    {
        throw std::invalid_argument("Vector size doesn't match column length");
    }
    for (size_t i = 0; i < size(); ++i)
    {
        (*this)[i] += vec[i];
    }
    return *this;
}

template <typename T>
inline col_proxy<T>& col_proxy<T>::operator-=(const col_proxy& other)
{
    if (size() != other.size())
    {
        throw std::invalid_argument("Column sizes don't match");
    }
    for (size_t i = 0; i < size(); ++i)
    {
        (*this)[i] -= other[i];
    }
    return *this;
}

template <typename T>
inline col_proxy<T>& col_proxy<T>::operator-=(const std::vector<T>& vec)
{
    if (vec.size() != size())
    {
        throw std::invalid_argument("Vector size doesn't match column length");
    }
    for (size_t i = 0; i < size(); ++i)
    {
        (*this)[i] -= vec[i];
    }
    return *this;
}

template <typename T>
inline col_proxy<T>& col_proxy<T>::operator*=(const T& scalar)
{
    for (size_t i = 0; i < size(); ++i)
    {
        (*this)[i] *= scalar;
    }
    return *this;
}

template <typename T>
inline col_proxy<T>& col_proxy<T>::operator/=(const T& scalar)
{
    if (scalar == T(0))
    {
        throw std::invalid_argument("Division by zero");
    }
    for (size_t i = 0; i < size(); ++i)
    {
        (*this)[i] /= scalar;
    }
    return *this;
}

template <typename T>
inline col_proxy<T>::operator std::vector<T>() const
{
    std::vector<T> result(size());
    for (size_t i = 0; i < size(); ++i)
    {
        result[i] = (*this)[i];
    }
    return result;
}

template <typename T>
inline matrix<T> col_proxy<T>::to_matrix() const
{
    matrix<T> result(size(), 1);
    for (size_t i = 0; i < size(); ++i)
    {
        result(i, 0) = (*this)[i];
    }
    return result;
}

template <typename T>
inline void col_proxy<T>::fill(const T& value)
{
    for (size_t i = 0; i < size(); ++i)
    {
        (*this)[i] = value;
    }
}

template <typename T>
inline void col_proxy<T>::swap(col_proxy& other)
{
    if (size() != other.size())
    {
        throw std::invalid_argument("Column sizes don't match");
    }
    for (size_t i = 0; i < size(); ++i)
    {
        std::swap((*this)[i], other[i]);
    }
}

template <typename T>
inline T col_proxy<T>::sum() const
{
    T result = T(0);
    for (size_t i = 0; i < size(); ++i)
    {
        result += (*this)[i];
    }
    return result;
}

template <typename T>
inline T col_proxy<T>::product() const
{
    T result = T(1);
    for (size_t i = 0; i < size(); ++i)
    {
        result *= (*this)[i];
    }
    return result;
}

template <typename T>
inline T col_proxy<T>::min() const
{
    if (size() == 0)
    {
        throw std::runtime_error("Cannot find min of empty column");
    }
    T result = (*this)[0];
    for (size_t i = 1; i < size(); ++i)
    {
        if ((*this)[i] < result)
        {
            result = (*this)[i];
        }
    }
    return result;
}

template <typename T>
inline T col_proxy<T>::max() const
{
    if (size() == 0)
    {
        throw std::runtime_error("Cannot find max of empty column");
    }
    T result = (*this)[0];
    for (size_t i = 1; i < size(); ++i)
    {
        if ((*this)[i] > result)
        {
            result = (*this)[i];
        }
    }
    return result;
}

template <typename T>
inline T col_proxy<T>::dot(const col_proxy& other) const
{
    if (size() != other.size())
    {
        throw std::invalid_argument("Column sizes don't match for dot product");
    }
    T result = T(0);
    for (size_t i = 0; i < size(); ++i)
    {
        result += (*this)[i] * other[i];
    }
    return result;
}

template <typename T>
inline T col_proxy<T>::dot(const std::vector<T>& vec) const
{
    if (size() != vec.size())
    {
        throw std::invalid_argument("Sizes don't match for dot product");
    }
    T result = T(0);
    for (size_t i = 0; i < size(); ++i)
    {
        result += (*this)[i] * vec[i];
    }
    return result;
}

// ============================================================================
// CONST_COL_PROXY
// ============================================================================

template <typename T>
inline const_col_proxy<T>::const_col_proxy(const matrix<T>& mat, size_t col_idx) : _mat(mat), _col_idx(col_idx)
{
}

template <typename T>
inline const T& const_col_proxy<T>::operator[](size_t row) const
{
    return _mat(row, _col_idx);
}

template <typename T>
inline size_t const_col_proxy<T>::size() const noexcept
{
    return _mat.rows();
}

template <typename T>
inline const_col_proxy<T>::operator std::vector<T>() const
{
    std::vector<T> result(size());
    for (size_t i = 0; i < size(); ++i)
    {
        result[i] = (*this)[i];
    }
    return result;
}

template <typename T>
inline matrix<T> const_col_proxy<T>::to_matrix() const
{
    matrix<T> result(size(), 1);
    for (size_t i = 0; i < size(); ++i)
    {
        result(i, 0) = (*this)[i];
    }
    return result;
}
