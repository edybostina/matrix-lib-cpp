#pragma once

// Template implementations for matrix_algorithms.hpp
// This file is included at the end of matrix_algorithms.hpp

/**
 * @brief Computes the determinant of a square matrix.
 *
 * Optimizations: Direct formulas for 1x1, 2x2, 3x3; Gaussian elimination for larger matrices.
 *
 * @return Determinant value as double
 * @throws std::invalid_argument If matrix is not square
 * @details Time O(n³) for n>3, O(1) for n≤3
 */
template <typename T>
double matrix<T>::determinant() const
{
    if (_rows != _cols)
    {
        std::ostringstream oss;
        oss << "Matrix is " << _rows << "x" << _cols << ", must be square to compute determinant";
        throw std::invalid_argument(oss.str());
    }
    if (_rows == 1)
    {
        return (*this)(0, 0);
    }
    if (_rows == 2)
    {
        return (*this)(0, 0) * (*this)(1, 1) - (*this)(0, 1) * (*this)(1, 0);
    }
    if (_rows == 3)
    {
        return (*this)(0, 0) * ((*this)(1, 1) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 1)) -
               (*this)(0, 1) * ((*this)(1, 0) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 0)) +
               (*this)(0, 2) * ((*this)(1, 0) * (*this)(2, 1) - (*this)(1, 1) * (*this)(2, 0));
    }

    double det = 1.0;
    matrix<double> temp = (matrix<double>)*this;
    for (size_t i = 0; i < _rows; ++i)
    {
        size_t pivot = i;
        for (size_t j = i + 1; j < _rows; ++j)
        {
            if (std::abs(temp(j, i)) > std::abs(temp(pivot, i)))
            {
                pivot = j;
            }
        }
        if (pivot != i)
        {
            temp.swapRows(i, pivot);
            det *= -1;
        }
        if (std::abs(temp(i, i)) < std::numeric_limits<double>::epsilon())
        {
            return 0;
        }
        det *= temp(i, i);
        for (size_t j = i + 1; j < _rows; ++j)
        {
            double factor = temp(j, i) / temp(i, i);
            for (size_t k = i + 1; k < _cols; ++k)
            {
                temp(j, k) -= factor * temp(i, k);
            }
        }
    }
    return det;
}

/**
 * @brief Computes the trace of a square matrix.
 *
 * Optimizations: Direct summation of diagonal elements.
 *
 * @return Trace value as T
 * @throws std::invalid_argument If matrix is not square
 * @details Time O(n), Space O(1)
 */
template <typename T>
T matrix<T>::trace() const
{
    if (_rows != _cols)
    {
        std::ostringstream oss;
        oss << "Matrix is " << _rows << "x" << _cols << ", must be square to compute trace";
        throw std::invalid_argument(oss.str());
    }
    T sum = 0;
    for (size_t i = 0; i < _rows; ++i)
    {
        sum += (*this)(i, i);
    }
    return sum;
}

/**
 * @brief Computes the transpose of the matrix.
 *
 * @return Transposed matrix
 * @details Time O(m*n), Space O(m*n)
 */
template <typename T>
matrix<T> matrix<T>::transpose() const
{
    matrix<T> result(_cols, _rows);
    for (size_t i = 0; i < _rows; ++i)
    {
        for (size_t j = 0; j < _cols; ++j)
        {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

/** * @brief Computes the cofactor matrix.
 *
 * @return Cofactor matrix
 * @details Time O(n^4), Space O(n^2)
 */
template <typename T>
matrix<T> matrix<T>::cofactor() const
{
    matrix<T> result(_rows, _cols);
    for (size_t i = 0; i < _rows; ++i)
    {
        for (size_t j = 0; j < _cols; ++j)
        {
            matrix<T> minor = this->minor(i, j);
            result(i, j) = ((i + j) % 2 == 0 ? 1 : -1) * minor.determinant();
        }
    }
    return result;
}

/** * @brief Computes the minor matrix by removing specified row and column.
 *
 * @param row Row index to remove
 * @param col Column index to remove
 * @return Minor matrix
 * @details Time O(n^2), Space O(n^2)
 */
template <typename T>
matrix<T> matrix<T>::minor(size_t row, size_t col) const
{
    matrix<T> result(_rows - 1, _cols - 1);
    for (size_t i = 0, r = 0; i < _rows; ++i)
    {
        if (i == row)
            continue;
        for (size_t j = 0, c = 0; j < _cols; ++j)
        {
            if (j == col)
                continue;
            result(r, c) = (*this)(i, j);
            ++c;
        }
        ++r;
    }
    return result;
}

/** * @brief Computes the adjoint (adjugate) of the matrix.
 *
 * @return Adjoint matrix
 * @throws std::invalid_argument If matrix is not square
 * @details Time O(n^4), Space O(n^2)
 */
template <typename T>
matrix<T> matrix<T>::adjoint() const
{
    if (_rows != _cols)
    {
        std::ostringstream oss;
        oss << "Matrix is " << _rows << "x" << _cols << ", must be square to compute adjoint";
        throw std::invalid_argument(oss.str());
    }
    matrix<T> result(_rows, _cols);
    result = this->cofactor().transpose();
    return result;
}

/** * @brief Computes the inverse of the matrix.
 *
 * @return Inverse matrix
 * @throws std::invalid_argument If matrix is not square or singular
 * @details Time O(n^3), Space O(n^2)
 */
template <typename T>
matrix<double> matrix<T>::inverse() const
{
    if (_rows != _cols)
    {
        std::ostringstream oss;
        oss << "Matrix is " << _rows << "x" << _cols << ", must be square to compute inverse";
        throw std::invalid_argument(oss.str());
    }
    double temp;
    matrix<double> augmented(_rows, 2 * _cols);
    matrix<double> result(_rows, _cols);
    for (size_t i = 0; i < _rows; ++i)
    {
        for (size_t j = 0; j < _cols; ++j)
        {
            augmented(i, j) = (*this)(i, j);
        }
        for (size_t j = _cols; j < 2 * _cols; ++j)
        {
            if (i == j - _cols)
            {
                augmented(i, j) = 1;
            }
            else
            {
                augmented(i, j) = 0;
            }
        }
    }
    for (size_t i = _rows - 1; i > 0; i--)
    {
        if (augmented(i - 1, 0) < augmented(i, 0))
        {
            augmented.swapRows(i - 1, i);
        }
    }

    for (size_t i = 0; i < _rows; ++i)
    {
        if (augmented(i, i) == 0)
        {
            throw std::invalid_argument("Matrix is singular and cannot be inverted");
        }
        for (size_t j = 0; j < _cols; ++j)
        {
            if (j != i)
            {
                temp = augmented(j, i) / augmented(i, i);
                for (size_t k = 0; k < 2 * _cols; ++k)
                {
                    augmented(j, k) -= augmented(i, k) * temp;
                }
            }
        }
    }
    for (size_t i = 0; i < _rows; ++i)
    {
        temp = augmented(i, i);
        for (size_t j = 0; j < 2 * _cols; ++j)
        {
            augmented(i, j) /= temp;
        }
    }
    for (size_t i = 0; i < _rows; ++i)
    {
        for (size_t j = 0; j < _cols; ++j)
        {
            result(i, j) = augmented(i, j + _cols);
        }
    }
    return result;
}

/**
 * @brief Computes the p-norm of the matrix.
 *
 * @param p Norm order (must be >= 1)
 * @return Norm value as double
 * @throws std::invalid_argument If p < 1
 * @details Time O(m*n), Space O(1)
 */
template <typename T>
double matrix<T>::norm(int p) const
{
    if (p < 1)
    {
        throw std::invalid_argument("Norm order must be greater than or equal to 1");
    }
    double norm = 0;
    for (size_t i = 0; i < _rows; ++i)
    {
        for (size_t j = 0; j < _cols; ++j)
        {
            norm += std::pow(std::abs((*this)(i, j)), p);
        }
    }
    return std::pow(norm, 1.0 / p);
}

/**
 * @brief Computes the rank of the matrix.
 *
 * Optimizations: Uses Gaussian elimination with numerical tolerance for stability.
 *
 * @return Rank as size_t
 * @details Time O(m*n*min(m,n)), Space O(m*n)
 */
template <typename T>
size_t matrix<T>::rank() const
{
    matrix<double> gaussian = this->gaussian_elimination();

    double max_val = 0.0;
    for (size_t i = 0; i < _rows; ++i)
    {
        for (size_t j = 0; j < _cols; ++j)
        {
            max_val = std::max(max_val, std::abs(gaussian(i, j)));
        }
    }

    double tolerance = std::max(1e-10 * max_val, std::numeric_limits<double>::epsilon());

    size_t rank = 0;
    for (size_t i = 0; i < _rows; ++i)
    {
        bool non_zero_row = false;
        for (size_t j = 0; j < _cols; ++j)
        {
            if (std::abs(gaussian(i, j)) > tolerance)
            {
                non_zero_row = true;
                break;
            }
        }
        if (non_zero_row)
        {
            ++rank;
        }
    }
    return rank;
}

/**
 * @brief Performs Gaussian elimination to obtain row echelon form.
 *
 * Optimizations: Partial pivoting for numerical stability.
 *
 * @return Matrix in row echelon form
 * @throws std::invalid_argument If matrix has zero dimensions
 * @details Time O(m*n*min(m,n)), Space O(m*n)
 */
template <typename T>
matrix<double> matrix<T>::gaussian_elimination() const
{
    if (_rows == 0 || _cols == 0)
    {
        std::ostringstream oss;
        oss << "Matrix dimensions must be positive, got " << _rows << "x" << _cols;
        throw std::invalid_argument(oss.str());
    }
    matrix<double> result = (matrix<double>)*this;
    for (size_t i = 0; i < _rows; ++i)
    {
        size_t pivot_row = i;
        for (size_t j = i + 1; j < _rows; ++j)
        {
            if (std::abs(result(j, i)) > std::abs(result(pivot_row, i)))
            {
                pivot_row = j;
            }
        }
        if (std::abs(result(pivot_row, i)) < std::numeric_limits<double>::epsilon())
        {
            continue;
        }
        if (pivot_row != i)
        {
            result.swapRows(i, pivot_row);
        }
        for (size_t j = i + 1; j < _rows; ++j)
        {
            double factor = result(j, i) / result(i, i);
            for (size_t k = i; k < _cols; ++k)
            {
                result(j, k) -= factor * result(i, k);
            }
        }
    }
    return result;
}

/**
 * @brief Computes LU decomposition (A = LU).
 *
 * @return Pair of (L, U) where L is lower triangular, U is upper triangular
 * @throws std::invalid_argument If matrix is not square
 * @details Time O(n³), Space O(n²)
 */
template <typename T>
std::pair<matrix<double>, matrix<double>> matrix<T>::LU_decomposition() const
{
    if (_rows != _cols)
    {
        std::ostringstream oss;
        oss << "Matrix is " << _rows << "x" << _cols << ", must be square for LU decomposition";
        throw std::invalid_argument(oss.str());
    }
    matrix<double> L(_rows, _cols);
    matrix<double> U = matrix<double>::eye(_rows, _cols);
    for (size_t p = 0; p < _rows; ++p)
    {
        for (size_t i = 0; i < p; ++i)
        {
            U(i, p) = (*this)(i, p) - (L.row(i) * U.col(p))(0, 0);
            U(i, p) /= L(i, i);
        }
        for (size_t i = p; i < _rows; ++i)
        {
            L(i, p) = (double)(*this)(i, p) - (L.row(i) * U.col(p))(0, 0);
        }
    }

    return std::make_pair(L, U);
}

/**
 * @brief Computes QR decomposition using Gram-Schmidt process (A = QR).
 *
 * Optimizations: Parallel computation using std::async for column operations.
 *
 * @return Pair of (Q, R) where Q is orthogonal, R is upper triangular
 * @details Time O(m*n²), Space O(m*n)
 */
template <typename T>
std::pair<matrix<double>, matrix<double>> matrix<T>::QR_decomposition() const
{
    matrix<double> Q = (matrix<double>)*this;
    matrix<double> R(_rows, _cols);

    double norm0 = this->col(0).norm(2);
    for (size_t i = 0; i < _rows; ++i)
        Q(i, 0) = (*this)(i, 0) / norm0;

    std::vector<std::future<void>> futures;
    for (size_t j = 0; j < _cols; ++j)
    {
        futures.emplace_back(std::async(std::launch::async, [&, j]()
                                        { R(0, j) = (Q.col(0).transpose() * (matrix<double>)(*this))(0, j); }));
    }
    for (auto& f : futures)
        f.get();

    for (size_t i = 0; i < _cols - 1; ++i)
    {
        futures.clear();
        for (size_t j = i + 1; j < _cols; ++j)
        {
            futures.emplace_back(std::async(std::launch::async,
                                            [&, i, j]()
                                            {
                                                matrix<double> new_col = Q.col(j);
                                                new_col -= (Q.col(j).transpose() * Q.col(i))(0, 0) * Q.col(i);
                                                for (size_t k = 0; k < _rows; ++k)
                                                    Q(k, j) = new_col(k, 0);
                                            }));
        }
        for (auto& f : futures)
            f.get();

        double norm = Q.col(i + 1).norm(2);
        for (size_t k = 0; k < _rows; ++k)
            Q(k, i + 1) = Q(k, i + 1) / norm;

        futures.clear();
        for (size_t k = 0; k < _cols; ++k)
        {
            futures.emplace_back(
                std::async(std::launch::async,
                           [&, i, k]() { R(i + 1, k) = (Q.col(i + 1).transpose() * (matrix<double>)(*this))(0, k); }));
        }
        for (auto& f : futures)
            f.get();
    }

    return std::make_pair(Q, R);
}

/**
 * @brief Computes eigenvalues using QR algorithm.
 *
 * @param max_iter Maximum iterations for convergence (default implementation)
 * @return Column matrix containing eigenvalues
 * @throws std::invalid_argument If matrix is not square
 * @details Time O(max_iter * n³), Space O(n²)
 * @note Does not support complex eigenvalues yet
 */
template <typename T>
matrix<double> matrix<T>::eigenvalues(int max_iter) const
{
    if (_rows != _cols)
    {
        std::ostringstream oss;
        oss << "Matrix is " << _rows << "x" << _cols << ", must be square to compute eigenvalues";
        throw std::invalid_argument(oss.str());
    }

    matrix<double> eigenvalues = (matrix<double>)(*this);
    for (int iter = 0; iter < max_iter; iter++)
    {
        matrix<double> Q, R;
        std::tie(Q, R) = eigenvalues.QR_decomposition();
        eigenvalues = R * Q;
    }
    matrix<double> result(_rows, 1);
    for (size_t i = 0; i < _rows; ++i)
    {
        result(i, 0) = eigenvalues(i, i);
    }
    return result;
}

/**
 * @brief Computes eigenvectors using QR algorithm.
 *
 * @param max_iter Maximum iterations for convergence (default implementation)
 * @return Matrix with eigenvectors as columns
 * @throws std::invalid_argument If matrix is not square
 * @details Time O(max_iter * n³), Space O(n²)
 * @note Does not support complex eigenvectors yet
 */
template <typename T>
matrix<double> matrix<T>::eigenvectors(int max_iter) const
{
    if (_rows != _cols)
    {
        std::ostringstream oss;
        oss << "Matrix is " << _rows << "x" << _cols << ", must be square to compute eigenvectors";
        throw std::invalid_argument(oss.str());
    }

    matrix<double> eigenvalues = (matrix<double>)(*this);
    matrix<double> eigenvectors = matrix<double>::eye(_rows, _cols);
    for (int iter = 0; iter < max_iter; iter++)
    {
        matrix<double> Q, R;
        std::tie(Q, R) = eigenvalues.QR_decomposition();
        eigenvalues = R * Q;
        eigenvectors = eigenvectors * Q;
    }
    return eigenvectors;
}