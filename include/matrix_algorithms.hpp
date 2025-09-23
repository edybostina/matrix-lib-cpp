#pragma once
#include "matrix_core.hpp"


// Determinant
template <typename T>
double matrix<T>::determinant() const
{
    if (_rows != _cols)
    {
        throw std::invalid_argument("Matrix must be square to compute determinant");
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
    // For larger matrices, use Gaussian elimination
    double det = 1.0;
    matrix<double> temp = (matrix<double>)*this;
    for (int i = 0; i < _rows; ++i)
    {
        int pivot = i;
        for (int j = i + 1; j < _rows; ++j)
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
        for (int j = i + 1; j < _rows; ++j)
        {
            double factor = temp(j, i) / temp(i, i);
            for (int k = i + 1; k < _cols; ++k)
            {
                temp(j, k) -= factor * temp(i, k);
            }
        }
    }
    return det;
}

// Trace
template <typename T>
T matrix<T>::trace() const
{
    if (_rows != _cols)
    {
        throw std::invalid_argument("Matrix must be square to compute trace");
    }
    T sum = 0;
    for (int i = 0; i < _rows; ++i)
    {
        sum += (*this)(i, i);
    }
    return sum;
}

// Transpose
template <typename T>
matrix<T> matrix<T>::transpose() const
{
    matrix<T> result(_cols, _rows);
    for (int i = 0; i < _rows; ++i)
    {
        for (int j = 0; j < _cols; ++j)
        {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

// Cofactor
template <typename T>
matrix<T> matrix<T>::cofactor() const
{
    matrix<T> result(_rows, _cols);
    for (int i = 0; i < _rows; ++i)
    {
        for (int j = 0; j < _cols; ++j)
        {
            matrix<T> minor = this->minor(i, j);
            result(i, j) = ((i + j) % 2 == 0 ? 1 : -1) * minor.determinant();
        }
    }
    return result;
}

// Minor
template <typename T>
matrix<T> matrix<T>::minor(int row, int col) const
{
    matrix<T> result(_rows - 1, _cols - 1);
    for (int i = 0, r = 0; i < _rows; ++i)
    {
        if (i == row)
            continue;
        for (int j = 0, c = 0; j < _cols; ++j)
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

// Adjoint
template <typename T>
matrix<T> matrix<T>::adjoint() const
{
    if (_rows != _cols)
    {
        throw std::invalid_argument("Matrix must be square to compute adjoint");
    }
    matrix<T> result(_rows, _cols);
    result = this->cofactor().transpose();
    return result;
}

// Inverse
template <typename T>
matrix<double> matrix<T>::inverse() const
{
    if (_rows != _cols)
    {
        throw std::invalid_argument("Matrix must be square to compute inverse");
    }
    double temp;
    matrix<double> augmented(_rows, 2 * _cols);
    matrix<double> result(_rows, _cols);
    for (int i = 0; i < _rows; ++i)
    {
        for (int j = 0; j < _cols; ++j)
        {
            augmented(i, j) = (*this)(i, j);
        }
        for (int j = _cols; j < 2 * _cols; ++j)
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
    for (int i = _rows - 1; i > 0; i--)
    {
        if (augmented(i - 1, 0) < augmented(i, 0))
        {
            augmented.swapRows(i - 1, i);
        }
    }

    for (int i = 0; i < _rows; ++i)
    {
        if (augmented(i, i) == 0)
        {
            throw std::invalid_argument("Matrix is singular and cannot be inverted");
        }
        for (int j = 0; j < _cols; ++j)
        {
            if (j != i)
            {
                temp = augmented(j, i) / augmented(i, i);
                for (int k = 0; k < 2 * _cols; ++k)
                {
                    augmented(j, k) -= augmented(i, k) * temp;
                }
            }
        }
    }
    for (int i = 0; i < _rows; ++i)
    {
        temp = augmented(i, i);
        for (int j = 0; j < 2 * _cols; ++j)
        {
            augmented(i, j) /= temp;
        }
    }
    for (int i = 0; i < _rows; ++i)
    {
        for (int j = 0; j < _cols; ++j)
        {
            result(i, j) = augmented(i, j + _cols);
        }
    }
    return result;
}

// Norm
template <typename T>
double matrix<T>::norm(int p) const
{
    if (p < 1)
    {
        throw std::invalid_argument("Norm order must be greater than or equal to 1");
    }
    double norm = 0;
    for (int i = 0; i < _rows; ++i)
    {
        for (int j = 0; j < _cols; ++j)
        {
            norm += std::pow(std::abs((*this)(i, j)), p);
        }
    }
    return std::pow(norm, 1.0 / p);
}

// Rank of the matrix
template <typename T>
int matrix<T>::rank() const
{
    matrix<double> gaussian = this->gaussian_elimination();
    int rank = 0;
    for (int i = 0; i < _rows; ++i)
    {
        bool non_zero_row = false;
        for (int j = 0; j < _cols; ++j)
        {
            if (std::abs(gaussian(i, j)) > std::numeric_limits<double>::epsilon())
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

template <typename T>
matrix<double> matrix<T>::gaussian_elimination() const
{
    if (_rows == 0 || _cols == 0)
    {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    matrix<double> result = (matrix<double>)*this;
    for (int i = 0; i < _rows; ++i)
    {
        int pivot_row = i;
        for (int j = i + 1; j < _rows; ++j)
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
        for (int j = i + 1; j < _rows; ++j)
        {
            double factor = result(j, i) / result(i, i);
            for (int k = i; k < _cols; ++k)
            {
                result(j, k) -= factor * result(i, k);
            }
        }
    }
    return result;
}

template <typename T>
std::pair<matrix<double>, matrix<double>> matrix<T>::LU_decomposition() const
{
    if (_rows != _cols)
    {
        throw std::invalid_argument("Matrix must be square for LU decomposition");
    }
    matrix<double> L(_rows, _cols);
    matrix<double> U = matrix<double>::eye(_rows, _cols);
    for (int p = 0; p < _rows; ++p)
    {
        for (int i = 0; i < p; ++i)
        {
            U(i, p) = (*this)(i, p) - (L.row(i) * U.col(p))(0, 0);
            U(i, p) /= L(i, i);
        }
        for (int i = p; i < _rows; ++i)
        {
            L(i, p) = (double)(*this)(i, p) - (L.row(i) * U.col(p))(0, 0);
        }
    }

    return std::make_pair(L, U);
}

// QR decomposition
// This function uses the Householder reflection method to compute the QR decomposition
// Q is an orthogonal matrix
// R is an upper triangular matrix
// returns make_pair(Q, R)

template <typename T>
std::pair<matrix<double>, matrix<double>> matrix<T>::QR_decomposition() const
{
    matrix<double> Q = (matrix<double>)*this;
    matrix<double> R(_rows, _cols);

    double norm0 = this->col(0).norm(2);
    for (int i = 0; i < _rows; ++i)
        Q(i, 0) = (*this)(i, 0) / norm0;

    std::vector<std::future<void>> futures;
    for (int j = 0; j < _cols; ++j)
    {
        futures.emplace_back(std::async(std::launch::async, [&, j]()
                                        { R(0, j) = (Q.col(0).transpose() * (matrix<double>)(*this))(0, j); }));
    }
    for (auto &f : futures)
        f.get();

    for (int i = 0; i < _cols - 1; ++i)
    {
        futures.clear();
        for (int j = i + 1; j < _cols; ++j)
        {
            futures.emplace_back(std::async(std::launch::async, [&, i, j]()
                                            {
                matrix<double> new_col = Q.col(j);
                new_col -= (Q.col(j).transpose() * Q.col(i))(0, 0) * Q.col(i);
                for (int k = 0; k < _rows; ++k)
                    Q(k, j) = new_col(k, 0); }));
        }
        for (auto &f : futures)
            f.get();

        double norm = Q.col(i + 1).norm(2);
        for (int k = 0; k < _rows; ++k)
            Q(k, i + 1) = Q(k, i + 1) / norm;

        futures.clear();
        for (int k = 0; k < _cols; ++k)
        {
            futures.emplace_back(std::async(std::launch::async, [&, i, k]()
                                            { R(i + 1, k) = (Q.col(i + 1).transpose() * (matrix<double>)(*this))(0, k); }));
        }
        for (auto &f : futures)
            f.get();
    }

    return std::make_pair(Q, R);
}

// Eigenvalues
// Currently, this function uses the QR algorithm to compute the eigenvalues
// Does not work for complex numbers yet.
template <typename T>
matrix<double> matrix<T>::eigenvalues(int max_iter) const
{
    if (_rows != _cols)
    {
        throw std::invalid_argument("Matrix must be square to compute eigenvalues");
    }

    matrix<double> eigenvalues = (matrix<double>)(*this);
    for (int iter = 0; iter < max_iter; iter++)
    {
        matrix<double> Q, R;
        std::tie(Q, R) = eigenvalues.QR_decomposition();
        eigenvalues = R * Q;
    }
    matrix<double> result(_rows, 1);
    for (int i = 0; i < _rows; ++i)
    {
        result(i, 0) = eigenvalues(i, i);
    }
    return result;
}

// Eigenvectors
// Currently, this function uses the QR algorithm to compute the eigenvectors
// Does not work for complex numbers yet.
template <typename T>
matrix<double> matrix<T>::eigenvectors(int max_iter) const
{
    if (_rows != _cols)
    {
        throw std::invalid_argument("Matrix must be square to compute eigenvectors");
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