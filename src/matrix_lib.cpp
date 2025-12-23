/**
 * @file matrix_lib.cpp
 * @brief Explicit template instantiations for compiled library builds
 *
 * This file provides explicit instantiations of the matrix template class
 * for common types when building as a static or shared library.
 *
 * For header-only builds, this file is not used.
 */

#include "matrix.hpp"

// Explicit template instantiations for floating-point types
template class matrix<float>;
template class matrix<double>;

// Explicit template instantiations for integer type
template class matrix<short>;
template class matrix<unsigned short>;

template class matrix<int>;
template class matrix<unsigned int>;

template class matrix<long>;
template class matrix<unsigned long>;

template class matrix<long long>;
template class matrix<unsigned long long>;

// add more here
