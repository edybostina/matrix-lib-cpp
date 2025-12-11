/**
 * @file matrix_instantiations.cpp
 * @brief Explicit template instantiations for common matrix types
 *
 * This file provides explicit instantiations of the matrix template
 * for commonly used types. This reduces compile time for users and
 * allows the library to be built as a true compiled library.
 *
 * By explicitly instantiating these types, users linking against the
 * compiled library don't need to recompile template code for these types.
 *
 * @author edybostina
 * @date 2025-12-11
 * @version 0.2.0
 */

// Force inclusion of all template implementations for explicit instantiation
#undef MATRIX_HEADER_ONLY
#include "../include/matrix.hpp"

// Force inclusion of all implementation files
#include "../include/impl/matrix_core.tpp"
#include "../include/impl/matrix_operators.tpp"
#include "../include/impl/matrix_algorithms.tpp"
#include "../include/impl/matrix_properties.tpp"
#include "../include/impl/matrix_utils.tpp"
#include "../include/impl/matrix_io.tpp"

template class matrix<float>;
template class matrix<double>;