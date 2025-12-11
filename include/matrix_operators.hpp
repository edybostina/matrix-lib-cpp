#pragma once

// Matrix operator declarations - all implementations moved to impl/matrix_operators.tpp
// This file is automatically included by matrix_core.hpp

// For header-only mode, include implementations
#ifndef MATRIX_EXPLICIT_INSTANTIATION
#include "impl/matrix_operators.tpp"
#endif
