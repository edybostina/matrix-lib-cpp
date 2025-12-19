/**
 * @file matrix_lib.cpp
 * @brief Library implementation file for matrix library
 *
 * This file exists to create a compiled library target.
 * Since the matrix library is header-only, this file contains
 * minimal implementation - mainly version information.
 *
 * @author edybostina
 * @date 2025-12-11
 * @version 0.2.0
 */

#include "../include/matrix.hpp"

namespace matrix_lib
{

const char* version()
{
    return "0.2.0";
}

const char* build_date()
{
    return __DATE__ " " __TIME__;
}

} // namespace matrix_lib
