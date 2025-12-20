# Matrix Lib C++

[![Language](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build Status](https://github.com/edybostina/matrix-lib-cpp/actions/workflows/build.yml/badge.svg)](https://github.com/edybostina/matrix-lib-cpp/actions)
[![Version](https://img.shields.io/badge/version-0.2.0-orange.svg)](CMakeLists.txt)

**Matrix Lib C++** is a lightweight, high-performance, header-only C++17 library designed for linear algebra operations.

It combines the ease of use of Python's NumPy with the raw performance of C++, utilizing SIMD intrinsics (AVX2/NEON), automatic multithreading, and optional BLAS integration.

## Key Features

- **High Performance:**
  - **SIMD Accelerated:** Native AVX2 (x86_64) and NEON (ARM64) implementations for arithmetic operations.
  - **Multithreaded:** Automatic parallelization for large matrix operations using `std::thread`.
  - **Cache Friendly:** Tiled matrix multiplication algorithms to optimize CPU cache usage.
- **Flexible Integration:**
  - **Header-Only:** Drop it into your project and go. No linking required by default.
  - **BLAS Support:** Optional linking with OpenBLAS, Intel MKL, or Apple Accelerate for maximum speed.
- **Comprehensive Linear Algebra:**
  - **Decompositions:** LU, QR, Cholesky (planned), Eigenvalues & Eigenvectors.
  - **Operations:** Determinant, Trace, Inverse, Rank, Norms, Transpose, Adjoint.
  - **Manipulation:** Submatrices, Resizing, Swapping Rows/Cols, Diagonal extraction.
- **Developer Friendly:**
  - **Modern C++:** Uses C++17 features like `std::optional`, `constexpr`, and structured bindings.
  - **Clean API:** Intuitive operator overloading (`A * B`, `A + B`) and method chaining.
  - **Tested:** Comprehensive unit test suite and benchmarks.

## Installation

### Option 1: CMake FetchContent (Recommended)

You can include this library directly in your `CMakeLists.txt` without manually downloading files:

```cmake
include(FetchContent)

FetchContent_Declare(
    matrix_lib
    GIT_REPOSITORY https://github.com/edybostina/matrix-lib-cpp.git
    GIT_TAG main
)

FetchContent_MakeAvailable(matrix_lib)
target_link_libraries(your_target PRIVATE matrix::matrix)
```

### Option 2: Copy Headers

Since the library is header-only by default, you can simply copy the `include/` directory to your project's source tree.

## Quick Start

```cpp
#include <matrix.hpp>
#include <iostream>

int main() {
    // Create matrices
    auto A = matrix<double>::random(3, 3, -1.0, 1.0);
    auto I = matrix<double>::eye(3, 3);

    // Arithmetic operations
    auto B = (A * 2.0) + I;

    try {
        // Linear Algebra
        double det = B.determinant();
        auto inverse = B.inverse();
        auto [L, U] = B.LU_decomposition();

        std::cout << "Matrix B:\n" << B << "\n";
        std::cout << "Determinant: " << det << "\n";
        std::cout << "Inverse:\n" << inverse << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }

    return 0;
}
```

## Performance

The library automatically selects the best available optimization strategy:

1.  **BLAS/LAPACK:** If linked (e.g., `-DMATRIX_USE_BLAS=ON`), it delegates heavy lifting to optimized libraries like OpenBLAS or Apple Accelerate.
2.  **SIMD Intrinsics:** If BLAS is not available, it uses hand-written AVX2/NEON kernels for element-wise operations.
3.  **Multithreading:** For large matrices (>10k elements), operations are parallelized across available cores.

## Documentation

- [Build Guide](docs/build.md)
- [API Reference](docs/api.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
