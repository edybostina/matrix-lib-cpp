# matrix-lib-cpp

A modern, high-performance C++17 matrix library featuring optimized linear algebra operations with SIMD acceleration, multi-threading support, and BLAS integration.

[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![CMake](https://img.shields.io/badge/CMake-3.14+-green.svg)](https://cmake.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.0-orange.svg)](CMakeLists.txt)

## Features

### Core Operations

- **Matrix Creation**: Initialize from arrays, initializer lists, or use factory methods (`zeros`, `ones`, `eye`, `random`)
- **Element Access**: Intuitive `operator()` for element and row access with bounds checking
- **Arithmetic Operations**: Addition, subtraction, multiplication (matrix-matrix, matrix-scalar)
- **Advanced Operations**: Transpose, inverse, determinant, trace, adjoint, cofactor, minor

### Linear Algebra Algorithms

- **Decompositions**: LU decomposition, QR decomposition (Gram-Schmidt)
- **Eigenanalysis**: Eigenvalues and eigenvectors computation (QR algorithm)
- **Matrix Properties**: Rank, norm (p-norm), Gaussian elimination
- **Utility Functions**: Row/column swapping, matrix resizing, submatrix extraction

### Matrix Properties

- Symmetry checking (`is_symmetric`)
- Triangular matrix detection (`is_lower_triangular`, `is_upper_triangular`)
- Diagonal matrix verification (`is_diagonal`)

### Performance Optimizations

- **SIMD Support**: AVX2 for x86_64, NEON for ARM64
- **Multi-threading**: Automatic parallelization for large matrices (>10k elements)
- **BLAS Integration**: Optional BLAS backend for matrix multiplication
- **Memory Efficiency**: Tiled matrix multiplication, direct memory access
- **Optimized Algorithms**: Binary exponentiation for matrix powers, cache-friendly layouts

### Additional Features

- **Matrix Utilities**: Power, exponential, Hadamard product, diagonal operations
- **Type Aliases**: `Matrixf` (float), `Matrixd` (double), `Matrixi` (int)
- **I/O Operations**: Stream input/output support
- **Header-Only Mode**: Easy integration into projects

## Documentation

- [Build Guide](docs/build.md)
- [API Reference](docs/api.md)

## Performance

The library includes extensive optimizations:

- **SIMD Vectorization**: 4x-8x speedup for arithmetic operations
- **Multi-threading**: Automatic parallelization for matrices >10,000 elements
- **Cache Optimization**: Tiled matrix multiplication for better cache utilization
- **BLAS Backend**: Up to 500x faster multiplication with optimized BLAS libraries

Run the benchmark to see performance on your system:

```bash
./matrix-benchmark
```

## Examples

See the [examples](examples/) directory for complete examples:

- [demo.cpp](examples/demo.cpp) - Basic usage and operations
- [benchmark.cpp](examples/benchmark.cpp) - Performance benchmarking


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Eduard Bostina (@edybostina)


