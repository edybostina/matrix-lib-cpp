# Matrix Library for C++

A modern C++ matrix library with comprehensive linear algebra operations.

## Features

- **Template-based design**: Flexible generic programming with C++17
- **Highly optimized**: Cache-blocking, SIMD vectorization (AVX2/NEON), and optional BLAS integration
- **Multithreading**: Automatic parallelization for large matrices
- **SIMD acceleration**: Native support for x86 AVX2 and ARM NEON intrinsics
- **BLAS support**: Optional integration with optimized BLAS libraries for maximum performance
- **Comprehensive operations**: Basic arithmetic, linear algebra, eigenvalues, decompositions
- **Easy integration**: CMake support with FetchContent, static/shared library builds
- **Type-safe**: Template-based design with compile-time checks
- **Flexible deployment**: Header-only, static library (.a/.lib), or shared library (.so/.dylib/.dll)

## Quick Start

### As a CMake Project

**Using just (recommended):**

```bash
git clone https://github.com/edybostina/matrix-lib-cpp.git
cd matrix-lib-cpp

just build
just test
just bench

# or everything:
just dev
```

**Manual build:**

```bash
git clone https://github.com/edybostina/matrix-lib-cpp.git
cd matrix-lib-cpp

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DMATRIX_USE_SIMD=ON
cmake --build .
```

**Note**: SIMD optimizations are enabled by default. For maximum performance with BLAS:

```bash
just blas          # Using just
```

### Using in Your Project

#### Option 1: CMake FetchContent (Recommended)

```cmake
include(FetchContent)

FetchContent_Declare(
    matrix
    GIT_REPOSITORY https://github.com/edybostina/matrix-lib-cpp.git
    GIT_TAG main
)

FetchContent_MakeAvailable(matrix)

add_executable(your_app main.cpp)
target_link_libraries(your_app PRIVATE matrix::matrix)
```

#### Option 2: Install Locally

```bash
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
sudo cmake --build . --target install
```

Then in your `CMakeLists.txt`:

```cmake
find_package(matrix REQUIRED)
target_link_libraries(your_app PRIVATE matrix::matrix)
```

#### Option 3: Header-only Direct Include

Simply copy the `include/` directory to your project and add it to your include path:

```cpp
#include "matrix.hpp"

int main() {
    matrix<double> A(3, 3);
    // Use the matrix...
}
```

## Usage Examples

### Basic Operations

```cpp
#include "matrix.hpp"

int main() {
    // Create matrices
    matrix<double> A(3, 3);
    matrix<double> B = matrix<double>::identity(3);

    // Fill with values
    A.fill(1.5);

    // Arithmetic operations
    matrix<double> C = A + B;
    matrix<double> D = A * B;
    matrix<double> E = A * 2.0;

    // Element access
    A(0, 0) = 5.0;
    double val = A(1, 2);

    // Display
    std::cout << A << std::endl;

    return 0;
}
```

### Linear Algebra

```cpp
// Matrix properties
double det = A.determinant();
int rank = A.rank();
double trace = A.trace();

// Matrix decompositions
auto [L, U, P] = A.lu_decomposition();
auto [Q, R] = A.qr_decomposition();

// Solve linear systems
matrix<double> x = A.solve(b);  // Solve Ax = b

// Inverse
matrix<double> A_inv = A.inverse();

// Eigenvalues and eigenvectors
auto [eigenvalues, eigenvectors] = A.eigenvalues_eigenvectors();
```

### Advanced Features

```cpp
// Transpose
matrix<double> At = A.transpose();

// Norms
double frobenius_norm = A.norm();
double infinity_norm = A.norm_inf();

// Condition number
double cond = A.condition_number();

// Matrix power
matrix<double> A_squared = A.power(2);

// Check properties
bool is_symmetric = A.is_symmetric();
bool is_positive_definite = A.is_positive_definite();
```

## Build Tools

This project supports multiple build methods:

### Just Commands (Recommended)

```bash
just build         # Standard release build
just blas          # Build with BLAS
just fast          # Maximum optimizations
just dev           # Build + test cycle
just clean         # Clean build directory
just --list        # See all commands
```

## CMake Options

When building the project, you can configure these options:

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DMATRIX_BUILD_SHARED=OFF \      # Build static library (default: OFF)
  -DMATRIX_BUILD_EXAMPLES=ON \     # Build example programs (default: ON)
  -DMATRIX_BUILD_TESTS=ON \        # Build test programs (default: ON)
  -DMATRIX_HEADER_ONLY=ON \        # Use header-only mode (default: ON)
  -DMATRIX_USE_SIMD=ON \           # Enable SIMD optimizations (default: ON)
  -DMATRIX_USE_BLAS=OFF \          # Use BLAS for matrix operations (default: OFF)
  -DMATRIX_INSTALL=ON              # Generate install target (default: ON)
```

### SIMD Optimizations

SIMD support is **enabled by default** and automatically detects your CPU architecture:

- **x86/x64**: Uses AVX2 and FMA instructions (8-wide float, 4-wide double)
- **ARM64**: Uses NEON instructions (4-wide float, 2-wide double)

To disable SIMD:

```bash
cmake .. -DMATRIX_USE_SIMD=OFF
```

### BLAS Integration

For maximum performance on large matrices, enable BLAS integration:

**macOS** (using Accelerate framework):

```bash
cmake .. -DMATRIX_USE_BLAS=ON
```

**macOS** (using OpenBLAS):

```bash
brew install openblas
cmake .. -DMATRIX_USE_BLAS=ON
```

**Linux**:

```bash
sudo apt-get install libblas-dev
cmake .. -DMATRIX_USE_BLAS=ON
```

BLAS provides **10-50x speedup** for large matrix multiplications by leveraging highly optimized vendor libraries.

## Library Types

The library supports three build modes:

### 1. Header-only (Default, Recommended)

All implementation in headers, no separate compilation needed:

```bash
cmake .. -DMATRIX_HEADER_ONLY=ON  # Default
```

**Pros**: Easy integration, no linking, template optimizations
**Cons**: Longer compilation times for including projects

### 2. Static Library

Compiled once, linked into your executable:

```bash
cmake .. -DMATRIX_HEADER_ONLY=OFF
```

Produces `.a` (Unix/macOS) or `.lib` (Windows).

**Pros**: Faster compilation for projects, single binary
**Cons**: Larger executables, less flexible

### 3. Shared Library

Compiled once, loaded at runtime:

```bash
cmake .. -DMATRIX_HEADER_ONLY=OFF -DMATRIX_BUILD_SHARED=ON
```

Produces `.so` (Linux), `.dylib` (macOS), or `.dll` (Windows).

**Pros**: Smaller executables, shared between programs, easier updates
**Cons**: Must distribute library file, runtime dependencies

## Performance

The library is highly optimized for modern CPUs with multiple performance features:

### Optimization Features

- **Cache-blocking**: 64×64 block tiling for optimal L1/L2 cache utilization
- **SIMD vectorization**: AVX2 (x86) or NEON (ARM) intrinsics for parallel operations
- **Multithreading**: Automatic parallelization for large matrices (>256×256)
- **Direct memory access**: Zero-overhead element access in hot loops
- **Loop unrolling**: Manual unrolling for better instruction-level parallelism
- **Optional BLAS**: Integration with vendor-optimized libraries (Intel MKL, OpenBLAS, Accelerate)

### Performance Benchmarks

Typical speedups on modern hardware:

| Configuration               | Speedup (vs naive) | Notes                       |
| --------------------------- | ------------------ | --------------------------- |
| **Base (no optimizations)** | 1.0x               | Simple triple-loop          |
| **Cache-blocking only**     | 5-10x              | Better cache utilization    |
| **+ SIMD**                  | 15-25x             | Vectorized operations       |
| **+ Multithreading**        | 40-100x            | 8-core CPU                  |
| **+ BLAS (OpenBLAS)**       | 100-500x           | Large matrices (>1024×1024) |

### Matrix Multiplication Performance

- **Small matrices** (<128×128): Cache-blocked SIMD
- **Medium matrices** (128-1024): Cache-blocked SIMD + multithreading
- **Large matrices** (>1024×1024): BLAS (if enabled) for maximum performance

Run the benchmark to see performance on your system:

```bash
cd build
./matrix-benchmark
```

## Requirements

- **C++17 compatible compiler**: GCC 7+, Clang 5+, MSVC 2017+
- **CMake 3.14+**: For building and installing
- **SIMD support** (optional): CPU with AVX2 (x86) or NEON (ARM64)
- **BLAS library** (optional): For maximum performance
  - macOS: Accelerate framework (built-in) or OpenBLAS (Accelerate works great out of the box)
  - Linux: OpenBLAS, ATLAS, or Intel MKL (I recommend OpenBLAS)
  - Windows: Intel MKL or OpenBLAS (I recommend OpenBLAS again)

### Compiler Flags

For optimal performance, the library automatically uses:

- `-O3`: Maximum optimization
- `-march=native`: CPU-specific optimizations
- `-mavx2 -mfma`: AVX2 and FMA support (x86 only)
- ARM NEON enabled automatically on ARM64

## Project Structure

```
matrix-lib-cpp/
├── include/              # Header files
│   ├── impl/             # Template implementations
│   ├── matrix.hpp        # Main include file
│   ├── matrix_core.hpp   # Core matrix class
│   ├── matrix_operators.hpp  # Operator overloads
│   ├── matrix_algorithms.hpp # Linear algebra algorithms
│   └── ...               # Other headers
├── src/                  # Library source
│   └── matrix_lib.cpp    # Library implementation
├── examples/             # Example programs
│   ├── demo.cpp          # Basic usage demo
│   └── benchmark.cpp     # Performance benchmarks
├── tests/                # Test suite
│   └── main.cpp          # Unit tests
├── docs/                 # Documentation
│   └── api.md            # API reference
└── CMakeLists.txt        # CMake configuration
```

## Documentation

For detailed API documentation, see [docs/api.md](docs/api.md).
For detailed build instructions, see [docs/BUILD_GUIDE.md](docs/BUILD_GUIDE.md).

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Eduard Bostina

## Version

Current version: 0.2.0 (December 2025)
