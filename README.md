# Matrix Library for C++

A modern C++ matrix library with comprehensive linear algebra operations.

## Features

- **Template-based design**: Flexible generic programming with C++17
- **Multithreading**: Automatic parallelization for large matrices
- **Comprehensive operations**: Basic arithmetic, linear algebra, eigenvalues, decompositions
- **Easy integration**: CMake support with static/shared library builds
- **Type-safe**: Template-based design with compile-time checks
- **Production ready**: Compiled library (.a/.so) for easy linking

## Quick Start

### As a CMake Project

```bash
# Clone the repository
git clone https://github.com/yourusername/matrix-lib-cpp.git
cd matrix-lib-cpp

# Build with CMake
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .

# Run tests
./matrix-test

# Run benchmark
./matrix-benchmark
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

## CMake Options

When building the project, you can configure these options:

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DMATRIX_BUILD_SHARED=OFF \      # Build static library (default: OFF)
  -DMATRIX_BUILD_EXAMPLES=ON \     # Build example programs (default: ON)
  -DMATRIX_BUILD_TESTS=ON \        # Build test programs (default: ON)
  -DMATRIX_INSTALL=ON              # Generate install target (default: ON)
```

## Library Types

By default, the project builds a **static library** (`.a` on Unix, `.lib` on Windows).

To build a **shared library** (`.so`/`.dylib`/`.dll`) instead:

```bash
cmake .. -DMATRIX_BUILD_SHARED=ON
```

## Performance

The library includes automatic multithreading for large matrices. Typical speedups:

- **Matrix operations**: 4-8x with multithreading on modern CPUs
- **Large matrices** (>1000x1000): Best performance with `-O3 -march=native` compiler flags

Run the benchmark to see performance on your system:

```bash
./matrix-benchmark
```

## Requirements

- **C++17 compatible compiler**: GCC 7+, Clang 5+, MSVC 2017+
- **CMake 3.14+**: For building and installing

## Project Structure

```
matrix-lib-cpp/
├── include/              # Header files
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
