# Build Guide

Comprehensive guide for building, testing, and integrating the Matrix Library for C++.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start with Just](#quick-start-with-just)
- [CMake Configuration](#cmake-configuration)
- [Build Options](#build-options)
- [Integration](#integration)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- **C++ Compiler:** GCC 7+, Clang 5+, MSVC 2019+ (C++17 support required)
- **CMake:** Version 3.14 or higher
- **Build System:** Make, Ninja, or Visual Studio
- **Optional:**
  - [Just](https://github.com/casey/just) (Command runner)
  - OpenBLAS / Intel MKL / Apple Accelerate (for BLAS support)

## Quick Start with Just

I recommend using [just](https://github.com/casey/just) for a simplified workflow.

```bash
# Install Just (if not installed)
# macOS: brew install just
# Linux: sudo apt install just

# Build in release mode
just build

# Run tests
just test

# Run benchmarks
just bench

# Clean build directory
just clean
```

**Common Commands:**

| Command | Description |
|---------|-------------|
| `just build` | Build library and examples in Release mode |
| `just debug` | Build in Debug mode with symbols |
| `just fast` | Build with all optimizations (SIMD + BLAS) |
| `just test` | Run the unit test suite |
| `just bench` | Run performance benchmarks |
| `just install` | Install library to system |

## CMake Configuration

If you prefer using CMake directly:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j\8
```

### Build Options

You can configure the build using the following CMake options:

| Option | Default | Description |
|--------|---------|-------------|
| `MATRIX_BUILD_EXAMPLES` | `ON` | Build example programs (demo, benchmark) |
| `MATRIX_BUILD_TESTS` | `ON` | Build unit tests |
| `MATRIX_HEADER_ONLY` | `ON` | Use header-only mode (no linking required) |
| `MATRIX_USE_SIMD` | `ON` | Enable AVX2/NEON SIMD optimizations |
| `MATRIX_USE_BLAS` | `ON` | Link against BLAS (OpenBLAS, MKL, Accelerate) |
| `MATRIX_BUILD_SHARED` | `OFF` | Build shared library (.so/.dylib/.dll) |

**Example: Optimized Build**

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DMATRIX_USE_SIMD=ON \
    -DMATRIX_USE_BLAS=ON
```

## Integration

### Method 1: Header-Only (Easiest)

Since the library is header-only by default, you can simply copy the `include/` directory to your project.

```cpp
// main.cpp
#include "matrix.hpp"
```

Compile with C++17 standard:
```bash
g++ main.cpp -std=c++17 -I/path/to/matrix-lib/include
```

### Method 2: CMake FetchContent

Add this to your `CMakeLists.txt`:

```cmake
include(FetchContent)

FetchContent_Declare(
    matrix_lib
    GIT_REPOSITORY https://github.com/edybostina/matrix-lib-cpp.git
    GIT_TAG main
)

FetchContent_MakeAvailable(matrix_lib)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE matrix::matrix)
```

### Method 3: System Installation

```bash
# Build and install
just install
# OR
cd build && sudo cmake --install .
```

Then in your `CMakeLists.txt`:

```cmake
find_package(matrix-lib REQUIRED)
target_link_libraries(my_app PRIVATE matrix::matrix)
```

## Troubleshooting

### BLAS Not Found

If CMake cannot find a BLAS library:
1.  Ensure OpenBLAS or MKL is installed.
2.  Provide the path manually: `-DBLAS_ROOT=/path/to/blas`.
3.  Disable BLAS to use internal implementation: `-DMATRIX_USE_BLAS=OFF`.

### SIMD Compilation Errors

If you encounter errors related to AVX2/NEON:
1.  Ensure your CPU supports the instruction set.
2.  Disable SIMD: `-DMATRIX_USE_SIMD=OFF`.

### Windows MSVC Issues

- Ensure you are using Visual Studio 2019 or later.
- Use the "x64 Native Tools Command Prompt" for 64-bit builds.
