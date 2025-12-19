# Build Guide

## Quick Start

**Using just (recommended):**

```bash
just build    # Build in release mode
just test     # Run tests
just bench    # Run benchmark
just dev      # Build + test
```

**Using build script:**

```bash
./build.sh
```

**Manual build:**

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./matrix-test
```

## Just Commands

[just](https://github.com/casey/just) is a modern command runner. Install with `cargo install just` or `brew install just`.

```bash
just               # List all commands
just build         # Release build
just debug         # Debug build
just blas          # Build with BLAS
just fast          # Clean + SIMD + BLAS
just clean         # Remove build directory
just rebuild       # Clean + build
just test          # Run tests
just bench         # Run benchmark
just dev           # Build + test
just info          # Show build info
```

## Build Script Options

```bash
./build.sh [OPTIONS]

OPTIONS:
  -h, --help          Show help
  -c, --clean         Clean before build
  -d, --debug         Debug build
  -b, --blas          Enable BLAS
  -n, --no-simd       Disable SIMD
  -t, --no-tests      Skip tests
  -e, --no-examples   Skip examples
```

**Examples:**

```bash
./build.sh                      # Standard release build
./build.sh --clean --blas       # Clean + BLAS
./build.sh --debug              # Debug build
```

## CMake Options

| Option                  | Default | Description                    |
| ----------------------- | ------- | ------------------------------ |
| `CMAKE_BUILD_TYPE`      | Release | `Release` or `Debug`           |
| `MATRIX_USE_SIMD`       | ON      | SIMD optimizations (AVX2/NEON) |
| `MATRIX_USE_BLAS`       | OFF     | Use BLAS for matrix ops        |
| `MATRIX_BUILD_TESTS`    | ON      | Build test suite               |
| `MATRIX_BUILD_EXAMPLES` | ON      | Build examples                 |
| `MATRIX_BUILD_SHARED`   | OFF     | Build shared lib (.so/.dylib)  |
| `MATRIX_HEADER_ONLY`    | ON      | Header-only mode               |

**Examples:**

```bash
# Standard optimized build
cmake .. -DCMAKE_BUILD_TYPE=Release

# With BLAS for maximum performance
cmake .. -DMATRIX_USE_BLAS=ON

# Debug build without optimizations
cmake .. -DCMAKE_BUILD_TYPE=Debug -DMATRIX_USE_SIMD=OFF

# Production build (shared lib, no tests/examples)
cmake .. -DMATRIX_BUILD_SHARED=ON -DMATRIX_BUILD_TESTS=OFF -DMATRIX_BUILD_EXAMPLES=OFF
```

## BLAS Setup

**macOS:**

```bash
brew install openblas
cmake .. -DMATRIX_USE_BLAS=ON
```

**Linux:**

```bash
sudo apt-get install libopenblas-dev
cmake .. -DMATRIX_USE_BLAS=ON
```
