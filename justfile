# Matrix Library C++ - Justfile
# Modern command runner for building and testing

# Default recipe (list all available commands)
default:
    @just --list

# Build the project in release mode
build:
    @echo "Building in Release mode..."
    @mkdir -p build
    @cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DMATRIX_USE_SIMD=ON
    @cd build && cmake --build . -j{{num_cpus()}}
    @echo "Build complete!"

# Build in debug mode
debug:
    @echo "Building in Debug mode..."
    @mkdir -p build
    @cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug -DMATRIX_USE_SIMD=ON
    @cd build && cmake --build . -j{{num_cpus()}}
    @echo "Debug build complete!"

# Build with BLAS support
blas:
    @echo "Building with BLAS support..."
    @mkdir -p build
    @cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DMATRIX_USE_SIMD=ON -DMATRIX_USE_BLAS=ON
    @cd build && cmake --build . -j{{num_cpus()}}
    @echo "BLAS build complete!"

# Build without SIMD optimizations
no-simd:
    @echo "Building without SIMD..."
    @mkdir -p build
    @cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DMATRIX_USE_SIMD=OFF
    @cd build && cmake --build . -j{{num_cpus()}}
    @echo "Build complete!"

# Clean build directory
clean:
    @echo "Cleaning build directory..."
    @rm -rf build
    @echo "Clean complete!"

# Rebuild from scratch
rebuild: clean build

# Run tests
test:
    @echo "Running tests..."
    @./build/matrix-test

# Run demo
demo:
    @echo "Running demo..."
    @./build/matrix-demo

# Run benchmark
bench:
    @echo "Running benchmark..."
    @./build/matrix-benchmark

# Build and run tests
check: build test

# Build and run all examples
run: build demo bench

# Install to system
install:
    @echo "Installing..."
    @cd build && sudo cmake --build . --target install
    @echo "Installation complete!"

# Configure only (no build)
config:
    @mkdir -p build
    @cd build && cmake .. -DCMAKE_BUILD_TYPE=Release

# Build with all optimizations (SIMD + BLAS)
fast: clean
    @echo "Building with maximum optimizations..."
    @mkdir -p build
    @cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DMATRIX_USE_SIMD=ON -DMATRIX_USE_BLAS=ON
    @cd build && cmake --build . -j{{num_cpus()}}
    @echo "Fast build complete!"

# Build production-ready shared library
prod: clean
    @echo "Building production shared library..."
    @mkdir -p build
    @cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DMATRIX_BUILD_SHARED=ON -DMATRIX_BUILD_TESTS=OFF -DMATRIX_BUILD_EXAMPLES=OFF
    @cd build && cmake --build . -j{{num_cpus()}}
    @echo "Production build complete!"

# Format all source files (if clang-format is available)
fmt:
    @echo "Formatting source files..."
    @find include -name "*.hpp" -o -name "*.tpp" | xargs clang-format -i
    @find src -name "*.cpp" | xargs clang-format -i
    @find examples -name "*.cpp" | xargs clang-format -i
    @find tests -name "*.cpp" | xargs clang-format -i
    @echo "Formatting complete!"

# Show build configuration
info:
    @echo "Matrix Library Build Information"
    @echo "================================="
    @echo "CPU cores: {{num_cpus()}}"
    @echo "OS: {{os()}}"
    @echo "Arch: {{arch()}}"
    @if [ -d build ]; then \
        echo "Build directory: exists"; \
        if [ -f build/CMakeCache.txt ]; then \
            echo "Configured: yes"; \
            grep CMAKE_BUILD_TYPE build/CMakeCache.txt | head -1; \
            grep MATRIX_USE_SIMD build/CMakeCache.txt | head -1; \
            grep MATRIX_USE_BLAS build/CMakeCache.txt | head -1; \
        else \
            echo "Configured: no"; \
        fi; \
    else \
        echo "Build directory: not found"; \
    fi

# Quick build and test cycle
dev: build test
    @echo "Development cycle complete!"

# Create release archive
archive:
    @echo "Creating release archive..."
    @tar czf matrix-lib-cpp-release.tar.gz \
        --exclude='build' \
        --exclude='.git' \
        --exclude='*.o' \
        --exclude='*.a' \
        --exclude='*.so' \
        --exclude='*.tar.gz' \
        .
    @echo "Archive created: matrix-lib-cpp-release.tar.gz"
