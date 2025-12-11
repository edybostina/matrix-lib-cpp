# Build Guide

## Quick Start

```bash
mkdir build && cd build
cmake ..
make
./matrix-test
```

This will now build in **header-only mode** by default, which supports all types including `matrix<int>`, `matrix<float>`, `matrix<double>`, etc.

## Build Modes

### 1. Header-Only Mode (Default) ✅

```bash
cmake ..
# or explicitly:
cmake -DMATRIX_HEADER_ONLY=ON ..
```

**Pros:**

- ✅ Supports ALL types: `int`, `float`, `double`, `long`, etc.
- ✅ No linking issues
- ✅ Works with examples and tests out of the box

**Cons:**

- ⚠️ Longer compilation time (templates compiled in every translation unit)

### 2. Compiled Library Mode (float/double only)

```bash
cmake -DMATRIX_HEADER_ONLY=OFF ..
```

**Pros:**

- ✅ Faster compilation (explicit instantiations pre-compiled)
- ✅ Smaller object files for users

**Cons:**

- ⚠️ Only supports `matrix<float>` and `matrix<double>`
- ⚠️ Integer types (`matrix<int>`, `matrix<long>`) not pre-compiled
- ⚠️ Examples/tests won't link (they use `matrix<int>`)

**Why only float/double?** Integer types have issues with:

- `std::uniform_real_distribution` in `random()` (requires floating-point)
- Ambiguous `std::abs()` calls with unsigned types

## Other CMake Options

```bash
# Build shared library (.dylib) instead of static (.a)
cmake -DMATRIX_BUILD_SHARED=ON ..

# Disable examples
cmake -DMATRIX_BUILD_EXAMPLES=OFF ..

# Disable tests
cmake -DMATRIX_BUILD_TESTS=OFF ..

# Disable install target
cmake -DMATRIX_INSTALL=OFF ..

# Combine multiple options
cmake -DMATRIX_HEADER_ONLY=OFF -DMATRIX_BUILD_EXAMPLES=OFF ..
```

## Recommended Setup

### For Development

```bash
cmake -DMATRIX_HEADER_ONLY=ON -DMATRIX_BUILD_EXAMPLES=ON -DMATRIX_BUILD_TESTS=ON ..
```

### For Production (using only float/double)

```bash
cmake -DMATRIX_HEADER_ONLY=OFF -DMATRIX_BUILD_EXAMPLES=OFF -DMATRIX_BUILD_TESTS=OFF ..
```

## Troubleshooting

### Error: "Undefined symbols for architecture arm64"

**Problem:** Building in compiled mode but using `matrix<int>` or other non-float/double types.

**Solution:** Use header-only mode:

```bash
cd build
cmake -DMATRIX_HEADER_ONLY=ON ..
make
```

### Error: Long compilation times

**Problem:** Header-only mode compiles templates in every file.

**Solution:** If you only use `float` and `double`, switch to compiled mode:

```bash
cd build
cmake -DMATRIX_HEADER_ONLY=OFF ..
make
```

## Default Configuration

The library now defaults to **header-only mode** (`MATRIX_HEADER_ONLY=ON`) to avoid linking issues with examples and tests. This provides the best out-of-the-box experience.

If you want faster builds and only need `float`/`double`, explicitly set `-DMATRIX_HEADER_ONLY=OFF`.
