# Matrix Library C++

A modern, header-only C++ matrix library for basic and advanced matrix operations. Supports creation, manipulation, and arithmetic operations on matrices with a clean, intuitive API.

---

## Features

- Generic matrix class (`matrix<T>`) supporting any numeric type
- Matrix creation: zeros, ones, identity
- Arithmetic operations: addition, subtraction, multiplication (matrix and scalar)
- Determinant and inverse calculation (for square matrices)
- Transpose, adjoint, and cofactor
- LU and QR decomposition
- Eigenvalues and eigenvectors
- Bounds-checked element access
- Header-only design (just include `matrix.hpp`)
- Multithreading
- More features to come!

---

## Getting Started

### Prerequisites

- C++17 compatible compiler
- Make (for building)
- CMake (optional, for building with CMake)

### Building

Use the provided `Makefile` or `CMake` to build the library and run tests.

```sh
make
```

or

```sh
cmake -S . -B build
cmake --build build
```

This will build the library and run the test suite in [`tests/main.cpp`](tests/main.cpp).

---

## Usage

Include the header in your project:

```cpp
#include "include/matrix.hpp"
```

### Example

```cpp
#include "include/matrix.hpp"
#include <iostream>
using namespace std;

int main() {
    matrix<double> A = {
    {1, 2},
    {3, 4}};

    matrix<double> B = A.transpose();
    matrix<double> C = A + B;
    matrix<double> D = A * 2.0;
    double det = A.determinant();
    matrix<double> inv = A.inverse();

    cout << "A:\n" << A;
    // ... and so on
}
```

---

## API

See [docs/api.md](docs/api.md) for full documentation.

---

## Project Structure

```
.
├── include/         # Header files (matrix.hpp)
├── tests/           # Test suite (main.cpp)
├── examples/        # Example usage (demo.cpp)
├── docs/            # API documentation
├── Makefile         # Build script
├── CMakeLists.txt   # CMake build script
└── README.md        # This file
└── LICENSE          # License file
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements, bug fixes, or new features.

## Acknowledgements

- [GeekforGeeks](https://www.geeksforgeeks.org/) for inspiration and examples.
- [Wikipedia](https://www.wikipedia.org/) for mathematical definitions and properties.
