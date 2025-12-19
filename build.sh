#!/bin/bash
# Matrix Library C++ Build Script

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
BUILD_TYPE="Release"
BUILD_DIR="build"
SIMD="ON"
BLAS="OFF"
TESTS="ON"
EXAMPLES="ON"
CLEAN=false

# Print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build the Matrix Library C++ project.

OPTIONS:
    -h, --help          Show this help message
    -c, --clean         Clean build directory before building
    -d, --debug         Build in Debug mode (default: Release)
    -b, --blas          Enable BLAS support (default: OFF)
    -n, --no-simd       Disable SIMD optimizations (default: ON)
    -t, --no-tests      Skip building tests (default: ON)
    -e, --no-examples   Skip building examples (default: ON)
    --build-dir DIR     Specify build directory (default: build)

EXAMPLES:
    $0                                      # Standard release build with SIMD
    $0 --clean --blas                       # Clean build with BLAS
    $0 --debug --no-simd                    # Debug build without SIMD
    $0 --clean --blas --no-tests            # Production build with BLAS, no tests

EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -d|--debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        -b|--blas)
            BLAS="ON"
            shift
            ;;
        -n|--no-simd)
            SIMD="OFF"
            shift
            ;;
        -t|--no-tests)
            TESTS="OFF"
            shift
            ;;
        -e|--no-examples)
            EXAMPLES="OFF"
            shift
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Print configuration
echo -e "${GREEN}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         Matrix Library C++ Build Script              ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  Build Type:    $BUILD_TYPE"
echo "  Build Dir:     $BUILD_DIR"
echo "  SIMD:          $SIMD"
echo "  BLAS:          $BLAS"
echo "  Tests:         $TESTS"
echo "  Examples:      $EXAMPLES"
echo ""

# Clean if requested
if [ "$CLEAN" = true ]; then
    echo -e "${YELLOW}Cleaning build directory...${NC}"
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo -e "${YELLOW}Configuring with CMake...${NC}"
cmake .. \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DMATRIX_USE_SIMD="$SIMD" \
    -DMATRIX_USE_BLAS="$BLAS" \
    -DMATRIX_BUILD_TESTS="$TESTS" \
    -DMATRIX_BUILD_EXAMPLES="$EXAMPLES"

# Build
echo ""
echo -e "${YELLOW}Building...${NC}"
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Success message
echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Build completed successfully!            ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"
echo ""

# Print next steps
if [ "$TESTS" = "ON" ]; then
    echo -e "${YELLOW}Run tests:${NC}        ./$BUILD_DIR/matrix-test"
fi
if [ "$EXAMPLES" = "ON" ]; then
    echo -e "${YELLOW}Run demo:${NC}         ./$BUILD_DIR/matrix-demo"
    echo -e "${YELLOW}Run benchmark:${NC}    ./$BUILD_DIR/matrix-benchmark"
fi
echo ""
