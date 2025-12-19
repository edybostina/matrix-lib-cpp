cmake_minimum_required(VERSION 3.14)
project(matrix-lib-cpp 
    VERSION 0.2.0 
    LANGUAGES CXX
    DESCRIPTION "A modern C++ matrix library"
)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags (compiler-specific)
if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /W4 /Zi")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /W4 /DNDEBUG")
else()
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -Wall -Wextra")
endif()

# Options
option(MATRIX_BUILD_EXAMPLES "Build example programs" ON)
option(MATRIX_BUILD_TESTS "Build test programs" ON)
option(MATRIX_HEADER_ONLY "Use header-only mode (recommended)" ON)
option(MATRIX_BUILD_SHARED "Build shared library instead of static" OFF)
option(MATRIX_USE_BLAS "Use BLAS for optimized matrix operations" ON)
option(MATRIX_USE_SIMD "Enable SIMD optimizations (AVX2/NEON)" ON)

# Find BLAS if enabled
if(MATRIX_USE_BLAS)
    find_package(BLAS)
    if(BLAS_FOUND)
        message(STATUS "BLAS found: ${BLAS_LIBRARIES}")
    else()
        message(WARNING "BLAS not found, disabling BLAS support")
        set(MATRIX_USE_BLAS OFF)
    endif()
endif()

# Create library target
if(MATRIX_HEADER_ONLY)
    message(STATUS "Building as header-only library")
    add_library(matrix INTERFACE)
    target_include_directories(matrix INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_compile_features(matrix INTERFACE cxx_std_17)
    
    if(MATRIX_USE_BLAS AND BLAS_FOUND)
        target_compile_definitions(matrix INTERFACE MATRIX_USE_BLAS)
        target_link_libraries(matrix INTERFACE ${BLAS_LIBRARIES})
    endif()
    
    if(MATRIX_USE_SIMD)
        target_compile_definitions(matrix INTERFACE MATRIX_USE_SIMD)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)")
            message(STATUS "Enabling ARM NEON SIMD support")
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|i686)")
            message(STATUS "Enabling AVX2 SIMD support")
            if(MSVC)
                target_compile_options(matrix INTERFACE /arch:AVX2)
            else()
                target_compile_options(matrix INTERFACE -mavx2 -mfma)
            endif()
        endif()
    endif()
else()
    # Build as compiled library
    if(MATRIX_BUILD_SHARED)
        message(STATUS "Building as shared library")
        add_library(matrix SHARED src/matrix_lib.cpp)
    else()
        message(STATUS "Building as static library")
        add_library(matrix STATIC src/matrix_lib.cpp)
    endif()
    
    target_include_directories(matrix PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_compile_features(matrix PUBLIC cxx_std_17)
    
    if(MATRIX_USE_BLAS AND BLAS_FOUND)
        target_compile_definitions(matrix PUBLIC MATRIX_USE_BLAS)
        target_link_libraries(matrix PUBLIC ${BLAS_LIBRARIES})
    endif()
    
    if(MATRIX_USE_SIMD)
        target_compile_definitions(matrix PUBLIC MATRIX_USE_SIMD)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)")
            message(STATUS "Enabling ARM NEON SIMD support")
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|i686)")
            message(STATUS "Enabling AVX2 SIMD support")
            if(MSVC)
                target_compile_options(matrix PUBLIC /arch:AVX2)
            else()
                target_compile_options(matrix PUBLIC -mavx2 -mfma)
            endif()
        endif()
    endif()
endif()

add_library(matrix::matrix ALIAS matrix)

if(MATRIX_BUILD_EXAMPLES)
    add_executable(matrix-demo examples/demo.cpp)
    target_link_libraries(matrix-demo PRIVATE matrix::matrix)
    
    add_executable(matrix-benchmark examples/benchmark.cpp)
    target_link_libraries(matrix-benchmark PRIVATE matrix::matrix)
endif()

if(MATRIX_BUILD_TESTS)
    add_executable(matrix-test tests/main.cpp)
    target_link_libraries(matrix-test PRIVATE matrix::matrix)
    
    enable_testing()
    add_test(NAME matrix-test COMMAND matrix-test)
endif()

if(MATRIX_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)
    
    install(DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING 
        PATTERN "*.hpp"
        PATTERN "*.tpp"
    )
    
    install(TARGETS matrix
        EXPORT matrix-targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    
    install(EXPORT matrix-targets
        FILE matrix-targets.cmake
        NAMESPACE matrix::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/matrix
    )
    
    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/matrix-config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/matrix-config.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/matrix
    )
    
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/matrix-config-version.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/matrix-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/matrix-config-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/matrix
    )
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "═══════════════════════════════════════════════════════")
message(STATUS "Matrix Library Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
if(MATRIX_HEADER_ONLY)
    message(STATUS "  Library type: Header-only (INTERFACE)")
elseif(MATRIX_BUILD_SHARED)
    message(STATUS "  Library type: Shared library (.so/.dylib/.dll)")
else()
    message(STATUS "  Library type: Static library (.a/.lib)")
endif()
message(STATUS "  Build examples: ${MATRIX_BUILD_EXAMPLES}")
message(STATUS "  Build tests: ${MATRIX_BUILD_TESTS}")
message(STATUS "  Install target: ${MATRIX_INSTALL}")
message(STATUS "  BLAS support: ${MATRIX_USE_BLAS}")
message(STATUS "  SIMD support: ${MATRIX_USE_SIMD}")
message(STATUS "═══════════════════════════════════════════════════════")
message(STATUS "")
