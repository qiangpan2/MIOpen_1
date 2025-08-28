#!/bin/bash

DEPS_PREFIX="${HOME}/miopen-deps"

# 配置 CMake with proper GPU target flags
echo "Configuring CMake..."
cmake -B build \
    -DCMAKE_PREFIX_PATH="${DEPS_PREFIX}" \
    -DCMAKE_INSTALL_PREFIX="${DEPS_PREFIX}" \
    -DMIOPEN_BACKEND=HIP \
    -DMIOPEN_USE_COMPOSABLEKERNEL=OFF \
    -DMIOPEN_USE_CKTILE_COMPOSABLEKERNEL=ON \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
    -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DGPU_TARGETS="gfx1100" \
    -DBUILD_TESTING=OFF \
    -G Ninja --debug-output

# 构建项目
cmake --build build -j8 > build.log 2>&1

# 安装项目
echo "Installing MIOpen..."
cmake --install build

echo "Build completed successfully!"
echo "Dependencies and MIOpen installed to ${DEPS_PREFIX}"
echo "To use with PyTorch, set CMAKE_PREFIX_PATH=${DEPS_PREFIX}"