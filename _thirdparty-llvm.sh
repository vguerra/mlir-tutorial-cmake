#!/bin/sh

set -e

git submodule update --init --recursive

# Xcode, Ninja
BUILD_SYSTEM=Ninja
BUILD_TAG=ninja

THIRDPARTY_LLVM_DIR=$PWD/thirdparty/llvm-project
BUILD_DIR=$THIRDPARTY_LLVM_DIR/build
INSTALL_DIR=$THIRDPARTY_LLVM_DIR/install

mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
pushd $BUILD_DIR

cmake ../llvm -G $BUILD_SYSTEM \
      -DCMAKE_CXX_COMPILER="$(xcrun --find clang++)" \
      -DCMAKE_C_COMPILER="$(xcrun --find clang)" \
      -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
      -DLLVM_LOCAL_RPATH=$INSTALL_DIR/lib \
      -DLLVM_PARALLEL_COMPILE_JOBS=7 \
      -DLLVM_PARALLEL_LINK_JOBS=1 \
      -DLLVM_BUILD_EXAMPLES=ON \
      -DLLVM_INSTALL_UTILS=ON \
      -DCMAKE_OSX_ARCHITECTURES="$(uname -m)" \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DLLVM_CCACHE_BUILD=ON \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DLLVM_ENABLE_PROJECTS='mlir' \
      -DDEFAULT_SYSROOT="$(xcrun --show-sdk-path)" \
      -DCMAKE_OSX_SYSROOT="$(xcrun --show-sdk-path)"
cmake --build . --target check-mlir

popd

