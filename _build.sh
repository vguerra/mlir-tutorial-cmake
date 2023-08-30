#!/bin/sh

rm -rf build
mkdir build

pushd build

LLVM_BUILD_DIR=$1

cmake -G Ninja .. \
    -DLLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm" \
    -DMLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_BUILD_TYPE=Debug

popd

cmake --build ./build --target MLIRAffineFullUnrollPasses
cmake --build ./build --target mlir-doc # or AffinePassesDocGen
cmake --build ./build --target tutorial-opt
cmake --build ./build --target check-mlir-tutorial

ln -fs ./build/compile_commands.json
ln -fs ./build/tablegen_compile_commands.yml