#!/bin/bash

set -ex

function repopulate_iree_dialect() {
  rm -Rf include/Dialect/$1 lib/Dialect/$1
  cp -R -f ../iree/llvm-external-projects/iree-dialects/include/iree-dialects/Dialect/$1 include/Dialect/
  cp -R -f ../iree/llvm-external-projects/iree-dialects/lib/Dialect/$1 lib/Dialect/
}

function repopulate_iree_dir() {
  rm -Rf include/$1 lib/$1
  cp -R -f ../iree/llvm-external-projects/iree-dialects/include/$1 include/
  cp -R -f ../iree/llvm-external-projects/iree-dialects/lib/$1 lib/
}

repopulate_iree_dialect LinalgExt
repopulate_iree_dialect LinalgTransform

repopulate_iree_dir Transforms


# Copy python files.
cp ../iree/llvm-external-projects/iree-dialects/python/iree/compiler/dialects/_iree_linalg_transform_ops_ext.py python/sandbox/dialects/
cp ../iree/llvm-external-projects/iree-dialects/python/iree/compiler/dialects/iree_linalg_ext.py python/sandbox/dialects/
cp ../iree/llvm-external-projects/iree-dialects/python/iree/compiler/dialects/iree_linalg_transform.py python/sandbox/dialects/
cp ../iree/llvm-external-projects/iree-dialects/python/iree/compiler/dialects/IreeLinalgExtBinding.td python/sandbox/dialects/
cp ../iree/llvm-external-projects/iree-dialects/python/iree/compiler/dialects/LinalgTransformBinding.td python/sandbox/dialects/

# Fix include paths.
git grep -l iree-dialects/Dialect/ | grep -v scripts | xargs sed -i "s:iree-dialects/Dialect/:Dialect/:g"

# Drop building of IREE's LinalgExt/Passes that depends on the IREEInputDialect.
git grep -l "add_subdirectory(Passes)" | grep LinalgExt | xargs sed -i "s:add_subdirectory(Passes)::g"

# Drop IREE python import, we do our own registration.
git grep -l "from .._mlir_libs._ireeDialects" | grep -v scripts | xargs sed -i "s:from .._mlir:\# from .._mlir:g"