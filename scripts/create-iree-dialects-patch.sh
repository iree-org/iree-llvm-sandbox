#!/bin/bash

set -ex

COMMIT=$1
echo ${COMMIT}
git show ${COMMIT} -- lib/Dialect/LinalgExt include/Dialect/LinalgExt test/Dialect/iree_linalg_ext \
  lib/Dialect/LinalgTransform include/Dialect/LinalgTransform test/Dialect/linalg_transform \
  > /tmp/patch.patch

sed -i "s:include/Dialect:include/iree-dialects/Dialect:g" /tmp/patch.patch

echo "Run:"
echo "cd ${IREE_SOURCE_DIR}/llvm-external-projects/iree-dialects/ && git status --porcelain && patch -p1 < /tmp/patch.patch"
