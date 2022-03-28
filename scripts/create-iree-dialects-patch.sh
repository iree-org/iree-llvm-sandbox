#!/bin/bash

set -ex

COMMIT=$1
echo ${COMMIT}
git show ${COMMIT} -- lib/Dialect/LinalgExt include/Dialect/LinalgExt test/Dialect/iree_linalg_ext \
  lib/Dialect/LinalgTransform include/Dialect/LinalgTransform test/Dialect/linalg_transform \
  > /tmp/patch.patch

sed -i "s:include/Dialect:include/iree-dialects/Dialect:g" /tmp/patch.patch

echo "Run: 'patch -p1 < /tmp/patch.patch from iree/llvm-external-projects/iree-dialects/'"
