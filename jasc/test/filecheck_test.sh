#!/bin/bash

die() { echo "$*" 1>&2 ; exit 1; }

TARGET=$1
TESTFILE="${TEST_SRCDIR}/jasc/test/$TARGET"
$TESTFILE || die "Failure in Python test: \"$TARGET\""
$TESTFILE | "$TEST_SRCDIR/llvm-project/llvm/FileCheck" "$TESTFILE.py" || die "Failure in FileCheck test: \"$TARGET\""

echo "PASS"
