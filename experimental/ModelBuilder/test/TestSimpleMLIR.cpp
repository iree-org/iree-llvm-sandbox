//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Most tests use the ModelBuilder to build a model and then
// use the ModelRunner on some test input and CHECK for the
// proper results in the output. The two tests in this file
// show how to CHECK the MLIR that is generated after building
// a model using the dump method on the resulting module.
//
// Both tests consists of computing the sum of two vectors. The
// first test builds the input vector values inside the function,
// using a dimensionless dynamically allocated memref for backing
// storage of the 8x128 vectors. The second test takes 1-D memrefs
// as parameters for backing storage of the 8x128 vectors.

// RUN: test-simple-mlir 2>&1 | IreeFileCheck %s

#include "experimental/ModelBuilder/MemRefUtils.h"
#include "experimental/ModelBuilder/ModelBuilder.h"
#include "experimental/ModelBuilder/ModelRunner.h"

using namespace mlir;  // NOLINT

void testValueVectorAdd() {
  constexpr unsigned M = 8, N = 128;

  ModelBuilder modelBuilder;
  constexpr StringLiteral kFuncName = "test_value_vector_add";
  auto f32 = modelBuilder.f32;
  auto mnVectorType = modelBuilder.getVectorType({M, N}, f32);
  auto typeA = modelBuilder.getMemRefType({}, mnVectorType);
  auto typeB = modelBuilder.getMemRefType({}, mnVectorType);
  auto typeC = modelBuilder.getMemRefType({}, mnVectorType);

  // 1. Build a simple vector_add.
  {
    // CHECK-LABEL: func @test_value_vector_add()
    auto f = modelBuilder.makeFunction(
        kFuncName, {}, {}, MLIRFuncOpConfig().setEmitCInterface(true));
    OpBuilder b(&f.getBody());
    edsc::ScopedContext scope(b, f.getLoc());

    // CHECK: %[[A:.*]] = memref.alloc() : memref<vector<8x128xf32>>
    // CHECK: %[[B:.*]] = memref.alloc() : memref<vector<8x128xf32>>
    // CHECK: %[[C:.*]] = memref.alloc() : memref<vector<8x128xf32>>
    auto bufferA = memref_alloc(typeA);
    auto bufferB = memref_alloc(typeB);
    auto bufferC = memref_alloc(typeC);

    MemRefIndexedValue A(bufferA), B(bufferB), C(bufferC);

    // CHECK: %[[C1:.*]] = arith.constant 1.000000e+00 : f32
    // CHECK: %[[S1:.*]] = vector.broadcast %[[C1]] : f32 to vector<8x128xf32>
    // CHECK: memref.store %[[S1]], %[[A]][] : memref<vector<8x128xf32>>
    auto one = std_constant_float(llvm::APFloat(1.0f), f32);
    A() = (vector_broadcast(mnVectorType, one));

    // CHECK: %[[C2:.*]] = arith.constant 2.000000e+00 : f32
    // CHECK: %[[S2:.*]] = vector.broadcast %[[C2]] : f32 to vector<8x128xf32>
    // CHECK: memref.store %[[S2]], %[[B]][] : memref<vector<8x128xf32>>
    auto two = std_constant_float(llvm::APFloat(2.0f), f32);
    B() = (vector_broadcast(mnVectorType, two));

    // CHECK-DAG: %[[a:.*]] = memref.load %[[A]][] : memref<vector<8x128xf32>>
    // CHECK-DAG: %[[b:.*]] = memref.load %[[B]][] : memref<vector<8x128xf32>>
    //     CHECK: %[[c:.*]] = arith.addf %[[a]], %[[b]] : vector<8x128xf32>
    //     CHECK: memref.store %[[c]], %[[C]][] : memref<vector<8x128xf32>>
    C() = A() + B();

    // CHECK: %[[p:.*]] = memref.load %[[C]][] : memref<vector<8x128xf32>>
    // CHECK: vector.print %[[p]] : vector<8x128xf32>
    (vector_print(C()));

    std_ret();
  }

  // 2. Dump IR for FileCheck.
  modelBuilder.getModuleRef()->dump();
}

void testMemRefVectorAdd() {
  constexpr unsigned M = 8, N = 128;

  ModelBuilder modelBuilder;
  constexpr StringLiteral kFuncName = "test_memref_vector_add";
  auto f32 = modelBuilder.f32;
  auto mnVectorType = modelBuilder.getVectorType({M, N}, f32);
  auto typeA = modelBuilder.getMemRefType({1}, mnVectorType);
  auto typeB = modelBuilder.getMemRefType({1}, mnVectorType);
  auto typeC = modelBuilder.getMemRefType({1}, mnVectorType);

  // 1. Build a simple vector_add.
  {
    // CHECK-LABEL: func @test_memref_vector_add(
    //       CHECK-SAME: %[[A:.*0]]: memref<1xvector<8x128xf32>>,
    //       CHECK-SAME: %[[B:.*1]]: memref<1xvector<8x128xf32>>,
    //       CHECK-SAME: %[[C:.*2]]: memref<1xvector<8x128xf32>>)
    auto f = modelBuilder.makeFunction(kFuncName, {}, {typeA, typeB, typeC});
    OpBuilder b(&f.getBody());
    edsc::ScopedContext scope(b, f.getLoc());

    // CHECK-DAG: %[[z:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[a:.*]] = memref.load %[[A]][%[[z]]] :
    // memref<1xvector<8x128xf32>> CHECK-DAG: %[[b:.*]] = memref.load
    // %[[B]][%[[z]]] : memref<1xvector<8x128xf32>>
    //     CHECK: %[[c:.*]] = arith.addf %[[a]], %[[b]] : vector<8x128xf32>
    //     CHECK: memref.store %[[c]], %[[C]][%[[z]]] :
    //     memref<1xvector<8x128xf32>>
    MemRefIndexedValue A(f.getArgument(0)), B(f.getArgument(1)),
        C(f.getArgument(2));
    Value idx_0 = std_constant_index(0);
    C(idx_0) = A(idx_0) + B(idx_0);

    // CHECK: %[[p:.*]] = memref.load %[[C]][%[[z]]] :
    // memref<1xvector<8x128xf32>> CHECK: vector.print %[[p]] :
    // vector<8x128xf32>
    (vector_print(C(idx_0)));

    std_ret();
  }

  // 2. Dump IR for FileCheck.
  modelBuilder.getModuleRef()->dump();
}

int main(int argc, char **argv) {
  testValueVectorAdd();
  testMemRefVectorAdd();
}
