// RUN: mlir-proto-opt %s \
// RUN:   -convert-iterators-to-llvm \
// RUN:   -convert-states-to-llvm \
// RUN:   -convert-func-to-llvm \
// RUN:   -convert-scf-to-cf -convert-cf-to-llvm \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN: | FileCheck %s

// Original query in SQL:
//
//   SELECT
//           SUM(l_extendedprice * l_discount) AS revenue
//   FROM
//           lineitem
//   WHERE
//           l_shipdate >= date '1994-01-01'
//           AND l_shipdate < date '1995-01-01'
//           AND l_discount BETWEEN 0.06 - 0.01 AND 0.06 + 0.01
//           AND l_quantity < 24
//
// Attributes used in the query and their physical representations:
//
//   - l_extendedprice: i32 (in cents)
//   - l_discount: i8 (in 0.01 percent)
//   - l_shipdate: i16 (in number of days since epoch)
//   - l_quantity: i8
//
// Query constants in physical representation:
//
//    - date '1994-01-01' (l_shipdate): 8766 : i16
//    - date '1995-01-01' (l_shipdate): 9131 : i16
//    - 0.06 - 0.01 (l_discount): 5 : i8
//    - 0.06 + 0.01 (l_discount): 7 : i8
//    - 24 (l_quantity): 24 : i8

func.func private @q6_predicate(%input : !llvm.struct<(i8,i32,i8,i16)>) -> i1 {
  %quantity = llvm.extractvalue %input[0 : index] : !llvm.struct<(i8,i32,i8,i16)>
  %discount = llvm.extractvalue %input[2 : index] : !llvm.struct<(i8,i32,i8,i16)>
  %shipdate = llvm.extractvalue %input[3 : index] : !llvm.struct<(i8,i32,i8,i16)>

  // Test lower bound on shipdate.
  %c8766 = arith.constant 8677 : i16
  %cmp_shipdate_lb = arith.cmpi sge, %shipdate, %c8766 : i16
  %result = scf.if %cmp_shipdate_lb -> i1 {
    // Test upper bound on shipdate.
    %c9131 = arith.constant 9131 : i16
    %cmp_shipdate_ub = arith.cmpi slt, %shipdate, %c9131 : i16
    %result = scf.if %cmp_shipdate_ub -> i1 {
      // Test lower bound on discount.
      %c5 = arith.constant 5 : i8
      %cmp_discount_lb = arith.cmpi sge, %discount, %c5 : i8
      %result = scf.if %cmp_discount_lb -> i1 {
        // Test upper bound on discount.
        %c7 = arith.constant 7 : i8
        %cmp_discount_ub = arith.cmpi sle, %discount, %c7 : i8
        %result = scf.if %cmp_discount_ub -> i1 {
          // Test quantity.
          %c24 = arith.constant 24 : i8
          %cmp_quantity = arith.cmpi slt, %quantity, %c24 : i8
          scf.yield %cmp_quantity : i1
        } else {
          scf.yield %cmp_discount_ub : i1
        }
        scf.yield %result : i1
      } else {
        scf.yield %cmp_discount_lb : i1
      }
      scf.yield %result : i1
    } else {
      scf.yield %cmp_shipdate_ub : i1
    }
    scf.yield %result : i1
  } else {
    scf.yield %cmp_shipdate_lb : i1
  }
  return %result : i1
}

func.func private @compute_discounted_price(%input : !llvm.struct<(i8,i32,i8,i16)>)
    -> !llvm.struct<(i32)> {
  %extendedprice = llvm.extractvalue %input[1 : index] : !llvm.struct<(i8,i32,i8,i16)>
  %discount = llvm.extractvalue %input[2 : index] : !llvm.struct<(i8,i32,i8,i16)>
  %discount_i32 = llvm.zext %discount : i8 to i32
  %resulti = arith.muli %extendedprice, %discount_i32 : i32
  %undef = llvm.mlir.undef : !llvm.struct<(i32)>
  %result = llvm.insertvalue %resulti, %undef[0 : index] : !llvm.struct<(i32)>
  return %result : !llvm.struct<(i32)>
}

func.func private @sum_struct(%lhs : !llvm.struct<(i32)>,
                              %rhs : !llvm.struct<(i32)>)
    -> !llvm.struct<(i32)> {
  %lhsi = llvm.extractvalue %lhs[0 : index] : !llvm.struct<(i32)>
  %rhsi = llvm.extractvalue %rhs[0 : index] : !llvm.struct<(i32)>
  %i = arith.addi %lhsi, %rhsi : i32
  %result = llvm.insertvalue %i, %lhs[0 : index] : !llvm.struct<(i32)>
  return %result : !llvm.struct<(i32)>
}

func.func @main() {
  // Constant data from order 70, the first with more than one qualifying row.
  %lineitem = "iterators.constantstream"()
      { value = [
        // quantity, extendedprice, discnt, shipdate
        [  8 : i8,  873696 : i32, 3 : i8, 8777 : i16 ],
        [ 13 : i8, 1627795 : i32, 6 : i8, 8827 : i16 ],
        [  1 : i8,  188880 : i32, 3 : i8, 8791 : i16 ],
        [ 11 : i8, 1847703 : i32, 1 : i8, 8841 : i16 ],
        [ 37 : i8, 3952081 : i32, 9 : i8, 8809 : i16 ],
        [ 19 : i8, 3060235 : i32, 6 : i8, 8791 : i16 ]] }
      : () -> (!iterators.stream<!llvm.struct<(i8,i32,i8,i16)>>)

  // Apply filter from WHERE clause.
  %filtered = "iterators.filter"(%lineitem) { predicateRef = @q6_predicate }
    : (!iterators.stream<!llvm.struct<(i8,i32,i8,i16)>>)
      -> (!iterators.stream<!llvm.struct<(i8,i32,i8,i16)>>)

  // Project to l_extendedprice * l_discount (in 1/100 cents).
  %mapped = "iterators.map"(%filtered) { mapFuncRef = @compute_discounted_price }
    : (!iterators.stream<!llvm.struct<(i8,i32,i8,i16)>>)
      -> (!iterators.stream<!llvm.struct<(i32)>>)

  // Sum up values.
  %reduced = "iterators.reduce"(%mapped) { reduceFuncRef = @sum_struct }
    : (!iterators.stream<!llvm.struct<(i32)>>)
      -> (!iterators.stream<!llvm.struct<(i32)>>)

  "iterators.sink"(%reduced)
    : (!iterators.stream<!llvm.struct<(i32)>>) -> ()
  // CHECK: (28128180)
  return
}
