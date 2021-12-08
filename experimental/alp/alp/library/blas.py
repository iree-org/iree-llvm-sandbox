typest = """
!memref_type_A = type tensor<_K_x_M_xf32>
!memref_type_B = type tensor<_K_x_N_xf32>
!memref_type_C = type tensor<_M_x_N_xf32>
"""

types = """
!memref_type_A = type tensor<_M_x_K_xf32>
!memref_type_B = type tensor<_K_x_N_xf32>
!memref_type_C = type tensor<_M_x_N_xf32>
"""

init_tensors = """
  %A0 = linalg.init_tensor [_M_,_K_] : !memref_type_A
  %B0 = linalg.init_tensor [_K_,_N_] : !memref_type_B
  %C = linalg.init_tensor [_M_, _N_] : !memref_type_C
"""

init_tensors_t = """
  %A0 = linalg.init_tensor [_K_,_M_] : !memref_type_A
  %B0 = linalg.init_tensor [_K_,_N_] : !memref_type_B
  %C = linalg.init_tensor [_M_, _N_] : !memref_type_C
"""


gemm_benchmark = f"""
func @main() -> i32 {{
  call @print_pid() : () -> ()
  __INIT_TENSORS__

  %elem = arith.constant 1.0 : f32
  %A = linalg.fill(%elem, %A0) : f32, !memref_type_A -> !memref_type_A
  %B = linalg.fill(%elem, %B0) : f32, !memref_type_B -> !memref_type_B

  %out = call @gemm(%A, %B, %C) : (!memref_type_A, !memref_type_B, !memref_type_C) -> !memref_type_C
  %reps = arith.constant _REPS_ : index
  %t_start = call @rtclock() : () -> f64
  affine.for %arg0 = 0 to %reps {{
    call @gemm(%A, %B, %C) : (!memref_type_A, !memref_type_B, !memref_type_C) -> !memref_type_C
  }}
  %t_end = call @rtclock() : () -> f64
  %repsi = arith.index_cast %reps : index to i64
  %repsf = arith.sitofp %repsi: i64 to f64
  %t_tot = arith.subf %t_end, %t_start : f64
  %t = arith.divf %t_tot, %repsf : f64

  call @print_time(%t) : (f64) -> ()

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %M = tensor.dim %C, %c0 : !memref_type_C
  %N = tensor.dim %C, %c1 : !memref_type_C
  %K = tensor.dim %A, %c0 : !memref_type_A

  %Mi32 = arith.index_cast %M: index to i64
  %Ni32 = arith.index_cast %N: index to i64
  %Ki32 = arith.index_cast %K: index to i64

  %c2 = arith.constant 2 : i64
  %f1 = arith.muli %Mi32, %Ni32 : i64
  %f2 = arith.muli %f1, %Ki32 : i64
  %f3 = arith.muli %c2, %f2 : i64

  // 2*M*N*K.
  %num_flops_f = arith.sitofp %f3: i64 to f64
  %flops = arith.divf %num_flops_f, %t : f64
  call @print_flops(%flops) : (f64) -> ()
   
   %i0 = arith.constant 0 : i32
  return %i0 : i32
}}


func private @print_flops(f64)
func private @print_time(f64)
func private @printNewline()
func private @print_pid()
func private @rtclock() -> f64
func private @print_memref_f32(memref<*xf32>)
func private @gemm(%A : !memref_type_A, %B : !memref_type_B, %C : !memref_type_C)  -> !memref_type_C
"""




GEMM = """
func @gemm(%A : !memref_type_A {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false}, 
           %B : !memref_type_B {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false}, 
           %C : !memref_type_C {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = true}) -> !memref_type_C {
    %0 = linalg.generic
      {indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                        affine_map<(m, n, k) -> (k, n)>,
                        affine_map<(m, n, k) -> (m, n)>],
       iterator_types = ["parallel", "parallel", "reduction"]}

      ins(%A, %B: !memref_type_A, !memref_type_B)
      outs(%C: !memref_type_C) {
      ^bb0(%a: f32, %b: f32, %c: f32) :
        %d = arith.mulf %a, %b: f32
        %e = arith.addf %c, %d: f32
        linalg.yield %e : f32
      } -> !memref_type_C
    return %0 : !memref_type_C
  }
"""

GEMM_T = """
func @gemm(%A : !memref_type_A {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false}, 
           %B : !memref_type_B {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false}, 
           %C : !memref_type_C {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = true}) -> !memref_type_C {
    %0 = linalg.generic
      {indexing_maps = [affine_map<(m, n, k) -> (k, m)>,
                        affine_map<(m, n, k) -> (k, n)>,
                        affine_map<(m, n, k) -> (m, n)>],
       iterator_types = ["parallel", "parallel", "reduction"]}

      ins(%A, %B: !memref_type_A, !memref_type_B)
      outs(%C: !memref_type_C) {
      ^bb0(%a: f32, %b: f32, %c: f32) :
        %d = arith.mulf %a, %b: f32
        %e = arith.addf %c, %d: f32
        linalg.yield %e : f32
      } -> !memref_type_C
    return %0 : !memref_type_C
  }
"""

def gemm(trA):
  if trA:
    bench = gemm_benchmark.replace("__INIT_TENSORS__", str(init_tensors_t))
    return (typest + bench, typest + GEMM_T)
  else:
    bench = gemm_benchmark.replace("__INIT_TENSORS__", str(init_tensors))
    return (types + bench, types+ GEMM)
