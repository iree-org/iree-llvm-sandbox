# This file contains a matmul microkernel performance search.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from ..contraction.definitions import EinsumProblem

################################################################################
# Compilation strategies.
################################################################################

all_experts = [
    e.print_ir(after_all=False, at_begin=False, llvm=False) for e in [
        Vectorize('matmul_on_tensors', 'linalg.generic')
        .then(LoweringOnlyExpert('matmul_on_tensors',
                                 'linalg.generic',
                                 transpose_lowering='eltwise')),
        Vectorize('matmul_on_tensors', 'linalg.generic')
        .then(LoweringOnlyExpert('matmul_on_tensors',
                                 'linalg.generic',
                                 transpose_lowering='shuffle')),
    ]]

################################################################################
# Problem instantiations.
################################################################################

keys = ['m', 'n', 'k']


def make_size_list(sizes: Sequence):
    return {k: v for k, v in zip(keys, sizes)}


def make_microkernel_sizes(min_m: int, max_m: int,
                           min_n: int, max_n: int,
                           min_k: int, max_k: int,
                           min_product_size: int, max_product_size: int,
                           multiple_of: int):
    import itertools
    from functools import reduce
    result = []
    for sizes in itertools.product(range(min_m, max_m + 1),
                                   range(min_n, max_n + 1),
                                   range(min_k, max_k + 1)):
        total = reduce(lambda x, y: x * y, sizes, 1)
        if total >= min_product_size and total <= max_product_size and \
                total % multiple_of == 0:
            result.append(list(sizes))
    import random
    random.shuffle(result)
    return result


def main():
    base_m = 5
    step_m = 30
    base_n = 5
    step_n = 30
    base_k = 5
    step_k = 30
    problem_size_list = make_microkernel_sizes(min_m=base_m, max_m=base_m+step_m,
                                               min_n=base_n, max_n=base_n+step_n,
                                               min_k=base_k, max_k=base_k+step_k,
                                               min_product_size=2048, max_product_size=10000,
                                               multiple_of=8)

    for runtime_only in [
        [],  # case 1: static at compile time
    ]:
        def filter_out_result(gf, gb):
          if gf < 110:
            return True
          return False

        test_harness(lambda s, t: EinsumProblem('mk,kn'), [[np.float32] * 3],
                     map(make_size_list, problem_size_list),
                     all_experts,
                     n_iters=1000,
                     runtime_only_sizes=set(runtime_only),
                     function_name='matmul_on_tensors',
                     # dump_ir_to_file='/tmp/abc.mlir',
                     # dump_obj_to_file='/tmp/abc.o'
                     filter_out_result=filter_out_result)


if __name__ == '__main__':
    main()
