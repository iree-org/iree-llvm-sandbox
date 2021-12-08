#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import sys
import argparse
from .utils import parse, run_command, print_command
from .compile_op import build_mlir


if __name__ == "__main__":
    parser = argparse.ArgumentParser("mlirc")

    # GEMM size
    parser.add_argument("--M", type=int)    
    parser.add_argument("--N", type=int)    
    parser.add_argument("--K", type=int)    

    # Outer tiling
    parser.add_argument("--tile-sizes", nargs='+', type=int)
    parser.add_argument("--reorder-tile-sizes", nargs='+', type=int)

    # Inner tiling
    parser.add_argument("--register-tile-sizes", nargs='+', type=int)
    parser.add_argument("--reorder-register-tile-sizes", nargs='+', type=int)
    parser.add_argument("--hoist-packing", nargs='+', type=int)

    # Vector lowering
    parser.add_argument("--unroll-vector-transfers", action="store_true")
    parser.add_argument("--split-vector-transfers-to")

    # micro-kernel transforms
    parser.add_argument("--extract-micro-kernel", action="store_true")
    parser.add_argument("--modulo-scheduling", action="store_true")

    # Verbosity
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--verbosity-level", type=int, default=0)
    parser.add_argument("--reps", type=int, default=1)

    args = parser.parse_args()

    stringify = lambda l : ','.join([str(e) for e in l])
    options = { "tile_sizes" : stringify(args.tile_sizes), 
                "register_tile_sizes" : stringify(args.register_tile_sizes),
                "split_vector_transfers_to" : args.split_vector_transfers_to,
                "unroll_vector_transfers" : args.unroll_vector_transfers,
                "reorder_tile_sizes": stringify(args.reorder_tile_sizes),
                "reorder_register_tile_sizes": stringify(args.reorder_register_tile_sizes),
                "hoist_packing": stringify(args.hoist_packing),
                "extract_micro_kernel": args.extract_micro_kernel,
                "modulo_scheduling": args.modulo_scheduling,
                "verbosity_level" : 0,
                "reps": args.reps
    }

    if (args.verbose):
        options["verbosity_level"]=1
    if (args.verbosity_level > 0):
        options["verbosity_level"]=args.verbosity_level
    build_mlir("gemm", args.M, args.N, args.K, options)
