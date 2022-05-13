#!/usr/bin/env python3

import argparse
import subprocess
from typing import Sequence

def parse_cli():
  parser = argparse.ArgumentParser(description="""
    Utility to provide an e2e compile, benchmark, report cycle that:
      1. extracts the dispatch regions from an iree model, 
      2. applies some transform-dialect-based transformations, 
      3. runs through iree-benchmarks.

    Given -benchmark-output-dir=/tmp/foo, creates intermediate files for each dispatch region:
       /tmp/foo/xxx_dispatch_xxx.mlir
       /tmp/foo/xxx_dispatch_xxx.mlir.vmfb
       /tmp/foo/xxx_dispatch_xxx.mlir.strategy.mlir

    Usage:
    ======

    python -m python.tools.iree_benchmarks \
      -iree-bin-dir=${IREE_BUILD_DIR} \
      -mlir-file=${IREE_SOURCE_DIR}/iree/test/e2e/models/fullyconnected.mlir \
      -benchmark-output-dir=/tmp/foo 
  """)

  parser.add_argument(
    '-benchmark-output-dir', 
    default='', 
    help='full path to benchmark output directory',
    required=True)
  parser.add_argument(
    '-iree-bin-dir', 
    default='', 
    help='full path to iree tools directory',
    required=True)
  parser.add_argument(
    '-iree-input-type', 
    default='mhlo',
    help='iree input IR type')
  parser.add_argument(
    '-iree-target-backends', 
    default='dylib', 
    help='target backends to compile for')
  parser.add_argument(
    '-mlir-file',
    default='',
    help='Full path to mlir file',
    required=True)
  parser.add_argument(
    '-benchmark-batch-size', 
    default=1, 
    help='benchmark batch size')
  parser.add_argument(
    '-benchmark-repetitions', 
    default=25,
    help='benchmark repetitions')
  parser.add_argument(
    '-benchmark-min-time', 
    default=0.01,
    help="""benchmark min time (note: this seems similar to gbench and broken 
    for autotuning; it seens to perform a number K of runs until min time is 
    reached and seemd to report the mean)""")
  parser.add_argument(
    '-debug', 
    default=False, action=argparse.BooleanOptionalAction, 
    help='debug (i.e. dump IR)')
  args = vars(parser.parse_args())
  return args


def generate_dispatch_regions(args) -> Sequence[str]:
  """Run iree-compile to produce the dispatch regions."""

  benchmark_output_dir = args['benchmark_output_dir']
  iree_compile_bin = args['iree_bin_dir'] + '/iree/tools/iree-compile'
  iree_input_type = args['iree_input_type']
  iree_target_backends = args['iree_target_backends']
  mlir_file = args['mlir_file']

  cmd_list = [iree_compile_bin] + \
    [mlir_file] + \
    ['--iree-hal-dump-executable-benchmarks-to=' + benchmark_output_dir] + \
    ['--iree-hal-target-backends=' + iree_target_backends] + \
    ['--iree-mlir-to-vm-bytecode-module'] + \
    ['--iree-input-type=' + iree_input_type] + \
    ['-o', '/dev/null']
  print(' '.join(cmd_list))
  subprocess.run(cmd_list)

  benchmark_output_dir = args['benchmark_output_dir']
  import glob
  return set(glob.glob(benchmark_output_dir + '/*dispatch*.mlir')) - \
         set(glob.glob(benchmark_output_dir + '/*strategy.mlir'))


def transform_dispatch_region(
    args, 
    dispatch_mlir_filename: str,
    strategy_filename : str = None) -> Sequence[float]:
  """Run iree-translate on a given dispatch region in an mlir file to produce .vmfb
     and run iree-benchmark-module on that .vmfb file.
     Return the list of times for the dispatch region.
     This is where we should inject transform dialect commands."""

  iree_target_backends = args['iree_target_backends']
  iree_translate_bin = args['iree_bin_dir'] + '/iree/tools/iree-translate'
  cmd_list = [iree_translate_bin] + \
    [dispatch_mlir_filename] + \
    ['-o', dispatch_mlir_filename + '.vmfb'] + \
    ['--iree-hal-target-backends=' + iree_target_backends] + \
    ['--iree-mlir-to-vm-bytecode-module']
  
  if strategy_filename is not None:
    cmd_list = cmd_list + \
      ['-iree-codegen-use-linalg-transform-interp'] + \
      ['-linalg-transform-file-name=' + strategy_filename]

  debug = args['debug']
  if debug:
    cmd_list = cmd_list + \
      ['-mlir-print-ir-after-all'] + \
      ['-mlir-print-ir-after-change']

  print(' '.join(cmd_list))
  subprocess.run(cmd_list)


def benchmark_dispatch_region(
    args, 
    dispatch_mlir_filename: str) -> Sequence[float]:
  """Run iree-benchmark-module to dump json and then parse the times."""

  iree_target_backends = args['iree_target_backends']
  iree_benchmark_module_bin = args['iree_bin_dir'] + '/iree/tools/iree-benchmark-module'
  benchmark_batch_size = args['benchmark_batch_size']
  benchmark_repetitions = args['benchmark_repetitions']
  benchmark_min_time = args['benchmark_min_time']
  cmd_list = [iree_benchmark_module_bin] + \
    ['--batch_size=' + str(benchmark_batch_size)] + \
    ['--module_file=' + dispatch_mlir_filename + '.vmfb'] + \
    ['--driver=' + iree_target_backends] + \
    ['--benchmark_format=json'] + \
    ['--benchmark_format_out=json'] + \
    ['--benchmark_out=/tmp/benchmarks_out'] + \
    ['--benchmark_report_aggregates_only=false'] + \
    ['--benchmark_display_aggregates_only=false'] + \
    ['--benchmark_repetitions=' + str(benchmark_repetitions)] + \
    ['--benchmark_min_time=' + str(benchmark_min_time)]
  print(' '.join(cmd_list))
  p = subprocess.Popen(cmd_list, stdout=subprocess.PIPE)
  p = subprocess.Popen(['grep', '\"real_time\"'], stdin=p.stdout, stdout=subprocess.PIPE)
  
  # Drop mean, median, stddev and cv.
  p = subprocess.Popen(['head', '-n', '-4'], stdin=p.stdout, stdout=subprocess.PIPE)
  output = p.communicate()[0].decode("utf-8") 
  p.stdout.close()
  output = output.replace('"real_time":', '').replace('\n', '').replace(',', '')
  times = [float(x) for x in output.split()]
  return times


def make_strategy_as_serialized_mlir(strategy_filename : str):
  """Create a fixed strategy with the transform dialect and save it to file."""

  import iree.compiler.dialects.transform as transform
  import iree.compiler.dialects.pdl as pdl
  import iree.compiler.ir as ir

  with ir.Context() as ctx, ir.Location.unknown(ctx):
    transform.register_dialect(ctx)

    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
      root = transform.WithPDLPatternsOp(root=None)
      with ir.InsertionPoint(root.body.blocks[0]):
        isa_matmul = pdl.PatternOp(benefit = 1, name = "isa_matmul")
        with ir.InsertionPoint(isa_matmul.body):
          args = pdl.OperandsOp()
          types = pdl.TypesOp()
          pdl_op = pdl.OperationOp(args=[args], types=[types])
          op_name = pdl.AttributeOp(value=ir.StringAttr.get("linalg.matmul"))
          pdl.ApplyNativeConstraintOp("isEquivalentToOp", args=[pdl_op, op_name])
          pdl.RewriteOp(pdl_op, "transform.dialect")

        sequence = transform.CanonicalizedSequenceOp(root.body.blocks[0].arguments[0])
        sequence_block = sequence.body.blocks[0]
        with ir.InsertionPoint(sequence_block):
          # transform.PrintOp(None, name="Initial IR")
          # ir.Operation.create(name="transform.iree.set_num_workgroups_to_one")
          target_match = transform.PDLMatchOp(sequence_block.arguments[0], "isa_matmul")
          # TODO: fuse...
          tiled = transform.TileOp(target=target_match, sizes=[0, 0, 1])
          # transform.PrintOp(None, name="After tiling")

          # TODO: peeling is disabled for now
          # transform.PeelLoopOp(tiled.results[1])
          # transform.PeelLoopOp(tiled.results[2])
          # TODO: Match dynamic matmul and scalarize.
          transform.VectorizeOp(vectorize_padding=False)
          # transform.PrintOp(None, name="After vectorization")

          ir.Operation.create(name="transform.iree.bufferize")
          # transform.PrintOp(None, name="After bufferization")
          
          stages = []
          for i in range(1, 8):
            stages.append(i)
            transform.LowerVectorsOp(contraction_lowering="outerproduct",
                                    multireduction_lowering="innerparallel",
                                    split_transfers="linalg-copy",
                                    stages=stages,
                                    transpose_avx2_lowering=False,
                                    transpose_lowering="shuffle",
                                    unroll_vector_transfers=True)
          transform.PrintOp(None, name="After lowering vectors")
          transform.YieldOp([])

    with open(strategy_filename, "w") as f:
      f.write(str(module))


args = parse_cli()
dispatch_regions = generate_dispatch_regions(args)
for dispatch_mlir_filename in dispatch_regions:
  strategy_filename = dispatch_mlir_filename + '.strategy.mlir'
  make_strategy_as_serialized_mlir(strategy_filename)
  transform_dispatch_region(args, dispatch_mlir_filename, strategy_filename)
  times = benchmark_dispatch_region(args, dispatch_mlir_filename)
  print(f'times: {times}\n')
