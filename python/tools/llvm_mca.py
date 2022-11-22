#!/usr/bin/env python3
import argparse
import subprocess

cpu_choices = [
    'nehalem', 'sandybridge', 'ivybridge', 'haswell', 'skylake-avx512',
    'cortex-a34', 'cortex-a53', 'cortex-a78'
]

arch_choices = ['x86-64', 'arm64']

opt_flags = lambda args: [
    '-O3',
    f'-march={args["a"]}',
    f'-mcpu={args["c"]}',
]

llc_flags = lambda args: [
    '-O3',
    '--function-sections',  # Important to isolate functions and pass to objdump
    '-filetype=obj',
    f'-march={args["a"]}',
    f'-mcpu={args["c"]}',
]

# Note: llvm-mca requires a processor to run properly,
# otherwise it will default to the host processor and make a mess.
llvm_mca_flags = lambda args: [
    '--all-stats',
    '--timeline',
    '--bottleneck-analysis',
    f'-march={args["a"]}',
    f'-mcpu={args["c"]}',
]

objdump_flags = lambda fn: [
    '-d',
    f'--disassemble-symbols={fn}',
    '--no-leading-headers',
    '--no-show-raw-insn',
    '--no-leading-addr',
    '-M',
    'att',
]


def objdump_and_llvm_mca(args, obj_file):
  fn = args['f']

  # Run llvm-objdump to produce the interesting portion of asm.
  asm_file = obj_file + '.S'
  f = open(asm_file, 'w')
  objdump = args['llvm_objdump']
  p = subprocess.Popen([objdump] + objdump_flags(fn) + [obj_file],
                       stdout=subprocess.PIPE)
  print(' '.join(p.args))
  p = subprocess.Popen(['tail', '-n', '+7'], stdin=p.stdout, stdout=f)
  p.wait()
  f.close()

  # Run llvm-objdump to produce the full asm for debugging.
  full_asm_file = obj_file + '-full.S'
  f = open(full_asm_file, 'w')
  subprocess.run([objdump] + ['-d', obj_file], stdout=f)
  f.close()

  # Run llvm-mca on asm
  llvm_mca_out_file = obj_file + '_llvm_mca.out'
  llvm_mca = args['llvm_mca']
  p = subprocess.run([llvm_mca] + llvm_mca_flags(args) + [asm_file] + ['-o'] +
                     [llvm_mca_out_file])
  print(' '.join(p.args))

  # Dump 10 lines of llvm-mca to stdout.
  subprocess.run(['head', '-n', '10', llvm_mca_out_file])


# Run opt and llc to produce a .o
def compile_to_object(args):
  mlir_file = args['mlir_file']
  mlir_translate = args['mlir_translate']
  ll_file = mlir_file + '.ll'
  prun = subprocess.run([mlir_translate] + ['--mlir-to-llvmir'] + [mlir_file] +
                        ['-o'] + [ll_file])
  print(' '.join(prun.args))

  llvm_opt = args['llvm_opt']
  prun = subprocess.Popen(['cat'] + [ll_file], stdout=subprocess.PIPE)
  prun = subprocess.Popen([llvm_opt] + opt_flags(args),
                          stdin=prun.stdout,
                          stdout=subprocess.PIPE)
  print(' '.join(prun.args))

  obj_file = mlir_file + '.o'
  f = open(obj_file, 'w')
  llvm_llc = args['llvm_llc']
  prun = subprocess.Popen([llvm_llc] + llc_flags(args) + ['-o'] + [obj_file],
                          stdin=prun.stdout,
                          stdout=prun.stdout)
  print(' '.join(prun.args))
  prun.wait()
  f.close()

  return obj_file


def main():
  parser = argparse.ArgumentParser(description="""
    Utility to compile to obj and instrument with llvm-mca.
    Given -mlir-file=/tmp/foo.mlir, this creates the intermediate files:
       /tmp/foo.mlir.ll
       /tmp/foo.mlir.o

    Given -obj-file=/tmp/foo.mlir.o, this creates the intermediate files
       /tmp/foo.mlir.o.S
       /tmp/foo.mlir-full.o.S
       /tmp/foo.mlir.o_llvm_mca.out

    Usage:
    ======

    python -m python.tools.llvm_mca \
      -mlir-translate=${LLVM_BUILD_DIR}/bin/mlir-translate \
      -llvm-objdump=${LLVM_BUILD_DIR}/bin/llvm-objdump \
      -llvm-llc=${LLVM_BUILD_DIR}/bin/llc \
      -llvm-opt=${LLVM_BUILD_DIR}/bin/opt \
      -llvm-mca=${LLVM_BUILD_DIR}/bin/llvm-mca \
      -f=fun_name_to_get_info_about \
      [-mlir-file=/tmp/foo.mlir] \
      [-obj-file=/tmp/foo.o] \
  """)
  parser.add_argument('-llvm-llc', default='', help='full path to llc')
  parser.add_argument('-llvm-mca', default='', help='full path to llvm-mca')
  parser.add_argument('-llvm-objdump',
                      default='',
                      help='full path to llvm-objdump')
  parser.add_argument('-llvm-opt', default='', help='full path to opt')
  parser.add_argument('-mlir-translate',
                      default='',
                      help='full path to mlir-translate')

  parser.add_argument(
      '-mlir-file',
      default='',
      help='(optional) full path to mlir file in the llvm dialect')
  parser.add_argument('-obj-file',
                      default='',
                      help='(optional) full path to obj file')

  parser.add_argument('-c',
                      '-cpu',
                      default='skylake-avx512',
                      choices=cpu_choices,
                      help='cpu to compile for')
  parser.add_argument('-a',
                      '-arch',
                      default='x86-64',
                      choices=arch_choices,
                      help='arch to compile for')
  parser.add_argument('-f',
                      '-fn',
                      help='name of the function to run through llvm_mca')
  args = vars(parser.parse_args())

  if 'obj_file' in args:
    objdump_and_llvm_mca(args, args['obj_file'])
  else:
    objdump_and_llvm_mca(args, compile_to_object(args))


main()
