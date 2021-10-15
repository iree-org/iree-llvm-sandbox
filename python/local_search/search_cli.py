# pytype: skip-file

import os
import argparse
import multiprocessing
import subprocess
import json
import hashlib
import pathlib
import shlex
from random import choice

from mlir.dialects import linalg

from ..core.compilation import scalar_types
from ..core.search_vars import collect_variables, instantiate_variables, are_constraints_satisfied
from ..core import experts, search_space


def parse_args(argv):
  parser = argparse.ArgumentParser(description='Command-line directed search.')
  search_space.extend_argument_parser(parser)
  parser.add_argument(
      '--invoke-cli',
      default='python3 invoke_cli',
      help='Location of the benchmark invocation command.')
  parser.add_argument(
      '--par',
      type=int,
      default=1,
      help='Number of search processes to run in parallel.')
  parser.add_argument(
      '--use-llvm-mca',
      type=bool,
      default=False,
      help='Use llvm-mca to report information about generated code.')
  return parser.parse_args(argv[1:])


def validate_args(args):
  no_errors = True

  def error(msg):
    nonlocal no_errors
    no_errors = False
    print(msg)

  if args.par < 1:
    error('Paralelism must be equal to 1 or larger.')

  no_errors = no_errors and search_space.validate_args(args)

  return no_errors


def run_in_parallel(target, kwargs, num_workers):
  for _ in range(num_workers):
    p = multiprocessing.Process(target=target, kwargs=kwargs)
    p.start()


def main(argv):
  args = parse_args(argv)
  if not validate_args(args):
    return

  if args.par > 1:
    run_in_parallel(target=search, kwargs={'args': args}, num_workers=args.par)
  else:
    search(args)


def search(args):
  def collect_sample():
    print('Searching for random assignment...')
    assignments = search_space.find_sample(args)
    expert_name = assignments['expert_name']
    print('Done: ' + str(assignments))
    invoke_subprocess(args.invoke_cli, args.op, expert_name, assignments,
                      args.output, args.timeout, args.iters, args.runs,
                      args.use_llvm_mca)

  if args.samples > 0:
    for _ in range(args.samples):
      collect_sample()
  else:
    while True:
      collect_sample()


def invoke_subprocess(invoke_cli, op, expert, assignments, output_dir, timeout,
                      iters, runs, use_llvm_mca):
  file_dirname = os.path.dirname(__file__)
  command = invoke_cli.split(' ') + [
      '--op', op, '--expert', expert, '--iters',
      str(iters), '--runs',
      str(runs), '--assign',
      json.dumps(assignments)
  ]

  def assignments_id():
    sorted_assignments = sorted(list(assignments.items()))
    key = json.dumps(sorted_assignments).encode(encoding='UTF-8')
    h = hashlib.new('sha256')
    h.update(key)
    return h.hexdigest()

  def ensure_exists(dir_path):
    path = pathlib.Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)

  def split_sections(output):
    section_name = ''
    sections = {}
    buf = []

    for line in output.split('\n'):
      if line.startswith('--- '):
        if len(buf) > 0:
          sections[section_name] = buf
          buf = []
        section_name = line[4:]
      else:
        buf.append(line)

    if len(buf) > 0:
      sections[section_name] = buf

    return sections

  def persist(status, output):
    uniq_id = assignments_id()

    result_dir = os.path.join(output_dir, op, expert, status)
    base_output = os.path.join(result_dir, uniq_id)
    ensure_exists(result_dir)

    with open(base_output + '.sh', 'w') as f:
      command_str = ' '.join(map(shlex.quote, command))
      f.write(command_str)

    for section, lines in split_sections(output).items():
      suffix = '.' + section if len(section) > 0 else ''
      with open(base_output + suffix, 'w') as f:
        f.write('\n'.join(lines))

    print(f'[{status.upper()}] {command_str}')
    return base_output

  try:
    print(command)
    result = subprocess.run(
        command,
        timeout=timeout,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        check=True)
    output_path = persist('ok', result.stdout.decode('utf-8'))
    if use_llvm_mca:
      invoke_llvm_mca(base_path)
  except subprocess.CalledProcessError as e:
    persist('fail', e.stderr.decode('utf-8'))
  except subprocess.TimeoutExpired as e:
    persist('timeout', '')


def invoke_llvm_mca(base_path):
  mlir_path = base_path + '.mlir'
  ll_path = base_path + '.ll'
  s_path = base_path + '.s'
  mca_path = base_path + '.mca'

  def annotate_llvm_mca_region():
    lines = open(s_path).readlines()
    did_begin = False
    did_end = False
    transformed = []
    for line in lines:
      if 'retq' in line and not did_end:
        transformed.append('# LLVM-MCA-END\n')
        did_end = True
      transformed.append(line)
      if 'prologue_end' in line and not did_begin:
        transformed.append('# LLVM-MCA-BEGIN\n')
        did_begin = True
    open(s_path, 'w').writelines(transformed)

  subp.run(['mlir-translate', '-mlir-to-llvmir', mlir_path, '-o', ll_path])
  subp.run(['llc', '-O3', '-mcpu=skylake-avx512', ll_path, '-o', s_path])
  annotate_llvm_mca_region()
  subp.run(['llvm-mca', '-mcpu=skylake-avx512', s_path, '-o', mca_path])


if __name__ == '__main__':
  import sys
  main(sys.argv)
