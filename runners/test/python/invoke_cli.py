import argparse
import json
from mlir.dialects import linalg
from compilation import compile_and_callback
from search import collect_variables
import experts
from search_cli import parse_assignments


def parse_args():
  parser = argparse.ArgumentParser(description='Command-line directed search.')
  parser.add_argument(
      '--op', type=str, help='Name of the linalg op to instantiate.')
  parser.add_argument(
      '--expert', type=str, help='Name of the expert to use for compilation.')
  parser.add_argument(
      '--assign',
      type=str,
      nargs='+',
      help='A sequence of K=V arguments to specify op or expert variables.')
  return parser.parse_args()


def validate_args(args):
  no_errors = True

  def error(msg):
    nonlocal no_errors
    no_errors = False
    print(msg)

  if not hasattr(linalg, args.op):
    error(f'Unknown op: {args.op}.')

  if not hasattr(experts, args.expert):
    error(f'Unknown expert name: {args.expert}')

  op = getattr(linalg, args.op)
  expert = getattr(experts, args.expert)
  variables = collect_variables(op, ['f32'], range(1, 2))
  variables.extend(expert.variables)
  assignments = parse_assignments(args)
  for variable in variables:
    if variable.name not in assignments:
      error(f'Variable {variable.name} was not assigned.')

  if no_errors:
    return (op, expert, assignments)
  else:
    return None


def invoke(op, expert, assignments):
  compile_and_callback(op, expert(**assignments), lambda x: None, **assignments)


def main():
  args = parse_args()
  validated = validate_args(args)
  if validated is None:
    return
  invoke(*validated)


if __name__ == '__main__':
  main()
