import argparse
from mlir.dialects import linalg
from compilation import scalar_types, compile_and_callback
from search import collect_variables, instantiate_variables
import experts


def parse_args():
  parser = argparse.ArgumentParser(description='Command-line directed search.')
  parser.add_argument(
      'op', type=str, help='Name of the linalg op to instantiate.')
  parser.add_argument(
      '--types',
      type=str,
      default='f32',
      help='Comma-separated list of scalar types.')
  parser.add_argument(
      '--range',
      type=str,
      default='128,1025,128',
      help='Comma-separated range for dimension variables.')
  parser.add_argument(
      '--expert',
      type=str,
      default='ExpertCompiler1',
      help='Name of the expert to use for compilation.')
  return parser.parse_args()


def validate_args(args):
  no_errors = True

  def error(msg):
    nonlocal no_errors
    no_errors = False
    print(msg)

  if not hasattr(linalg, args.op):
    error(f'Unknown op: {args.op}.')

  range_parts = args.range.split(',')
  if len(range_parts) < 1 or len(range_parts) > 3:
    error(f'Value range should be defined using 1, 2 or 3 integer values.')
  for part in range_parts:
    try:
      i = int(part)
    except:
      error(f'Failed to parse integer value: {part}')
  range_parts = map(int, range_parts)
  value_range = range(*range_parts)
  if len(value_range) < 1:
    error(f'Range must not be empty: {value_range}')

  for type_name in args.types.split(','):
    if type_name not in scalar_types:
      error(f'Unknown type: {type_name}')

  if not hasattr(experts, args.expert):
    error(f'Unknown expert name: {args.expert}')

  return no_errors


def main():
  args = parse_args()
  if not validate_args(args):
    return
  print(args)
  op = getattr(linalg, args.op)
  range_parts = map(int, args.range.split(','))
  value_range = range(*range_parts)
  types = [scalar_types[n] for n in args.types.split(',')]
  expert = getattr(experts, args.expert)
  variables = collect_variables(op, types, value_range)
  variables.extend(expert.variables)
  assignments = instantiate_variables(variables)
  print(assignments)
  compile_and_callback(op, expert(**assignments), lambda x: None, **assignments)


if __name__ == '__main__':
  main()
