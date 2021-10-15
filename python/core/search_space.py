# pytype: skip-file

from random import choice

from mlir.dialects import linalg

from ..core.search_vars import collect_variables, instantiate_variables, are_constraints_satisfied
from ..core.compilation import scalar_types
from ..core import experts


def extend_argument_parser(parser):
  parser.add_argument(
      '--op',
      type=str,
      default='matmul',
      help='Name of the linalg op to instantiate.')
  parser.add_argument(
      '--types',
      type=str,
      default='f32',
      help='Comma-separated list of scalar types.')
  parser.add_argument(
      '--dim_range',
      type=str,
      default='128,513,128',
      help='Range of potential dimension values.')
  parser.add_argument(
      '--int_range',
      type=str,
      default='128,1025,128',
      help='Range of potential int values.')
  parser.add_argument(
      '--tsize_length_range',
      type=str,
      default='default=3,4,1',
      help='Ranges of potential lengths for tiling sizes either specified for '
      'all tiling hierarchies =\"default=3,4,1\" or for all of '
      'them =\"sizes3=3,4,1 sizes2=3,4,1 sizes=3,4,1\"')
  parser.add_argument(
      '--tsize_value_range',
      type=str,
      default='sizes3=8,33,8 sizes2=32,129,16 sizes=128,513,32',
      help='Ranges of potential values for tiling sizes either specified for '
      'all tiling hierarchies =\"default=0,513,32\" or for all of '
      'them =\"sizes3=8,33,8 sizes2=32,129,16 sizes=128,513,32\"')
  parser.add_argument(
      '--tsize_register_tile_bound',
      type=int,
      default=256,
      help='Upper bound for the register tiling sizes.')
  parser.add_argument(
      '--ppad_length_range',
      type=str,
      default='default=3,4,1',
      help='Ranges of potential operand indices to pack either specified for '
      'all tiling hierarchies =\"default=3,4,1\" or for all of '
      'them =\"pack_padding0=0,3,4 pack_padding1=0,1,1\"')
  parser.add_argument(
      '--hpad_length_range',
      type=str,
      default='default=2,3,1',
      help='Ranges of potential lengths for hoist padding depths specified for '
      'all hoist padding variables =\"default==0,3,1\" or for all of '
      'them =\"hoist_padding0=0,3,1 hoist_padding1=0,3,1\"')
  parser.add_argument(
      '--hpad_value_range',
      type=str,
      default='default=0,4,1',
      help='Ranges of potential values for hoist padding depths specified for '
      'all hoist padding variables =\"default=0,513,32\" or for all of '
      'them =\"hoist_padding0=0,4,1 hoist_padding1=0,7,1\"')
  parser.add_argument(
      '--experts',
      type=str,
      default='SingleTilingExpert',
      help='Comma-separated list of possible experts to use for compilation.')
  parser.add_argument(
      '--samples',
      type=int,
      default=100,
      help='Number of samples to collect in each search process (0 for unlimited).'
  )
  parser.add_argument(
      '--timeout',
      type=int,
      default=120,
      help='Timeout for running a subprocess in seconds.')
  parser.add_argument(
      '--output',
      type=str,
      default='/tmp/sandbox-output',
      help='The output directory to accumulate search results.')
  parser.add_argument(
      '--iters',
      type=int,
      default=100,
      help='Number of iterations of the MLIR loop.')
  parser.add_argument(
      '--runs',
      type=int,
      default=10,
      help='Number of times the MLIR program is run to measure runtime.')


def validate_args(args):
  no_errors = True

  def error(msg):
    nonlocal no_errors
    no_errors = False
    print(msg)

  def validate_range_str(range_str):
    range_parts = range_str.split(',')
    if len(range_parts) < 1 or len(range_parts) > 3:
      error(
          f'{range_name} range should be defined using 1, 2 or 3 integer values.'
      )
    for part in range_parts:
      try:
        i = int(part)
      except:
        error(f'Failed to parse element in {range_name} range: {part}')
    range_parts = map(int, range_parts)
    value_range = range(*range_parts)
    if len(value_range) < 1:
      error(f'Must have at least one value in {range_name} range.')

  def validate_range(range_name):
    range_str = getattr(args, range_name)
    validate_range_str(range_str)

  def validate_named_range(named_range_name):
    named_range_str = getattr(args, named_range_name)
    str_parts = named_range_str.split(' ')
    for str_part in str_parts:
      assign_parts = str_part.split('=')
      if len(assign_parts) != 2:
        error(
            f'{assign_parts} should contain assignments of the form name=range.'
        )
      validate_range_str(assign_parts[1])

  validate_range('int_range')
  validate_range('dim_range')
  validate_named_range('tsize_length_range')
  validate_named_range('tsize_value_range')
  validate_named_range('ppad_length_range')
  validate_named_range('hpad_length_range')
  validate_named_range('hpad_value_range')

  if not hasattr(linalg, args.op):
    error(f'Unknown op: {args.op}.')

  for type_name in args.types.split(','):
    if type_name not in scalar_types:
      error(f'Unknown type: {type_name}')

  for expert in args.experts.split(','):
    if not hasattr(experts, expert):
      error(f'Unknown expert name: {expert}')

  if args.samples < 0:
    error('Number of samples must be non-negative.')

  if args.timeout < 1:
    error('Timeout must be equal to 1 or larger.')

  if args.iters < 0:
    error(f'Number of iterations must be non-negative.')

  if args.runs < 0:
    error(f'Number of runs must be non-negative.')

  return no_errors


def parse_named_ranges(named_range_str):
  str_parts = named_range_str.split(' ')
  named_ranges = {}
  for str_part in str_parts:
    [name, range_str] = str_part.split('=')
    named_ranges[name] = parse_range(range_str)
  return named_ranges


def parse_range(range_str):
  str_parts = range_str.split(',')
  int_parts = map(int, str_parts)
  return range(*int_parts)


def parse_settings(args):
  settings = dict()
  settings['types'] = args.types.split(',')
  settings['int_range'] = parse_range(args.int_range)
  settings['dim_range'] = parse_range(args.dim_range)
  settings['tsize_length_range'] = parse_named_ranges(args.tsize_length_range)
  settings['tsize_value_range'] = parse_named_ranges(args.tsize_value_range)
  settings['tsize_register_tile_bound'] = args.tsize_register_tile_bound
  settings['ppad_length_range'] = parse_named_ranges(args.ppad_length_range)
  settings['hpad_length_range'] = parse_named_ranges(args.hpad_length_range)
  settings['hpad_value_range'] = parse_named_ranges(args.hpad_value_range)
  return settings


def find_sample(args):
  op = getattr(linalg, args.op)
  op_variables = collect_variables(op)
  settings = parse_settings(args)
  expert_names = args.experts.split(',')
  expert_name = choice(expert_names)
  expert = getattr(experts, expert_name)
  variables = {}
  variables.update(op_variables)
  variables.update(expert.variables)

  while True:
    assignments = dict()
    assignments.update(instantiate_variables(variables, **settings))
    if are_constraints_satisfied(assignments, variables, **settings):
      break

  assignments['expert_name'] = expert_name

  return assignments
