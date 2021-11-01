from glob import glob
import argparse


def parse_args(argv):
  parser = argparse.ArgumentParser(description='Command-line directed search.')
  parser.add_argument(
      '--op', type=str, default='matmul', help='Name of the op.')
  parser.add_argument(
      '--by', type=str, default='time', help='Name of the metric to rank by.')
  parser.add_argument(
      '--input',
      type=str,
      default='runtime',
      help='Extension of the output file to read metric from.')
  parser.add_argument(
      '--limit', type=int, default='20', help='Number of results to show.')
  return parser.parse_args(argv)


def main(argv=None):
  args = parse_args(argv)

  candidates = []
  metric = args.by.lower()

  def find_metric_value(path):
    with open(path) as f:
      for (i, line) in enumerate(f):
        if i > 20:
          break
        if metric in line.lower() and ':' in line:
          _, v = line.split(':')
          return float(v)
    return None

  for path in glob(f'output/{args.op}/*/ok/*.{args.input}'):
    value = find_metric_value(path)
    if value is not None:
      base_path = path[:-len(args.input) - 1]
      candidates.append((value, base_path))

  print(f'Top {args.limit} results for {args.op} op, ranked by {args.by}:')
  for (i, (v, candidate)) in enumerate(sorted(candidates)[:args.limit]):
    command = open(candidate + '.sh').readlines()[0]
    args = ' '.join(command.split(' ')[4:])
    print(f'#{i} @ {v}: {candidate}')
    print(f'  $ {command}')
    print('')


if __name__ == '__main__':
  main()
