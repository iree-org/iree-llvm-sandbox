import argparse, pandas, seaborn, sys
from unicodedata import name

names = {
    'gflop_per_s_per_iter': 'gflops / s / iter',
    'runtime_problem_sizes_dict': 'problem sizes',
}


def _parse_arguments() -> argparse.Namespace:
  """Plot argument parser.

  Arguments:
  filename: Benchmark path prefix.
  peak_throughput: Peak throughput of the benchmark system.
  peak_bandwidth: Peak bandwidth of the benchmark system.
  """
  parser = argparse.ArgumentParser(description="Plot")
  parser.add_argument("--input",
                      "-i",
                      type=str,
                      required=True,
                      help="input data filename (e.g., -i input)")
  parser.add_argument("--output",
                      "-o",
                      type=str,
                      required=True,
                      help="output plot filename (e.g., -o output)")
  parser.add_argument("--name",
                      "-n",
                      type=str,
                      required=True,
                      help="plot name (e.g., -n name)")
  parser.add_argument("--peak_compute",
                      "-t",
                      type=int,
                      nargs="?",
                      help="peak compute (e.g., -t 192)",
                      default=192)
  parser.add_argument("--peak_bandwidth",
                      "-b",
                      type=str,
                      nargs="?",
                      help="peak bandwidth (e.g., -t 87,26.4,12)",
                      default='87,26.4,12')
  return parser.parse_args(sys.argv[1:])


def main():
  args = _parse_arguments()
  all_data = pandas.read_json(args.input)

  # Drop the first index matching every key_value (i.e. the first measurement)
  key = all_data.keys()[0]
  val = all_data.keys()[1]
  x = all_data[key].drop_duplicates()
  for key_value in x:
    locs_with_key_value = all_data.loc[all_data[key] == key_value]
    first_loc_with_key_value = locs_with_key_value.head(1)
    index_to_drop = first_loc_with_key_value.index.values[0]
    all_data = all_data.drop(index_to_drop)

  plot = seaborn.violinplot(
      x=all_data.keys()[0],
      y=all_data.keys()[1],
      # TODO, filter the slowest to isolate the compulsory miss effects
      data=all_data,
      width=1.25)

  if all_data.keys()[1] == 'gflop_per_s_per_iter':
    plot.set(ylim=(0, args.peak_compute + 10))
    plot.axhline(args.peak_compute,
                 label=f'Peak Compute ({args.peak_compute} GFlop/s)')
  if all_data.keys()[1] == 'gbyte_per_s_per_iter':
    plot.set(ylim=(0, args.peak_bandwidth_l1 + 10))
    plot.axhline(args.peak_bandwidth.split(',')[0],
                 label=f'Peak L2 ({args.peak_bandwidth[0]} GB/s)')
    plot.axhline(args.peak_bandwidth.split(',')[1],
                 label=f'Peak L3 ({args.peak_bandwidth[1]} GB/s)')
    plot.axhline(args.peak_bandwidth.split(',')[2],
                 label=f'Peak DRAM ({args.peak_bandwidth[2]} GB/s)')

  plot.set_title(args.name)
  plot.set_xlabel(names[all_data.keys()[0]])
  plot.set_ylabel(names[all_data.keys()[1]])
  plot.set_xticklabels(plot.get_xticklabels(), rotation=30)
  plot.legend(bbox_to_anchor=(1.0, 1), loc='upper center')
  fig = plot.get_figure()
  fig.set_size_inches(12, 12)
  fig.savefig(args.output)


if __name__ == '__main__':
  main()
