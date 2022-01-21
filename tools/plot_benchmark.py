import argparse, pandas, seaborn, sys
from unicodedata import name

names_to_translate = {
    'gflop_per_s_per_iter': 'gflops / s / iter',
    'gbyte_per_s_per_iter': 'gbytes / s / iter',
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
                      type=int,
                      nargs="?",
                      help="peak bandwidth (e.g., -t 281)",
                      default=281)
  return parser.parse_args(sys.argv[1:])


def main():
  args = _parse_arguments()
  all_data = pandas.read_json(args.input)

  # Filter the slowest to isolate the compulsory miss effects.
  # Drop the first index matching every key_value (i.e. the first measurement)
  key = all_data.keys()[0]
  x = all_data[key].drop_duplicates()
  for key_value in x:
    locs_with_key_value = all_data.loc[all_data[key] == key_value]
    first_loc_with_key_value = locs_with_key_value.head(1)
    index_to_drop = first_loc_with_key_value.index.values[0]
    all_data = all_data.drop(index_to_drop)

  for key in all_data.keys():
    if key not in ['gflop_per_s_per_iter', 'gbyte_per_s_per_iter']:
      continue
    plot = seaborn.violinplot(x=all_data.keys()[0],
                              y=key,
                              data=all_data,
                              width=1.25)

    if key == 'gflop_per_s_per_iter':
      plot.set(ylim=(0, args.peak_compute + 10))
      plot.axhline(args.peak_compute,
                   label=f'Peak Compute ({args.peak_compute} GFlop/s)')
    elif key == 'gbyte_per_s_per_iter':
      plot.set(ylim=(0, args.peak_bandwidth * 1.1))
      plot.axhline(args.peak_bandwidth,
                   label=f'Peak BW ({args.peak_bandwidth} GB/s)')

    plot.set_title(args.name)
    plot.set_xlabel(names_to_translate[all_data.keys()[0]])
    plot.set_ylabel(names_to_translate[key])
    plot.set_xticklabels(plot.get_xticklabels(), rotation=15)
    plot.legend(bbox_to_anchor=(1.0, 1), loc='upper center')
    fig = plot.get_figure()
    fig.set_size_inches(12, 12)
    fig.savefig(args.output.replace('.pdf', '_' + key + '.pdf'))


if __name__ == '__main__':
  main()
