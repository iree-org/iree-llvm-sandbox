import argparse, pandas, seaborn, sys
import numpy as np
from unicodedata import name

names_to_translate = {
    'gflop_per_s_per_iter': 'gflops / s / iter',
    'gbyte_per_s_per_iter': 'gbytes / s / iter',
    'runtime_problem_sizes_dict': 'problem sizes',
}


def _parse_arguments() -> argparse.Namespace:
  """Plot argument parser.
  """
  parser = argparse.ArgumentParser(description="Plot")
  parser.add_argument("--input",
                      type=str,
                      required=True,
                      help="input data filename (e.g., --input input)")
  parser.add_argument("--output",
                      type=str,
                      required=True,
                      help="output plot filename (e.g., --output output)")
  parser.add_argument("--plot_name",
                      type=str,
                      required=True,
                      help="plot name (e.g., --plot_name name)")
  parser.add_argument("--print_available_benchmarks",
                      type=bool,
                      required=False,
                      help="print the existing list of benchmarks in the data")
  parser.add_argument("--benchmarks_to_plot",
                      type=str,
                      required=False,
                      help="comma-separated names of benchmarks to plot",
                      default='all')
  parser.add_argument("--metric_to_plot",
                      type=str,
                      required=True,
                      choices=["gflop_per_s_per_iter", "gbyte_per_s_per_iter"])

  ###############################################################################
  # Not used atm
  ###############################################################################
  parser.add_argument("--peak_compute",
                      type=int,
                      nargs="?",
                      help="peak compute (e.g., --peak_compute 192)",
                      default=192)
  parser.add_argument("--peak_bandwidth_hi",\
                      type=int,
                      nargs="?",
                      help="high peak bandwidth (e.g., --peak_bandwidth_hi 281)",
                      default=281)
  parser.add_argument("--peak_bandwidth_lo",
                      type=int,
                      nargs="?",
                      help="low peak bandwidth (e.g., -peak_bandwidth_lo 281)",
                      default=281)

  return parser.parse_args(sys.argv[1:])


def add_peak_lines(args, plot, key):
  if key == 'gflop_per_s_per_iter':
    plot.set(ylim=(0, args.peak_compute + 10))
    plot.axhline(args.peak_compute,
                 label=f'Peak Compute ({args.peak_compute} GFlop/s)')
  elif key == 'gbyte_per_s_per_iter':
    plot.set(ylim=(0, args.peak_bandwidth_hi * 1.1))
    plot.axhline(args.peak_bandwidth_hi,
                 label=f'Peak BW ({args.peak_bandwidth_hi} GB/s)')
    if args.peak_bandwidth_lo != args.peak_bandwidth_hi:
      plot.axhline(args.peak_bandwidth_lo,
                   label=f'Peak BW ({args.peak_bandwidth_lo} GB/s (low range))')


###############################################################################
# End Not used atm
###############################################################################


def available_benchmark_key(data):
  return data.keys()[0]


def problem_size_key(data):
  return data.keys()[1]


def get_unique_benchmarks(data):
  return np.unique(data[available_benchmark_key(data)].values)


def print_available_benchmarks_and_exit(data, args):
  print(get_unique_benchmarks(data))
  exit()


def get_benchmarks_to_plot(data, args):
  if args.benchmarks_to_plot != 'all':
    specified_benchmarks = args.benchmarks_to_plot.split(',')
    available_benchmarks = get_unique_benchmarks(data)
    return list(
        filter(lambda x: x in available_benchmarks, specified_benchmarks))
  return get_unique_benchmarks(data)


def main():
  args = _parse_arguments()
  data = pandas.read_json(args.input)

  if args.print_available_benchmarks:
    print_available_benchmarks_and_exit(data, args)

  benchmarks_to_plot = get_benchmarks_to_plot(data, args)

  ordering_dict = {
      benchmarks_to_plot[i]: i for i in range(len(benchmarks_to_plot))
  }
  data_to_plot = data[data[available_benchmark_key(data)].isin(
      benchmarks_to_plot)]
  data_to_plot = data_to_plot.sort_values(by=[available_benchmark_key(data)],
                                          kind='stable',
                                          key=lambda x: x.map(ordering_dict))
  facetgrid = seaborn.catplot(x=problem_size_key(data_to_plot),
                              y=args.metric_to_plot,
                              hue=available_benchmark_key(data_to_plot),
                              data=data_to_plot,
                              kind="bar")

  # Matplotlib way but facetgrid isn't such a unicorn...
  # plot.set_title(args.plot_name)
  # plot.set_xlabel(names_to_translate[problem_size_key(data)])
  # plot.set_ylabel(names_to_translate[args.metric_to_plot])
  # plot.set_xticklabels(plot.get_xticklabels(), rotation=45)
  # Facetgrid way.
  facetgrid.set_axis_labels( \
    names_to_translate[problem_size_key(data)],
    names_to_translate[args.metric_to_plot])

  # facetgrid actually contains plots within its axes[0].
  for plot in facetgrid.axes[0]:
    plot.set_xticklabels(plot.get_xticklabels(), rotation=45)

  fig = facetgrid.fig
  fig.set_size_inches(12, 12)
  output_file = args.output.replace(
      '.pdf', '-'.join(x for x in benchmarks_to_plot) + '-' +
      args.metric_to_plot + '.pdf')

  print(f'Save plot to {output_file}')
  fig.savefig(output_file)


if __name__ == '__main__':
  main()
