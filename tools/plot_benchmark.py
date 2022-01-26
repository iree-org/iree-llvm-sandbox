import argparse, pandas, os, seaborn, sys
import numpy as np
from unicodedata import name

import matplotlib.pyplot as plt

names_to_translate = {
    'gflop_per_s_per_iter': 'Throughput [Gflop/s]',
    'gbyte_per_s_per_iter': 'Bandwidth [GB/s]',
    'runtime_problem_sizes_dict': 'Problem Size',

    # Add benchmark function names.
    'copy_2d': 'Copy2D',
    'transpose_2d': 'Transpose2D',
    'row_reduction_2d': 'RowRed2D',
    'column_reduction_2d': 'ColRed2D',
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
  parser.add_argument("--sizes_to_plot",
                      type=str,
                      required=False,
                      help="semicolon-separated lost of problem sizes to plot "
                      "(e.g., --sizes_to_plot=\"m=32,n=48;m=90,n=32\")",
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


#### Tools to query benchmarks info from dataframe
def benchmark_key(data):
  return data.keys()[0]


def get_unique_benchmarks(data):
  return np.unique(data[benchmark_key(data)].values)


def print_available_benchmarks_and_exit(data, args):
  print(get_unique_benchmarks(data))
  exit()


def get_benchmarks_to_plot(data, args):
  if args.benchmarks_to_plot != 'all':
    specified_benchmarks = args.benchmarks_to_plot.split(',')
    print(f'Specified benchmark filter: {specified_benchmarks}')
    available_benchmarks = get_unique_benchmarks(data)
    print(f'Available benchmarks in the data set: {available_benchmarks}')
    return list(
        filter(lambda x: x in available_benchmarks, specified_benchmarks))
  return get_unique_benchmarks(data)


#### Tools to query problem_size info from dataframe
def problem_size_key(data):
  return data.keys()[1]


def get_unique_sizes(data):
  return np.unique(data[problem_size_key(data)].values)


def print_available_sizes_and_exit(data, args):
  print(get_unique_sizes(data))


def get_sizes_to_plot(data, args):
  if args.sizes_to_plot != 'all':
    specified_sizes = args.sizes_to_plot.split(';')
    print(f'Specified size filter: {specified_sizes}')
    available_sizes = get_unique_sizes(data)
    print(f'Available sizes in the data set: {available_sizes}')
    return list(filter(lambda x: x in available_sizes, specified_sizes))
  return get_unique_sizes(data)


#### Start
def main():
  args = _parse_arguments()

  if not os.path.exists(args.input):
    print(f'{args.input} does not exist')
    return

  data = pandas.read_json(args.input)

  if args.print_available_benchmarks:
    print_available_benchmarks_and_exit(data, args)

  benchmarks_to_plot = get_benchmarks_to_plot(data, args)
  print(f'Benchmarks to plot: {benchmarks_to_plot}')

  sizes_to_plot = get_sizes_to_plot(data, args)
  print(f'Sizes to plot: {sizes_to_plot}')

  data_to_plot = data
  data_to_plot = data_to_plot[data_to_plot[benchmark_key(data_to_plot)].isin(
      benchmarks_to_plot)]
  data_to_plot = data_to_plot[data_to_plot[problem_size_key(data_to_plot)].isin(
      sizes_to_plot)]

  # Add helper column that computes the problem volume.
  def compute_volume(problem_size):
    sizes = [int(size.split('=')[1]) for size in problem_size.split(',')]
    return np.prod(sizes)
  data_to_plot['problem_volume'] = data_to_plot[problem_size_key(
      data_to_plot)].apply(compute_volume)
  # Add helper column that maps benchmark name to its index.
  data_to_plot['benchmark_index'] = data_to_plot[benchmark_key(
      data_to_plot)].apply(lambda x: benchmarks_to_plot.index(x))
  # Sort by problem volume and benchmark index.
  data_to_plot = data_to_plot.sort_values(
      by=['problem_volume', 'benchmark_index'], ascending=(True, True))

  # Keep only the relevant columns.
  data_to_plot = data_to_plot[[benchmark_key(
      data_to_plot), problem_size_key(data_to_plot), args.metric_to_plot]]

  # Compute the quartiles.
  data_lower_quart = data_to_plot.groupby(
      [benchmark_key(data_to_plot), problem_size_key(data_to_plot)]).quantile(0.25)
  data_upper_quart = data_to_plot.groupby(
      [benchmark_key(data_to_plot), problem_size_key(data_to_plot)]).quantile(0.75)

  fig = plt.figure(figsize=(9.66, 6))
  ax = seaborn.barplot(x=problem_size_key(data_to_plot),
                  y=args.metric_to_plot,
                  hue=benchmark_key(data_to_plot),
                  data=data_to_plot,
                  ci=None)

  maximum = 0
  for idx, p in enumerate(ax.patches):
    benchmark = benchmarks_to_plot[idx//len(sizes_to_plot)]
    size = sizes_to_plot[idx%len(sizes_to_plot)]
    lower = data_lower_quart.loc[benchmark, size][args.metric_to_plot]
    upper = data_upper_quart.loc[benchmark, size][args.metric_to_plot]
    maximum = max(maximum, upper)
    ax.errorbar(p.get_x() + p.get_width() / 2, lower, upper - lower, color="k")
    ax.annotate(format(p.get_height(), '.1f'),
                (p.get_x() + p.get_width() / 2., upper),
                ha='center', va='bottom',
                xytext=(0, 2),
                textcoords='offset points',
                rotation=90,
                fontsize=8)

  ax.tick_params(axis="x", rotation=20)
  ax.legend(labels=[names_to_translate[text.get_text()]
            for text in ax.get_legend().get_texts()], title='', frameon=False)
  ax.set_ylim(bottom=0, top=maximum*1.1)
  ax.margins(x=0.01)
  plt.xlabel(names_to_translate[problem_size_key(data)])
  plt.ylabel(names_to_translate[args.metric_to_plot])



  fig.tight_layout()
  print(f'Save plot to {args.output}')
  plt.savefig(args.output)


if __name__ == '__main__':
  main()
