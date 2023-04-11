import argparse, pandas, os, random, seaborn, sys, re
import numpy as np
from unicodedata import name
from numpy import median

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
    'matmul_kmkn': '$A^TB$',
    'matmul_mkkn': '$AB$',
    'matmul_mknk': '$AB^T$',
    'conv_1d_nwc_wcf_main': 'Conv1D',
    'conv_2d_nhwc_hwcf_main': 'Conv2D',
    'depthwise_conv_2d_nhwc_hwc': 'DepthwiseConv2D',
    'depthwise_conv_1d_nwc_wc_1_1': 'str.=[1],dil.=[1]',
    'depthwise_conv_1d_nwc_wc_1_2': 'str.=[1],dil.=[2]',
    'depthwise_conv_1d_nwc_wc_2_1': 'str.=[2],dil.=[1]',
    'depthwise_conv_1d_nwc_wc_2_2': 'str.=[2],dil.=[2]',
    'conv_1d_nwc_wcf_main_1_1': 'str.=[1],dil.=[1]',
    'conv_1d_nwc_wcf_main_2_1': 'str.=[2],dil.=[1]',
    'conv_1d_nwc_wcf_main_1_2': 'str.=[1],dil.=[2]',
    'conv_1d_nwc_wcf_main_2_2': 'str.=[2],dil.=[2]',
    'conv_2d_nhwc_hwcf_main_11_11': 'str.=[1,1],dil.=[1,1]',
    'conv_2d_nhwc_hwcf_main_22_11': 'str.=[2,2],dil.=[1,1]',
    'conv_2d_nhwc_hwcf_main_11_22': 'str.=[1,1],dil.=[2,2]',
    'conv_2d_nhwc_hwcf_main_22_22': 'str.=[2,2],dil.=[2,2]',
}


def _parse_arguments() -> argparse.Namespace:
  """Plot argument parser.
  """
  parser = argparse.ArgumentParser(description="Plot")
  parser.add_argument(
      "--inputs",
      type=str,
      required=True,
      help=
      "comma-separated list of input data filenames (e.g., --input input1,input2)\n"
      + "The data for multiple files is concatenated into a single graph.")
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
  parser.add_argument("--num_sizes_to_plot",
                      type=int,
                      required=False,
                      help="sample the given number of problem sizes to plot",
                      default=-1)
  parser.add_argument("--metric_to_plot",
                      type=str,
                      required=True,
                      choices=["gflop_per_s_per_iter", "gbyte_per_s_per_iter"])
  parser.add_argument("--group_by_strides_and_dilations",
                      type=bool,
                      required=False,
                      help="plot separate bars for strides and dilations")

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


#### Tools to compute labels
def compress_problem_sizes_label(labels):
  """Shorten the problem size lables by removing redundant information.

  Plotting the entire problem size configuration for every axis tick
  requires a lot of space and results in overlapping labels. The method
  identifies the dimensions that take different values and filters out
  the dimensions that are constant for the entire plot. Additionally,
  the dimension names (it suffices to plot them once) and sizes values
  are returned seperately.

  Example:
  ["H=64,W=32", "H=64,W=64"]
  ->
  ["W"], ["32", "64"]
  """
  label_dicts = []
  for label in labels:
    groups = re.findall(r"""([a-zA-Z]+)=(\d+|\[[0-9, ]+\])""", label)
    label_dicts.append(dict(groups))
  # Collect all values for a specific key.
  value_dict = {}
  for label_dict in label_dicts:
    for k, v in label_dict.items():
      if k in value_dict:
        value_dict[k].add(v)
      else:
        value_dict[k] = set([v])
  # Collect the keys that have multiple values.
  keys = []
  for k, v in value_dict.items():
    if len(v) != 1:
      keys.append(k)
  # Collect the keys for every label
  new_labels = []
  for label_dict in label_dicts:
    new_labels.append(",".join([label_dict[k] for k in keys]))
  return keys, new_labels


def get_strides_and_dilations(size):
  match1 = re.search(r"""strides=\[([0-9, ]+)\]""", size)
  match2 = re.search(r"""dilations=\[([0-9, ]+)\]""", size)
  suffixes = []
  if match1:
    suffixes.append("".join([x.strip() for x in match1.group(1).split(",")]))
  if match2:
    suffixes.append("".join([x.strip() for x in match2.group(1).split(",")]))
  if suffixes:
    return "_" + "_".join([suffix.strip() for suffix in suffixes])
  return ""


def remove_strides_and_dilations(size):
  size = re.sub(r"""[,]*strides=\[[0-9, ]+\]""", "", size)
  size = re.sub(r"""[,]*dilations=\[[0-9, ]+\]""", "", size)
  return size


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
  return list(get_unique_benchmarks(data))


def get_translated_name(name):
  return names_to_translate.get(name, name)


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
  if args.num_sizes_to_plot <= 0:
    return get_unique_sizes(data)
  random.seed(42)
  return random.sample(list(get_unique_sizes(data)), args.num_sizes_to_plot)


#### Start
def main():
  args = _parse_arguments()

  data = None
  for file in args.inputs.split(','):
    print(f'Processing {file}')
    if not os.path.exists(file):
      print(f'{file} does not exist')
      return
    read_data = pandas.read_json(file)
    print(read_data)
    data = read_data if data is None else pandas.concat([data, read_data])

  # Add strides and dilations to the function name.
  if args.group_by_strides_and_dilations:
    data[benchmark_key(data)] = data[[
        benchmark_key(data), problem_size_key(data)
    ]].apply(lambda x: x[0] + get_strides_and_dilations(x[1]), axis=1)
    data[problem_size_key(data)] = data[problem_size_key(data)].apply(
        lambda x: remove_strides_and_dilations(x))

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
    sizes = []
    for size in problem_size.split(','):
      try:
        size = size.split('=')[1]
        sizes.append(int(size))
      except Exception:
        pass
    return np.prod(sizes)

  def get_index(x):
    idx = -1
    for v in benchmarks_to_plot:
      idx = idx + 1
      if v == x:
        return idx
    assert False, f'Could not find {x} in {benchmarks_to_plot}'

  data_to_plot['problem_volume'] = data_to_plot[problem_size_key(
      data_to_plot)].apply(compute_volume)
  # Add helper column that maps benchmark name to its index.
  print(data_to_plot[benchmark_key(data_to_plot)])
  data_to_plot['benchmark_index'] = data_to_plot[benchmark_key(
      data_to_plot)].apply(get_index)
  # Sort by problem volume and benchmark index.
  data_to_plot = data_to_plot.sort_values(
      by=['problem_volume', 'benchmark_index'], ascending=(True, True))
  sizes_to_plot = list(
      data_to_plot[problem_size_key(data_to_plot)].drop_duplicates())

  # # Sort by problem volume and benchmark index.
  # data_to_plot = data_to_plot.sort_values(by=['problem_volume'], kind='stable')
  # print(data_to_plot)
  # sizes_to_plot = list(
  #     data_to_plot[problem_size_key(data_to_plot)].drop_duplicates())

  # Keep only the relevant columns.
  data_to_plot = data_to_plot[[
      benchmark_key(data_to_plot),
      problem_size_key(data_to_plot), args.metric_to_plot
  ]]

  # Compute the quartiles.
  data_lower_quart = data_to_plot.groupby(
      [benchmark_key(data_to_plot),
       problem_size_key(data_to_plot)]).quantile(0.25)
  data_upper_quart = data_to_plot.groupby(
      [benchmark_key(data_to_plot),
       problem_size_key(data_to_plot)]).quantile(0.75)
  print(data_lower_quart)
  print(data_upper_quart)

  fig = plt.figure(figsize=(9.66, 6))
  ax = seaborn.barplot(x=problem_size_key(data_to_plot),
                       y=args.metric_to_plot,
                       hue=benchmark_key(data_to_plot),
                       data=data_to_plot,
                       estimator=median,
                       ci=None)

  maximum = 0
  for idx, p in enumerate(ax.patches):
    benchmark = benchmarks_to_plot[idx // len(sizes_to_plot)]
    size = sizes_to_plot[idx % len(sizes_to_plot)]
    lower = data_lower_quart.loc[benchmark, size][args.metric_to_plot]
    upper = data_upper_quart.loc[benchmark, size][args.metric_to_plot]
    maximum = max(maximum, upper)
    ax.errorbar(p.get_x() + p.get_width() / 2, lower, upper - lower, color="k")
    ax.annotate(format(p.get_height(), '.1f'),
                (p.get_x() + p.get_width() / 2., upper),
                ha='center',
                va='bottom',
                xytext=(0, 2),
                textcoords='offset points',
                rotation=90,
                fontsize=8)

  keys, new_labels = compress_problem_sizes_label(
      [text.get_text() for text in ax.get_xticklabels()])
  ax.set_xticklabels(labels=new_labels)

  ax.tick_params(axis="x", rotation=30)
  ax.legend(ncol=4,
            loc='upper right',
            labels=[
                names_to_translate[text.get_text()]
                for text in ax.get_legend().get_texts()
            ],
            title='',
            frameon=False)
  ax.set_ylim(bottom=0, top=maximum * 1.15)
  ax.margins(x=0.01)
  plt.xlabel(
      str.format(
          f"{get_translated_name(problem_size_key(data))} [{','.join(keys)}]"))
  plt.ylabel(get_translated_name(args.metric_to_plot))

  fig.tight_layout()
  print(f'Save plot to {args.output}')
  plt.savefig(args.output)


if __name__ == '__main__':
  main()
