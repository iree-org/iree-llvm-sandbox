#!/usr/bin/env python3

import argparse
import math as m

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def plot_throughput_by_num_elements_dtype(df, ax, dtype):
  # Filter.
  df = df[df.method == dtype]

  # Set up plot.
  ax.set_xscale('log', base=2)

  ax.set_ylabel('Throughput [GElems/s]')
  ax.set_xlabel('Number of elements')

  # Plot.
  lines = sorted(df['dtype'].unique())
  for dtype in lines:
    df_series = df[df['dtype'] == dtype]
    ax.errorbar(df_series.num_elements,
                df_series['throughput_median'] / 10**9,
                yerr=df_series['throughput_std'] / 10**9,
                label=dtype)
  ax.legend()

  # Ticks and limits.
  ax.set_ylim(ymin=0)


def plot_throughput_by_num_elements_method(df, ax, dtype):
  # Filter.
  df = df[df['dtype'] == dtype]

  # Set up plot.
  ax.set_xscale('log', base=2)

  ax.set_ylabel('Throughput [GElems/s]')
  ax.set_xlabel('Number of elements')

  # Plot.
  lines = sorted(df.method.unique())
  for method in lines:
    df_series = df[df.method == method]
    ax.errorbar(df_series.num_elements,
                df_series['throughput_median'] / 10**9,
                yerr=df_series['throughput_std'] / 10**9,
                label=method)
  ax.legend()

  # Ticks and limits.
  ax.set_ylim(ymin=0)


def plot_time_by_num_elements(df, ax, dtype, phase):
  # Filter.
  df = df[df['dtype'] == dtype]

  # Set up plot.
  ax.set_xscale('log', base=2)
  ax.set_yscale('log')

  ax.set_ylabel('Running time')
  ax.set_xlabel('Number of elements')

  median_column_name = phase + '_time_us_median'
  std_column_name = phase + '_time_us_std'

  # Plot.
  lines = sorted(df.method.unique())
  for method in lines:
    df_series = df[df.method == method]
    ax.errorbar(df_series.num_elements,
                df_series[median_column_name],
                yerr=df_series[std_column_name],
                label=method)
  ax.legend()

  # Ticks and limits.
  ax.set_yticks([10**i for i in range(8)])
  ax.set_yticklabels(['1us', '10us', '.1ms', '1ms', '10ms', '.1s', '1s', '10s'])
  ymin = 10**m.floor(m.log(df[median_column_name].min() * 0.8, 10))
  ymax = 10**m.ceil(m.log(df[median_column_name].max() / 0.8, 10))
  ax.set_ylim(ymin, ymax)


def plot_time_by_method_dtype(df, ax, num_elements, phase):
  # Filter.
  df = df[df.num_elements == num_elements]

  median_column_name = phase + '_time_us_median'
  std_column_name = phase + '_time_us_std'

  # Compute unit prefix.
  unit_factor = 1000**m.floor(m.log(df[median_column_name].max() / 0.8, 1000))
  if unit_factor == 1:
    unit_prefix = 'u'
  elif unit_factor == 1000:
    unit_prefix = 'm'
  else:
    unit_prefix = ''
    unit_factor = 10**6

  # Set up plot.
  ax.set_ylabel('Running time [{}s]'.format(unit_prefix))
  ax.set_xlabel('Element type')

  # Plot.
  bars = sorted(df.method.unique())
  num_bars = len(bars)
  groups = sorted(df['dtype'].unique())
  num_groups = len(groups)
  bar_width = 0.8 / num_bars
  indexes = np.arange(num_groups)
  for i, method in enumerate(bars):
    df_series = df[df.method == method].sort_values(['dtype'])
    ax.bar(indexes - ((num_bars - 1) / 2.0 - i) * bar_width,
           df_series[median_column_name] / unit_factor,
           bar_width,
           yerr=df_series[std_column_name] / unit_factor,
           label=method)
  ax.legend()

  # Ticks and limits.
  ax.set_xticks(indexes)
  ax.set_xticklabels(groups)


def parse_args():
  """Parse the command line arguments using `argparse` and return an
  `argparse.Namespace` object with the bound argument values."""

  parser = argparse.ArgumentParser(
      description='Plot measurements of the inner product benchmark.')
  parser.add_argument('plot',
                      metavar='PLOT',
                      choices=[
                          'time_by_num_elements',
                          'throughput_by_num_elements_dtype',
                          'throughput_by_num_elements_method',
                          'time_by_method_dtype',
                      ],
                      help='Name of the plot that should be produced.')
  parser.add_argument('-i',
                      '--input-file',
                      metavar='FILE',
                      type=argparse.FileType('r'),
                      required=True,
                      help='Path to the JSON lines input file.')
  parser.add_argument('-o',
                      '--output-file',
                      metavar='FILE',
                      type=argparse.FileType('wb'),
                      required=True,
                      help='Path to resulting PDF file.')
  parser.add_argument('-t',
                      '--dtype',
                      metavar='TYPE',
                      type=str,
                      default='int32',
                      help='dtype by which to filter.')
  parser.add_argument('-m',
                      '--method',
                      metavar='METHOD',
                      type=str,
                      default='iterators',
                      help='Method by which to filter.')
  parser.add_argument('-n',
                      '--num-elements',
                      metavar='N',
                      type=int,
                      default=2**25,
                      help='Number of elements by which to filter.')
  parser.add_argument('-p',
                      '--phase',
                      metavar='PHASE',
                      choices=['run', 'prepare', 'compile'],
                      default='run',
                      help='Phase whose timing should be plotted.')
  return parser.parse_args()


def main():
  # Parse arguments.
  args = parse_args()

  # Read measurement data.
  df = pd.read_json(args.input_file, orient='records', lines=True)

  # Unnest nested data.
  df = df.explode(['results', 'run_times_ns'])
  df = df.rename(columns={'results': 'result', 'run_times_ns': 'run_time_ns'})
  df.run_time_ns = df.run_time_ns.astype(np.int64)

  # Compute times in us.
  df['run_time_us'] = df.run_time_ns / 1000
  df['compile_time_us'] = df.compile_time_ns / 1000
  df['prepare_time_us'] = df.prepare_time_ns / 1000

  df['throughput'] = df.num_elements / df.run_time_ns * 10**9

  # Aggregate.
  df = df \
    .groupby(['method', 'dtype', 'num_elements']) \
    .aggregate({metric: ['median', 'std']
                for metric in [
                  'run_time_us',
                  'prepare_time_us',
                  'compile_time_us',
                  'throughput',
                ]}) \
    .reset_index() \
    .sort_values(['method', 'dtype', 'num_elements'])

  # Flatten hierarchical column names.
  df.columns = [
      '_'.join([v for v in col if v != '']) for col in df.columns.values
  ]

  # Plot.
  plt.rcParams.update({
      'errorbar.capsize': 2,
  })

  fig = plt.figure(figsize=(6, 4))
  ax = fig.add_subplot(1, 1, 1)

  if args.plot == 'time_by_num_elements':
    plot_time_by_num_elements(df, ax, args.dtype, args.phase)
  if args.plot == 'throughput_by_num_elements_method':
    assert args.phase == 'run'
    plot_throughput_by_num_elements_method(df, ax, args.dtype)
  if args.plot == 'throughput_by_num_elements_dtype':
    assert args.phase == 'run'
    plot_throughput_by_num_elements_dtype(df, ax, args.method)
  elif args.plot == 'time_by_method_dtype':
    plot_time_by_method_dtype(df, ax, args.num_elements, args.phase)

  plt.savefig(args.output_file, format='pdf', bbox_inches='tight', pad_inches=0)
  plt.close()


if __name__ == '__main__':
  main()
