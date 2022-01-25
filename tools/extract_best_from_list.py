import argparse, pandas, sys
import numpy as np

from ast import literal_eval


def _parse_arguments() -> argparse.Namespace:
  """Plot argument parser.
  """
  parser = argparse.ArgumentParser(description="Plot")
  parser.add_argument("--input",
                      type=str,
                      required=True,
                      help="input data filename (e.g., --input input)")

  return parser.parse_args(sys.argv[1:])


problem_size_column_name = 'problem_size'
expert_column_name = 'expert'
p75_column_name = 'p75'


def get_unique_problem_size(data):
  return np.unique(data[problem_size_column_name].values)


class ParserState(object):

  def __init__(self):
    self.count = -1
    self.data = None
    self.reset()

  def reset(self):
    self.compile_time_problem_size_dict = None
    self.compilation_expert = []
    self.p75 = []

  def parse_compile_time_problem_size(self, line: str):
    prefix = 'Compile-time problem size'
    if not line.startswith(prefix):
      return False
    self.compile_time_problem_size_dict = literal_eval(
        line[len(prefix):].strip())
    return True

  def parse_compilation_expert(self, line: str):
    prefix = 'Compilation expert'
    if not line.startswith(prefix):
      return False
    self.compilation_expert.append(line[len(prefix):].strip())
    return True

  def parse_p75(self, line: str, metric: str = 'GBs/s'):
    if line.find(metric) < 0:
      return False
    p75 = line[:-len(metric) - 1].strip().split()
    self.p75.append(p75[5])
    return True

  def parse_end(self, line: str):
    return line.find('###############') >= 0

  def parse_next(self, line: str):
    # Need a new problem size to start parsing other stuff.
    if self.compile_time_problem_size_dict is None:
      return self.parse_compile_time_problem_size(line)
    # If we have a problem size, try to add an expert.
    if self.parse_compilation_expert(line):
      return True
    # If we have a problem size, try to add a p75.
    if self.parse_p75(line):
      return True
    # If we reach here, try to find the end to concat to the data frame.
    if self.parse_end(line):
      # Sanity check.
      assert len(self.compilation_expert) == len(
          self.p75), 'must have parsed a compilation expert first'
      self.concat_new_data()
      return True
    return False

  def concat_new_data(self):
    compile_time_problem_size = ','.join(
        str(v) for k, v in self.compile_time_problem_size_dict.items())

    for expert, p75 in zip(self.compilation_expert, self.p75):
      self.count = self.count + 1
      new_data = pandas.DataFrame(
          {
              problem_size_column_name: compile_time_problem_size,
              expert_column_name: expert,
              p75_column_name: float(p75)
          },
          index=[self.count])
      self.data = new_data if self.data is None else pandas.concat(
          [self.data, new_data])
    self.reset()


def main():
  args = _parse_arguments()

  parser = ParserState()
  with open(args.input, "r") as f:
    for line in f:
      stripped = line.strip()
      parser.parse_next(line)

  # Group by problem size, take the p75-max idx.
  best_experts = parser.data.loc[parser.data.groupby(
      problem_size_column_name).p75.idxmax()]
  # Sort_index puts us back into original file order (i.e. experiment run order)
  best_experts = best_experts.sort_index()
  print(best_experts)

  for index, row in best_experts.iterrows():
    print('${COMMAND} ' + \
          f'--expert_list {row[expert_column_name]} ' +
          f'--problem_sizes_list {row[problem_size_column_name]}')


if __name__ == '__main__':
  main()
