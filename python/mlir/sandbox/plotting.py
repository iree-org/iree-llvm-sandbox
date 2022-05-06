import matplotlib.pyplot as plt
import os

class Plotting:
  """
  A helper class for plotting proposals.

  After a proposal has been evaluated, it can be added as a data point. Once all
  proposals have been added, a graph can be generated for a given dimension. All
  other dimensions are fixed to the value that resulted in the best performance.
  """

  def __init__(self):
    self.configurations = []
    self.throughputs = []

  def get_features(self):
    return self.configurations[0].keys()

  def add_data_point(self, configuration, throughput):
    self.configurations.append(configuration)
    self.throughputs.append(throughput)

  def find_best_configuration(self):
    best_config = self.configurations[0]
    best_throughput = self.throughputs[0]
    for i in range(len(self.configurations)):
      if self.throughputs[i] > best_throughput:
        best_config = self.configurations[i]
        best_throughput = self.throughputs[i]
    return best_config

  def should_plot_point(self, dimension: str, configuration, best):
    """
    Return true if the configuration should be plotted, assuming that we are
    only interested in changes of `dimension`. All other features should be
    equal to `best`.
    """

    for param in configuration.items():
      if param[0] == dimension:
        continue
      if param[1] != best[param[0]]:
        return False
    return True

  def plot(self, dimension: str, output_dir: str):
    """
    Take the best configuration and plot changes in throughput when varying
    one feature at a time.
    """

    best_configuration = self.find_best_configuration()
    labels = []
    values = []

    for i in range(len(self.configurations)):
      configuration = self.configurations[i]
      throughput = self.throughputs[i]
      if not self.should_plot_point(dimension, configuration,
                                    best_configuration):
        continue
      label = configuration[dimension]
      if isinstance(label, (tuple, list)) and len(label) == 1:
        # Extract the single value of the tuple or list.
        label = label[0]
      else:
        label = str(label)
      labels.append(label)
      values.append(throughput)

    # Sort data points by label.
    zipped = zip(labels, values)
    sorted_points = sorted(zipped)
    tuples = zip(*sorted_points)
    labels, values = [list(tuple) for tuple in tuples]

    # Create and save the plot.
    plt.scatter(labels, values)
    plt.ylabel("GFlop/s")
    plt.xlabel(dimension)
    output_file = os.path.join(output_dir, dimension + ".png")
    plt.savefig(output_file)
    plt.clf()
