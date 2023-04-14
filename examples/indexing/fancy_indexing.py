import numpy as np
from numpy.random import randint

# %out = tensor.gather %input[%indices] gather_along_dimension(1) :
#   tensor<2x3x4xf32>[tensor<5x6x 1xindex>] -> tensor<5x6x2x1x4xf32>

input = randint(0, 10, (2, 3, 4))
indices = randint(0, 2, (5, 6))
out = input[:, indices, :]
out_ = np.moveaxis(out, (1, 2), (0, 1))

for index in np.ndindex(5, 6):
  coord = indices[index]
  assert np.allclose(out_[index], input[:, coord, :])

# %out = tensor.gather %input[%indices] gather_along_dimensions(0, 2) :
#   (tensor<2x3x4xf32>, tensor<5x6x 2xindex>) -> tensor<5x6x3xf32>

indices = randint(0, 1, (5, 6, 2))
out = input[indices[:, :, 0], :, indices[:, :, 1]]

for index in np.ndindex(5, 6):
  coord = indices[index]
  assert np.allclose(out[index], input[coord[0], :, coord[1]])

# %out = scatter %input into %dest[%indices] :
#   tensor<1x2xf32> into tensor<4x4x4xf32>[tensor<1x2x 3xindex>] -> tensor<4x4x4xf32>

input = randint(0, 10, (1, 2))
dest = randint(0, 10, (4, 4, 4))
indices = randint(0, 4, (1, 2, 3))
indices_ = np.moveaxis(indices, (0, 1, 2), (1, 2, 0))
dest[*(*indices_,)] = input

for index in np.ndindex(1, 2):
  coord = indices[*index]
  assert np.allclose(dest[*coord], input[*index])

# %out = scatter %input into %dest[%indices] coordinates = [1] :
#   tensor<5x6x4x4xf32> into tensor<4x100x4xf32>[tensor<5x6x 1xindex>] -> tensor<4x100x4xf32>

input = randint(0, 10, (5, 6, 4, 4))
dest = np.zeros((4, 100, 4))
indices = randint(0, 100, (5, 6))
while len(np.unique(indices)) != 30:
  indices = randint(0, 100, (5, 6))
input_ = np.moveaxis(input, (0, 1), (1, 2))
dest[:, indices, :] = input_

for index in np.ndindex(5, 6):
  coord = indices[*index]
  assert np.allclose(dest[:, coord, :], input[*index])

# %out = scatter %input into %dest[%indices] scatter_along_dimensions([0, 2]) :
#   tensor<5x6x4xf32> into tensor<1000x4x1000xf32>[tensor<5x6x 2xindex>] -> tensor<1000x4x1000xf32>

input = randint(0, 10, (5, 6, 4))
dest = np.zeros((1000, 4, 1000))
indices = randint(0, 1000, (5, 6, 2))
while len(np.unique(indices)) != 60:
  indices = randint(0, 1000, (5, 6, 2))
dest[indices[:, :, 0], :, indices[:, :, 1]] = input

for index in np.ndindex(5, 6):
  coord = indices[*index]
  assert np.allclose(dest[coord[0], :, coord[1]], input[index])
assert dest.shape == (1000, 4, 1000)
