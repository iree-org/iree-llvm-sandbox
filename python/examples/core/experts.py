from .transforms import *
from .transform import TransformListFactory, TransformationList

# TODO: After DecomposeToLowerDimensionalNamedOp the op_name to anchor on
# changes: we need a better control mechanism.

###############################################################################
# Experts provide end-to-end batteries including bufferization, vector lowering
# and lowering to LLVM. All options must be passed inline, in particular it is
# not possible to deactivate bufferization, vectorization or lowering or make
# them apply partially.
###############################################################################
LoweringOnlyExpert = Bufferize.then(LowerVectors).then(LowerToLLVM)
SingleTilingExpert = Tile.then(DecomposeToLowerDimensionalNamedOp).then(
    Vectorize).then(LoweringOnlyExpert)
DoubleTilingExpert = Tile.then(SingleTilingExpert)
TripleTilingExpert = Tile.then(DoubleTilingExpert)

###############################################################################
# Coarser-grained transformations, composed of other transformations.
# No bufferization, vector lowering or lowering to LLVM is applied.
###############################################################################
DoubleTile = Tile.then(Tile)
TripleTile = Tile.then(DoubleTile)
TileAndDecompose = Tile.then(DecomposeToLowerDimensionalNamedOp)
DoubleTileAndDecompose = Tile.then(TileAndDecompose)
TripleTileAndDecompose = Tile.then(DoubleTileAndDecompose)


# Expert compiler that applies the whole sparse compiler.
class ExpertSparseCompiler(TransformationList):
  variables = {'options': str}

  def __init__(self, options):
    self.options = options
    TransformationList.__init__(self, **{'transforms': [Sparsify(options)]})
