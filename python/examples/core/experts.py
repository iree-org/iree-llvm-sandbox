from .transforms import *
from .transform import TransformListFactory, TransformationList

# TODO: After DecomposeToLowerDimensionalNamedOp the op_name to anchor on
# changes: we need a better control mechanism.
LoweringOnlyExpert = Bufferize.then(LowerVectors).then(LowerToLLVM)
SingleTilingExpert = Tile.then(DecomposeToLowerDimensionalNamedOp).then(
    Vectorize).then(LoweringOnlyExpert)
DoubleTilingExpert = Tile.then(SingleTilingExpert)
TripleTilingExpert = Tile.then(DoubleTilingExpert)

TileAndDecompose = Tile.then(DecomposeToLowerDimensionalNamedOp)
DoubleTileAndDecompose = Tile.then(TileAndDecompose)

# Expert compiler that applies the whole sparse compiler.
class ExpertSparseCompiler(TransformationList):
  variables = {'options': str}

  def __init__(self, options):
    self.options = options
    TransformationList.__init__(self, **{'transforms': [Sparsify(options)]})
