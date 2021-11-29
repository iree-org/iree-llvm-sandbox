from .transforms import *
from .transform import TransformListFactory, TransformationList


def LowerVectorFactory(stage):
  """Create a new Transformation class that binds the lowering stage to the
  given number at class construction time rather than at object construction."""

  def init(self, **kwargs):
    LowerVectors.__init__(self, stage, **kwargs)

  return type('LowerVectors' + str(stage), (LowerVectors,), {'__init__': init})


VectorLowering = TransformListFactory('VectorLowering',
                                      [LowerVectorFactory(i) for i in range(7)])

# TODO: After DecomposeToLowerDimensionalNamedOp the op_name to anchor on
# changes: we need a better control mechanism.
LoweringOnlyExpert = Bufferize.then(VectorLowering).then(LowerToLLVM)
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
