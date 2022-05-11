# Smoke check tests for the existence of iree dialects.

# 1. linalg_ext dialect (ReverseOp, FFtOp, InParallelOp etc).
# Dialect, bindings etc are well defined, no problem here.
import iree.compiler.dialects.iree_linalg_ext as linalg_ext
assert hasattr(linalg_ext, 'InParallelOp'), 'Could not load linalg_ext.InParallelOp'

# 2. transform ops for the linalg_ext dialect.
# FuseProducersOp, TileToLinalgExtTileOp, FuseIntoContainingOp, 
# RewriteLinalgExtTileToScfForOp, RewriteLinalgExtTileToInParallelOp,
# RewriteLinalgExtInParallelToAsyncOp, RewriteLinalgExtInParallelToHALOp and
# RewriteLinalgExtInParallelToScfForOp are:
#   a. defined in Tablegen in:
#     iree-dialects/Dialect/LinalgExt/TransformOps/LinalgExtTransformOps.td
#   b. use the Tablegen transform dialect extension mechanism.
#   c. on the Python side, a stale version seems to live in:
#     iree/compiler/dialects/_iree_linalg_transform_ops_ext.py
#   d. there is currently no entry in any *transform_ops_gen.py
#
# TODO: 1. clarify where in python such transform dialect op extensions should live
#       2. autogenerate the *ops_gen.py (e.g. linalg_ext_transform_ops_gen.py)
#       3. move the stale *ops_ext.py version to the correct place.
#
# Alternatively, we could use a custom op to get started:
#
# import iree.compiler.dialects.iree_linalg_transform as linalg_transform
# assert hasattr(linalg_transform, 'TileToLinalgExtTileOp'), \
#  'Could not load linalg_transform.TileToLinalgExtTileOp'

# 3. transform ops for IREE abstractions.
# IREEBufferizeOp, IREESetNumWorkgroupToOneOp, InParallelToHALOp (future) are:
#   a. defined in CPP (there is no Tablegen for these ops atm) in:
#       compiler/src/iree/compiler/Codegen/TransformDialectExtensions/TransformDialectExtensions.cpp
#   b. use the transform dialect extension mechanism (C++ registerTransformOps).
#   c. there is no python for these ops atm. What 
#     python/examples/iree/simple_matmul_sequential_transform.py does is to 
#     create a custom op:
#       e.g. ir.Operation.create(name="transform.iree.bufferize")
# TODO: evaluate whether custom op is good enough for the foreseeable future.

# 4. iree transform dialect.
# ScopeOp, ForwardOp and ExpertOp are:
#   a. defined in Tablegen in:
#     llvm-external-projects/iree-dialects/include/iree-dialects/Dialect/LinalgTransform/LinalgTransformOps.td
#   b. DO NOT use the transform dialect extension mechanism but instead live in
#     the iree_linalg_transform dialect under the names `util.scope`, `util.forward`, `expert`.
#   c. DO NOT have a python version
# TODO: Are probably on the path to deprecation

# 5. structured transform ops that should be upstreamed in the fullness of time
# PDLMatchOp, WithPDLPatternsOp, YieldOp, BufferizeOp, CanonicalizedSequenceOp,
# TileOp, DecomposeOp, FuseOp, GeneralizeOp, GetParentLoopOp, InterchangeOp, 
# LowerToLLVMOp, LowerVectorsOp, OutlineLoopOp, PadOp, PeelLoopOp, PipelineLoopOp, 
# PrintOp, ScalarizeOp, UnrollLoopOp, VectorizeOp, 
#   a. defined in Tablegen in:
#    iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.td
#   b. use the Tablegen transform dialect extension mechanism.
#   c. on the python side live in a new TmpTransformDialect for those extensions.
#      Use the auto-generated and manually copy-pasted
#      iree-dialects/python/iree/compiler/dialects/_transform_ops_gen.py
#      
# TODO: 1. clarify where in python such transform dialect op extensions should live
#       2. autogenerate _transform_ops_gen.py
#       2. update the _transform_ops_ext.py
#       3. delete the stale version from iree/compiler/dialects/_iree_linalg_transform_ops_ext.py
import iree.compiler.dialects.transform as transform
assert hasattr(transform, 'LowerToLLVMOp'), 'Could not load transform.LowerToLLVMOp'