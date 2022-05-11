import mlir.sandbox.iree_sandbox as sandbox
import iree.compiler.ir as ir
import iree.compiler.dialects.transform as transform
import iree.compiler.dialects.pdl as pdl


def run(f):
  print(f"TEST: {f.__name__}")
  with ir.Context() as ctx, ir.Location.unknown(ctx):
    import iree.compiler.dialects.iree_linalg_ext as linalg_ext
    import iree.compiler.dialects.transform as transform
    linalg_ext.register_dialect(ctx)
    transform.register_dialect(ctx)

    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
      f(module)
    print(module)


# CHECK-LABEL: TEST: tile_once
@run
def tile_once(module):
  root = transform.WithPDLPatternsOp(root=None)
  with ir.InsertionPoint(root.body.blocks[0]):
    foo_pattern = pdl.PatternOp(benefit = 1, name = "foo")
    with ir.InsertionPoint(foo_pattern.body):
      pdl_op = pdl.OperationOp()
      pdl.RewriteOp(pdl_op, "transform.dialect")
    sequence = transform.CanonicalizedSequenceOp(root.body.blocks[0].arguments[0])
    with ir.InsertionPoint(sequence.body.blocks[0]):
      target = transform.PDLMatchOp(sequence.body.blocks[0].arguments[0], "foo")
      tiled = transform.TileOp(target, sizes=[32, 16])
      padded = transform.PadOp(tiled.results[0])
      transform.VectorizeOp(padded, vectorize_padding=True)
      transform.BufferizeOp()
      transform.LowerVectorsOp(multireduction_lowering="innerreduce")
      transform.LowerToLLVMOp()
      transform.YieldOp([])
      print(module)

  code = str(sequence)
  assert "match @foo" in code
  assert "tile %" in code
  assert "sizes = [32, 16]" in code
  assert "pad %" in code
  assert "vectorize" in code
  assert "vectorize_padding = true" in code


# CHECK-LABEL: TEST: tile_twice
@run
def tile_twice(module):
  sequence = transform.CanonicalizedSequenceOp(target=None)
  with ir.InsertionPoint(sequence.body.blocks[0]):
    target = transform.PDLMatchOp(sequence.body.blocks[0].arguments[0], "foo")
    tiled1 = transform.TileOp(target, sizes=[128, 32])
    tiled2 = transform.TileOp(tiled1.results[0], sizes=[32, 16])
    padded = transform.PadOp(tiled2.results[0])
    transform.VectorizeOp(padded, vectorize_padding=True)
    transform.BufferizeOp()
    transform.LowerVectorsOp(multireduction_lowering="innerreduce")
    transform.LowerToLLVMOp()
    transform.YieldOp([])

  code = str(sequence)
  assert "match @foo" in code
  assert "tile %" in code
  assert "sizes = [128, 32]" in code
  assert "sizes = [32, 16]" in code
  assert "pad %" in code
  assert "vectorize" in code


# CHECK-LABEL: TEST: fuse_once
@run
def fuse_once(module):
  sequence = transform.CanonicalizedSequenceOp(target=None)
  with ir.InsertionPoint(sequence.body.blocks[0]):
    target = transform.PDLMatchOp(sequence.body.blocks[0].arguments[0], "foo")
    tiled = transform.FuseOp(target, tile_sizes=[16, 32])
    transform.PeelLoopOp(tiled.results[1])
    transform.PeelLoopOp(tiled.results[2])
    transform.YieldOp([])

  code = str(sequence)
  assert "match @foo" in code
  assert "fuse %" in code
  assert "tile_sizes = [16, 32]" in code
  assert "peel_loop %" in code
  assert "peel_loop %" in code
