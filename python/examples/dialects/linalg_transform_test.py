import mlir.iree_sandbox as sandbox
import mlir.ir as ir
import mlir.dialects.iree_linalg_transform as transform


def run(f):
  print(f"TEST: {f.__name__}")
  with ir.Context() as ctx, ir.Location.unknown(ctx):
    import mlir.dialects.iree_linalg_ext as linalg_ext
    import mlir.dialects.iree_linalg_transform as transform
    from mlir.iree_sandbox import register_sandbox_passes_and_dialects
    linalg_ext.register_dialect(ctx)
    transform.register_dialect(ctx)
    register_sandbox_passes_and_dialects(ctx)

    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
      f()
    print(module)


# CHECK-LABEL: TEST: tile_once
@run
def tile_once():
  sequence = transform.SequenceOp()
  with ir.InsertionPoint(sequence.body.blocks[0]):
    target = transform.MatchOp("foo")
    tiled = transform.TileOp(target, sizes=[32, 16])
    padded = transform.PadOp(tiled.results[0])
    transform.VectorizeOp(padded, vectorize_padding=True)
    transform.BufferizeOp()
    transform.LowerVectorsOp(multireduction_lowering="innerreduce")
    transform.LowerToLLVMOp()

  code = str(sequence)
  assert "match @foo" in code
  assert "tile %" in code
  assert "sizes = [32, 16]" in code
  assert "pad %" in code
  assert "vectorize" in code
  assert "vectorize_padding = true" in code


# CHECK-LABEL: TEST: tile_twice
@run
def tile_twice():
  sequence = transform.SequenceOp()
  with ir.InsertionPoint(sequence.body.blocks[0]):
    target = transform.MatchOp("foo")
    tiled1 = transform.TileOp(target, sizes=[128, 32])
    tiled2 = transform.TileOp(tiled1.results[0], sizes=[32, 16])
    padded = transform.PadOp(tiled2.results[0])
    transform.VectorizeOp(padded, vectorize_padding=True)
    transform.BufferizeOp()
    transform.LowerVectorsOp(multireduction_lowering="innerreduce")
    transform.LowerToLLVMOp()

  code = str(sequence)
  assert "match @foo" in code
  assert "tile %" in code
  assert "sizes = [128, 32]" in code
  assert "sizes = [32, 16]" in code
  assert "pad %" in code
  assert "vectorize" in code


# CHECK-LABEL: TEST: fuse_once
@run
def fuse_once():
  sequence = transform.SequenceOp()
  with ir.InsertionPoint(sequence.body.blocks[0]):
    target = transform.MatchOp("foo")
    tiled = transform.FuseOp(target, tile_sizes=[16, 32])
    transform.PeelLoopOp(tiled.results[1])
    transform.PeelLoopOp(tiled.results[2])

  code = str(sequence)
  assert "match @foo" in code
  assert "fuse %" in code
  assert "tile_sizes = [16, 32]" in code
  assert "peel_loop %" in code
  assert "peel_loop %" in code
