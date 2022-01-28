import mlir.iree_sandbox as sandbox
import mlir.ir as ir
import mlir.dialects.linalg_transform as transform


def run(f):
  print(f"TEST: {f.__name__}")
  with ir.Context() as ctx, ir.Location.unknown(ctx):
    sandbox.register_sandbox_passes_and_dialects(ctx)

    # TODO: this is necessary to force-load the dialect, otherwise op creation
    # complains about "unregistered dialect" despite the registration call just
    # above.
    ctx.dialects["linalg_transform"]

    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
      f()
    print(module)


# CHECK-LABEL: TEST: tile_once
@run
def tile_once():
  sequence = transform.SequenceOp()
  with ir.InsertionPoint(sequence.body.blocks[0]):
    tiled = transform.TileOp("foo", sizes=[32, 16], pad=True)
    transform.VectorizeOp(tiled, vectorize_padding=True)
    transform.BufferizeOp()
    transform.LowerVectorsOp(multireduction_lowering="innerreduce")
    transform.LowerToLLVMOp()

  code = str(sequence)
  assert "tile when @foo" in code
  assert "sizes = [32, 16]" in code
  assert "pad = true" in code
  assert "vectorize" in code


# CHECK-LABEL: TEST: tile_twice
@run
def tile_twice():
  sequence = transform.SequenceOp()
  with ir.InsertionPoint(sequence.body.blocks[0]):
    tiled1 = transform.TileOp("foo", sizes=[128, 32], pad=True)
    tiled2 = transform.TileOp(tiled1, sizes=[32, 16], pad=True)
    transform.VectorizeOp(tiled2, vectorize_padding=True)
    transform.BufferizeOp()
    transform.LowerVectorsOp(multireduction_lowering="innerreduce")
    transform.LowerToLLVMOp()

  code = str(sequence)
  assert "tile when @foo" in code
  assert "sizes = [128, 32]" in code
  assert "sizes = [32, 16]" in code
  assert "pad = true" in code
  assert "vectorize" in code
