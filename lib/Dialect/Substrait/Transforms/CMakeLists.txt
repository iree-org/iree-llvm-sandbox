add_mlir_dialect_library(MLIRSubstraitTransforms
  EmitDeduplication.cpp

  DEPENDS
  MLIRSubstraitTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRRewrite
  MLIRSubstraitDialect
  MLIRTransforms
  MLIRTransformUtils
)
