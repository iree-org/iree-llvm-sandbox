add_mlir_library(IteratorsOneToNTypeConversionTestPass
  OneToNTypeConversion.cpp
  OneToNTypeConversionFunc.cpp
  OneToNTypeConversionSCF.cpp
  OneToNTypeConversionTestPass.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRSCFDialect
  MLIRTestDialect
  MLIRTransformUtils
)

target_include_directories(IteratorsOneToNTypeConversionTestPass
  PRIVATE
  ${MLIR_MAIN_SRC_DIR}/test/lib/Dialect/Test/
  ${MLIR_TABLEGEN_OUTPUT_DIR}/../test/lib/Dialect/Test
  )
