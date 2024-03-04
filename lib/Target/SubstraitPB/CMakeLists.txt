add_mlir_translation_library(MLIRTargetSubstraitPB
  Export.cpp
  Import.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSubstraitDialect
  MLIRSupport
  MLIRTranslateLib
  Substrait
  )
