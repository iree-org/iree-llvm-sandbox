add_mlir_library(MLIRLinalgExt
  LinalgExtDialect.cpp
  LinalgExtInterfaces.cpp
  LinalgExtOps.cpp

  DEPENDS
  MLIRLinalgExtInterfacesIncGen
  MLIRLinalgExtOpsIncGen
  MLIRLinalgExtPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
