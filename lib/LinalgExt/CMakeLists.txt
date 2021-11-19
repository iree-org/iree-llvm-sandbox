add_mlir_library(MLIRLinalgExt
  LinalgExtDialect.cpp
  LinalgExtInterfaces.cpp
  LinalgExtOps.cpp
  Tiling.cpp

  DEPENDS
  MLIRLinalgExtInterfacesIncGen
  MLIRLinalgExtOpsIncGen
  MLIRLinalgExtPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
