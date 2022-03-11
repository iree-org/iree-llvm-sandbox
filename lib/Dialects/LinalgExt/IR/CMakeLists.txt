add_mlir_library(MLIRLinalgExt
  LinalgExtDialect.cpp
  LinalgExtInterfaces.cpp
  LinalgExtOps.cpp

  DEPENDS
  mlir-headers
  MLIRLinalgExtInterfacesIncGen
  MLIRLinalgExtOpsIncGen

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRIR
  MLIRMemRef
  MLIRTensor
)
