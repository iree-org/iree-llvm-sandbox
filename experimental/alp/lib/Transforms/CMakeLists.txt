add_mlir_library(ExperimentalAlpTransforms
  extract_kernel_pass.cpp
  modulo_scheduling_pass.cpp
  legalize_vector.cpp

  LINK_LIBS PRIVATE
  MLIRLinalg
  MLIRLinalgTransforms

  DEPENDS
  ALPPassIncGen
  MLIRLinalg  # TODO: Why needed here?
  MLIRLinalgTransforms  # TODO: Why needed here?
)
