add_subdirectory(VectorExt)

if(NOT SANDBOX_ENABLE_IREE_DIALECTS)
  add_subdirectory(LinalgExt)
  add_subdirectory(LinalgTransform)
endif()
