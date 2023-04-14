include(AddMLIRPython)
mlir_configure_python_dev_packages()
mlir_detect_pybind11_install()

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_indexing.")

# find the directory where the python bindings IRModule.h lives
# and add it as an include directory in order to extend/play with the bindings
get_target_property(mlir_python_sources MLIRPythonExtension.Core INTERFACE_SOURCES)
list(GET mlir_python_sources 0 HEADER)
get_filename_component(header_dir ${HEADER} DIRECTORY)
string(REPLACE "$<BUILD_INTERFACE:" "" header_dir_clean "${header_dir}")
set(header_dir_interface "${header_dir}>")

execute_process(COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/regex_mlir_bindings.py ${header_dir_clean}/IRModule.h)
execute_process(COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/regex_mlir_bindings.py ${header_dir_clean}/IRCore.cpp)
execute_process(COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/regex_mlir_bindings.py ${header_dir_clean}/IRAffine.cpp)

add_compile_definitions(MY_PYBIND11_EXPORT_SYMBOLS)
add_compile_definitions(MY_PYBIND11_EXPORT_MODULE)
declare_mlir_python_sources(IndexingPythonSources)
declare_mlir_dialect_python_bindings(
        ADD_TO_PARENT IndexingPythonSources
        ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_indexing"
        TD_FILE dialects/IndexingOps.td
        SOURCES
        dialects/indexing.py
        DIALECT_NAME indexing)

declare_mlir_python_extension(IndexingPythonSources.Extension
        MODULE_NAME _indexingDialects
        ADD_TO_PARENT IndexingPythonSources

        SOURCES
        IndexingExtension.cpp
        IRTypes.cpp

        PRIVATE_LINK_LIBS
        MLIRIndexing
        )
target_include_directories(IndexingPythonSources.Extension INTERFACE ${header_dir_interface})
target_sources(IndexingPythonSources.Extension INTERFACE "${header_dir_clean}/PybindUtils.cpp")

set(_source_components
        MLIRPythonExtension.Core
        MLIRPythonExtension.RegisterEverything
        MLIRPythonSources
        IndexingPythonSources
        )

add_mlir_python_common_capi_library(IndexingPythonCAPI
        INSTALL_COMPONENT IndexingPythonModules
        INSTALL_DESTINATION python_packages/indexing/mlir_indexing/_mlir_libs
        OUTPUT_DIRECTORY "${INDEXING_BINARY_DIR}/python_packages/mlir_indexing/_mlir_libs"
        RELATIVE_INSTALL_ROOT "../../../.."
        DECLARED_SOURCES ${_source_components}
        )

add_mlir_python_modules(IndexingPythonModules
        ROOT_PREFIX "${INDEXING_BINARY_DIR}/python_packages/mlir_indexing"
        INSTALL_PREFIX "python_packages/mlir_indexing"
        DECLARED_SOURCES ${_source_components}
        COMMON_CAPI_LINK_LIBS
        IndexingPythonCAPI
        )

