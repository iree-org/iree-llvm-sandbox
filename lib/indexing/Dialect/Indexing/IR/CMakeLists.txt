add_mlir_library(MLIRIndexing
        Indexing.cpp

        LINK_LIBS PUBLIC
        MLIRFuncDialect
        MLIRInferTypeOpInterface
        MLIRIR
        MLIRTensorDialect

        DEPENDS
        MLIRIndexingOpsIncGen
        )
