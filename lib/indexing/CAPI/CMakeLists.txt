add_mlir_public_c_api_library(IndexingCAPI
        Dialects.cpp
        LINK_LIBS PUBLIC
        MLIRIndexing
        MLIRPass
        )
