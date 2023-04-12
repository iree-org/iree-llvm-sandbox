add_executable(ScanAndPrint
  ScanAndPrint.cpp
)
target_link_libraries(ScanAndPrint
  PRIVATE
    DatabaseIterators
)
