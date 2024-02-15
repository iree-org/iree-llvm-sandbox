set(SUBSTRAIT_ROOT_DIR ${STRUCTURED_MAIN_SRC_DIR}/third_party/substrait)
set(SUBSTRAIT_PROTO_DIR ${SUBSTRAIT_ROOT_DIR}/proto)
set(SUBSTRAIT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(SUBSTRAIT_PROTO_NAMES
  algebra
  capabilities
  extended_expression
  extensions/extensions
  function
  parameterized_types
  plan
  type_expressions
  type
  )

# Compute file paths.
set(SUBSTRAIT_PROTO_FILES "")
foreach(SUBSTRAIT_PROTO_NAME ${SUBSTRAIT_PROTO_NAMES})
  list(APPEND SUBSTRAIT_PROTO_FILES
       "${SUBSTRAIT_PROTO_DIR}/substrait/${SUBSTRAIT_PROTO_NAME}.proto")
endforeach(SUBSTRAIT_PROTO_NAME ${SUBSTRAIT_PROTO_NAMES})

# Add targets for each individual proto file.
set(SUBSTRAIT_SOURCES "")
set(SUBSTRAIT_HEADERS "")
foreach(SUBSTRAIT_PROTO_NAME ${SUBSTRAIT_PROTO_NAMES})
  set(SUBSTRAIT_SOURCE "${SUBSTRAIT_BINARY_DIR}/substrait/${SUBSTRAIT_PROTO_NAME}.pb.cc")
  set(SUBSTRAIT_HEADER "${SUBSTRAIT_BINARY_DIR}/substrait/${SUBSTRAIT_PROTO_NAME}.pb.h")
  set(SUBSTRAIT_PROTO_FILE "${SUBSTRAIT_PROTO_DIR}/substrait/${SUBSTRAIT_PROTO_NAME}.proto")
  list(APPEND SUBSTRAIT_SOURCES ${SUBSTRAIT_SOURCE})
  list(APPEND SUBSTRAIT_HEADERS ${SUBSTRAIT_HEADER})
  add_custom_command(
    OUTPUT
    ${SUBSTRAIT_SOURCE}
    ${SUBSTRAIT_HEADER}

    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
    "-I${SUBSTRAIT_PROTO_DIR}"
    "--cpp_out=${SUBSTRAIT_BINARY_DIR}"
    "${SUBSTRAIT_PROTO_FILE}"
    DEPENDS ${SUBSTRAIT_PROTO_FILES}
    )
endforeach(SUBSTRAIT_PROTO_NAME ${SUBSTRAIT_PROTO_NAMES})

# Library target.
add_mlir_library(Substrait
  ${SUBSTRAIT_SOURCES}

  ADDITIONAL_HEADERS
  ${SUBSTRAIT_HEADERS}

  LINK_LIBS PUBLIC
  protobuf::libprotobuf
  )
target_include_directories(Substrait
  PUBLIC
  "$<BUILD_INTERFACE:${SUBSTRAIT_BINARY_DIR}>"
  )
