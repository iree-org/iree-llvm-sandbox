
#include "capi.h"
#include "mlir/CAPI/Registration.h"
#include "dialect.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Jasc, jasc, jasc::JascDialect)
