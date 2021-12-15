// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <pybind11/pybind11.h>

#include "CAPI.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace py = pybind11;

// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------

PYBIND11_MODULE(_ireeSandbox, m) {
  m.doc() = "IREE LLVM Sandbox Module";

  m.def(
      "register_sandbox_passes_and_dialects",
      [](MlirContext context) { ireeLlvmSandboxRegisterAll(context); },
      py::arg("context"));
}
