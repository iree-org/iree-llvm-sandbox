//===-- call_kernel.cc - Runtime glue for JAX kernels -----------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sys/stat.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/lib/Bindings/Python/IRModule.h"
#include "pybind11/cast.h"
#include "pybind11/pybind11.h"
#include "pybind11_abseil/status_casters.h"
#include "status_macros.h"

#include "mlir_lowering.h"

#define VLOG(X) std::cerr

namespace jasc {
namespace {

// A "truncated" definition from `struct StridedMemRefType` defined in
// "mlir/include/mlir/ExecutionEngine/CRunnerUtils.h" without `barePtr` and
// `data`. The value of those pointers are not available by the time when the
// metadata is constructed until the CpuKernel is actually invoked.
struct StridedMemRefMD {
  int64_t offset;
  llvm::SmallVector<int64_t> sizes;
  llvm::SmallVector<int64_t> strides;
  explicit StridedMemRefMD(size_t rank) : offset(0), sizes(rank, 0), strides(rank, 0) {}
};

// A compiled CPU kernel ready to be executed. This class is responsible for
// holding the compiled code. It maintaining a global registry of CPU kernels
// identified by unique ID.
class CpuKernel {
 public:
  CpuKernel(std::unique_ptr<mlir::ExecutionEngine> execution_engine,
            llvm::SmallVector<StridedMemRefMD, 8> &&memref_metadata,
            int num_inputs, int num_ouputs)
      : execution_engine_(std::move(execution_engine)),
        memref_metadata_(std::move(memref_metadata)),
        num_inputs_(num_inputs),
        num_outputs_(num_ouputs) {
    absl::WriterMutexLock lock(&global_registry_mutex_);
    if (global_registry_ == nullptr) {
      global_registry_ = new absl::flat_hash_map<int, CpuKernel *>();
    }
    identifier_ = next_kernel_id_++;
    global_registry_->emplace(identifier_, this);

    VLOG(1) << "allocated kernel " << identifier_ << "\n";
  }

  ~CpuKernel() {
    absl::WriterMutexLock lock(&global_registry_mutex_);
    global_registry_->erase(identifier_);
    VLOG(1) << "deallocated kernel " << identifier_ << "\n";
  }

  // A unique identifier for the kernel.
  int identifier() const { return identifier_; }

  // Retrieve a kernel given its identifier.
  static const CpuKernel *GetKernelById(int id) {
    absl::ReaderMutexLock lock(&global_registry_mutex_);
    auto it = global_registry_->find(id);
    if (it == global_registry_->end()) {
      LOG(FATAL) << "unable to find kernel " << id;
    }
    return it->second;
  }

  void Call(void *out, void **ins) const {
    std::vector<void *> args;
    // Each input has one basePtr, one data, and one offset + an array of sizes
    // and strides depending on the rank.
    size_t flat_args_count = (num_inputs_ + num_outputs_) * 3;
    for (const StridedMemRefMD &MD : memref_metadata_) {
      flat_args_count += MD.sizes.size();
      flat_args_count += MD.strides.size();
    }
    args.reserve(flat_args_count);

    // Reconstructs a memref descriptor structure from a bare pointer. See
    // `struct StridedMemRefType` in "mlir/ExecutionEngine/CRunnerUtils.h"
    auto pack_args = [&](void *ptr, const StridedMemRefMD& MD) {
      args.push_back(ptr);  // basePtr
      args.push_back(ptr);  // data
      StridedMemRefMD &tmp_MD = const_cast<StridedMemRefMD &>(MD);
      args.push_back(reinterpret_cast<void *>(&tmp_MD.offset));
      for (auto &sz : tmp_MD.sizes) {
        args.push_back(reinterpret_cast<void *>(&sz));
      }
      for (auto &sd : tmp_MD.strides) {
        args.push_back(reinterpret_cast<void *>(&sd));
      }
    };

    for (int i = 0; i < num_inputs_; ++i) {
      pack_args(&ins[i], memref_metadata_[i]);
    }

    if (num_outputs_ == 1) {
      pack_args(&out, memref_metadata_[num_inputs_]);
    } else {
      void **out_ptrs = reinterpret_cast<void **>(out);
      for (int i = 0; i < num_outputs_; ++i) {
        pack_args(&out_ptrs[i], memref_metadata_[num_inputs_ + i]);
      }
    }

    assert(args.size() == flat_args_count);
    llvm::cantFail(execution_engine_->invokePacked("main", args));
  }

 private:
  static absl::Mutex global_registry_mutex_;
  static absl::flat_hash_map<int, CpuKernel *> *global_registry_
      ABSL_GUARDED_BY(global_registry_mutex_);
  static int next_kernel_id_ ABSL_GUARDED_BY(global_registry_mutex_);

  std::unique_ptr<mlir::ExecutionEngine> execution_engine_;
  llvm::SmallVector<StridedMemRefMD> memref_metadata_;
  int num_inputs_;
  int num_outputs_;
  int identifier_;
};

absl::Mutex CpuKernel::global_registry_mutex_(absl::kConstInit);
absl::flat_hash_map<int, CpuKernel *> *CpuKernel::global_registry_ = nullptr;
int CpuKernel::next_kernel_id_ = 0;

llvm::SmallVector<StridedMemRefMD> PopulateMemrefMetaData(
    mlir::FunctionType ftp) {
  // Modified from `fill_sizes_and_strides` defined in cpu_executable.cc
  auto fill_metadata = [&](mlir::ArrayRef<int64_t> shape) -> StridedMemRefMD {
    StridedMemRefMD MD(shape.size());
    size_t multiplier = 1;
    for (int i = static_cast<int>(shape.size()); i > 0; --i) {
      size_t position = i - 1;
      // Payload using `position` instead of `i`.
      size_t size = shape[position];
      MD.sizes[position] = size;
      MD.strides[position] = multiplier;
      multiplier *= size;
    }
    return MD;
  };

  llvm::SmallVector<StridedMemRefMD> ret;
  for (auto t :
       llvm::concat<const mlir::Type>(ftp.getInputs(), ftp.getResults())) {
    auto stp = t.cast<mlir::ShapedType>();
    ret.push_back(fill_metadata(stp.getShape()));
  }
  return ret;
}

absl::StatusOr<std::unique_ptr<CpuKernel>> CreateCpuKernel(
    mlir::python::PyModule &py_module, int num_inputs, int num_outputs,
    bool dump_ir) {
  mlir::ModuleOp module = unwrap(py_module.get());
  // Fills in the memref metadata according to the function type.
  auto entry = llvm::cast<mlir::func::FuncOp>(module.lookupSymbol("main"));
  auto MDs = PopulateMemrefMetaData(entry.getFunctionType());
  assert(MDs.size() == num_inputs + num_outputs);

  RETURN_IF_ERROR(LowerStableHloToCpuLLVM(module, dump_ir));
  mlir::ExecutionEngineOptions engine_opts;
  // TODO(ulysse): Select LLVM opt level.
  static constexpr std::array<llvm::StringRef, 1> sharedLibPaths = {
      "libmlir_c_runner_utils.so"};
  engine_opts.sharedLibPaths = sharedLibPaths;
  engine_opts.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Default;
  auto engineOrError = mlir::ExecutionEngine::create(module, engine_opts);
  if (!engineOrError) {
    llvm::handleAllErrors(
        engineOrError.takeError(), [&](const llvm::StringError &err) {
          LOG(FATAL) << "Error while creating execution engine: "
                     << err.getMessage();
        });
  }
  return std::make_unique<CpuKernel>(llvm::cantFail(std::move(engineOrError)),
                                     std::move(MDs), num_inputs, num_outputs);
}

// XLA custom call callback that calls a kernel on CPU. The first input is the
// identifier of the kernel. Subsequent inputs are the inputs of the kernel.
void CpuCallback(void *out, void **ins) {
  int64_t identifier = *reinterpret_cast<int64_t *>(ins[0]);
  const CpuKernel *kernel = CpuKernel::GetKernelById(identifier);
  kernel->Call(out, (ins + 1));
}

/// Clears the `PyOperation` (representing Python-level handles to
/// `Operation *`s) that are tracked by the context. This function should be
/// called by any entry point that may modify the IR, which could cause above
/// handles to be dangling.
// void clearOperationsInside(mlir::python::PyModule &py_module) {
//   llvm::errs() << "clearOperationsInside\n";
//   MlirOperation op = mlirModuleGetOperation(py_module.get());
//   auto py_op = mlir::python::PyOperation::forOperation(
//       py_module.getContext(), op, py_module.getCapsule());
//   llvm::errs() << "got py_op\n";
//   py_module.getContext()->clearOperationsInside(py_op->getOperation());
// }

namespace py = ::pybind11;

PYBIND11_MODULE(call_kernel, m) {
  pybind11::google::ImportStatusModule();

  // Initializes LLVM targets. Must be called before CreateCpuKernel.
  m.def("init_llvm", []() {
    LLVMInitializeNativeTarget();
    LLVMInitializeNativeAsmPrinter();
    LLVMInitializeNativeAsmParser();
  });
  m.def(
      "apply_schedule",
      [](mlir::python::PyModule &py_module, bool dump_ir) {
        // py_module.getContext()->clearOperationsInside(py_module);
        mlir::ModuleOp module = unwrap(py_module.get());
        return ApplyTransformScript(module, dump_ir);
      },
      py::arg("module"), py::arg("dump_ir") = false);

  py::class_<CpuKernel>(m, "CpuKernel")
      .def_property_readonly("identifier", &CpuKernel::identifier);

  m.def(
      "create_cpu_kernel",
      [](mlir::python::PyModule &py_module, int num_inputs, int num_outputs,
         bool dump_ir) {
        // py_module.getContext()->clearOperationsInside(py_module);
        return CreateCpuKernel(py_module, num_inputs, num_outputs, dump_ir);
      },
      py::arg("module"), py::arg("num_inputs"), py::arg("num_outputs"),
      py::arg("dump_ir") = false);

  m.def("get_cpu_callback", []() {
    return pybind11::capsule(reinterpret_cast<void *>(&CpuCallback),
                             "xla._CUSTOM_CALL_TARGET");
  });

  m.def(
      "lower_to_linalg",
      [](mlir::python::PyModule &py_module, bool dump_ir) {
        // clearOperationsInside(py_module);
        mlir::ModuleOp module = unwrap(py_module.get());
        return LowerStableHloToLinalg(module, dump_ir);
      },
      py::arg("module"), py::arg("dump_ir") = false);
}

}  // namespace
}  // namespace jasc
