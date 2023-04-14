import re
import sys
from os.path import exists

symbol_guard = """\
#ifdef MY_PYBIND11_EXPORT_SYMBOLS
#  define MY_PYBIND11_EXPORT __attribute__((visibility("default")))
#else
#  define MY_PYBIND11_EXPORT
#endif
"""

module_guard = """\
#ifdef MY_PYBIND11_EXPORT_MODULE
#  define MY_PYBIND11_LOCAL_MODULE
#else
#  define MY_PYBIND11_LOCAL_MODULE , py::module_local()
#endif
"""

def ir_module(fp):
  with open(fp) as f:
    f = f.read()
  if not exists(f"{fp}.bkup"):
    with open(f"{fp}.bkup", "w") as bkup:
      bkup.write(f)
    with open(fp, "w") as ff:
      ff.write(symbol_guard)
      res = re.sub(
          r"^class ((?!MY_PYBIND11_EXPORT).*) {$",
          r"class MY_PYBIND11_EXPORT \1 {",
          f,
          flags=re.MULTILINE,
      )
      res = res.replace(
          "class DefaultingPyMlirContext",
          "class MY_PYBIND11_EXPORT DefaultingPyMlirContext",
      )
      ff.write(res)


def ir_core(fp):
  with open(fp) as f:
    f = f.read()
  if not exists(f"{fp}.bkup"):
    with open(f"{fp}.bkup", "w") as bkup:
      bkup.write(f)
    with open(fp, "w") as ff:
      ff.write(module_guard)
      res = f.replace(", py::module_local()", " MY_PYBIND11_LOCAL_MODULE")
      ff.write(res)


if __name__ == "__main__":
  fp = sys.argv[1]
  if fp.endswith("IRModule.h") or fp.endswith("PybindUtils.h"):
    ir_module(fp)
  elif fp.endswith("IRCore.cpp") or fp.endswith("IRAffine.cpp"):
    ir_core(fp)
  else:
    raise NotImplementedError(f"unknown fp {fp=}")
