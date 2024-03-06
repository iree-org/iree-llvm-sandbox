# RUN: %PYTHON %s 2>&1 | FileCheck %s

import json

from mlir_structured.dialects import substrait as ss, arith
from mlir_structured.ir import Context, Location

JSON_PLAN = '''
  {
    "version": {
      "minorNumber": 42,
      "patchNumber": 1,
    }
  }
'''


def run(f):
  print("\nTEST:", f.__name__)
  with Context(), Location.unknown():
    ss.register_dialect()
    f()
  return f


# CHECK-LABEL: TEST: testJsonFormat
@run
def testJsonFormat():
  plan_module = ss.from_json(JSON_PLAN)
  print(plan_module)
  # CHECK: substrait.plan version

  json_plan = ss.to_json(plan_module.operation)
  print(json_plan)
  # CHECK: {"version":{"minorNumber":42,"patchNumber":1}}

  json_plan = json.dumps(json.loads(json_plan))
  print(json_plan)
  # CHECK: {"version": {"minorNumber": 42, "patchNumber": 1}}

  json_plan = ss.to_json(plan_module.operation, pretty=True)
  print(json_plan)
  # CHECK:      "version": {
  # CHECK-NEXT:   "minorNumber": 42,
  # CHECK-NEXT:   "patchNumber": 1


# CHECK-LABEL: TEST: testTextPB
@run
def testTextPB():
  plan_module = ss.from_json(JSON_PLAN)

  text_plan = ss.to_textpb(plan_module.operation)
  print(text_plan)
  # CHECK:      version {
  # CHECK-NEXT:   minor_number: 42
  # CHECK-NEXT:   patch_number: 1

  plan_module = ss.from_textpb(text_plan)
  print(plan_module)
  # CHECK: substrait.plan version


# CHECK-LABEL: TEST: testBinPB
@run
def testBinPB():
  plan_module = ss.from_json(JSON_PLAN)

  bin_plan = ss.to_binpb(plan_module.operation)
  print(bin_plan)
  # CHECK: 2

  plan_module = ss.from_binpb(bin_plan)
  print(plan_module)
  # CHECK: substrait.plan version


# CHECK-LABEL: TEST: testInvalid
@run
def testInvalid():
  try:
    ss.from_json('this is not json')
    # CHECK-NEXT: error: could not deserialize JSON as 'Plan' message:
    # CHECK-NEXT: Unexpected token.
    # CHECK-NEXT: this is not json
  except ValueError as ex:
    print(ex)
    # CHECK:      Could not import Substrait plan

  const_op = arith.ConstantOp.create_index(42)
  try:
    ss.to_json(const_op)
    # CHECK-NEXT: error: 'arith.constant' op not supported for export
  except ValueError as ex:
    print(ex)
    # CHECK-NEXT: Could not export Substrait plan
