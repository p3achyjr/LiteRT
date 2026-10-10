# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pathlib

import flatbuffers

from absl.testing import absltest as googletest
from litert.tools import litert_lint_rules
from litert.python import schema_py_generated as schema  # pylint:disable=g-direct-tensorflow-import


def _make_tensor(
    shape, buffer_index: int = 0, tensor_type=schema.TensorType.FLOAT32
):
  tensor = schema.TensorT()
  tensor.shape = shape
  tensor.type = tensor_type
  tensor.buffer = buffer_index
  return tensor


class LiteRtLintRulesTest(googletest.TestCase):

  def test_multisig_weight_sharing_pass(self):
    # Two subgraphs sharing the exact same buffer index 1 (150 KB)
    shared_data = b"\xaa" * (150 * 1024)
    model = schema.ModelT()
    model.version = 3
    model.buffers = [schema.BufferT(), schema.BufferT()]
    model.buffers[1].data = shared_data

    sg1 = schema.SubGraphT()
    sg1.tensors = [_make_tensor([150, 1024], buffer_index=1)]
    sg1.operators = []

    sg2 = schema.SubGraphT()
    sg2.tensors = [_make_tensor([150, 1024], buffer_index=1)]
    sg2.operators = []

    model.subgraphs = [sg1, sg2]

    result = litert_lint_rules.lint_model(model)
    self.assertTrue(result.is_passing())
    self.assertEmpty(result.errors)
    self.assertEqual(result.total_weight_bytes, 150 * 1024)
    self.assertEqual(result.shared_weight_bytes, 150 * 1024)

  def test_multisig_weight_sharing_fail(self):
    # Two subgraphs with separate buffers holding identical 150 KB payload
    duplicate_data = b"\xbb" * (150 * 1024)
    model = schema.ModelT()
    model.version = 3
    model.buffers = [schema.BufferT(), schema.BufferT(), schema.BufferT()]
    model.buffers[1].data = duplicate_data
    model.buffers[2].data = duplicate_data

    sg1 = schema.SubGraphT()
    sg1.tensors = [_make_tensor([150, 1024], buffer_index=1)]
    sg1.operators = []

    sg2 = schema.SubGraphT()
    sg2.tensors = [_make_tensor([150, 1024], buffer_index=2)]
    sg2.operators = []

    model.subgraphs = [sg1, sg2]

    result = litert_lint_rules.lint_model(model)
    self.assertFalse(result.is_passing())
    self.assertEqual(result.total_weight_bytes, 2 * 150 * 1024)
    self.assertEqual(result.shared_weight_bytes, 0)
    self.assertTrue(
        any("NO_UNSHARED_MULTISIG_WEIGHTS" in err for err in result.errors)
    )

  def test_activation_arena_budget_pass(self):
    # Subgraph with peak activation of 2 MB when budget is 10 MB
    model = schema.ModelT()
    model.version = 3
    model.buffers = [schema.BufferT()]

    op_code = schema.OperatorCodeT()
    op_code.builtinCode = schema.BuiltinOperator.FULLY_CONNECTED
    model.operatorCodes = [op_code]

    sg = schema.SubGraphT()
    # 500k float32 elements = 2 MB
    sg.tensors = [
        _make_tensor([500000], buffer_index=0),
        _make_tensor([500000], buffer_index=0),
    ]

    op = schema.OperatorT()
    op.opcodeIndex = 0
    op.inputs = [0]
    op.outputs = [1]
    sg.operators = [op]

    model.subgraphs = [sg]

    result = litert_lint_rules.lint_model(model, max_activation_mb=10.0)
    self.assertTrue(result.is_passing())
    self.assertEmpty(result.errors)

  def test_activation_arena_budget_fail(self):
    # Subgraph with activation tensor requiring 40 MB when budget is 30 MB
    model = schema.ModelT()
    model.version = 3
    model.buffers = [schema.BufferT()]

    op_code = schema.OperatorCodeT()
    op_code.builtinCode = schema.BuiltinOperator.FULLY_CONNECTED
    model.operatorCodes = [op_code]

    sg = schema.SubGraphT()
    # 10M float32 elements = 40 MB
    sg.tensors = [
        _make_tensor([10 * 1024 * 1024], buffer_index=0),
        _make_tensor([10 * 1024 * 1024], buffer_index=0),
    ]

    op = schema.OperatorT()
    op.opcodeIndex = 0
    op.inputs = [0]
    op.outputs = [1]
    sg.operators = [op]

    model.subgraphs = [sg]

    result = litert_lint_rules.lint_model(model, max_activation_mb=30.0)
    self.assertFalse(result.is_passing())
    self.assertTrue(
        any("MAX_ACTIVATION_PEAK_BUDGET" in err for err in result.errors)
    )

  def test_partition_fragmentation_pass(self):
    # Subgraph with homogeneous supported ops
    model = schema.ModelT()
    model.version = 3
    model.buffers = [schema.BufferT()]

    op_code = schema.OperatorCodeT()
    op_code.builtinCode = schema.BuiltinOperator.FULLY_CONNECTED
    model.operatorCodes = [op_code]

    sg = schema.SubGraphT()
    sg.tensors = [_make_tensor([10], buffer_index=0) for _ in range(5)]
    sg.operators = []
    for i in range(4):
      op = schema.OperatorT()
      op.opcodeIndex = 0
      op.inputs = [i]
      op.outputs = [i + 1]
      sg.operators.append(op)

    model.subgraphs = [sg]

    result = litert_lint_rules.lint_model(model, max_partitions=3)
    self.assertEqual(result.partition_count, 1)
    self.assertEmpty(result.warnings)

  def test_partition_fragmentation_warn(self):
    # Subgraph alternating supported and unsupported ops: 5 partitions
    model = schema.ModelT()
    model.version = 3
    model.buffers = [schema.BufferT()]

    code_supported = schema.OperatorCodeT()
    code_supported.builtinCode = schema.BuiltinOperator.FULLY_CONNECTED

    code_unsupported = schema.OperatorCodeT()
    code_unsupported.builtinCode = schema.BuiltinOperator.CUSTOM
    code_unsupported.customCode = "MY_CUSTOM_OP"

    model.operatorCodes = [code_supported, code_unsupported]

    sg = schema.SubGraphT()
    sg.tensors = [_make_tensor([10], buffer_index=0) for _ in range(6)]
    sg.operators = []

    # Alternate: op0 (supp), op1 (unsupp), op2 (supp), op3 (unsupp), op4 (supp)
    op_indices = [0, 1, 0, 1, 0]
    for i, code_idx in enumerate(op_indices):
      op = schema.OperatorT()
      op.opcodeIndex = code_idx
      op.inputs = [i]
      op.outputs = [i + 1]
      sg.operators.append(op)

    model.subgraphs = [sg]

    result = litert_lint_rules.lint_model(model, max_partitions=3)
    self.assertGreater(result.partition_count, 3)
    self.assertTrue(
        any("PARTITION_FRAGMENTATION_CHECK" in w for w in result.warnings)
    )

  def test_end_to_end_model_file(self):
    temp_dir = pathlib.Path(self.create_tempdir().full_path)
    model_path = temp_dir / "test_model.tflite"

    model = schema.ModelT()
    model.version = 3
    model.buffers = [schema.BufferT()]
    op_code = schema.OperatorCodeT()
    op_code.builtinCode = schema.BuiltinOperator.ADD
    model.operatorCodes = [op_code]

    sg = schema.SubGraphT()
    sg.tensors = [
        _make_tensor([10, 16], buffer_index=0),
        _make_tensor([10, 16], buffer_index=0),
        _make_tensor([10, 16], buffer_index=0),
    ]
    op = schema.OperatorT()
    op.opcodeIndex = 0
    op.inputs = [0, 1]
    op.outputs = [2]
    sg.operators = [op]
    model.subgraphs = [sg]

    builder = flatbuffers.Builder(1024)
    root = model.Pack(builder)
    builder.Finish(root, file_identifier=b"TFL3")
    model_path.write_bytes(bytes(builder.Output()))

    result = litert_lint_rules.lint_model_file(str(model_path))
    self.assertTrue(result.is_passing())
    self.assertEqual(result.subgraphs_count, 1)
    self.assertGreater(result.peak_activation_bytes, 0)

  def test_model_file_not_found(self):
    result = litert_lint_rules.lint_model_file(
        "/nonexistent/path/to/model.tflite"
    )
    self.assertFalse(result.is_passing())
    self.assertTrue(any("not found" in err for err in result.errors))

  def test_scalar_and_empty_tensors(self):
    model = schema.ModelT()
    model.version = 3
    model.buffers = [schema.BufferT()]
    sg = schema.SubGraphT()
    # Scalar tensor with shape []
    sg.tensors = [_make_tensor([], buffer_index=0)]
    sg.operators = []
    model.subgraphs = [sg]

    result = litert_lint_rules.lint_model(model)
    self.assertTrue(result.is_passing())
    self.assertEmpty(result.errors)

  def test_single_subgraph_weight_sharing_ignored(self):
    # In a single-subgraph model, duplicate buffers are not flagged by
    # multisig rule.
    duplicate_data = b"\xbb" * (150 * 1024)
    model = schema.ModelT()
    model.version = 3
    model.buffers = [schema.BufferT(), schema.BufferT(), schema.BufferT()]
    model.buffers[1].data = duplicate_data
    model.buffers[2].data = duplicate_data

    sg = schema.SubGraphT()
    sg.tensors = [
        _make_tensor([150, 1024], buffer_index=1),
        _make_tensor([150, 1024], buffer_index=2),
    ]
    sg.operators = []
    model.subgraphs = [sg]

    result = litert_lint_rules.lint_model(model)
    self.assertTrue(result.is_passing())
    self.assertEmpty(result.errors)

  def test_orphan_subgraphs_warn_and_call_once_reachability(self):
    model = schema.ModelT()
    model.version = 3
    model.buffers = [schema.BufferT()]

    op_code = schema.OperatorCodeT()
    op_code.builtinCode = schema.BuiltinOperator.CALL_ONCE
    model.operatorCodes = [op_code]

    # Subgraph 0 (signature "serving_default") -> CALL_ONCE(Subgraph 1);
    # Subgraph 2 is an unreferenced orphan subgraph.
    sg0 = schema.SubGraphT()
    sg0.tensors = []
    call_once = schema.OperatorT()
    call_once.opcodeIndex = 0
    call_once_opts = schema.CallOnceOptionsT()
    call_once_opts.initSubgraphIndex = 1
    call_once.builtinOptionsType = schema.BuiltinOptions.CallOnceOptions
    call_once.builtinOptions = call_once_opts
    sg0.operators = [call_once]

    sg1 = schema.SubGraphT()
    sg1.tensors = []
    sg1.operators = []

    sg2 = schema.SubGraphT()
    sg2.tensors = []
    sg2.operators = []

    model.subgraphs = [sg0, sg1, sg2]

    sig = schema.SignatureDefT()
    sig.signatureKey = "serving_default"
    sig.subgraphIndex = 0
    model.signatureDefs = [sig]

    result = litert_lint_rules.lint_model(model)
    self.assertTrue(result.is_passing())
    self.assertEqual(result.orphan_subgraphs_count, 1)
    self.assertTrue(
        any("ORPHAN_SUBGRAPH_CHECK" in w and "#2" in w for w in result.warnings)
    )


if __name__ == "__main__":
  googletest.main()
