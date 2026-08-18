// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/runtime/tfl_utils.h"

#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/internal/litert_dispatch_delegate_vendor_options.h"
#include "litert/core/build_stamp.h"
#include "litert/core/dispatch_op_schema.h"
#include "litert/core/model/model.h"
#include "litert/test/matchers.h"
#include "tflite/internal/signature_def.h"
#include "tflite/interpreter.h"

namespace tflite {
// Test helper taking advantage of friend class tflite::InterpreterTest on
// tflite::Interpreter.
class InterpreterTest {
 public:
  static void SetSignatureDefs(Interpreter* interpreter,
                               std::vector<internal::SignatureDef> defs) {
    interpreter->SetSignatureDef(std::move(defs));
  }
};
}  // namespace tflite

namespace litert::internal {
namespace {

TEST(TflUtilsTest, BuildDispatchDelegateVendorOptionsNullInterpreter) {
  LiteRtModelT model;
  Expected<DispatchDelegateVendorOptions> options =
      BuildDispatchDelegateVendorOptions(/*interpreter=*/nullptr, model);
  EXPECT_FALSE(options.HasValue());
  EXPECT_EQ(options.Error().Status(), kLiteRtStatusErrorInvalidArgument);
}

TEST(TflUtilsTest, BuildDispatchDelegateVendorOptionsEmptySignatures) {
  tflite::Interpreter interpreter;
  LiteRtModelT model;
  LITERT_ASSERT_OK_AND_ASSIGN(
      DispatchDelegateVendorOptions options,
      BuildDispatchDelegateVendorOptions(&interpreter, model));
  EXPECT_TRUE(options.FunctionMappings().empty());
}

TEST(TflUtilsTest, BuildDispatchDelegateVendorOptionsInvalidSubgraphIndex) {
  tflite::Interpreter interpreter;
  tflite::internal::SignatureDef sig_def;
  sig_def.signature_key = "serving_default";
  sig_def.subgraph_index = 5;
  tflite::InterpreterTest::SetSignatureDefs(&interpreter, {sig_def});

  LiteRtModelT model;
  Expected<DispatchDelegateVendorOptions> options =
      BuildDispatchDelegateVendorOptions(&interpreter, model);
  EXPECT_FALSE(options.HasValue());
  EXPECT_EQ(options.Error().Status(), kLiteRtStatusErrorNotFound);
}

TEST(TflUtilsTest, BuildDispatchDelegateVendorOptionsSuccess) {
  tflite::Interpreter interpreter;
  tflite::internal::SignatureDef sig_def1;
  sig_def1.signature_key = "signature_vision";
  sig_def1.subgraph_index = 0;

  tflite::internal::SignatureDef sig_def2;
  sig_def2.signature_key = "signature_audio";
  sig_def2.subgraph_index = 1;

  tflite::InterpreterTest::SetSignatureDefs(&interpreter,
                                            {sig_def1, sig_def2});

  LiteRtModelT model;

  // Subgraph 0: contains a non-custom op and a dispatch op with input/output
  // tensors.
  LiteRtSubgraphT& sg0 = model.EmplaceSubgraph();
  LiteRtOpT& non_custom_op = sg0.EmplaceOp();
  non_custom_op.SetOpCode(kLiteRtOpCodeTflAdd);

  LiteRtOpT& op0 = sg0.EmplaceOp();
  op0.SetOpCode(kLiteRtOpCodeTflCustom);
  op0.SetCustomCode(std::string(kLiteRtDispatchOpCustomName));
  DispatchOpOptions op_opts0;
  op_opts0.name = "vision_func";
  op0.SetCustomOptions(MakeDispatchOpOptions(std::move(op_opts0)));

  LiteRtTensorT& in0 = sg0.EmplaceTensor();
  in0.SetName("image_input");
  op0.Inputs().push_back(&in0);

  LiteRtTensorT& out0 = sg0.EmplaceTensor();
  out0.SetName("vision_output");
  op0.Outputs().push_back(&out0);

  // Subgraph 1: contains another dispatch op with input/output tensors.
  LiteRtSubgraphT& sg1 = model.EmplaceSubgraph();
  LiteRtOpT& op1 = sg1.EmplaceOp();
  op1.SetOpCode(kLiteRtOpCodeTflCustom);
  op1.SetCustomCode(std::string(kLiteRtDispatchOpCustomName));
  DispatchOpOptions op_opts1;
  op_opts1.name = "audio_func";
  op1.SetCustomOptions(MakeDispatchOpOptions(std::move(op_opts1)));

  LiteRtTensorT& in1 = sg1.EmplaceTensor();
  in1.SetName("audio_input");
  op1.Inputs().push_back(&in1);

  LiteRtTensorT& out1 = sg1.EmplaceTensor();
  out1.SetName("audio_output");
  op1.Outputs().push_back(&out1);

  LITERT_ASSERT_OK_AND_ASSIGN(
      DispatchDelegateVendorOptions vendor_options,
      BuildDispatchDelegateVendorOptions(&interpreter, model));

  absl::Span<const FunctionMapping> mappings =
      vendor_options.FunctionMappings();
  ASSERT_EQ(mappings.size(), 2);

  EXPECT_EQ(mappings[0].function_name, "vision_func");
  EXPECT_EQ(mappings[0].signature_name, "signature_vision");
  ASSERT_EQ(mappings[0].input_tensor_ports.size(), 1);
  EXPECT_EQ(mappings[0].input_tensor_ports[0].tensor_name, "image_input");
  EXPECT_EQ(mappings[0].input_tensor_ports[0].port_index, 0);
  ASSERT_EQ(mappings[0].output_tensor_ports.size(), 1);
  EXPECT_EQ(mappings[0].output_tensor_ports[0].tensor_name, "vision_output");
  EXPECT_EQ(mappings[0].output_tensor_ports[0].port_index, 0);

  EXPECT_EQ(mappings[1].function_name, "audio_func");
  EXPECT_EQ(mappings[1].signature_name, "signature_audio");
  ASSERT_EQ(mappings[1].input_tensor_ports.size(), 1);
  EXPECT_EQ(mappings[1].input_tensor_ports[0].tensor_name, "audio_input");
  EXPECT_EQ(mappings[1].input_tensor_ports[0].port_index, 0);
  ASSERT_EQ(mappings[1].output_tensor_ports.size(), 1);
  EXPECT_EQ(mappings[1].output_tensor_ports[0].tensor_name, "audio_output");
  EXPECT_EQ(mappings[1].output_tensor_ports[0].port_index, 0);
}

}  // namespace
}  // namespace litert::internal
