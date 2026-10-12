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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_GATHER_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_GATHER_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {

inline LiteRtStatus InferGather(const LiteRtOpT& op,
                                absl::Span<const Dims> input_shapes,
                                std::vector<Dims>& output_shapes) {
  constexpr size_t kGatherMinArgs = 2;
  constexpr size_t kInputArgIndex = 0;
  constexpr size_t kIndicesArgIndex = 1;

  if (input_shapes.size() < kGatherMinArgs) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto& input_shape = input_shapes[kInputArgIndex];
  const auto& indices_shape = input_shapes[kIndicesArgIndex];

  const auto& opts = GetTflOptions(op);
  const auto* gather_opts = opts.AsGatherOptions();
  int32_t axis = gather_opts ? gather_opts->axis : 0;

  if (axis < 0) axis += input_shape.size();
  if (axis < 0 || axis >= input_shape.size()) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  Dims out_shape;
  // Output = input_shape[:axis] + indices_shape + input_shape[axis+1:]
  for (int i = 0; i < axis; ++i) out_shape.push_back(input_shape[i]);
  for (auto d : indices_shape) out_shape.push_back(d);
  for (int i = axis + 1; i < input_shape.size(); ++i)
    out_shape.push_back(input_shape[i]);

  output_shapes[0] = std::move(out_shape);
  return kLiteRtStatusOk;
}

inline LiteRtStatus InferGatherNd(const LiteRtOpT& op,
                                  absl::Span<const Dims> input_shapes,
                                  std::vector<Dims>& output_shapes) {
  constexpr size_t kGatherNdMinArgs = 2;
  constexpr size_t kInputArgIndex = 0;
  constexpr size_t kIndicesArgIndex = 1;

  if (input_shapes.size() < kGatherNdMinArgs) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto& input_shape = input_shapes[kInputArgIndex];
  const auto& indices_shape = input_shapes[kIndicesArgIndex];

  if (indices_shape.empty()) return kLiteRtStatusErrorInvalidArgument;
  int32_t index_rank = indices_shape.back();

  // Output = indices_shape[:-1] + input_shape[index_rank:]
  Dims out_shape;
  for (size_t i = 0; i < indices_shape.size() - 1; ++i) {
    out_shape.push_back(indices_shape[i]);
  }

  if (index_rank < 0) {
    return kLiteRtStatusErrorUnsupported;
  }

  for (size_t i = index_rank; i < input_shape.size(); ++i) {
    out_shape.push_back(input_shape[i]);
  }

  output_shapes[0] = std::move(out_shape);
  return kLiteRtStatusOk;
}

inline LiteRtStatus InferEmbeddingLookup(const LiteRtOpT& op,
                                         absl::Span<const Dims> input_shapes,
                                         std::vector<Dims>& output_shapes) {
  // EmbeddingLookup is essentially Gather(params, ids) with axis=0.
  // Inputs: ids, params.
  constexpr size_t kEmbeddingLookupMinArgs = 2;
  constexpr size_t kIdsArgIndex = 0;
  constexpr size_t kParamsArgIndex = 1;

  if (input_shapes.size() < kEmbeddingLookupMinArgs) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto& ids_shape = input_shapes[kIdsArgIndex];
  const auto& params_shape = input_shapes[kParamsArgIndex];

  Dims out_shape = ids_shape;
  for (size_t i = 1; i < params_shape.size(); ++i) {
    out_shape.push_back(params_shape[i]);
  }

  output_shapes[0] = std::move(out_shape);
  return kLiteRtStatusOk;
}

template <typename T, typename IndexT = int32_t>
inline void ReferenceGather(const T* input_data, const int32_t* input_dims,
                            int input_rank, const IndexT* indices_data,
                            const int32_t* indices_dims, int indices_rank,
                            int axis, int batch_dims, T* output_data) {
  if (axis < 0) axis += input_rank;
  if (batch_dims < 0) batch_dims += indices_rank;

  int64_t batch_size = 1;
  for (int i = 0; i < batch_dims; ++i) batch_size *= input_dims[i];
  int64_t outer_size = 1;
  for (int i = batch_dims; i < axis; ++i) outer_size *= input_dims[i];
  int64_t inner_size = 1;
  for (int i = axis + 1; i < input_rank; ++i) inner_size *= input_dims[i];
  int64_t num_indices = 1;
  for (int i = batch_dims; i < indices_rank; ++i) {
    num_indices *= indices_dims[i];
  }

  int64_t axis_size = input_dims[axis];
  for (int64_t b = 0; b < batch_size; ++b) {
    for (int64_t o = 0; o < outer_size; ++o) {
      for (int64_t i = 0; i < num_indices; ++i) {
        int64_t idx = indices_data[b * num_indices + i];
        const T* src =
            input_data + ((b * outer_size + o) * axis_size + idx) * inner_size;
        std::memcpy(output_data, src, inner_size * sizeof(T));
        output_data += inner_size;
      }
    }
  }
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_GATHER_H_
