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

#import <Foundation/Foundation.h>

#import "third_party/odml/litert/litert/objc/apis/LRTTensorBuffer.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * Describes the allocation requirements for an input or output @c LRTTensorBuffer, as specified by
 * the compiled model and hardware accelerator.
 */
@interface LRTTensorBufferRequirements : NSObject <NSCopying>

/**
 * Buffer storage types (@c LRTTensorBufferType boxed in @c NSNumber) supported for this tensor.
 */
@property(nonatomic, copy, readonly) NSArray<NSNumber *> *supportedBufferTypes;

/** Minimum required buffer size in bytes. */
@property(nonatomic, readonly) NSUInteger bufferSize;

/** Required memory alignment in bytes. */
@property(nonatomic, readonly) NSUInteger alignment;

/**
 * Tensor buffer strides per dimension (boxed as @c NSNumber), or an empty array when strides are
 * not specified.
 */
@property(nonatomic, copy, readonly) NSArray<NSNumber *> *strides;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

/**
 * Initializes tensor buffer requirements with the specified storage types, size, alignment, and
 * strides.
 *
 * @param supportedBufferTypes Supported @c LRTTensorBufferType values boxed in @c NSNumber.
 * @param bufferSize Minimum required buffer size in bytes.
 * @param alignment Required memory alignment in bytes.
 * @param strides Per-dimension buffer strides boxed in @c NSNumber.
 * @return An initialized @c LRTTensorBufferRequirements instance.
 */
- (instancetype)initWithSupportedBufferTypes:(NSArray<NSNumber *> *)supportedBufferTypes
                                  bufferSize:(NSUInteger)bufferSize
                                   alignment:(NSUInteger)alignment
                                     strides:(NSArray<NSNumber *> *)strides
    NS_DESIGNATED_INITIALIZER;

/**
 * Returns whether the receiver is equal to another @c LRTTensorBufferRequirements instance.
 *
 * @param otherRequirements The requirements instance to compare with the receiver.
 * @return @c YES if both instances have identical buffer types, size, alignment, and strides.
 */
- (BOOL)isEqualToRequirements:(LRTTensorBufferRequirements *)otherRequirements;

@end

NS_ASSUME_NONNULL_END
