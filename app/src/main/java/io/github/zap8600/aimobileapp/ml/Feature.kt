/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
package io.github.zap8600.aimobileapp.ml

import com.google.common.primitives.Ints

/** Feature to be fed into the Bert model.  */
class Feature(
    inputIds: List<Int?>?,
    inputMask: List<Int?>?,
    segmentIds: List<Int?>?,
    origTokens: List<String>,
    tokenToOrigMap: Map<Int, Int>
) {
    val inputIds: IntArray
    private val inputMask: IntArray
    private val segmentIds: IntArray
    val origTokens: List<String>
    val tokenToOrigMap: Map<Int, Int>

    init {
        this.inputIds = Ints.toArray(inputIds)
        this.inputMask = Ints.toArray(inputMask)
        this.segmentIds = Ints.toArray(segmentIds)
        this.origTokens = origTokens
        this.tokenToOrigMap = tokenToOrigMap
    }
}