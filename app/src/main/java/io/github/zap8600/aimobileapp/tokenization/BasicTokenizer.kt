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
package io.github.zap8600.aimobileapp.tokenization

import com.google.common.collect.Iterables
import io.github.zap8600.aimobileapp.tokenization.CharChecker.isControl
import io.github.zap8600.aimobileapp.tokenization.CharChecker.isInvalid
import io.github.zap8600.aimobileapp.tokenization.CharChecker.isPunctuation
import io.github.zap8600.aimobileapp.tokenization.CharChecker.isWhitespace
import java.util.*

/** Basic tokenization (punctuation splitting, lower casing, etc.)  */
class BasicTokenizer(private val doLowerCase: Boolean) {
    fun tokenize(text: String?): List<String> {
        val cleanedText = cleanText(text)
        val origTokens = whitespaceTokenize(cleanedText)
        val stringBuilder = StringBuilder()
        for (token in origTokens) {
            if (doLowerCase) {
                token.let { it.lowercase() }
            }
            val list = runSplitOnPunc(token)
            for (subToken in list) {
                stringBuilder.append(subToken).append(" ")
            }
        }
        return whitespaceTokenize(stringBuilder.toString())
    }

    companion object {
        /* Performs invalid character removal and whitespace cleanup on text. */
        fun cleanText(text: String?): String {
            if (text == null) {
                throw NullPointerException("The input String is null.")
            }
            val stringBuilder = StringBuilder("")
            for (element in text) {

                // Skip the characters that cannot be used.
                if (isInvalid(element) || isControl(element)) {
                    continue
                }
                if (isWhitespace(element)) {
                    stringBuilder.append(" ")
                } else {
                    stringBuilder.append(element)
                }
            }
            return stringBuilder.toString()
        }

        /* Runs basic whitespace cleaning and splitting on a piece of text. */
        fun whitespaceTokenize(text: String?): List<String> {
            if (text == null) {
                throw NullPointerException("The input String is null.")
            }
            return listOf(*text.split(" ".toRegex()).dropLastWhile { it.isEmpty() }
                .toTypedArray())
        }

        /* Splits punctuation on a piece of text. */
        fun runSplitOnPunc(text: String?): List<String> {
            if (text == null) {
                throw NullPointerException("The input String is null.")
            }
            val tokens: MutableList<String> = ArrayList()
            var startNewWord = true
            for (element in text) {
                if (isPunctuation(element)) {
                    tokens.add(element.toString())
                    startNewWord = true
                } else {
                    if (startNewWord) {
                        tokens.add("")
                        startNewWord = false
                    }
                    tokens[tokens.size - 1] = Iterables.getLast(tokens) + element
                }
            }
            return tokens
        }
    }
}