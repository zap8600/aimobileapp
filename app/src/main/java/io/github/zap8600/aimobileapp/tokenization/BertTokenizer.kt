package io.github.zap8600.aimobileapp.tokenization

import java.util.StringTokenizer

//Tokenizer class for BERT. Has a parameter for the vocabulary, which is a .txt file that is passed to the class as Map<String, Int>.
//It will be able to encode and decode text.
class BertTokenizer(private val vocab: Map<String, Int>) {
    // Normalize function that normalizes the text and splits it into tokens.
    fun normalize(text: String): List<String> {
        val normalized = text.lowercase()
            .replace("[^a-z0-9'.,!?\\s]".toRegex(), "")
            .replace("\\s+".toRegex(), " ")
            .trim()
        return normalized.split(" ")
    }
    
    fun pretokenize(tokens: List<String>): List<String> {
        val pretokenized = mutableListOf<String>()
        pretokenized.add("[CLS]")
        for (token in tokens) {
            val matches = "\\w+|[^\uAC00-\uD7A3\u3130-\u318F\\s]+".toRegex().findAll(token)
            for (match in matches) {
                pretokenized.add(match.value)
            }
        }
        pretokenized.add("[SEP]")
        return pretokenized
    }

    //encode function that encodes the text into a list of integers. It uses the normalize and pretokenize functions. For example, encoding "Who was Jim Henson?" will return [101, 2040, 2001, 3958, 27227, 1029, 102].
    fun encode(text: String): List<Int> {
        val normalized = normalize(text)
        val pretokenized = pretokenize(normalized)
        return pretokenized.map { vocab[it] ?: 100 }
    }
}