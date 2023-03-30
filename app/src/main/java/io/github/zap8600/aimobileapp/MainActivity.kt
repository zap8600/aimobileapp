package io.github.zap8600.aimobileapp

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.JsonReader
import io.github.zap8600.aimobileapp.tokenization.GPT2Tokenizer
import java.io.BufferedReader
import java.io.InputStreamReader

class MainActivity : AppCompatActivity() {
    private lateinit var tokenizer: GPT2Tokenizer

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        //Initialize input field
        val userPrompt = findViewById<android.widget.EditText>(R.id.userPrompt)
        val generate = findViewById<android.widget.Button>(R.id.generate)

        //Initialize tokenizer
        tokenizer = loadTokenizer()

        //Add listener to generate button
        generate.setOnClickListener {
            val prompt = userPrompt.text.toString()
            val tokenizedPrompt = tokenizer.encode(prompt)
            //Set input field to tokenized prompt
            userPrompt.setText(tokenizedPrompt.toString())
        }
    }

    private fun loadTokenizer(): GPT2Tokenizer {
        //Create encoder of Map<String, Int>
        val encoder = mutableMapOf<String, Int>()
        val vocabStream = application.assets.open("vocab.json")
        vocabStream.use {
            val vocabReader = JsonReader(InputStreamReader(it, "UTF-8"))
            vocabReader.beginObject()
            while (vocabReader.hasNext()) {
                val key = vocabReader.nextName()
                val value = vocabReader.nextInt()
                encoder[key] = value
            }
            vocabReader.close()
        }
        val decoder  = encoder.entries.associateBy({ it.value }, { it.key })
        val bpeRanks = mutableMapOf<Pair<String, String>, Int>()
        val mergesStream = application.assets.open("merges.txt")
        mergesStream.use { stream ->
            val mergesReader = BufferedReader(InputStreamReader(stream))
            mergesReader.useLines { seq ->
                seq.drop(1).forEachIndexed { i, s ->
                    val list = s.split(" ")
                    val keyTuple = list[0] to list[1]
                    bpeRanks[keyTuple] = i
                }
            }
        }
        return GPT2Tokenizer(encoder, decoder, bpeRanks)
    }
}