package io.github.zap8600.aimobileapp.ml

import android.app.Application
import android.text.Spannable
import android.text.SpannableStringBuilder
import android.util.JsonReader
import android.widget.TextView
import androidx.core.content.res.ResourcesCompat
import androidx.databinding.BindingAdapter
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import com.google.common.base.Joiner
import io.github.zap8600.aimobileapp.R
import io.github.zap8600.aimobileapp.tokenization.GPT2Tokenizer
import io.github.zap8600.aimobileapp.tokenization.BertTokenizer
import kotlinx.coroutines.*
import org.tensorflow.lite.Interpreter
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.InputStreamReader
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.random.Random

private const val GPT2SEQUENCE_LENGTH  = 64
private const val VOCAB_SIZE       = 50257
private const val NUM_HEAD         = 12
private const val NUM_LITE_THREADS = 4
private const val GPT2MODEL       = "gpt2.tflite"
private const val GPT2VOCAB       = "gpt2-vocab.json"
private const val GPT2MERGES      = "gpt2-merges.txt"

private const val BERTSEQUENCE_LENGTH  = 384
private const val BERTANSWER_LENGTH = 32
private const val BERTPREDICT_ANS_NUM = 5
private const val BERTOUTPUT_OFFSET = 1
private const val BERTDO_LOWER_CASE = true
private const val BERTMAX_QUERY_LEN = 64
private const val BERTMODEL       = "distilbert.tflite"
private const val BERTDICT       = "bert-vocab.json"

private const val TAG              = "Client"

private typealias Predictions = Array<Array<FloatArray>>

enum class StrategyEnum { GREEDY, TOPK }
data class Strategy(val strategy: StrategyEnum, val value: Int = 0)

private val BERTSPACE_JOINER = Joiner.on(" ")

class Client(application: Application) : AndroidViewModel(application) {
    private val initJob: Job
    private var generateJob: Job? = null
    private lateinit var gpt2Tokenizer: GPT2Tokenizer
    private lateinit var model: Interpreter
    private lateinit var bertTokenizer: BertTokenizer

    private val _prompt = MutableLiveData("Your prompt will go here and text will be generated with it, once you hit \"Generate\" after entering your prompt.")
    val prompt: LiveData<String> = _prompt

    private val _completion = MutableLiveData("")
    val completion: LiveData<String> = _completion

    //model will be a string that will be passed in from the main activity

    private var strategy = Strategy(StrategyEnum.TOPK, 40)

    init {
        initJob = viewModelScope.launch {
            val gpt2Encoder  = loadEncoder(GPT2VOCAB)
            val gpt2Decoder  = gpt2Encoder.entries.associateBy({ it.value }, { it.key })
            val gpt2BpeRanks = loadBpeRanks(GPT2MERGES)

            gpt2Tokenizer = GPT2Tokenizer(gpt2Encoder, gpt2Decoder, gpt2BpeRanks)

            bertTokenizer = BertTokenizer(loadDictionary(BERTDICT))
        }
    }

    override fun onCleared() {
        super.onCleared()
        model.close()
    }

    fun launchGeneration(text: String, usrModel: String) {
        generateJob = viewModelScope.launch {
            initJob.join()
            generateJob?.cancelAndJoin()
            _completion.value = ""
            _prompt.value = text
            if(usrModel == "GPT-2"){
                model = loadModel(GPT2MODEL)
                generateGPT2(_prompt.value!!)
            }
            else if(usrModel == "DistilBERT"){
                model = loadModel(BERTMODEL)
                generateBERT(_prompt.value!!, "Extractive Question Answering is the task of extracting an answer from a text given a question. An example of an question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune a model on a SQuAD task, you may leverage the examples/pytorch/question-answering/run_squad.py script.")
            }
        }
    }

    private suspend fun generateGPT2(text: String, nbTokens: Int = 100) = withContext(Dispatchers.Default) {
        val tokens = gpt2Tokenizer.encode(text)
        repeat (nbTokens) {
            val maxTokens    = tokens.takeLast(GPT2SEQUENCE_LENGTH).toIntArray()
            val paddedTokens = maxTokens + IntArray(GPT2SEQUENCE_LENGTH - maxTokens.size)
            val inputIds     = Array(1) { paddedTokens }

            val predictions: Predictions = Array(1) { Array(GPT2SEQUENCE_LENGTH) { FloatArray(VOCAB_SIZE) } }
            val outputs = mutableMapOf<Int, Any>(0 to predictions)

            model.runForMultipleInputsOutputs(arrayOf(inputIds), outputs)
            val outputLogits = predictions[0][maxTokens.size-1]

            val nextToken: Int = when (strategy.strategy) {
                StrategyEnum.TOPK -> {
                    val filteredLogitsWithIndexes = outputLogits
                            .mapIndexed { index, fl -> (index to fl) }
                            .sortedByDescending { it.second }
                            .take(strategy.value)

                    // Softmax computation on filtered logits
                    val filteredLogits = filteredLogitsWithIndexes.map { it.second }
                    val maxLogitValue  = filteredLogits.max()
                    val logitsExp      = filteredLogits.map { exp(it - maxLogitValue) }
                    val sumExp         = logitsExp.sum()
                    val probs          = logitsExp.map { it.div(sumExp) }

                    val logitsIndexes = filteredLogitsWithIndexes.map { it.first }
                    sample(logitsIndexes, probs)
                }
                else -> outputLogits.argmax()
            }

            tokens.add(nextToken)
            val decodedToken = gpt2Tokenizer.decode(listOf(nextToken))
            _completion.postValue(_completion.value + decodedToken)

            yield()
        }
    }

    private suspend fun generateBERT(query: String, content: String) = withContext(Dispatchers.Default) {
        val tokenizedText = bertTokenizer.encode(query)
        val bertVocab = loadEncoder(BERTDICT)

        _completion.postValue(bertVocab.toString())
        yield()
    }

    private suspend fun loadModel(file: String): Interpreter = withContext(Dispatchers.IO) {
        val assetFileDescriptor = getApplication<Application>().assets.openFd(file)
        assetFileDescriptor.use {
            val fileChannel = FileInputStream(assetFileDescriptor.fileDescriptor).channel
            val modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, it.startOffset, it.declaredLength)

            val opts = Interpreter.Options()
            opts.numThreads = NUM_LITE_THREADS
            return@use Interpreter(modelBuffer, opts)
        }
    }

    private suspend fun loadEncoder(file: String): Map<String, Int> = withContext(Dispatchers.IO) {
        hashMapOf<String, Int>().apply {
            val vocabStream = getApplication<Application>().assets.open(file)
            vocabStream.use {
                val vocabReader = JsonReader(InputStreamReader(it, "UTF-8"))
                vocabReader.beginObject()
                while (vocabReader.hasNext()) {
                    val key = vocabReader.nextName()
                    val value = vocabReader.nextInt()
                    put(key, value)
                }
                vocabReader.close()
            }
        }
    }

    private suspend fun loadBpeRanks(file: String):Map<Pair<String, String>, Int> = withContext(Dispatchers.IO) {
        hashMapOf<Pair<String, String>, Int>().apply {
            val mergesStream = getApplication<Application>().assets.open(file)
            mergesStream.use { stream ->
                val mergesReader = BufferedReader(InputStreamReader(stream))
                mergesReader.useLines { seq ->
                    seq.drop(1).forEachIndexed { i, s ->
                        val list = s.split(" ")
                        val keyTuple = list[0] to list[1]
                        put(keyTuple, i)
                    }
                }
            }
        }
    }
    //loadDictionary function. Loads BERT's vocab.txt from the file. Turns it into a Map<String, Int>. [PAD] is 0, [unused1] through [unused99] are 1 through 99, [UNK] is 100, [CLS] is 101, [SEP] is 102, [MASK] is 103, [unused100] and [unused101] are 104 and 105, and the rest are 106 through 30521. The file is not a .json file, so it is read as a .txt file.
    private suspend fun loadDictionary(file: String): Map<String, Int> = withContext(Dispatchers.IO) {
        hashMapOf<String, Int>().apply {
            val vocabStream = getApplication<Application>().assets.open(file)
            vocabStream.use {
                val vocabReader = BufferedReader(InputStreamReader(it))
                vocabReader.useLines { seq ->
                    seq.forEachIndexed { i, s ->
                        put(s, i)
                    }
                }
            }
        }
    }
}

private fun randomIndex(probs: List<Float>): Int {
    val rnd = probs.sum() * Random.nextFloat()
    var acc = 0f

    probs.forEachIndexed { i, fl ->
        acc += fl
        if (rnd < acc) {
            return i
        }
    }

    return probs.size - 1
}

private fun sample(indexes: List<Int>, probs: List<Float>): Int {
    val i = randomIndex(probs)
    return indexes[i]
}

private fun FloatArray.argmax(): Int {
    var bestIndex = 0
    repeat(size) {
        if (this[it] > this[bestIndex]) {
            bestIndex = it
        }
    }

    return bestIndex
}

@BindingAdapter("prompt", "completion")
fun TextView.formatCompletion(prompt: String, completion: String): Unit {
    text = when {
        completion.isEmpty() -> prompt
        else -> {
            val str = SpannableStringBuilder(prompt + completion)
            val bgCompletionColor = ResourcesCompat.getColor(resources, R.color.colorPrimary, context.theme)
            str.setSpan(android.text.style.BackgroundColorSpan(bgCompletionColor), prompt.length, str.length, Spannable.SPAN_EXCLUSIVE_EXCLUSIVE)

            str
        }
    }
}