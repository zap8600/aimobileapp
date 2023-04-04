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
private const val BERTDICT       = "bert-vocab.txt"

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
    private lateinit var featureConverter: FeatureConverter

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

            featureConverter = FeatureConverter(loadDictionary(BERTDICT), BERTDO_LOWER_CASE, BERTMAX_QUERY_LEN, BERTSEQUENCE_LENGTH)
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
        val feature: Feature = featureConverter.convert(query, content)
        val inputIds = Array(1) {
            IntArray(BERTSEQUENCE_LENGTH)
        }
        val startLogits = Array(1) {
            FloatArray(BERTSEQUENCE_LENGTH)
        }
        val endLogits = Array(1) {
            FloatArray(BERTSEQUENCE_LENGTH)
        }

        for (j in 0 until BERTSEQUENCE_LENGTH) {
            inputIds[0][j] = feature.inputIds[j]
        }

        val output: MutableMap<Int, Any> = HashMap()
        output[0] = startLogits
        output[1] = endLogits

        model.runForMultipleInputsOutputs(arrayOf<Any>(inputIds), output)

        val answers = getBestAnswers(startLogits[0], endLogits[0], feature)

        if (answers != null) {
            if(answers.isNotEmpty()) {
                val topAnswer = answers[0]
                _completion.postValue(_completion.value + topAnswer.text)
            }
        }

        yield()
    }

    private fun getBestAnswers(
        startLogits: FloatArray, endLogits: FloatArray, feature: Feature
    ): List<QaAnswer>? {
        // Model uses the closed interval [start, end] for indices.
        val startIndexes: IntArray = getBestIndex(startLogits, feature.tokenToOrigMap)
        val endIndexes: IntArray = getBestIndex(endLogits, feature.tokenToOrigMap)
        val origResults: MutableList<QaAnswer.Pos> = ArrayList()
        for (start in startIndexes) {
            for (end in endIndexes) {
                if (end < start) {
                    continue
                }
                val length = end - start + 1
                if (length > BERTANSWER_LENGTH) {
                    continue
                }
                origResults.add(QaAnswer.Pos(start, end, startLogits[start] + endLogits[end]))
            }
        }
        origResults.sort()
        val answers: MutableList<QaAnswer> = ArrayList()
        for (i in origResults.indices) {
            if (i >= BERTPREDICT_ANS_NUM) {
                break
            }
            var convertedText: String = if (origResults[i].start > 0) ({
                convertBack(
                    feature,
                    origResults[i].start,
                    origResults[i].end
                )
            }).toString() else {
                ""
            }
            val ans = QaAnswer(convertedText, origResults[i])
            answers.add(ans)
        }
        return answers
    }

    private fun getBestIndex(logits: FloatArray, tokenToOrigMap: Map<Int, Int>): IntArray {
        val tmpList: MutableList<QaAnswer.Pos> = ArrayList()
        for (i in 0 until BERTSEQUENCE_LENGTH) {
            if (tokenToOrigMap.containsKey(i + BERTOUTPUT_OFFSET)) {
                tmpList.add(QaAnswer.Pos(i, i, logits[i]))
            }
        }
        tmpList.sort()
        val indexes =
            IntArray(BERTPREDICT_ANS_NUM)
        for (i in 0 until BERTPREDICT_ANS_NUM) {
            indexes[i] = tmpList[i].start
        }
        return indexes
    }

    private fun convertBack(feature: Feature, start: Int, end: Int): String? {
        // Shifted index is: index of logits + offset.
        val shiftedStart: Int =
            start + BERTOUTPUT_OFFSET
        val shiftedEnd: Int =
            end + BERTOUTPUT_OFFSET
        val startIndex = feature.tokenToOrigMap[shiftedStart]!!
        val endIndex = feature.tokenToOrigMap[shiftedEnd]!!
        // end + 1 for the closed interval.
        return BERTSPACE_JOINER.join(
            feature.origTokens.subList(
                startIndex,
                endIndex + 1
            )
        )
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

    private suspend fun loadDictionary(file: String): Map<String, Int> = withContext(Dispatchers.IO) {
        hashMapOf<String, Int>().apply {
            val dictStream = getApplication<Application>().assets.open(file)
            dictStream.use { stream ->
                val dictReader = BufferedReader(InputStreamReader(stream))
                dictReader.useLines { seq ->
                    seq.drop(1).forEachIndexed { i, s ->
                        val list = s.split(" ")
                        val keyTuple = list[0]
                        put(keyTuple, i)
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