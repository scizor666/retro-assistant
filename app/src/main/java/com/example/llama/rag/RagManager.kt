package com.example.llama.rag

import android.content.Context
import android.util.Log
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.text.textembedder.TextEmbedder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.sqrt

/**
 * Provides retrieval-augmented context for user queries using the same pipeline
 * as the Ubuntu evaluate.py:
 *
 *  1. Embed query with MediaPipe TextEmbedder (text_embedder.tflite asset)
 *  2. L2-normalise the query vector
 *  3. Dot-product against pre-normalised corpus vectors (rag_index.bin asset)
 *  4. Return top-k chunks where cosine similarity > 0.65, joined by "\n---\n"
 *     or null when nothing passes the threshold
 *
 * Parameters are identical to evaluate.py:
 *   top_k = 2, threshold = 0.65
 */
class RagManager(private val context: Context) {

    companion object {
        private const val TAG = "RagManager"

        private const val ASSET_EMBEDDER = "text_embedder.tflite"
        private const val ASSET_BIN      = "rag_index.bin"
        private const val ASSET_JSON     = "rag_texts.json"

        // Must match evaluate.py constants exactly
        private const val TOP_K = 2
        private const val THRESHOLD = 0.4f
        private const val CONTEXT_SEPARATOR = "\n---\n"
    }

    // -----------------------------------------------------------------------
    // Lazy-loaded singletons — initialised once on first use (IO thread)
    // -----------------------------------------------------------------------

    /** MediaPipe TextEmbedder backed by the bundled TFLite model */
    private val embedder: TextEmbedder by lazy {
        Log.i(TAG, "Initialising MediaPipe TextEmbedder from $ASSET_EMBEDDER …")
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath(ASSET_EMBEDDER)
            .build()
        val options = TextEmbedder.TextEmbedderOptions.builder()
            .setBaseOptions(baseOptions)
            .build()
        TextEmbedder.createFromOptions(context, options).also {
            Log.i(TAG, "TextEmbedder ready")
        }
    }

    /** Holds the vector data and its dimensions together to ensure atomicity */
    private data class CorpusData(
        val vectors: FloatArray,
        val numRows: Int,
        val numCols: Int
    )

    private val corpus: CorpusData by lazy { loadBinaryVectors() }

    /** Parallel text chunks from rag_texts.json, same ordering as vectors. */
    private val corpusTexts: List<String> by lazy { loadTextsJson() }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    suspend fun getContext(query: String): String? = withContext(Dispatchers.IO) {
        try {
            // Ensure assets are loaded
            val (vectors, rows, cols) = corpus
            
            // 1. Embed query
            val result = embedder.embed(query)
            val embedding = result.embeddingResult().embeddings()[0]
            val rawVec: FloatArray = embedding.floatEmbedding()
                ?: run {
                    Log.w(TAG, "Embedder returned null floatEmbedding, skipping RAG")
                    return@withContext null
                }

            // 2. L2-normalise query vector
            val qNorm = l2Norm(rawVec)
            val qVec = if (qNorm > 0f) FloatArray(rawVec.size) { rawVec[it] / qNorm }
                       else rawVec

            // 3. Cosine similarities via dot-product
            val scores = FloatArray(rows) { row ->
                dotProduct(vectors, row * cols, qVec)
            }

            // 4. Argsort descending, take top-k
            val topIndices = scores.indices
                .sortedByDescending { scores[it] }
                .take(TOP_K)

            val topScore = topIndices.firstOrNull()?.let { scores[it] } ?: 0f
            Log.i(TAG, "Top context score for '$query': ${"%.4f".format(topScore)}")

            // 5. Filter by threshold
            val chunks = topIndices
                .filter { scores[it] > THRESHOLD }
                .mapNotNull { corpusTexts.getOrNull(it) }

            if (chunks.isEmpty()) {
                Log.i(TAG, "No context above threshold $THRESHOLD (best: ${"%.4f".format(topScore)})")
                return@withContext null
            }

            Log.i(TAG, "Injecting context (${chunks.size} chunks, ${chunks.sumOf { it.length }} chars)")
            chunks.joinToString(CONTEXT_SEPARATOR)

        } catch (e: Exception) {
            Log.e(TAG, "RAG lookup failed: ${e.message}", e)
            null
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    private fun loadBinaryVectors(): CorpusData {
        Log.i(TAG, "Loading $ASSET_BIN …")
        context.assets.open(ASSET_BIN).use { input ->
            val headerBytes = ByteArray(8)
            input.read(headerBytes)
            
            val headerBuf = ByteBuffer.wrap(headerBytes).order(ByteOrder.LITTLE_ENDIAN)
            val rows = headerBuf.int
            val cols = headerBuf.int

            Log.i(TAG, "Header: $rows vectors of dim $cols")

            val totalElements = rows * cols
            val dataBytes = ByteArray(totalElements * 4)
            
            // Read in chunks as .read(byteArray) isn't guaranteed to fill the whole buffer in one call for large streams
            var bytesRead = 0
            while (bytesRead < dataBytes.size) {
                val n = input.read(dataBytes, bytesRead, dataBytes.size - bytesRead)
                if (n == -1) break
                bytesRead += n
            }

            val floatArr = FloatArray(totalElements)
            ByteBuffer.wrap(dataBytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(floatArr)
            
            Log.i(TAG, "Binary vectors loaded successfully")
            return CorpusData(floatArr, rows, cols)
        }
    }

    private fun loadTextsJson(): List<String> {
        Log.i(TAG, "Loading $ASSET_JSON …")
        val raw = context.assets.open(ASSET_JSON).bufferedReader().readText()
        val arr = JSONArray(raw)
        val texts = ArrayList<String>(arr.length())
        for (i in 0 until arr.length()) {
            texts.add(arr.getJSONObject(i).optString("content", ""))
        }
        Log.i(TAG, "Loaded ${texts.size} text chunks")
        return texts
    }

    /** L2 (Euclidean) norm of a FloatArray */
    private fun l2Norm(vec: FloatArray): Float {
        var sum = 0f
        for (v in vec) sum += v * v
        return sqrt(sum)
    }

    /**
     * Dot-product of [query] against a single row in [matrix].
     * [rowStart] is the flat index of the first element of that row.
     */
    private fun dotProduct(matrix: FloatArray, rowStart: Int, query: FloatArray): Float {
        var sum = 0f
        for (i in query.indices) sum += matrix[rowStart + i] * query[i]
        return sum
    }
}
