package com.example.llama.rag

import android.content.Context
import android.util.Log
import com.google.mediapipe.tasks.text.textembedder.TextEmbedder
import com.google.mediapipe.tasks.text.textembedder.TextEmbedder.TextEmbedderOptions
import com.google.mediapipe.tasks.core.BaseOptions
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.sqrt

class RagManager(private val context: Context) {

    companion object {
        private const val TAG = "RagManager"
    }

    private var embedder: TextEmbedder? = null
    private var vectorArray: FloatArray? = null
    private var numRows: Int = 0
    private var numCols: Int = 0

    // Memory-mapped metadata
    private var metaIndexBuffer: ByteBuffer? = null 
    private var metaDataBuffer: ByteBuffer? = null
    private var metaTotalRows: Int = 0

    private var isInitialized = false

    private suspend fun initialize() = withContext(Dispatchers.IO) {
        if (isInitialized) return@withContext

        try {
            val startTime = System.currentTimeMillis()
            // 1. Load Text Embedder
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("text_embedder.tflite")
                .build()
            val options = TextEmbedderOptions.builder()
                .setBaseOptions(baseOptions)
                .build()
            embedder = TextEmbedder.createFromOptions(context, options)

            // 2. Load Vector Index into Heap (requires largeHeap="true")
            val vecFile = copyAssetToFile("rag_index.bin")
            RandomAccessFile(vecFile, "r").use { raf ->
                val channel = raf.channel
                val size = channel.size()
                val buffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, size).order(ByteOrder.LITTLE_ENDIAN)
                
                numRows = buffer.int
                numCols = buffer.int
                
                Log.i(TAG, "Allocating ${numRows * numCols * 4} bytes for vectors in heap...")
                val floatArray = FloatArray(numRows * numCols)
                buffer.position(8)
                buffer.asFloatBuffer().get(floatArray)
                
                Log.i(TAG, "Pre-normalizing vectors...")
                for (r in 0 until numRows) {
                    var sumSq = 0f
                    val offset = r * numCols
                    for (c in 0 until numCols) {
                        val v = floatArray[offset + c]
                        sumSq += v * v
                    }
                    val norm = sqrt(sumSq)
                    if (norm > 0) {
                        for (c in 0 until numCols) {
                            floatArray[offset + c] /= norm
                        }
                    }
                }
                vectorArray = floatArray
                Log.i(TAG, "Vectors loaded and normalized in ${System.currentTimeMillis() - startTime}ms")
            }

            // 3. Memory-map Metadata (.bin)
            val metaFile = copyAssetToFile("rag_metadata.bin")
            val rafMeta = RandomAccessFile(metaFile, "r")
            val channelMeta = rafMeta.channel
            val sizeMeta = channelMeta.size()
            val metaFullBuffer = channelMeta.map(FileChannel.MapMode.READ_ONLY, 0, sizeMeta).order(ByteOrder.LITTLE_ENDIAN)
            
            metaTotalRows = metaFullBuffer.int
            val indexTableSize = metaTotalRows * 8 
            
            metaFullBuffer.position(4)
            metaIndexBuffer = (metaFullBuffer.slice().limit(indexTableSize) as ByteBuffer).order(ByteOrder.LITTLE_ENDIAN)
            
            metaFullBuffer.position(4 + indexTableSize)
            metaDataBuffer = (metaFullBuffer.slice() as ByteBuffer).order(ByteOrder.LITTLE_ENDIAN)
            
            Log.i(TAG, "Mapped metadata: $metaTotalRows rows. Total init: ${System.currentTimeMillis() - startTime}ms")
            isInitialized = true
        } catch (e: Exception) {
            Log.e(TAG, "Initialization failed", e)
        }
    }

    private fun copyAssetToFile(assetName: String): File {
        val file = File(context.filesDir, assetName)
        if (!file.exists()) {
            Log.i(TAG, "Copying $assetName to internal storage...")
            context.assets.open(assetName).use { input ->
                file.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }
        return file
    }

    suspend fun getContext(query: String, topK: Int = 2, threshold: Float = 0.78f): String? = withContext(Dispatchers.IO) {
        if (!isInitialized) initialize()

        val currentEmbedder = embedder ?: return@withContext null
        val vArr = vectorArray ?: return@withContext null
        val mIdxBuf = metaIndexBuffer ?: return@withContext null
        val mDataBuf = metaDataBuffer ?: return@withContext null

        try {
            val startTime = System.currentTimeMillis()
            // Embed query
            val result = currentEmbedder.embed(query)
            val queryVector = result.embeddingResult().embeddings()[0].floatEmbedding()
            
            // L2 Normalize query vector
            var qNorm = 0f
            for (v in queryVector) qNorm += v * v
            qNorm = sqrt(qNorm)
            if (qNorm > 0) {
                for (i in queryVector.indices) queryVector[i] /= qNorm
            }

            // Compute cosine similarities using Heap-resident vectors (Fastest)
            val scores = FloatArray(numRows)
            for (r in 0 until numRows) {
                var dot = 0f
                val rowOffset = r * numCols
                for (c in 0 until numCols) {
                    dot += queryVector[c] * vArr[rowOffset + c]
                }
                scores[r] = dot
            }

            // Get top K indices above threshold and with sufficient content length
            val topIndices = scores.indices
                .filter { scores[it] > threshold }
                .filter { idx ->
                    val length = mIdxBuf.getInt(idx * 8 + 4)
                    length > 100 
                }
                .sortedByDescending { scores[it] }
                .take(topK)

            Log.d(TAG, "Search completed in ${System.currentTimeMillis() - startTime}ms. Found ${topIndices.size} matches.")

            if (topIndices.isEmpty()) {
                return@withContext null
            }

            // Fetch strings
            return@withContext topIndices.joinToString("\n---\n") { idx ->
                val offset = mIdxBuf.getInt(idx * 8)
                val length = mIdxBuf.getInt(idx * 8 + 4)
                val bytes = ByteArray(length)
                val slice = mDataBuf.duplicate()
                slice.position(offset)
                slice.get(bytes)
                String(bytes, Charsets.UTF_8)
            }

        } catch (e: Exception) {
            Log.e(TAG, "getContext failed", e)
            return@withContext null
        }
    }
}
