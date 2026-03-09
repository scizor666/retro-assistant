package com.example.llama.rag

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import kotlin.math.sqrt

/**
 * Pure-JVM unit tests for the math helpers in RagManager.
 *
 * These replicate the Python evaluate.py behaviour:
 *   - l2Norm / normalize
 *   - dotProduct / cosine similarity
 *   - threshold gating (< 0.65 → null context)
 */
class RagManagerTest {

    // ---- Helpers copied from RagManager (package-private equivalents) -------

    private fun l2Norm(vec: FloatArray): Float {
        var sum = 0f
        for (v in vec) sum += v * v
        return sqrt(sum)
    }

    private fun normalize(vec: FloatArray): FloatArray {
        val norm = l2Norm(vec)
        return if (norm > 0f) FloatArray(vec.size) { vec[it] / norm } else vec
    }

    private fun dotProduct(matrix: FloatArray, rowStart: Int, query: FloatArray): Float {
        var sum = 0f
        for (i in query.indices) sum += matrix[rowStart + i] * query[i]
        return sum
    }

    // ---- Tests -----------------------------------------------------------------

    @Test
    fun `l2Norm of 3-4-0 vector is 5`() {
        val norm = l2Norm(floatArrayOf(3f, 4f, 0f))
        assertEquals(5f, norm, 1e-6f)
    }

    @Test
    fun `normalize produces unit vector`() {
        val unit = normalize(floatArrayOf(3f, 4f))
        assertEquals(0.6f, unit[0], 1e-6f)
        assertEquals(0.8f, unit[1], 1e-6f)
        // Norm of result should be 1.0
        assertEquals(1f, l2Norm(unit), 1e-6f)
    }

    @Test
    fun `normalize of zero vector returns it unchanged`() {
        val zero = normalize(floatArrayOf(0f, 0f))
        assertEquals(0f, zero[0], 0f)
        assertEquals(0f, zero[1], 0f)
    }

    @Test
    fun `identical vectors have cosine similarity of 1`() {
        val vec = normalize(floatArrayOf(1f, 2f, 3f))
        val matrix = vec.copyOf()              // one-row matrix
        val score = dotProduct(matrix, rowStart = 0, query = vec)
        assertEquals(1f, score, 1e-6f)
    }

    @Test
    fun `orthogonal vectors have cosine similarity of 0`() {
        val a = normalize(floatArrayOf(1f, 0f))
        val b = normalize(floatArrayOf(0f, 1f))
        val matrix = a.copyOf()
        val score = dotProduct(matrix, rowStart = 0, query = b)
        assertEquals(0f, score, 1e-6f)
    }

    @Test
    fun `opposite vectors have cosine similarity of -1`() {
        val a = normalize(floatArrayOf(1f, 0f))
        val b = normalize(floatArrayOf(-1f, 0f))
        val matrix = a.copyOf()
        val score = dotProduct(matrix, rowStart = 0, query = b)
        assertEquals(-1f, score, 1e-6f)
    }

    /**
     * Simulate the threshold gate used in RagManager.getContext():
     *   scores below 0.65 must produce null (no context injected)
     */
    @Test
    fun `threshold gating filters scores below 0_65`() {
        val threshold = 0.65f
        val scores = floatArrayOf(0.90f, 0.50f, 0.70f)     // only indices 0 and 2 pass
        val texts = listOf("chunk-A", "chunk-B", "chunk-C")

        val chunks = scores.indices
            .sortedByDescending { scores[it] }
            .take(2)
            .filter { scores[it] > threshold }
            .mapNotNull { texts.getOrNull(it) }

        // index 0 (0.90) and index 2 (0.70) pass — ordered descending
        assertEquals(listOf("chunk-A", "chunk-C"), chunks)
    }

    @Test
    fun `all scores below threshold yields empty list equivalent to null`() {
        val threshold = 0.65f
        val scores = floatArrayOf(0.30f, 0.20f)
        val texts = listOf("chunk-A", "chunk-B")

        val chunks = scores.indices
            .sortedByDescending { scores[it] }
            .take(2)
            .filter { scores[it] > threshold }
            .mapNotNull { texts.getOrNull(it) }

        assert(chunks.isEmpty()) { "Expected empty context list but got: $chunks" }
        // Matches RagManager returning null when chunks.isEmpty()
        val result: String? = if (chunks.isEmpty()) null else chunks.joinToString("\n---\n")
        assertNull(result)
    }

    @Test
    fun `context chunks joined by separator`() {
        val chunks = listOf("Super Mario World is a platformer.", "SNES launched in 1990.")
        val joined = chunks.joinToString("\n---\n")
        assertEquals("Super Mario World is a platformer.\n---\nSNES launched in 1990.", joined)
    }
}
