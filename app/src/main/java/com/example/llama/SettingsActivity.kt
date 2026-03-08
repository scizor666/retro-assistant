package com.example.llama

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.llama.data.AppDatabase
import com.example.llama.data.ConversationRepository
import com.arm.aichat.gguf.GgufMetadataReader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream

class SettingsActivity : AppCompatActivity() {

    private lateinit var modelMetadataTv: TextView
    private lateinit var changeModelBtn: Button
    private lateinit var clearHistoryBtn: Button
    private lateinit var clearAllDataBtn: Button
    private lateinit var processingLayout: android.view.View
    private lateinit var processingStatusTv: TextView
    private lateinit var repository: ConversationRepository

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)

        val database = AppDatabase.getDatabase(applicationContext)
        repository = ConversationRepository(database)

        modelMetadataTv = findViewById(R.id.model_metadata_tv)
        changeModelBtn = findViewById(R.id.change_model_btn)
        clearHistoryBtn = findViewById(R.id.clear_history_btn)
        clearAllDataBtn = findViewById(R.id.clear_all_data_btn)
        processingLayout = findViewById(R.id.processing_layout)
        processingStatusTv = findViewById(R.id.processing_status_tv)

        // Load current metadata
        val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        modelMetadataTv.text = prefs.getString(MainActivity.PREF_LAST_MODEL_METADATA, "No model documentation available.")

        changeModelBtn.setOnClickListener {
            getContent.launch(arrayOf("*/*"))
        }

        clearHistoryBtn.setOnClickListener {
            lifecycleScope.launch {
                repository.clearAllHistory()
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@SettingsActivity, "Conversation history cleared", Toast.LENGTH_SHORT).show()
                }
            }
        }

        clearAllDataBtn.setOnClickListener {
            lifecycleScope.launch {
                repository.clearAllHistory()
                getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE).edit().clear().apply()
                withContext(Dispatchers.Main) {
                    modelMetadataTv.text = "No model documentation available."
                    Toast.makeText(this@SettingsActivity, "All data and settings cleared", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private val getContent = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        uri?.let { handleSelectedModel(it) }
    }

    private fun handleSelectedModel(uri: Uri) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                withContext(Dispatchers.Main) {
                    processingLayout.visibility = android.view.View.VISIBLE
                    processingStatusTv.text = "Parsing GGUF metadata..."
                    changeModelBtn.isEnabled = false
                }

                contentResolver.openInputStream(uri)?.use {
                    GgufMetadataReader.create().readStructuredMetadata(it)
                }?.let { metadata ->
                    val modelName = metadata.filename() + ".gguf"
                    
                    withContext(Dispatchers.Main) {
                        processingStatusTv.text = "Copying model to private storage..."
                    }

                    contentResolver.openInputStream(uri)?.use { input ->
                        ensureModelFile(modelName, input)
                    }?.let { modelFile ->
                        val metadataStr = metadata.toString()
                        
                        getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE).edit()
                            .putString(MainActivity.PREF_LAST_MODEL_PATH, modelFile.absolutePath)
                            .putString(MainActivity.PREF_LAST_MODEL_METADATA, metadataStr)
                            .putBoolean(MainActivity.PREF_MODEL_CHANGED, true)
                            .apply()

                        withContext(Dispatchers.Main) {
                            modelMetadataTv.text = metadataStr
                            processingLayout.visibility = android.view.View.GONE
                            changeModelBtn.isEnabled = true
                            Toast.makeText(this@SettingsActivity, "Model updated: $modelName", Toast.LENGTH_SHORT).show()
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e("SettingsActivity", "Failed to load model", e)
                withContext(Dispatchers.Main) {
                    processingLayout.visibility = android.view.View.GONE
                    changeModelBtn.isEnabled = true
                    Toast.makeText(this@SettingsActivity, "Failed to load model", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private suspend fun ensureModelFile(modelName: String, input: InputStream) =
        withContext(Dispatchers.IO) {
            val modelsDir = File(filesDir, "models")
            if (!modelsDir.exists()) modelsDir.mkdir()
            File(modelsDir, modelName).also { file ->
                if (!file.exists()) {
                    FileOutputStream(file).use { input.copyTo(it) }
                }
            }
        }

    companion object {
        const val PREFS_NAME = "com.example.llama_preferences"
    }
}
