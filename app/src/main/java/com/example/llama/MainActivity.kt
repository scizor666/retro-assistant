package com.example.llama

import android.net.Uri
import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.widget.EditText
import android.widget.ImageButton
import android.widget.TextView
import android.widget.Toast
import androidx.activity.addCallback
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.arm.aichat.AiChat
import com.arm.aichat.InferenceEngine
import com.arm.aichat.gguf.GgufMetadata
import com.arm.aichat.gguf.GgufMetadataReader
import com.google.android.material.floatingactionbutton.FloatingActionButton
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.onCompletion
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.util.UUID
import com.example.llama.data.AppDatabase
import com.example.llama.data.ConversationRepository
import com.example.llama.data.ConversationEntity
import com.example.llama.data.MessageEntity
import androidx.drawerlayout.widget.DrawerLayout
import com.google.android.material.navigation.NavigationView
import android.widget.Button

class MainActivity : AppCompatActivity() {

    // Android views
    private lateinit var messagesRv: RecyclerView
    private lateinit var userInputEt: EditText
    private lateinit var userActionFab: FloatingActionButton
    private lateinit var clearChatBtn: ImageButton
    private lateinit var openSettingsBtn: ImageButton
    private lateinit var loadingPb: android.widget.ProgressBar

    // Arm AI Chat inference engine
    private lateinit var engine: InferenceEngine
    private var generationJob: Job? = null

    // Conversation states
    private var isModelReady = false
    private val messages = mutableListOf<Message>()
    private val lastAssistantMsg = StringBuilder()
    private val messageAdapter = MessageAdapter(messages)

    // History states
    private lateinit var repository: ConversationRepository
    private var currentConversationId: Long? = null
    private lateinit var drawerLayout: DrawerLayout
    private lateinit var historyRv: RecyclerView
    private lateinit var historyAdapter: HistoryAdapter
    private lateinit var newChatBtn: Button
    private lateinit var openHistoryBtn: ImageButton
    private var messageCollectionJob: Job? = null
    private var isRefreshingModel = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        // View model boilerplate and state management is out of this basic sample's scope
        onBackPressedDispatcher.addCallback { Log.w(TAG, "Ignore back press for simplicity") }

        // Find views
        messagesRv = findViewById(R.id.messages)
        messagesRv.layoutManager = LinearLayoutManager(this).apply { stackFromEnd = true }
        messagesRv.adapter = messageAdapter
        userInputEt = findViewById(R.id.user_input)
        userActionFab = findViewById(R.id.fab)
        clearChatBtn = findViewById(R.id.clear_chat)
        drawerLayout = findViewById(R.id.drawer_layout)
        historyRv = findViewById(R.id.history_rv)
        newChatBtn = findViewById(R.id.new_chat_btn)
        openHistoryBtn = findViewById(R.id.open_history)
        openSettingsBtn = findViewById(R.id.open_settings)
        loadingPb = findViewById(R.id.loading_pb)

        // Arm AI Chat initialization
        lifecycleScope.launch(Dispatchers.Default) {
            val database = AppDatabase.getDatabase(applicationContext)
            repository = ConversationRepository(database)
            
            engine = AiChat.getInferenceEngine(applicationContext)
            
            // Observe conversation history
            launch(Dispatchers.Main) {
                setupHistoryUI()
            }
            
            // Check for last selected model
            val prefs = getSharedPreferences(SettingsActivity.PREFS_NAME, Context.MODE_PRIVATE)
            val lastPath = prefs.getString(PREF_LAST_MODEL_PATH, null)
            
            if (lastPath != null && File(lastPath).exists()) {
                try {
                    withContext(Dispatchers.Main) {
                        loadingPb.visibility = android.view.View.VISIBLE
                    }
                    loadModel("Last Model", File(lastPath))
                    withContext(Dispatchers.Main) {
                        onModelReady()
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to restore last model", e)
                }
            }
        }

        // Upon CTA button tapped
        userActionFab.setOnClickListener {
            if (isModelReady) {
                // If model is ready, validate input and send to engine
                handleUserInput()
            } else {
                // Otherwise, prompt user to go to settings
                Toast.makeText(this, "Please select a model in Settings first", Toast.LENGTH_SHORT).show()
                startActivity(Intent(this, SettingsActivity::class.java))
            }
        }

        openSettingsBtn.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }

        clearChatBtn.setOnClickListener {
            handleClearChat()
        }

        openHistoryBtn.setOnClickListener {
            drawerLayout.openDrawer(android.view.Gravity.START)
        }

        newChatBtn.setOnClickListener {
            startNewChat()
            drawerLayout.closeDrawers()
        }
    }

    override fun onResume() {
        super.onResume()
        val prefs = getSharedPreferences(SettingsActivity.PREFS_NAME, Context.MODE_PRIVATE)
        if (prefs.getBoolean(PREF_MODEL_CHANGED, false) && !isRefreshingModel) {
            isRefreshingModel = true
            prefs.edit().putBoolean(PREF_MODEL_CHANGED, false).apply()
            
            val lastPath = prefs.getString(PREF_LAST_MODEL_PATH, null)
            if (lastPath != null && File(lastPath).exists()) {
                loadingPb.visibility = android.view.View.VISIBLE
                lifecycleScope.launch {
                    try {
                        loadModel("New Model", File(lastPath))
                        onModelReady()
                        startNewChat()
                        Toast.makeText(this@MainActivity, "Model reloaded", Toast.LENGTH_SHORT).show()
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to refresh model", e)
                    } finally {
                        isRefreshingModel = false
                    }
                }
            } else {
                isRefreshingModel = false
            }
        }
    }

    private fun setupHistoryUI() {
        historyAdapter = HistoryAdapter(emptyList()) { conversation ->
            switchConversation(conversation)
            drawerLayout.closeDrawers()
        }
        historyRv.layoutManager = LinearLayoutManager(this)
        historyRv.adapter = historyAdapter

        lifecycleScope.launch {
            repository.getAllConversations().collect { conversations ->
                historyAdapter.updateData(conversations)
            }
        }
    }

    private fun startNewChat() {
        currentConversationId = null
        generationJob?.cancel()
        messageCollectionJob?.cancel()
        messages.clear()
        messageAdapter.notifyDataSetChanged()
        engine.resetChat()
        clearChatBtn.visibility = android.view.View.GONE
        userInputEt.text = null
        Toast.makeText(this, "Started new chat", Toast.LENGTH_SHORT).show()
    }

    private fun switchConversation(conversation: ConversationEntity) {
        currentConversationId = conversation.id
        generationJob?.cancel()
        messageCollectionJob?.cancel()
        
        messageCollectionJob = lifecycleScope.launch {
            // Restore model if needed (simplified: just log it for now as loading models is expensive)
            // Ideally we'd compare modelPath with current loaded model
            
            repository.getMessagesForConversation(conversation.id).collect { dbMessages ->
                withContext(Dispatchers.Main) {
                    messages.clear()
                    dbMessages.forEach {
                        messages.add(Message(it.id.toString(), it.content, it.isUser))
                    }
                    messageAdapter.notifyDataSetChanged()
                    messagesRv.scrollToPosition(messages.size - 1)
                    
                    if (messages.isNotEmpty()) {
                        clearChatBtn.visibility = android.view.View.VISIBLE
                    }
                }
            }
        }
    }

    private fun handleClearChat() {
        if (!isModelReady) return

        // Reset engine context
        engine.resetChat()

        // Hide clear button
        clearChatBtn.visibility = android.view.View.GONE

        // If we have a conversation, we should probably delete its messages in DB too
        // or just start a new session. For now, let's just clear the current UI session.
        currentConversationId?.let { id ->
            lifecycleScope.launch {
                // repository.deleteMessagesForConversation(id) // Optional: keep history but clear content?
                // Let's just start a new chat for simplicity
                startNewChat()
            }
        } ?: run {
            startNewChat()
        }

        Toast.makeText(this, "Chat session reset", Toast.LENGTH_SHORT).show()
    }

    private val getContent = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        Log.i(TAG, "Selected file uri:\n $uri")
        uri?.let { handleSelectedModel(uri) }
    }

    private fun handleSelectedModel(uri: Uri) {
        lifecycleScope.launch(Dispatchers.IO) {
            contentResolver.openInputStream(uri)?.use {
                GgufMetadataReader.create().readStructuredMetadata(it)
            }?.let { metadata ->
                val modelName = metadata.filename() + FILE_EXTENSION_GGUF
                contentResolver.openInputStream(uri)?.use { input ->
                    ensureModelFile(modelName, input)
                }?.let { modelFile ->
                    loadModel(modelName, modelFile)
                    withContext(Dispatchers.Main) {
                        getSharedPreferences(SettingsActivity.PREFS_NAME, Context.MODE_PRIVATE).edit()
                            .putString(PREF_LAST_MODEL_PATH, modelFile.absolutePath)
                            .putString(PREF_LAST_MODEL_METADATA, metadata.toString())
                            .apply()
                        onModelReady()
                    }
                }
            }
        }
    }

    private fun onModelReady() {
        isModelReady = true
        loadingPb.visibility = android.view.View.GONE
        userInputEt.hint = "Type and send a message!"
        userInputEt.isEnabled = true
        userActionFab.setImageResource(R.drawable.outline_send_24)
        userActionFab.isEnabled = true
        if (messages.isNotEmpty()) {
            clearChatBtn.visibility = android.view.View.VISIBLE
        }
    }

    private fun isUninterruptible(): Boolean {
        // Simple check to avoid picking model while generating
        return generationJob?.isActive == true
    }

    /**
     * Prepare the model file within app's private storage
     */
    private suspend fun ensureModelFile(modelName: String, input: InputStream) =
        withContext(Dispatchers.IO) {
            File(ensureModelsDirectory(), modelName).also { file ->
                // Copy the file into local storage if not yet done
                if (!file.exists()) {
                    Log.i(TAG, "Start copying file to $modelName")
                    withContext(Dispatchers.Main) {
                        userInputEt.hint = "Copying file..."
                    }

                    FileOutputStream(file).use { input.copyTo(it) }
                    Log.i(TAG, "Finished copying file to $modelName")
                } else {
                    Log.i(TAG, "File already exists $modelName")
                }
            }
        }

    /**
     * Load the model file from the app private storage
     */
    private suspend fun loadModel(modelName: String, modelFile: File) =
        withContext(Dispatchers.IO) {
            Log.i(TAG, "Loading model $modelName")
            withContext(Dispatchers.Main) {
                userInputEt.hint = "Loading model..."
            }
            engine.loadModel(modelFile.path)
        }

    /**
     * Validate and send the user message into [InferenceEngine]
     */
    private fun handleUserInput() {
        userInputEt.text.toString().also { userMsg ->
            if (userMsg.isEmpty()) {
                Toast.makeText(this, "Input message is empty!", Toast.LENGTH_SHORT).show()
            } else {
                userInputEt.text = null
                userInputEt.isEnabled = false
                userActionFab.isEnabled = false

                // Update message states
                val insertPos = messages.size
                messages.add(Message(UUID.randomUUID().toString(), userMsg, true))
                lastAssistantMsg.clear()
                messages.add(Message(UUID.randomUUID().toString(), lastAssistantMsg.toString(), false))
                
                messageAdapter.notifyItemRangeInserted(insertPos, 2)
                messagesRv.scrollToPosition(messages.size - 1)

                // Show clear button when first message is added
                clearChatBtn.visibility = android.view.View.VISIBLE

                lifecycleScope.launch {
                    val prefs = getSharedPreferences(SettingsActivity.PREFS_NAME, Context.MODE_PRIVATE)
                    val modelPath = prefs.getString(PREF_LAST_MODEL_PATH, "") ?: ""
                    
                    if (currentConversationId == null) {
                        currentConversationId = repository.createNewConversation(modelPath, userMsg.take(20))
                    }
                    repository.addMessage(currentConversationId!!, userMsg, true)
                }

                generationJob = lifecycleScope.launch(Dispatchers.Default) {
                    engine.sendUserPrompt(userMsg)
                        .onCompletion {
                            withContext(Dispatchers.Main) {
                                if (lastAssistantMsg.isEmpty()) {
                                    val messageCount = messages.size
                                    if (messageCount > 0 && !messages[messageCount - 1].isUser) {
                                        messages.removeAt(messageCount - 1).copy(
                                            content = "(No response from model)"
                                        ).let { messages.add(it) }
                                        messageAdapter.notifyItemChanged(messages.size - 1)
                                    }
                                }
                                userInputEt.isEnabled = true
                                userActionFab.isEnabled = true
                            }
                        }.collect { token ->
                            Log.d(TAG, "Token received: '$token'")
                            withContext(Dispatchers.Main) {
                                val messageCount = messages.size
                                if (messageCount > 0 && !messages[messageCount - 1].isUser) {
                                    messages.removeAt(messageCount - 1).copy(
                                        content = lastAssistantMsg.append(token).toString()
                                    ).let { messages.add(it) }

                                    messageAdapter.notifyItemChanged(messages.size - 1)
                                    messagesRv.scrollToPosition(messages.size - 1)
                                }
                            }
                        }
                    
                    // Persist assistant message once finished
                    currentConversationId?.let { id ->
                        repository.addMessage(id, lastAssistantMsg.toString(), false)
                    }
                }
            }
        }
    }

    /**
     * Run a benchmark with the model file
     */
    @Deprecated("This benchmark doesn't accurately indicate GUI performance expected by app developers")
    private suspend fun runBenchmark(modelName: String, modelFile: File) =
        withContext(Dispatchers.Default) {
            Log.i(TAG, "Starts benchmarking $modelName")
            withContext(Dispatchers.Main) {
                userInputEt.hint = "Running benchmark..."
            }
            engine.bench(
                pp=BENCH_PROMPT_PROCESSING_TOKENS,
                tg=BENCH_TOKEN_GENERATION_TOKENS,
                pl=BENCH_SEQUENCE,
                nr=BENCH_REPETITION
            ).let { result ->
                messages.add(Message(UUID.randomUUID().toString(), result, false))
                withContext(Dispatchers.Main) {
                    messageAdapter.notifyItemChanged(messages.size - 1)
                }
            }
        }

    /**
     * Create the `models` directory if not exist.
     */
    private fun ensureModelsDirectory() =
        File(filesDir, DIRECTORY_MODELS).also {
            if (it.exists() && !it.isDirectory) { it.delete() }
            if (!it.exists()) { it.mkdir() }
        }

    override fun onStop() {
        generationJob?.cancel()
        messageCollectionJob?.cancel()
        super.onStop()
    }

    override fun onDestroy() {
        engine.destroy()
        super.onDestroy()
    }

    companion object {
        private val TAG = MainActivity::class.java.simpleName

        private const val DIRECTORY_MODELS = "models"
        private const val FILE_EXTENSION_GGUF = ".gguf"

        private const val BENCH_PROMPT_PROCESSING_TOKENS = 512
        private const val BENCH_TOKEN_GENERATION_TOKENS = 128
        private const val BENCH_SEQUENCE = 1
        private const val BENCH_REPETITION = 3

        const val PREF_LAST_MODEL_PATH = "last_model_path"
        const val PREF_LAST_MODEL_METADATA = "last_model_metadata"
        const val PREF_MODEL_CHANGED = "model_changed_flag"
    }
}

fun GgufMetadata.filename() = when {
    basic.name != null -> {
        basic.name?.let { name ->
            basic.sizeLabel?.let { size ->
                "$name-$size"
            } ?: name
        }
    }
    architecture?.architecture != null -> {
        architecture?.architecture?.let { arch ->
            basic.uuid?.let { uuid ->
                "$arch-$uuid"
            } ?: "$arch-${System.currentTimeMillis()}"
        }
    }
    else -> {
        "model-${java.lang.Long.toHexString(System.currentTimeMillis())}"
    }
}
