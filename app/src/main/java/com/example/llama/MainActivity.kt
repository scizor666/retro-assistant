package com.example.llama

import android.net.Uri
import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.widget.EditText
import android.widget.ImageButton
import android.widget.TextView
import android.text.Editable
import android.text.TextWatcher
import android.view.View
import android.view.inputmethod.InputMethodManager
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
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.onCompletion
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.ExperimentalCoroutinesApi

import kotlinx.coroutines.flow.onStart

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
import com.example.llama.rag.RagManager

class MainActivity : AppCompatActivity() {

    // Android views
    private lateinit var messagesRv: RecyclerView
    private lateinit var userInputEt: EditText
    private lateinit var userActionFab: FloatingActionButton
    private lateinit var clearChatBtn: ImageButton
    private lateinit var openSettingsBtn: ImageButton
    private lateinit var pickImageBtn: ImageButton
    private lateinit var loadingPb: android.widget.ProgressBar

    // Arm AI Chat inference engine
    private lateinit var engine: InferenceEngine
    private var generationJob: Job? = null
    private var selectedImageBytes: ByteArray? = null

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
    private lateinit var historySearchEt: EditText
    private lateinit var openHistoryBtn: ImageButton
    private var messageCollectionJob: Job? = null
    private var isRefreshingModel = false
    private val searchQuery = MutableStateFlow("")

    // RAG: lazily initialised so it doesn't block onCreate
    private val ragManager by lazy { RagManager(applicationContext) }

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
        pickImageBtn = findViewById(R.id.pick_image_btn)
        clearChatBtn = findViewById(R.id.clear_chat)
        drawerLayout = findViewById(R.id.drawer_layout)
        historyRv = findViewById(R.id.history_rv)
        newChatBtn = findViewById(R.id.new_chat_btn)
        historySearchEt = findViewById(R.id.history_search)
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

        pickImageBtn.setOnClickListener {
            pickImageActivity.launch("image/*")
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

        drawerLayout.addDrawerListener(object : androidx.drawerlayout.widget.DrawerLayout.SimpleDrawerListener() {
            override fun onDrawerClosed(drawerView: View) {
                super.onDrawerClosed(drawerView)
                historySearchEt.clearFocus()
                val imm = getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager
                imm.hideSoftInputFromWindow(drawerView.windowToken, 0)
                
                if (userInputEt.isEnabled) {
                    userInputEt.requestFocus()
                }
            }
        })
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

        historySearchEt.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
                searchQuery.value = s?.toString() ?: ""
            }
            override fun afterTextChanged(s: Editable?) {}
        })

        @OptIn(ExperimentalCoroutinesApi::class)
        lifecycleScope.launch {
            searchQuery.flatMapLatest { query ->
                if (query.isEmpty()) {
                    repository.getAllConversations()
                } else {
                    repository.searchConversations(query)
                }
            }.collect { conversations ->
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
        selectedImageBytes = null
        Toast.makeText(this, "Started new chat", Toast.LENGTH_SHORT).show()
    }

    private fun switchConversation(conversation: ConversationEntity) {
        Log.i(TAG, "Switching to conversation ID: ${conversation.id}, Title: ${conversation.title}")
        currentConversationId = conversation.id
        generationJob?.cancel()
        messageCollectionJob?.cancel()
        
        // Reset engine context before switching
        Log.d(TAG, "Calling engine.resetChat()")
        engine.resetChat()
        
        messageCollectionJob = lifecycleScope.launch {
            Log.d(TAG, "Starting message collection for ${conversation.id}")
            val dbMessages = repository.getMessagesForConversation(conversation.id).first()
            Log.d(TAG, "Received ${dbMessages.size} messages from DB for ${conversation.id}")
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
            
            // The engine context is reset via engine.resetChat() earlier.
            // Rebuilding full context by re-generating every message is too expensive.
            // Ideally the SDK has a method to inject history without generating.
            // For now, it will just act as a renewed session.
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
        uri?.let { handleSelectedModel(it) }
    }

    private val pickImageActivity = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let {
            contentResolver.openInputStream(uri)?.use { inputStream ->
                selectedImageBytes = inputStream.readBytes()
                Toast.makeText(this, "Image attached (${selectedImageBytes?.size} bytes). Ready to send.", Toast.LENGTH_SHORT).show()
                // Change UI to reflect image attachment (e.g. change hint or show an icon)
                userInputEt.hint = "Image attached! Ask a question about it..."
            }
        }
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
                        Log.i(TAG, "Starting NEW conversation")
                        currentConversationId = repository.createNewConversation(modelPath, userMsg.take(20))
                    } else {
                        Log.i(TAG, "Continuing conversation ID: $currentConversationId")
                    }
                    repository.addMessage(currentConversationId!!, userMsg, true)
                }

                Log.i(TAG, "Launching generation job for prompt: '${userMsg.take(50)}...'")
                generationJob = lifecycleScope.launch(Dispatchers.Default) {
                    // RAG: retrieve context on IO, prepend to prompt if found.
                    // UI always shows userMsg; only the engine receives the augmented prompt.
                    Log.i(TAG, "RAG: Requesting context for user input...")
                    val ragContext = ragManager.getContext(userMsg)
                    if (ragContext != null) {
                        Log.i(TAG, "RAG: SUCCESS. Injected context of length ${ragContext.length}")
                    } else {
                        Log.i(TAG, "RAG: SKIPPED. No relevant context found.")
                    }
                    val promptToSend = if (ragContext != null) {
                        "Context:\n$ragContext\n\nQuestion:\n$userMsg"
                    } else {
                        userMsg
                    }

                    val imageBytes = selectedImageBytes
                    val flowResult = if (imageBytes != null) {
                        Log.i(TAG, "Pipeline: sendImagePrompt")
                        engine.sendImagePrompt(imageBytes, promptToSend)
                    } else {
                        Log.i(TAG, "Pipeline: sendUserPrompt")
                        engine.sendUserPrompt(promptToSend)
                    }

                    // clear attachment so we don't accidentally send it twice
                    withContext(Dispatchers.Main) {
                        selectedImageBytes = null
                        userInputEt.hint = "Type and send a message!"
                    }

                    flowResult
                        .onStart { Log.d(TAG, "Generation started") }
                        .onCompletion { error: Throwable? ->
                            Log.d(TAG, "Generation completed. Total assistant tokens: ${lastAssistantMsg.length}, Error: $error")
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
                                
                                // Persist assistant message once finished
                                currentConversationId?.let { id ->
                                    val finalMsg = if (lastAssistantMsg.isEmpty()) "(No response from model)" else lastAssistantMsg.toString()
                                    // Use a separate coroutine or scope if needed, but we are inside launch(Dispatchers.Default) usually, 
                                    // however we are inside withContext(Dispatchers.Main) here.
                                    lifecycleScope.launch {
                                        repository.addMessage(id, finalMsg, false)
                                    }
                                }
                            }
                        }.collect { token: String ->
                            Log.i(TAG, "Token received: '$token'")
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
