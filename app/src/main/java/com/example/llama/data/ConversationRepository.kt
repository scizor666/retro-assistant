package com.example.llama.data

import kotlinx.coroutines.flow.Flow
import android.util.Log
import java.util.UUID

private const val TAG = "ConversationRepository"

class ConversationRepository(private val db: AppDatabase) {
    private val conversationDao = db.conversationDao()
    private val messageDao = db.messageDao()

    fun getAllConversations(): Flow<List<ConversationEntity>> = conversationDao.getAllConversations()

    fun getMessagesForConversation(conversationId: Long): Flow<List<MessageEntity>> =
        messageDao.getMessagesForConversation(conversationId)

    suspend fun createNewConversation(modelPath: String, title: String = "New Chat"): Long {
        Log.d(TAG, "Creating new conversation: Title='$title', Model='$modelPath'")
        val conversation = ConversationEntity(title = title, modelPath = modelPath)
        val id = conversationDao.insertConversation(conversation)
        Log.d(TAG, "Conversation created with ID: $id")
        conversationDao.deleteOldConversations()
        return id
    }

    suspend fun addMessage(conversationId: Long, content: String, isUser: Boolean) {
        Log.d(TAG, "Adding message to conv $conversationId: IsUser=$isUser, Length=${content.length}")
        val message = MessageEntity(conversationId = conversationId, content = content, isUser = isUser)
        messageDao.insertMessage(message)
        
        // Update lastUpdatedAt for the conversation
        conversationDao.getConversationById(conversationId)?.let {
            Log.d(TAG, "Updating lastUpdatedAt for conv $conversationId")
            conversationDao.updateConversation(it.copy(lastUpdatedAt = System.currentTimeMillis()))
        }
    }

    suspend fun clearAllHistory() {
        conversationDao.deleteAllConversations()
        messageDao.deleteAllMessages()
    }
}
