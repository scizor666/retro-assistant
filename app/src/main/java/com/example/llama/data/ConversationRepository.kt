package com.example.llama.data

import kotlinx.coroutines.flow.Flow
import java.util.UUID

class ConversationRepository(private val db: AppDatabase) {
    private val conversationDao = db.conversationDao()
    private val messageDao = db.messageDao()

    fun getAllConversations(): Flow<List<ConversationEntity>> = conversationDao.getAllConversations()

    fun getMessagesForConversation(conversationId: Long): Flow<List<MessageEntity>> =
        messageDao.getMessagesForConversation(conversationId)

    suspend fun createNewConversation(modelPath: String, title: String = "New Chat"): Long {
        val conversation = ConversationEntity(title = title, modelPath = modelPath)
        val id = conversationDao.insertConversation(conversation)
        conversationDao.deleteOldConversations()
        return id
    }

    suspend fun addMessage(conversationId: Long, content: String, isUser: Boolean) {
        val message = MessageEntity(conversationId = conversationId, content = content, isUser = isUser)
        messageDao.insertMessage(message)
        
        // Update lastUpdatedAt for the conversation
        conversationDao.getConversationById(conversationId)?.let {
            conversationDao.updateConversation(it.copy(lastUpdatedAt = System.currentTimeMillis()))
        }
    }

    suspend fun clearAllHistory() {
        conversationDao.deleteAllConversations()
        messageDao.deleteAllMessages()
    }
}
