package com.example.llama

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.example.llama.data.ConversationEntity
import java.text.SimpleDateFormat
import java.util.*

class HistoryAdapter(
    private var conversations: List<ConversationEntity>,
    private val onItemClick: (ConversationEntity) -> Unit
) : RecyclerView.Adapter<HistoryAdapter.ViewHolder>() {

    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val titleTv: TextView = view.findViewById(R.id.conv_title)
        val dateTv: TextView = view.findViewById(R.id.conv_date)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_history, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val conv = conversations[position]
        holder.titleTv.text = conv.title
        val sdf = SimpleDateFormat("MMM dd, yyyy HH:mm", Locale.getDefault())
        holder.dateTv.text = sdf.format(Date(conv.lastUpdatedAt))
        holder.itemView.setOnClickListener { onItemClick(conv) }
    }

    override fun getItemCount(): Int = conversations.size

    fun updateData(newConversations: List<ConversationEntity>) {
        conversations = newConversations
        notifyDataSetChanged()
    }
}
