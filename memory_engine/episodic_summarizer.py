"""
memory_engine/episodic_summarizer.py

Component for summarizing conversation episodes into concise memories.
Implements requirement #2 of Phase 9: Implement episodic summarization.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from datetime import datetime

from memory_engine.models import MemoryType
from memory_engine.llm_client import QwenClient


class EpisodicSummarizer:
    """
    Summarizes conversation episodes into concise memories for storage.
    """

    def __init__(self, llm_client: Optional[QwenClient] = None):
        """
        Args:
            llm_client: LLM client to use for summarization. If None, creates a new one.
        """
        self.llm_client = llm_client or QwenClient()

    async def summarize_conversation(
        self,
        conversation_history: List[Dict[str, str]],
        user_id: str,
        session_id: str,
        max_length: int = 200,
    ) -> Dict[str, Any]:
        """
        Summarize a conversation history into an episodic memory.

        Args:
            conversation_history: List of {"role": "user/assistant", "content": "..."} dicts
            user_id: User ID
            session_id: Session ID
            max_length: Maximum length of the summary in characters

        Returns:
            Dictionary containing the summary memory data suitable for creating a MemoryRecord
        """
        if not conversation_history:
            return {}

        # Format conversation for summarization
        formatted_conversation = self._format_conversation(conversation_history)

        # Create summarization prompt
        prompt = f"""Please provide a concise summary of the following conversation between a user and an AI assistant.
Focus on capturing the key information exchanged, any important facts learned about the user,
and any preferences or decisions made during the conversation.

Maximum length: {max_length} characters.

Conversation:
{formatted_conversation}

Summary:"""

        # Get summary from LLM
        summary_text = await self.llm_client.chat(
            user_message=prompt,
            retrieved_memories=[],  # No additional context needed for summarization
            conversation_history=[],  # No history for the summarization task itself
            max_tokens=150,  # Roughly corresponds to max_length characters
            temperature=0.3,  # Lower temperature for more focused summarization
        )

        # Truncate if necessary
        if len(summary_text) > max_length:
            summary_text = summary_text[:max_length].rsplit(' ', 1)[0] + '...'

        # Create memory data for the summary
        summary_memory = {
            "content": summary_text,
            "memory_type": MemoryType.EPISODE,
            "user_id": user_id,
            "session_id": session_id,
            "importance_score": 0.6,  # Episodic memories have moderate baseline importance
            "tags": ["summary", "conversation"],
            "source_turn": 0,  # Will be set by the caller
        }

        return summary_memory

    def _format_conversation(self, conversation_history: List[Dict[str, str]]) -> str:
        """Format conversation history for the summarization prompt."""
        formatted_lines = []
        for turn in conversation_history:
            role = turn.get("role", "unknown").upper()
            content = turn.get("content", "")
            formatted_lines.append(f"{role}: {content}")
        return "\n".join(formatted_lines)

    async def summarize_turn(
        self,
        user_query: str,
        assistant_reply: str,
        user_id: str,
        session_id: str,
        turn_number: int,
    ) -> Dict[str, Any]:
        """
        Summarize a single turn (user query + assistant reply) into an episodic memory.

        Args:
            user_query: The user's input
            assistant_reply: The assistant's response
            user_id: User ID
            session_id: Session ID
            turn_number: The turn number in the session

        Returns:
            Dictionary containing the summary memory data
        """
        conversation_history = [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": assistant_reply}
        ]

        return await self.summarize_conversation(
            conversation_history=conversation_history,
            user_id=user_id,
            session_id=session_id,
            max_length=150  # Shorter for single turn summaries
        )


# Factory function for easy instantiation
def create_episodic_summarizer(llm_client: Optional[QwenClient] = None) -> EpisodicSummarizer:
    """
    Create an EpisodicSummarizer instance.

    Args:
        llm_client: Optional LLM client to use

    Returns:
        EpisodicSummarizer instance
    """
    return EpisodicSummarizer(llm_client=llm_client)