"""
memory_engine/llm_client.py

Qwen LLM client for Mnemosyne.

Uses the OpenAI-compatible Alibaba Cloud Model Studio endpoint.
Injects retrieved memories into the system prompt so the agent
reasons over its memory context on every turn.
"""

from __future__ import annotations
import os
from typing import Optional
from openai import AsyncOpenAI

from memory_engine.models import MemoryType


# Memory type labels for prompt formatting
MEMORY_TYPE_LABELS = {
    MemoryType.PREFERENCE: "User Preference",
    MemoryType.FACT:       "Known Fact",
    MemoryType.EPISODE:    "Past Episode",
    MemoryType.RULE:       "Behavioural Rule",
}

BASE_SYSTEM_PROMPT = """You are Mnemosyne, an AI agent with persistent memory.
You remember details about the user across sessions and use that knowledge
to give accurate, personalised responses.

When recalling memories, be natural — do not announce that you are reading
from memory. Just use what you know.
"""


class QwenClient:
    """
    Async Qwen client using the OpenAI-compatible Alibaba Cloud endpoint.
    """

    def __init__(
        self,
        api_key:  Optional[str] = None,
        base_url: Optional[str] = None,
        model:    Optional[str] = None,
    ):
        self.api_key  = api_key  or os.getenv("QWEN_API_KEY")
        self.base_url = base_url or os.getenv("QWEN_BASE_URL",
                            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        self.model    = model    or os.getenv("QWEN_MODEL", "qwen-plus")

        if not self.api_key:
            raise ValueError("QWEN_API_KEY is not set. Add it to your .env file.")

        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    async def chat(
        self,
        user_message:     str,
        retrieved_memories: list[dict] = [],
        conversation_history: list[dict] = [],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """
        Send a message to Qwen with memory context injected.

        Args:
            user_message:          The user's current message
            retrieved_memories:    List of memory dicts from /turn retrieval
            conversation_history:  Prior turns: [{"role": "user"|"assistant", "content": "..."}]
            max_tokens:            Max tokens for response
            temperature:           Sampling temperature

        Returns:
            The assistant's reply as a string
        """
        system_prompt = self._build_system_prompt(retrieved_memories)

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_message})

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return response.choices[0].message.content

    async def chat_raw(
        self,
        messages:    list[dict],
        max_tokens:  int   = 1024,
        temperature: float = 0.7,
    ) -> str:
        """
        Send a raw messages list to Qwen — no memory injection.
        Useful for internal agent tasks (summarisation, classification, etc.)
        """
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content

    def _build_system_prompt(self, memories: list[dict]) -> str:
        """
        Build a system prompt that includes retrieved memories
        formatted by type.
        """
        if not memories:
            return BASE_SYSTEM_PROMPT

        # Group memories by type
        grouped: dict[str, list[str]] = {}
        for mem in memories:
            payload = mem.get("payload", mem)  # handle both raw and Qdrant-wrapped
            raw_type = payload.get("memory_type", "fact")

            try:
                mtype = MemoryType(raw_type)
                label = MEMORY_TYPE_LABELS[mtype]
            except (ValueError, KeyError):
                label = "Memory"

            content = payload.get("content", "")
            if content:
                grouped.setdefault(label, []).append(content)

        # Format memory block
        memory_lines = ["## What you remember about this user:\n"]
        for label, contents in grouped.items():
            memory_lines.append(f"**{label}:**")
            for c in contents:
                memory_lines.append(f"  - {c}")
            memory_lines.append("")

        memory_block = "\n".join(memory_lines)
        return f"{BASE_SYSTEM_PROMPT}\n{memory_block}"