"""
memory_engine/memory_extractor.py

Autonomous memory extraction — the missing piece identified in your app.py
audit: the frontend always sends "memories": [], and episodic_summarizer
only runs on req.history (which starts empty every session), so nothing
from actual conversation content was ever being captured.

This module reads the user_message + assistant_reply for a single turn
and asks Qwen to pull out anything worth remembering long-term, typed as
PREFERENCE / FACT / RULE / EPISODE / PLANNING. TOOL_USAGE exists in
MemoryType but is intentionally left out of extraction here since the
current Qwen setup has no tools — add it back to the prompt and
type_map below if that changes later. This is meant to run
unconditionally inside POST /turn, after the reply is generated,
regardless of what the caller passed in `memories` or `history`.
"""

from __future__ import annotations
import json
import re
from typing import List, Dict, Any, Optional

from memory_engine.models import MemoryType
from memory_engine.llm_client import QwenClient


class AutonomousMemoryExtractor:
    """
    Extracts typed memories directly from a conversation turn using the LLM,
    rather than relying on the caller to supply an explicit memories array.
    """

    def __init__(
        self,
        llm_client: Optional[QwenClient] = None,
        min_importance: float = 0.3,
        max_memories_per_turn: int = 5,
    ):
        self.llm_client = llm_client or QwenClient()
        self.min_importance = min_importance
        self.max_memories_per_turn = max_memories_per_turn

    async def extract_memories(
        self,
        user_message: str,
        assistant_reply: str,
        user_id: str,
        session_id: str,
        turn_number: int,
    ) -> List[Dict[str, Any]]:
        """
        Returns a list of memory dicts ready to hand to MemoryWriter.write(),
        i.e. {content, memory_type, importance_score, tags, source_turn}.
        Returns [] if nothing in the exchange is worth storing, or if
        extraction fails for any reason (never raises — a broken extraction
        should never break a turn).
        """
        if not user_message or not user_message.strip():
            return []

        prompt = self._build_prompt(user_message, assistant_reply)

        try:
            raw = await self.llm_client.chat(
                user_message=prompt,
                retrieved_memories=[],
                conversation_history=[],
                max_tokens=500,
                temperature=0.2,
            )
        except Exception as e:
            print(f"AutonomousMemoryExtractor: LLM call failed: {e}")
            return []

        items = self._parse_response(raw)

        memories: List[Dict[str, Any]] = []
        for item in items:
            mem = self._to_memory_dict(item, turn_number)
            if mem is not None:
                memories.append(mem)
            if len(memories) >= self.max_memories_per_turn:
                break

        return memories

    # ------------------------------------------------------------------ #

    def _build_prompt(self, user_message: str, assistant_reply: str) -> str:
        return f"""You extract long-term memories from a single conversation turn.

Categories:
- PREFERENCE: how the user likes things done (style, tone, format, workflow habits)
- FACT: a stable fact about the user, their projects, or their situation
- RULE: an explicit instruction or constraint the user gave that should be followed in future turns
- EPISODE: a notable event or decision from this exchange worth recalling later
- PLANNING: a stated intention, plan, or next step the user or agent committed to

Only extract things that would still be useful to know in a future, unrelated conversation.
Skip small talk, transient debugging details, and anything already obvious or generic.
If nothing qualifies, return an empty array.

Respond with ONLY a JSON array, no prose, no markdown fences. Each element:
{{"type": "PREFERENCE"|"FACT"|"RULE"|"EPISODE"|"PLANNING", "content": "<short, self-contained statement>", "importance": <0.0-1.0>, "tags": ["..."]}}

USER: {user_message}
ASSISTANT: {assistant_reply}

JSON:"""

    def _parse_response(self, raw: str) -> List[Dict[str, Any]]:
        if not raw:
            return []

        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

        # Model sometimes adds a stray sentence before/after the array — grab
        # the first [...] block rather than failing the whole parse.
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            print(f"AutonomousMemoryExtractor: could not parse JSON: {cleaned[:200]!r}")
            return []

        if not isinstance(data, list):
            return []
        return [x for x in data if isinstance(x, dict)]

    def _to_memory_dict(self, item: Dict[str, Any], turn_number: int) -> Optional[Dict[str, Any]]:
        try:
            type_str = str(item.get("type", "")).strip().lower()
            content = str(item.get("content", "")).strip()
            importance = float(item.get("importance", 0.5))
            tags = item.get("tags", [])
            if not isinstance(tags, list):
                tags = []

            if not content:
                return None
            if importance < self.min_importance:
                return None

            type_map = {
                "preference": MemoryType.PREFERENCE,
                "fact": MemoryType.FACT,
                "rule": MemoryType.RULE,
                "episode": MemoryType.EPISODE,
                "planning": MemoryType.PLANNING,
            }
            memory_type = type_map.get(type_str, MemoryType.EPISODE)

            return {
                "content": content,
                "memory_type": memory_type,
                "importance_score": max(0.0, min(1.0, importance)),
                "tags": [str(t) for t in tags] + ["auto_extracted"],
                "source_turn": turn_number,
            }
        except Exception as e:
            print(f"AutonomousMemoryExtractor: skipping malformed item {item!r}: {e}")
            return None


def create_memory_extractor(llm_client: Optional[QwenClient] = None) -> AutonomousMemoryExtractor:
    return AutonomousMemoryExtractor(llm_client=llm_client)