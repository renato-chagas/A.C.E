"""LLM-backed extraction of durable facts from a conversation turn.

The extractor takes a single (user, assistant) exchange and asks a small
local model to distill any long-term, user-specific facts worth remembering.
Extraction defaults to best effort for callers on the request path. A durable
worker can request strict extraction and handle inference or response errors
separately from successful exchanges with nothing worth retaining.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Optional

from openjarvis.core.types import Message, Role

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "You extract durable, long-term facts about the user from a single "
    "conversation exchange. A good fact is stable over time and useful in "
    "future conversations: preferences, identity, goals, ongoing projects, "
    "constraints, relationships, and important project decisions or outcomes "
    "that are likely to matter again. Keep decisions and outcomes concise and "
    "preserve their current status when the user states it. Ignore one-off task "
    "details, small talk, and anything the assistant said about itself. "
    "Never extract passwords, "
    "API keys, access tokens, recovery codes, or full financial account "
    "identifiers. Do not infer sensitive facts that the user did not state.\n\n"
    "Respond with ONLY a JSON array of short fact strings (each under 200 "
    "characters). If there is nothing worth remembering, respond with []."
)


class FactExtractor:
    """Extract memory-worthy facts from a conversation turn via an engine."""

    def __init__(
        self,
        engine: Any,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_facts_per_turn: int = 10,
        max_fact_chars: int = 200,
        system_prompt: Optional[str] = None,
    ) -> None:
        self._engine = engine
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_facts_per_turn = max_facts_per_turn
        self._max_fact_chars = max_fact_chars
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT

    def extract(
        self, user_text: str, assistant_text: str = "", *, strict: bool = False
    ) -> List[str]:
        """Return durable facts; strict mode raises on inference or response errors."""
        user_text = (user_text or "").strip()
        if not user_text:
            return []

        exchange = f"User: {user_text}"
        if assistant_text and assistant_text.strip():
            exchange += f"\nAssistant: {assistant_text.strip()}"

        messages = [
            Message(role=Role.SYSTEM, content=self._system_prompt),
            Message(role=Role.USER, content=exchange),
        ]

        try:
            result = self._engine.generate(
                messages,
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except Exception:
            if strict:
                raise RuntimeError("Memory fact inference failed.") from None
            logger.debug("Memory extraction failed")
            return []

        if isinstance(result, dict):
            content = result.get("content", "") or ""
        else:
            content = str(result)

        if strict and not isinstance(content, str):
            content = ""
        return self._parse(content, strict=strict)

    # -- parsing ------------------------------------------------------------

    def _parse(self, content: str, *, strict: bool = False) -> List[str]:
        """Parse model output into a clean, deduped, capped list of facts."""
        if not strict and (not content or not content.strip()):
            return []

        raw = self._coerce_to_list(content, strict=strict)

        facts: List[str] = []
        seen: set[str] = set()
        for item in raw:
            fact = self._clean_fact(item)
            if not fact:
                continue
            key = fact.lower()
            if key in seen:
                continue
            seen.add(key)
            facts.append(fact)
            if len(facts) >= self._max_facts_per_turn:
                break
        return facts

    def _coerce_to_list(self, content: str, *, strict: bool = False) -> List[str]:
        """Best-effort conversion of model output to a list of strings."""
        # In strict mode accept one JSON value, optionally in a Markdown fence.
        # Other callers can also recover arrays from surrounding prose.
        if strict:
            candidate = re.sub(
                r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.I
            )
        else:
            match = re.search(r"\[.*\]", content, re.DOTALL)
            candidate = match.group(0) if match else ""
        if candidate:
            try:
                parsed = json.loads(candidate)
            except (ValueError, RecursionError):
                parsed = None
            if isinstance(parsed, list) and (
                not strict or all(isinstance(item, str) for item in parsed)
            ):
                return [str(item) for item in parsed]

        if strict:
            raise ValueError("Memory fact extraction returned an invalid response.")

        # 2. Fall back to line-based parsing (markdown bullets / numbered).
        #    Deliberately permissive: small local models routinely answer with
        #    one bare fact per line, and rejecting those would quietly stop
        #    memory capture. Injected text can trivially add list markers, so
        #    filtering here buys no defence — provenance tagging and the
        #    injection scanner in MemoryService are the real gate.
        items: List[str] = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line)
            items.append(line)
        return items

    def _clean_fact(self, item: str) -> str:
        fact = str(item).strip().strip("\"'").strip()
        # Drop obvious non-facts the model sometimes emits.
        if not fact or fact.lower() in ("[]", "none", "n/a", "null"):
            return ""
        if len(fact) > self._max_fact_chars:
            fact = fact[: self._max_fact_chars].rstrip()
        return fact


__all__ = ["FactExtractor"]
