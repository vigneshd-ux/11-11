from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import tiktoken


MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
SYSTEM_PROMPT = (
    "You are a concise assistant. Use facts the user gave earlier in this conversation. "
    "If a fact was not provided in this conversation, say you do not know."
)
CONVERSATION_FILE = Path(__file__).with_name("conversation_6_turns.json")
_ENCODING: Any | None = None
_ENCODING_TRIED = False
_TOKEN_COUNT_IS_ESTIMATE = True


def tokenizer_for(model: str):
    global _ENCODING, _ENCODING_TRIED, _TOKEN_COUNT_IS_ESTIMATE
    if _ENCODING_TRIED:
        return _ENCODING
    _ENCODING_TRIED = True
    try:
        _ENCODING = tiktoken.encoding_for_model(model)
    except Exception:
        # tiktoken downloads tokenizer data the first time; offline users may
        # not have that data cached, so the local demonstration stays usable.
        try:
            _ENCODING = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENCODING = None
    _TOKEN_COUNT_IS_ESTIMATE = _ENCODING is None
    return _ENCODING


class Conversation:
    """Stores and resends the complete role-tagged history on every turn."""

    def __init__(
        self,
        model: str = MODEL,
        messages: list[dict[str, str]] | None = None,
        token_counts: list[int] | None = None,
    ):
        self.model = model
        self.messages = messages or [{"role": "system", "content": SYSTEM_PROMPT}]
        self.token_counts = token_counts or []
        if not self.messages or self.messages[0].get("role") != "system":
            raise ValueError("Conversation history must start with a system message")

    def reset(self) -> None:
        """Keep the system message and clear all user/assistant messages."""
        self.messages[:] = [self.messages[0]]
        self.token_counts.clear()

    def token_count(self) -> int:
        """Estimate tokens in the full chat-formatted history.

        Counts each message's role and text with tiktoken plus the standard
        Chat Completions message framing estimate. Exact server accounting can
        vary by model and API formatting.
        """
        enc = tokenizer_for(self.model)
        total = 3  # Assistant reply priming.
        for message in self.messages:
            total += 3  # Per-message framing estimate.
            role = message.get("role", "")
            content = message.get("content", "")
            if enc is None:
                total += len(re.findall(r"\w+|[^\w\s]", role + " " + content, flags=re.UNICODE))
            else:
                total += len(enc.encode(role)) + len(enc.encode(content))
        return total

    async def ask(self, text: str, client: Any | None = None, *, offline: bool = False) -> str:
        self.messages.append({"role": "user", "content": text})

        if offline:
            answer = offline_answer(text, self.messages)
        else:
            # Pass a copy of every message accumulated so far on every request.
            response = await client.chat.completions.create(
                model=self.model,
                messages=[dict(message) for message in self.messages],
            )
            answer = response.choices[0].message.content or ""

        self.messages.append({"role": "assistant", "content": answer})
        self.token_counts.append(self.token_count())
        return answer

    def save(self, path: Path = CONVERSATION_FILE) -> None:
        payload = {
            "model": self.model,
            "messages": self.messages,
            "token_counts_after_each_turn": self.token_counts,
            "token_count_method": "tiktoken" if not _TOKEN_COUNT_IS_ESTIMATE else "local regex estimate",
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path = CONVERSATION_FILE) -> "Conversation":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            model=payload["model"],
            messages=payload["messages"],
            token_counts=payload.get("token_counts_after_each_turn", []),
        )


def offline_answer(text: str, history: list[dict[str, str]]) -> str:
    """Predictable stand-in responses for demonstrating the message flow."""
    stripped_history = [m for m in history if m["role"] in ("user", "assistant")]
    if len(stripped_history) == 1 and text == "My name is Arun.":
        return "Nice to meet you, Arun."
    if len(stripped_history) == 1 and text == "What is my name?":
        return "I don't know your name because this request has no earlier conversation history."
    answers = {
        "My name is Arun.": "Nice to meet you, Arun.",
        "What is my name?": "Your name is Arun.",
        "I live in Chennai.": "Got it. You live in Chennai.",
        "Which city do I live in?": "You live in Chennai.",
        "My favorite color is blue.": "I'll remember that your favorite color is blue.",
        "What is my favorite color?": "Your favorite color is blue.",
    }
    return answers.get(text, "I don't know that yet.")


async def make_client(live: bool):
    if not live:
        return None
    from openai import AsyncOpenAI

    return AsyncOpenAI()


async def stateless_demonstration(client: Any | None, *, offline: bool, model: str) -> None:
    print("=== Two separate calls, with no prior history ===")
    first_prompt = "My name is Arun."
    second_prompt = "What is my name?"
    if offline:
        first = offline_answer(first_prompt, [{"role": "user", "content": first_prompt}])
        second = offline_answer(second_prompt, [{"role": "user", "content": second_prompt}])
    else:
        # These are two fresh API requests. Each contains only the current user message.
        first_response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": first_prompt}],
        )
        second_response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": second_prompt}],
        )
        first = first_response.choices[0].message.content or ""
        second = second_response.choices[0].message.content or ""
    print(f"Call 1 messages: [{{'role': 'user', 'content': {first_prompt!r}}}]")
    print(f"Assistant: {first}")
    print(f"Call 2 messages: [{{'role': 'user', 'content': {second_prompt!r}}}]")
    print(f"Assistant: {second}")


async def six_turn_demonstration(client: Any | None, *, offline: bool, model: str) -> Conversation:
    print("\n=== Six turns with full conversation history ===")
    conversation = Conversation(model=model)
    turns = [
        "My name is Arun.",
        "What is my name?",
        "I live in Chennai.",
        "Which city do I live in?",
        "My favorite color is blue.",
        "What is my favorite color?",
    ]
    for turn_number, user_text in enumerate(turns, start=1):
        answer = await conversation.ask(user_text, client, offline=offline)
        print(f"Turn {turn_number} user: {user_text}")
        print(f"Turn {turn_number} assistant: {answer}")
        count = conversation.token_counts[-1]
        method = "estimated" if _TOKEN_COUNT_IS_ESTIMATE else "tiktoken"
        print(f"History tokens after turn {turn_number} ({method}): {count}")

    conversation.save()
    loaded = Conversation.load()
    print(f"Saved and loaded: {CONVERSATION_FILE.name} ({len(loaded.messages)} messages)")
    return conversation


async def interactive_chat(client: Any | None, *, offline: bool, model: str) -> None:
    conversation = Conversation(model=model)
    print("Interactive chat. Enter /reset to clear history, or /quit to exit.")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text == "/quit":
            break
        if text == "/reset":
            conversation.reset()
            print("History cleared; the system prompt remains.")
            print(f"History tokens: {conversation.token_count()}")
            continue
        if not text:
            continue
        answer = await conversation.ask(text, client, offline=offline)
        print(f"Assistant: {answer}")
        print(f"History tokens: {conversation.token_count()}")
        conversation.save()
    conversation.save()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Use the OpenAI API")
    parser.add_argument("--interactive", action="store_true", help="Start a chat with /reset support")
    parser.add_argument("--model", default=MODEL, help="Model for --live mode")
    args = parser.parse_args()

    client = await make_client(args.live)
    if args.interactive:
        await interactive_chat(client, offline=not args.live, model=args.model)
        return

    await stateless_demonstration(client, offline=not args.live, model=args.model)
    await six_turn_demonstration(client, offline=not args.live, model=args.model)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
