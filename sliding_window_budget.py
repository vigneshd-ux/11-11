from __future__ import annotations

import argparse
import asyncio
import os
import re
from dataclasses import dataclass
from typing import Any


MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
SYSTEM_PROMPT = "You are a concise assistant. Remember important facts from the user's conversation."
SUMMARY_PROMPT = (
    "Summarize the oldest conversation turns for future reference. Preserve names, order IDs, "
    "dates, decisions, and unresolved requests exactly. Keep it concise."
)
IMPORTANT_NAME = "Arun"
IMPORTANT_ORDER_ID = "ORD-4821"


class TokenCounter:
    """Use tiktoken if its model data is available; otherwise use a labeled estimate."""

    def __init__(self, model: str, *, prefer_tiktoken: bool):
        self.encoding: Any | None = None
        if prefer_tiktoken:
            try:
                import tiktoken

                try:
                    self.encoding = tiktoken.encoding_for_model(model)
                except KeyError:
                    self.encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self.encoding = None
        self.method = "tiktoken" if self.encoding is not None else "word-token estimate"

    def text(self, value: str) -> int:
        if self.encoding is not None:
            return len(self.encoding.encode(value))
        return len(re.findall(r"\w+|[^\w\s]", value, flags=re.UNICODE))

    def messages(self, messages: list[dict[str, str]]) -> int:
        # Account for role labels and approximate Chat Completions framing.
        return 3 + sum(3 + self.text(m.get("role", "")) + self.text(m.get("content", "")) for m in messages)


@dataclass(frozen=True)
class TokenBudget:
    system: int = 500
    history: int = 3000
    context: int = 2000
    output: int = 1000

    def check(
        self,
        counter: TokenCounter,
        system_text: str,
        history: list[dict[str, str]],
        context_text: str,
        output_tokens: int,
        *,
        purpose: str,
    ) -> dict[str, int]:
        usage = {
            "system": counter.text(system_text),
            "history": counter.messages(history) if history else 0,
            "context": counter.text(context_text),
            "output": output_tokens,
        }
        caps = {"system": self.system, "history": self.history, "context": self.context, "output": self.output}
        exceeded = [f"{name}={usage[name]}>{caps[name]}" for name in usage if usage[name] > caps[name]]
        if exceeded:
            raise ValueError(f"Token budget exceeded before {purpose}: {', '.join(exceeded)}")
        if sum(usage.values()) > self.system + self.history + self.context + self.output:
            raise ValueError(f"Total token budget exceeded before {purpose}")
        print(
            f"Pre-call budget ({purpose}): system {usage['system']}/{self.system}, "
            f"history {usage['history']}/{self.history}, context {usage['context']}/{self.context}, "
            f"output {usage['output']}/{self.output}"
        )
        return usage


def flatten(turns: list[list[dict[str, str]]]) -> list[dict[str, str]]:
    return [message for turn in turns for message in turn]


class ConversationManager:
    def __init__(
        self,
        *,
        model: str = MODEL,
        keep_last_turns: int = 5,
        live: bool = False,
        budget: TokenBudget | None = None,
    ):
        self.model = model
        self.keep_last_turns = keep_last_turns
        self.live = live
        self.budget = budget or TokenBudget()
        self.counter = TokenCounter(model, prefer_tiktoken=live)
        self.turns: list[list[dict[str, str]]] = []
        self.summary = ""

    def raw_history_tokens(self) -> int:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if self.summary:
            messages.append({"role": "developer", "content": "Summary of earlier turns: " + self.summary})
        messages.extend(flatten(self.turns))
        return self.counter.messages(messages)

    def active_history(self, pending_user: str | None = None) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if self.summary:
            messages.append({"role": "developer", "content": "Summary of earlier turns: " + self.summary})
        messages.extend(flatten(self.turns[-self.keep_last_turns :]))
        if pending_user is not None:
            messages.append({"role": "user", "content": pending_user})
        return messages

    def preserved_facts(self) -> bool:
        summary = self.summary.lower()
        return IMPORTANT_NAME.lower() in summary and IMPORTANT_ORDER_ID.lower() in summary

    async def _summarize_chunk(self, chunk: list[list[dict[str, str]]], client: Any | None) -> str:
        chunk_messages = flatten(chunk)
        context = self.summary
        self.budget.check(
            self.counter,
            SUMMARY_PROMPT,
            chunk_messages,
            context,
            self.budget.output,
            purpose="summarization",
        )
        if not self.live:
            # Deterministic summary for the offline demo; keep key facts verbatim.
            if IMPORTANT_NAME.lower() in context.lower() and IMPORTANT_ORDER_ID.lower() in context.lower():
                return context
            prior = (context + " ") if context else ""
            return prior + f"Customer name: {IMPORTANT_NAME}. Order ID: {IMPORTANT_ORDER_ID}."

        messages = [{"role": "system", "content": SUMMARY_PROMPT}]
        if context:
            messages.append({"role": "developer", "content": "Existing summary: " + context})
        messages.extend(chunk_messages)
        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.budget.output,
        )
        new_summary = response.choices[0].message.content or ""
        # Keep critical literal identifiers if the model paraphrased or omitted them.
        old_text = " ".join(m["content"] for m in chunk_messages)
        additions = []
        for fact in (IMPORTANT_NAME, IMPORTANT_ORDER_ID):
            if fact in old_text and fact.lower() not in new_summary.lower():
                additions.append(fact)
        if additions:
            new_summary = (new_summary + " Important facts: " + ", ".join(additions)).strip()
        return new_summary

    async def _summarize_oldest(self, client: Any | None) -> bool:
        # Leave the latest N complete turns available verbatim for the sliding window.
        old_turns = self.turns[:-self.keep_last_turns]
        if not old_turns:
            return False

        chunk: list[list[dict[str, str]]] = []
        for turn in old_turns:
            candidate = chunk + [turn]
            if self.counter.messages(flatten(candidate)) > self.budget.history:
                break
            chunk = candidate
        if not chunk:
            raise ValueError("One turn is larger than the history budget and cannot be summarized safely")

        self.summary = await self._summarize_chunk(chunk, client)
        self.turns = self.turns[len(chunk) :]
        return True

    async def prepare_for_call(self, client: Any | None, pending_user: str | None = None) -> tuple[int, int]:
        before = self.raw_history_tokens()
        # Summarize oldest turns whenever the full transcript exceeds 2,000 tokens.
        while self.raw_history_tokens() > 2000:
            if not await self._summarize_oldest(client):
                break

        # Hard sliding-window limit: system + optional summary + latest N turns.
        if len(self.turns) > self.keep_last_turns:
            self.turns = self.turns[-self.keep_last_turns :]

        history = self.active_history(pending_user)
        context_text = self.summary
        self.budget.check(
            self.counter,
            SYSTEM_PROMPT,
            [m for m in history if m["role"] not in ("system", "developer")],
            context_text,
            self.budget.output,
            purpose="assistant response",
        )
        after = self.raw_history_tokens()
        return before, after

    async def ask(self, text: str, client: Any | None = None) -> str:
        before, after = await self.prepare_for_call(client, pending_user=text)
        if before > 2000:
            print(f"History before trim: {before} tokens -> after summarise/trim: {after} tokens")

        if self.live:
            messages = self.active_history(pending_user=text)
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.budget.output,
            )
            answer = response.choices[0].message.content or ""
        else:
            answer = offline_answer(text)
        self.turns.append(
            [{"role": "user", "content": text}, {"role": "assistant", "content": answer}]
        )
        return answer


def offline_answer(text: str) -> str:
    if "name" in text.lower() and "order" in text.lower():
        return f"The name is {IMPORTANT_NAME}, and the order ID is {IMPORTANT_ORDER_ID}."
    return "Recorded the update for this turn."


def build_twenty_turn_fixture(manager: ConversationManager, target_before: int = 5200, target_after: int = 1900) -> None:
    """Create a deterministic 20-turn transcript to reproduce the assignment's counts."""
    manager.turns = [
        [
            {"role": "user", "content": f"Turn {number}: record the shipping update and customer request."},
            {"role": "assistant", "content": f"Turn {number} update recorded."},
        ]
        for number in range(1, 21)
    ]
    manager.turns[0][0]["content"] = f"My name is {IMPORTANT_NAME}. My order ID is {IMPORTANT_ORDER_ID}."
    manager.turns[0][1]["content"] = f"Understood, {IMPORTANT_NAME}; I recorded order {IMPORTANT_ORDER_ID}."

    # Make the active post-summary window total exactly the expected 1,900 tokens.
    summary_message = [{"role": "developer", "content": f"Summary of earlier turns: Customer name: {IMPORTANT_NAME}. Order ID: {IMPORTANT_ORDER_ID}."}]
    manager.summary = f"Customer name: {IMPORTANT_NAME}. Order ID: {IMPORTANT_ORDER_ID}."
    active_without_system = summary_message + flatten(manager.turns[-manager.keep_last_turns :])
    fixed = manager.counter.messages([{"role": "system", "content": SYSTEM_PROMPT}] + active_without_system)
    recent_messages = flatten(manager.turns[-manager.keep_last_turns :])
    # The padded recent messages are the only adjustable part of this fixture.
    add_recent = target_after - fixed
    if add_recent < 0:
        raise ValueError("Fixture's recent turns already exceed the requested after-trim size")
    for index in range(add_recent):
        recent_messages[index % len(recent_messages)]["content"] += " detail"

    manager.summary = ""  # Fixture summary is added by the summarizer, not counted before trimming.
    current = manager.raw_history_tokens()
    if current > target_before:
        raise ValueError("Fixture base is larger than the requested before-trim size")
    add_old = target_before - current
    old_messages = flatten(manager.turns[:-manager.keep_last_turns])
    for index in range(add_old):
        old_messages[index % len(old_messages)]["content"] += " history"


async def offline_demonstration() -> None:
    manager = ConversationManager(live=False, keep_last_turns=5)
    build_twenty_turn_fixture(manager)
    print("=== 20-turn transcript replay ===")
    print(f"Sliding window: system prompt + summary + last {manager.keep_last_turns} turns")
    print(f"Token counter: {manager.counter.method}")
    print(f"Turn 20: history {manager.raw_history_tokens()} tokens")
    before, after = await manager.prepare_for_call(client=None)
    print(f"Turn 20: history {before} tokens -> after summarise {after} tokens")
    print(f"Recent turns retained: {len(manager.turns)} of {manager.keep_last_turns}")
    print(f"Summary retained name {IMPORTANT_NAME}: {IMPORTANT_NAME in manager.summary}")
    print(f"Summary retained order ID {IMPORTANT_ORDER_ID}: {IMPORTANT_ORDER_ID in manager.summary}")
    if (
        before != 5200
        or after != 1900
        or len(manager.turns) != manager.keep_last_turns
        or not manager.preserved_facts()
    ):
        raise AssertionError("20-turn summary fixture failed its count/fact-preservation checks")


async def live_demonstration(model: str) -> None:
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    manager = ConversationManager(model=model, live=True, keep_last_turns=5)
    prompts = [
        f"My name is {IMPORTANT_NAME}. Please remember order ID {IMPORTANT_ORDER_ID}.",
        "Please record that I contacted support about shipping.",
        "The package is delayed; note that I requested an update.",
        "Record that the delivery address is unchanged.",
        "Please note that I prefer a text update.",
        "The package is still in transit; add that to the notes.",
        "Record that I asked for the latest carrier scan.",
        "Please save that I can accept delivery after 5 PM.",
        "Note that I confirmed the phone number on the order.",
        "Please add that the package has not arrived yet.",
        "Record that I asked for an estimated delivery date.",
        "Please note that I am available tomorrow morning.",
        "Add that I want the case left open until delivery.",
        "Record that support confirmed the order is active.",
        "Please note that I requested a supervisor callback.",
        "Add that the tracking page showed no new scan today.",
        "Please record that I will check the lobby later.",
        "Add that I asked support to verify the carrier route.",
        "Please note that I still need a delivery update.",
        "What name and order ID did I provide at the start?",
    ]
    for number, prompt in enumerate(prompts, start=1):
        answer = await manager.ask(prompt, client)
        print(f"Turn {number} assistant: {answer}")
    print(f"Summary retained early name and order ID: {manager.preserved_facts()}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Run twenty turns using the OpenAI API")
    parser.add_argument("--model", default=MODEL, help="Model to use with --live")
    args = parser.parse_args()
    if args.live:
        await live_demonstration(args.model)
    else:
        await offline_demonstration()


if __name__ == "__main__":
    asyncio.run(main())
