from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from typing import Any


ORDERS = {
    1001: {"status": "shipped", "tracking": "YOHO-1001"},
    1002: {"status": "processing", "tracking": None},
}
CUSTOMERS = {5: {"name": "Avery Shah", "tier": "gold"}}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Return the status and tracking code for an order.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "integer"}},
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_customer",
            "description": "Return a customer's profile.",
            "parameters": {
                "type": "object",
                "properties": {"customer_id": {"type": "integer"}},
                "required": ["customer_id"],
                "additionalProperties": False,
            },
        },
    },
]


class ToolError(Exception):
    """An expected, user-correctable tool error returned to the model."""


def _require_int(args: dict[str, Any], name: str) -> int:
    value = args.get(name)
    # bool is a subclass of int in Python, so reject it explicitly.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolError(f"Invalid argument '{name}': expected an integer")
    return value


async def run_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Dispatch one tool. Unknown names and bad inputs become tool results."""
    if name == "get_order_status":
        order_id = _require_int(args, "order_id")
        order = ORDERS.get(order_id)
        if order is None:
            raise ToolError(f"Order {order_id} not found")
        return {"order_id": order_id, **order}
    if name == "get_customer":
        customer_id = _require_int(args, "customer_id")
        customer = CUSTOMERS.get(customer_id)
        if customer is None:
            raise ToolError(f"Customer {customer_id} not found")
        return {"customer_id": customer_id, **customer}
    raise ToolError(f"Tool '{name}' does not exist")


def error_result(error: Exception) -> str:
    return json.dumps({"error": str(error)})


def parse_tool_arguments(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ToolError(f"Invalid JSON arguments: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ToolError("Invalid arguments: expected a JSON object")
    return value


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class ModelReply:
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class OfflineModel:
    """Small scripted model so all four scenarios run without credentials."""

    async def first_reply(self, prompt: str) -> ModelReply:
        if prompt == "Status of orders 1001 and 1002 and customer 5":
            return ModelReply(
                tool_calls=[
                    ToolCall("call_order_1001", "get_order_status", '{"order_id":1001}'),
                    ToolCall("call_order_1002", "get_order_status", '{"order_id":1002}'),
                    ToolCall("call_customer_5", "get_customer", '{"customer_id":5}'),
                ]
            )
        if prompt == "Status of order 9999":
            return ModelReply(
                tool_calls=[ToolCall("call_missing_order", "get_order_status", '{"order_id":9999}')]
            )
        if prompt == "Offline invalid argument case":
            return ModelReply(
                tool_calls=[ToolCall("call_bad_argument", "get_order_status", '{"order_id":"abc"}')]
            )
        if prompt == "Offline unknown tool case":
            return ModelReply(
                tool_calls=[ToolCall("call_unknown_tool", "lookup_invoice", '{"invoice_id":7}')]
            )
        if prompt == "Offline malformed arguments case":
            return ModelReply(
                tool_calls=[ToolCall("call_bad_json", "get_order_status", '{"order_id":')]
            )
        return ModelReply(content="I couldn't determine which tools to call.")

    async def final_reply(self, prompt: str, results: list[tuple[str, str]]) -> str:
        values = [json.loads(result) for _, result in results]
        first_error = next((v["error"] for v in values if isinstance(v, dict) and "error" in v), None)
        if prompt == "Status of orders 1001 and 1002 and customer 5":
            return "Order 1001 is shipped (YOHO-1001); order 1002 is processing; customer 5 is Avery Shah (gold tier)."
        if first_error:
            if "9999" in first_error:
                return "Sorry, I couldn't find order 9999. Could you provide the correct order ID?"
            if "does not exist" in first_error:
                return "Sorry, I couldn't use that tool. Please rephrase your request and I’ll try another way."
            return f"Sorry, I couldn't process that request: {first_error}. Could you correct the details and try again?"
        return "Done."


class LiveModel:
    def __init__(self, model: str):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI()
        self.model = model

    async def first_reply(self, prompt: str) -> ModelReply:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            tools=TOOLS,
            tool_choice="auto",
            parallel_tool_calls=True,
        )
        msg = response.choices[0].message
        calls = [ToolCall(c.id, c.function.name, c.function.arguments) for c in (msg.tool_calls or [])]
        return ModelReply(content=msg.content, tool_calls=calls or None)

    async def final_reply(self, prompt: str, results: list[tuple[str, str]]) -> str:
        # Send tool outputs back under their original tool_call_id values.
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        for call_id, result in results:
            messages.append({"role": "tool", "tool_call_id": call_id, "content": result})
        # Live conversation history must include the assistant's tool_calls message.
        # The caller provides this history through _messages to preserve the model round.
        messages = getattr(self, "_messages", messages)
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            parallel_tool_calls=True,
        )
        return response.choices[0].message.content or ""


def assistant_tool_message(calls: list[ToolCall]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in calls
        ],
    }


async def execute_round(calls: list[ToolCall]) -> list[tuple[str, str]]:
    async def execute_one(call: ToolCall) -> tuple[str, str]:
        try:
            args = parse_tool_arguments(call.arguments)
            value = await run_tool(call.name, args)
            result = json.dumps(value)
        except ToolError as exc:
            result = error_result(exc)
        return call.id, result

    # These independent tool invocations start together.
    return await asyncio.gather(*(execute_one(call) for call in calls))


async def run_scenario(
    model: OfflineModel | LiveModel,
    prompt: str,
    *,
    label: str | None = None,
    injected_call: ToolCall | None = None,
) -> None:
    print(f"\n=== {label or prompt} ===")
    # Injected calls exercise malformed or impossible model outputs consistently
    # in both offline and live modes; the final recovery message is still model generated in live mode.
    first = ModelReply(tool_calls=[injected_call]) if injected_call else await model.first_reply(prompt)
    if not first.tool_calls:
        print(f"Assistant: {first.content or ''}")
        return

    calls = first.tool_calls
    print(f"Tool calls in one round: {len(calls)}")
    for call in calls:
        print(f"  {call.id}: {call.name}({call.arguments})")

    results = await execute_round(calls)
    for call_id, result in results:
        print(f"Tool result [{call_id}]: {result}")

    if isinstance(model, LiveModel):
        model._messages = [
            {"role": "user", "content": prompt},
            assistant_tool_message(calls),
            *[{"role": "tool", "tool_call_id": call_id, "content": result} for call_id, result in results],
        ]
    final = await model.final_reply(prompt, results)
    print(f"Assistant: {final}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Use the OpenAI API instead of the offline scripted model")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model to use with --live")
    args = parser.parse_args()

    model: OfflineModel | LiveModel = LiveModel(args.model) if args.live else OfflineModel()
    await run_scenario(model, "Status of orders 1001 and 1002 and customer 5")

    # Four recoverable error paths; each result is returned to the model as a tool result.
    error_cases = [
        ("Status of order 9999", "Unknown order", None),
        (
            "Look up order abc",
            "Invalid argument",
            ToolCall("call_bad_argument", "get_order_status", '{"order_id":"abc"}'),
        ),
        (
            "Look up invoice 7",
            "Tool name does not exist",
            ToolCall("call_unknown_tool", "lookup_invoice", '{"invoice_id":7}'),
        ),
        (
            "Look up an order with malformed tool arguments",
            "Malformed JSON arguments",
            ToolCall("call_bad_json", "get_order_status", '{"order_id":'),
        ),
    ]
    for prompt, label, injected_call in error_cases:
        await run_scenario(model, prompt, label=label, injected_call=injected_call)


if __name__ == "__main__":
    asyncio.run(main())
