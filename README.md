# Parallel tools and error handling

This demo shows a model requesting multiple tools in one round, concurrent tool execution with `asyncio.gather`, and returning every result with its original `tool_call_id`. Expected tool failures are sent back to the model as JSON tool results so it can recover.

## Run without an API key

The default mode uses a small scripted model and runs the main request plus four error scenarios locally:

```bash
python3 parallel_tools.py
```

The scenarios cover a missing order (`9999`), an invalid argument, a tool name that is not registered, and malformed JSON arguments.

## Run with the OpenAI API

Install the dependency and set `OPENAI_API_KEY` in your shell, then run:

```bash
python3 -m pip install -r requirements.txt
export OPENAI_API_KEY="..."
python3 parallel_tools.py --live
```

In live mode, the model chooses tools for the normal request and the unknown-order case. The other three error cases supply synthetic invalid tool calls to exercise the dispatcher, then ask the model to produce the recovery response. No key is stored by the demo.

## Task 2: conversation memory

Run the offline six-turn conversation and stateless comparison:

```bash
python3 conversation_memory.py
```

Run it with the OpenAI API:

```bash
python3 conversation_memory.py --live
```

Start an interactive chat with `/reset` support:

```bash
python3 conversation_memory.py --interactive --live
```

Each API request includes the conversation's full `system`/`user`/`assistant` message history. After each user/assistant exchange, the script prints a token estimate and saves the six-turn sample to `conversation_6_turns.json`. It uses `tiktoken` when tokenizer data is available and a labeled local estimate when the machine is offline. The saved JSON can be loaded with `Conversation.load()`.

## Task 3: sliding window, summary, and budget

Run the deterministic 20-turn fixture offline:

```bash
python3 sliding_window_budget.py
```

Expected key output:

```text
Turn 20: history 5200 tokens -> after summarise 1900 tokens
Summary retained name Arun: True
Summary retained order ID ORD-4821: True
```

Run a live 20-turn conversation with the OpenAI API:

```bash
python3 sliding_window_budget.py --live
```

The manager retains a summary plus the latest five turns. Before each summarization or assistant call it checks the system, history, context, and output limits (500, 3000, 2000, and 1000 tokens). If tokenizer data is unavailable, it labels and uses a word-token estimate for the budget demonstration.

## Task 4: FastAPI routes and Pydantic models

Install dependencies and start the API:

```bash
python3 -m pip install -r requirements.txt
uvicorn fastapi_app:app --reload
```

Open `http://127.0.0.1:8000/docs` to try `GET /sessions/{session_id}` and `POST /chat`. The chat body requires a message between 1 and 2,000 characters. An empty string returns HTTP 422. Middleware echoes a supplied `X-Request-ID` or generates one and adds it to the response.

Try these in Swagger UI:

- `GET /sessions/{session_id}` with `session_id=demo-123`
- `POST /chat` with `{"message":"hello"}`
- `POST /chat` with `{"message":""}` to see the validation error:

```json
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": ["body", "message"],
      "msg": "String should have at least 1 character",
      "input": "",
      "ctx": {"min_length": 1}
    }
  ]
}
```

In `POST /chat`, FastAPI validates the incoming `ChatRequest`. The route uses `ChatResponse.model_validate()` to validate its response data and `model_dump()` to serialize that model to a dictionary for JSON output.
