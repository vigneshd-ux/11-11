from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI
from pydantic import BaseModel, Field


app = FastAPI(
    title="YOHO Chat API",
    description="Session lookup and validated chat request examples.",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    reply: str


class SessionResponse(BaseModel):
    session_id: str
    status: str


@app.middleware("http")
async def add_request_id(request, call_next):
    """Echo a supplied request ID or generate one for this response."""
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/sessions/{session_id}", response_model=SessionResponse, tags=["sessions"])
async def get_session(session_id: str) -> SessionResponse:
    """Return a small status object for the path parameter."""
    return SessionResponse(session_id=session_id, status="active")


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def post_chat(payload: ChatRequest) -> dict[str, str]:
    """Accept a validated message and return a serialized response."""
    # FastAPI validates ChatRequest before entering the route. This explicit
    # model_validate call also demonstrates validating data received internally.
    response = ChatResponse.model_validate({"reply": f"Received: {payload.message}"})
    # model_dump converts the Pydantic model into a JSON-serializable dict.
    return response.model_dump()
