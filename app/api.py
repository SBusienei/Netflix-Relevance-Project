"""Serve bounded, cached search and validated local-agent responses through one index."""

from functools import lru_cache
import os

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.agent import ask
from src.index import build_index
from src.search import Mode, search

app = FastAPI(title="Netflix hybrid retrieval")


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=500, pattern=r"\w")
    mode: Mode = "hybrid"
    k: int = Field(default=10, ge=1, le=50)
    depth: int = Field(default=50, ge=1, le=100)

    @model_validator(mode="after")
    def valid_depth(self) -> "Request":
        if self.k > self.depth:
            raise ValueError("k must not exceed candidate depth")
        return self


@lru_cache(maxsize=256)
def cached(query: str, mode: Mode, k: int, depth: int) -> tuple[tuple[str, float], ...]:
    return tuple((hit["show_id"], hit["score"]) for hit in search(build_index(), query, mode, k, depth))


@app.post("/search")
def search_endpoint(request: Request) -> list[dict]:
    try:
        scores = cached(request.query, request.mode, request.k, request.depth)
        records = {row["show_id"]: row for row in build_index().records}
        return [records[show_id] | {"score": score} for show_id, score in scores]
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@app.post("/rerank")
def rerank_endpoint(request: Request) -> list[dict]:
    return search_endpoint(request.model_copy(update={"mode": "rerank"}))


@app.post("/ask")
def ask_endpoint(request: Request) -> dict:
    if not os.environ.get("NETFLIX_AGENT_MODEL"):
        raise HTTPException(503, "Agent disabled: configure NETFLIX_AGENT_MODEL and local Ollama")
    try:
        return ask(build_index(), request.query)
    except ValueError as error:
        raise HTTPException(502, str(error)) from error
    except httpx.HTTPError as error:
        raise HTTPException(503, f"Local Ollama request failed: {error}") from error
