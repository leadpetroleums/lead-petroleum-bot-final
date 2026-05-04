"""main.py — FastAPI server exposing the chatbot as a REST API.

    uvicorn main:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health                   health check
    POST /api/chat                 chat endpoint (streaming)
    GET  /api/products             list all products (with optional filters)
    GET  /api/products/{id}        get a single product
    GET  /api/categories           list product categories
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag import RAGEngine


app = FastAPI(title="Lead Petroleum AI Assistant", version="0.1.0")

# CORS — allow leadpetroleum.com and localhost for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://leadpetroleum.com",
        "https://www.leadpetroleum.com",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single engine instance reused across requests
_engine: RAGEngine | None = None


def get_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine


# ─── Models ──────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str
    stream: bool = True


class ChatResponse(BaseModel):
    text: str
    intent: str
    citations: list[dict[str, Any]]
    confidence: float


# ─── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/chat")
def chat(req: ChatRequest):
    engine = get_engine()

    if req.stream:
        def iter_response():
            for token in engine.stream(req.query):
                yield token
        return StreamingResponse(iter_response(), media_type="text/plain")

    answer = engine.answer(req.query)
    return ChatResponse(
        text=answer.text,
        intent=answer.intent.value,
        citations=[{"product": c.product_name, "page": c.page_ref} for c in answer.citations],
        confidence=answer.confidence,
    )


@app.get("/api/products")
def list_products(
    category: str | None = Query(None),
    api_level: str | None = Query(None),
    viscosity: str | None = Query(None),
) -> list[dict]:
    engine = get_engine()
    products = engine.retriever.all_products()
    if category:
        products = [p for p in products if p["category"].lower() == category.lower()]
    if api_level:
        products = [
            p
            for p in products
            if (p.get("api_level") or "").upper().find(api_level.upper()) >= 0
        ]
    if viscosity:
        products = [
            p
            for p in products
            if viscosity.upper() in (p.get("viscosity_grades") or [])
        ]
    # Return a slim list for UI consumption
    return [
        {
            "id": p["id"],
            "name": p["name"],
            "category": p["category"],
            "page_ref": p["page_ref"],
            "api_level": p.get("api_level"),
            "viscosity_grades": p.get("viscosity_grades", []),
        }
        for p in products
    ]


@app.get("/api/products/{product_id}")
def get_product(product_id: str) -> dict:
    engine = get_engine()
    products = engine.retriever.all_products()
    for p in products:
        if p["id"] == product_id:
            return p
    raise HTTPException(status_code=404, detail="Product not found")


@app.get("/api/categories")
def list_categories() -> list[dict]:
    engine = get_engine()
    products = engine.retriever.all_products()
    counts: dict[str, int] = {}
    for p in products:
        counts[p["category"]] = counts.get(p["category"], 0) + 1
    return [
        {"name": name, "product_count": n}
        for name, n in sorted(counts.items(), key=lambda x: -x[1])
    ]
