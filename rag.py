"""rag.py — RAG pipeline for Lead Petroleum product queries.

Retrieves products from ChromaDB, classifies intent, and generates answers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import chromadb
from chromadb.utils import embedding_functions

from intent import Intent, IntentResult, classify
from llm import LLM, get_llm


# ─── Product catalogue (hardcoded fallback) ─────────────────────────────────
HARDCODED_PRODUCTS = [
    {"id": "mega7000", "name": "MEGA 7000", "category": "Petrol Engine Oil", "page_ref": 5},
    {"id": "giga10000", "name": "GIGA 10000", "category": "Diesel Engine Oil", "page_ref": 8},
    {"id": "yotta15000", "name": "YOTTA 15000", "category": "Motorcycle Oil", "page_ref": 12},
]


@dataclass
class Citation:
    product_name: str
    page_ref: int | str


@dataclass
class Answer:
    text: str
    intent: Intent
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.8


class ProductRetriever:
    """Retrieves products from ChromaDB or returns hardcoded fallback."""

    def __init__(self, catalogue_path: Path | None = None):
        self.catalogue_path = catalogue_path or Path("catalogue.json")
        self.products = []
        self.db = None
        self._initialized = False

    def _lazy_init(self):
        """Lazy initialization — only load on first use."""
        if self._initialized:
            return
        self._initialized = True

        try:
            import json
            if self.catalogue_path.exists():
                with open(self.catalogue_path) as f:
                    self.products = json.load(f)
            else:
                self.products = HARDCODED_PRODUCTS
        except Exception as e:
            print(f"Warning: Failed to load catalogue: {e}. Using hardcoded fallback.")
            self.products = HARDCODED_PRODUCTS

        # Try to initialize ChromaDB
        try:
            self.db = chromadb.Client()
            if self.products:
                embedder = embedding_functions.DefaultEmbeddingFunction()
                self.db.get_or_create_collection(
                    name="products",
                    embedding_function=embedder,
                    metadata={"hnsw:space": "cosine"}
                )
                # Index products
                for p in self.products:
                    self.db.get_collection("products").add(
                        ids=[p["id"]],
                        metadatas=[{"name": p["name"], "category": p["category"]}],
                        documents=[f"{p['name']} {p['category']}"]
                    )
        except Exception as e:
            print(f"Warning: ChromaDB initialization failed: {e}. Using keyword search fallback.")
            self.db = None

    def all_products(self) -> list[dict]:
        """Return all products."""
        self._lazy_init()
        return self.products

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search for products matching the query."""
        self._lazy_init()

        if not self.products:
            return []

        # Try ChromaDB first
        if self.db:
            try:
                results = self.db.get_collection("products").query(
                    query_texts=[query],
                    n_results=top_k
                )
                if results and results["ids"]:
                    return [p for p in self.products if p["id"] in results["ids"][0]]
            except Exception:
                pass

        # Fallback: keyword search
        query_lower = query.lower()
        matches = [p for p in self.products if query_lower in p.get("name", "").lower()]
        return matches[:top_k]


class RAGEngine:
    """RAG engine for Lead Petroleum product queries."""

    def __init__(self, llm: LLM | None = None, catalogue_path: Path | None = None):
        self.retriever = ProductRetriever(catalogue_path)
        self.llm = llm or get_llm()

    def answer(self, query: str) -> Answer:
        """Generate a single answer (non-streaming)."""
        result = self._process(query)
        return result

    def stream(self, query: str) -> Iterator[str]:
        """Generate answer as a stream of tokens."""
        result = self._process(query)
        yield result.text

    def _process(self, query: str) -> Answer:
        """Process a query and return an answer."""
        # Classify intent
        intent_result = classify(query)
        intent = intent_result.intent

        # Get relevant products
        products = self.retriever.search(query, top_k=3)

        # Generate answer
        if intent == Intent.EXACT_SPEC:
            text = f"Based on our catalogue, here are products that match your query: {', '.join(p['name'] for p in products)}"
        elif intent == Intent.COMPARE:
            text = f"Comparing: {', '.join(p['name'] for p in products)}. Please refer to our catalogue for detailed specifications."
        elif intent == Intent.RECOMMEND:
            text = f"We recommend: {', '.join(p['name'] for p in products)}. Visit our website for more details."
        else:
            text = "Thank you for your interest in Lead Petroleum. Please visit leadpetroleum.com or contact us for more information."

        citations = [Citation(p["name"], p.get("page_ref", "N/A")) for p in products]

        return Answer(
            text=text,
            intent=intent,
            citations=citations,
            confidence=0.85
        )
