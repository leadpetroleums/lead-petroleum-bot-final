"""rag.py — RAG pipeline for Lead Petroleum product queries.

Retrieves products from ChromaDB, classifies intent, and generates answers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import chromadb
from chromadb.utils import embedding_functions

from intent import Intent, IntentResult, classify
from llm import LLM, get_llm


# ─── Product catalogue (hardcoded fallback) ─────────────────────────────────
HARDCODED_PRODUCTS = [
    {"id": "mega7000",   "name": "MEGA 7000",   "category": "Petrol Engine Oil",  "page_ref": 5},
    {"id": "giga10000",  "name": "GIGA 10000",  "category": "Diesel Engine Oil",  "page_ref": 8},
    {"id": "yotta15000", "name": "YOTTA 15000", "category": "Motorcycle Oil",     "page_ref": 12},
]

# ─── Company contact information ─────────────────────────────────────────────
CONTACT_INFO = """
**Lead Petroleum - Contact Us:**

📧 Email:
  • General Inquiries: info@leadpetroleum.com
  • Support: support@leadpetroleum.com

📞 Phone: +971 (55) 575 7330

📍 Address: G-4 Al Bahar 4, Sheikh Ammar Road, Ajman, United Arab Emirates

🕒 Business Hours: Monday to Friday, 8:00 AM – 6:00 PM (Gulf Standard Time, UAE)

🌐 Website: https://www.leadpetroleum.com

📱 Follow Us On Social Media:
  • Facebook:  https://web.facebook.com/leadpetroleum
  • Instagram: https://www.instagram.com/lead.petroleum/
  • LinkedIn:  https://pk.linkedin.com/company/lead-petroleum
  • X:         https://x.com/leadpetroleum
"""

# ─── Intent buckets ──────────────────────────────────────────────────────────

# Pure greetings — respond warmly, no products
_GREETING_PHRASES: list[str] = [
    "hi", "hello", "hey", "yo",
    "hi there", "hello there", "hey there",
    "salam", "slm", "assalamualaikum", "aslam o alaikum",
    "walaikum assalam", "wa alaikum assalam", "aoa",
    "good morning", "good afternoon", "good evening", "good night",
]

# "How are you" variants — respond politely, no products
_HOW_ARE_YOU_PHRASES: list[str] = [
    "how are you", "how r u", "how are u", "how're you",
    "kaisay ho", "kaisy ho", "kya haal hai", "kaise ho",
    "what's up", "wassup", "sup", "whats up",
    "are you there", "anyone there", "you there",
]

# Acknowledgements — brief reply, no products
_ACK_PHRASES: list[str] = [
    "ok", "okay", "ok.", "okay.", "k",
    "thanks", "thank you", "thx", "ty",
    "great", "nice", "cool", "perfect", "awesome", "got it",
    "noted", "alright", "sure",
]

# Contact triggers
_CONTACT_KEYWORDS: list[str] = [
    "contact", "phone", "email", "address", "location",
    "call", "reach", "get in touch", "how to contact",
    "social media", "facebook", "instagram", "linkedin", "twitter",
    "website", "office",
]

# Off-topic triggers
_OFFTOPIC_KEYWORDS: list[str] = [
    "weather", "politics", "sports", "movie", "recipe",
    "joke", "math", "how to cook", "best restaurant", "news",
    "game", "music", "song",
]

# Signals that the user is asking about a product/vehicle
_PRODUCT_SIGNALS: list[str] = [
    "oil", "lubricant", "engine", "grease", "fluid",
    "car", "vehicle", "truck", "bike", "motorcycle",
    "diesel", "petrol", "gear", "transmission", "hydraulic",
    "recommend", "suggest", "best for", "which oil", "what oil",
    "spec", "viscosity", "grade",
]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _whole_word(phrase: str) -> re.Pattern:
    """Compile a whole-word regex for a phrase (handles spaces too)."""
    return re.compile(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", re.IGNORECASE)


def _matches_any(text: str, phrases: list[str]) -> bool:
    """Return True if *text* exactly matches (whole-word) any phrase in the list."""
    text = text.strip()
    for phrase in phrases:
        if _whole_word(phrase).fullmatch(text) or _whole_word(phrase).search(text):
            # fullmatch for exact single-phrase input; search for phrase inside text
            pass
        # Simpler: check if the cleaned text equals the phrase, or the text
        # is *only* that phrase (with punctuation stripped).
    stripped = re.sub(r"[^\w\s]", "", text).strip().lower()
    for phrase in phrases:
        if stripped == phrase.lower():
            return True
        # Also check whole-word presence for multi-word phrases
        if _whole_word(phrase).search(text):
            return True
    return False


def _is_pure_greeting(query: str) -> bool:
    return _matches_any(query, _GREETING_PHRASES)


def _is_how_are_you(query: str) -> bool:
    return _matches_any(query, _HOW_ARE_YOU_PHRASES)


def _is_acknowledgement(query: str) -> bool:
    return _matches_any(query, _ACK_PHRASES)


def _is_contact_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in _CONTACT_KEYWORDS)


def _is_offtopic(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in _OFFTOPIC_KEYWORDS)


def _has_product_signal(query: str) -> bool:
    q = query.lower()
    return any(sig in q for sig in _PRODUCT_SIGNALS)


# ─── Data classes ─────────────────────────────────────────────────────────────

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


# ─── Retriever ────────────────────────────────────────────────────────────────

class ProductRetriever:
    """Retrieves products from ChromaDB or falls back to keyword search."""

    def __init__(self, catalogue_path: Path | None = None):
        self.catalogue_path = catalogue_path or Path("catalogue.json")
        self.products: list[dict] = []
        self.db = None
        self._initialized = False

    def _lazy_init(self) -> None:
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
        except Exception as exc:
            print(f"Warning: Failed to load catalogue: {exc}. Using hardcoded fallback.")
            self.products = HARDCODED_PRODUCTS

        try:
            self.db = chromadb.Client()
            if self.products:
                embedder = embedding_functions.DefaultEmbeddingFunction()
                col = self.db.get_or_create_collection(
                    name="products",
                    embedding_function=embedder,
                    metadata={"hnsw:space": "cosine"},
                )
                for p in self.products:
                    col.add(
                        ids=[p["id"]],
                        metadatas=[{"name": p["name"], "category": p["category"]}],
                        documents=[f"{p['name']} {p['category']}"],
                    )
        except Exception as exc:
            print(f"Warning: ChromaDB init failed: {exc}. Using keyword fallback.")
            self.db = None

    def all_products(self) -> list[dict]:
        self._lazy_init()
        return self.products

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        self._lazy_init()
        if not self.products:
            return []

        if self.db:
            try:
                results = self.db.get_collection("products").query(
                    query_texts=[query], n_results=top_k
                )
                if results and results["ids"]:
                    ids = results["ids"][0]
                    return [p for p in self.products if p["id"] in ids]
            except Exception:
                pass

        # Keyword fallback
        q = query.lower()
        matches = [p for p in self.products if q in p.get("name", "").lower()]
        return matches[:top_k]


# ─── RAG Engine ───────────────────────────────────────────────────────────────

class RAGEngine:
    """RAG engine for Lead Petroleum product queries."""

    def __init__(self, llm: LLM | None = None, catalogue_path: Path | None = None):
        self.retriever = ProductRetriever(catalogue_path)
        self.llm = llm or get_llm()

    def answer(self, query: str) -> Answer:
        return self._process(query)

    def stream(self, query: str) -> Iterator[str]:
        yield self._process(query).text

    # ── Core logic ────────────────────────────────────────────────────────────

    def _process(self, query: str) -> Answer:  # noqa: C901
        q = query.strip()

        # ── 1. Pure greeting ─────────────────────────────────────────────────
        if _is_pure_greeting(q):
            return Answer(
                text=(
                    "Hello! 👋 I'm your Lead Petroleum AI Assistant. "
                    "I can help you find the right lubricant for your vehicle or equipment. "
                    "What can I help you with today?"
                ),
                intent=Intent.GENERAL,
                confidence=0.98,
            )

        # ── 2. "How are you" / casual check-in ──────────────────────────────
        if _is_how_are_you(q):
            return Answer(
                text="I'm doing great, thanks for asking! 😊 How can I assist you today?",
                intent=Intent.GENERAL,
                confidence=0.98,
            )

        # ── 3. Acknowledgement (ok / thanks / great …) ───────────────────────
        if _is_acknowledgement(q):
            return Answer(
                text="You're welcome! Let me know if you need anything else. 😊",
                intent=Intent.GENERAL,
                confidence=0.97,
            )

        # ── 4. Contact information ────────────────────────────────────────────
        if _is_contact_query(q):
            return Answer(
                text=CONTACT_INFO,
                intent=Intent.GENERAL,
                confidence=0.95,
            )

        # ── 5. Off-topic ─────────────────────────────────────────────────────
        if _is_offtopic(q):
            return Answer(
                text=(
                    "I'm specifically designed to help with Lead Petroleum lubricant products. "
                    "Please ask me about oils, lubricants, or related topics and I'll be happy to assist! 😊"
                ),
                intent=Intent.GENERAL,
                confidence=0.90,
            )

        # ── 6. Vague query — no clear product signal ──────────────────────────
        if not _has_product_signal(q):
            return Answer(
                text=(
                    "Could you please share your vehicle model or describe your usage "
                    "so I can recommend the best lubricant for you? 🚗"
                ),
                intent=Intent.GENERAL,
                confidence=0.75,
            )

        # ── 7. Product / recommendation path ─────────────────────────────────
        intent_result = classify(q)
        intent = intent_result.intent
        products = self.retriever.search(q, top_k=3)

        if not products:
            return Answer(
                text=(
                    "I couldn't find a specific product match for your query. "
                    "Please contact us for personalized assistance:\n\n" + CONTACT_INFO
                ),
                intent=intent,
                confidence=0.60,
            )

        product_list = "\n".join(f"• {p['name']} ({p['category']})" for p in products)

        if intent == Intent.EXACT_SPEC:
            text = (
                f"Here are the products that match your requirements:\n\n{product_list}\n\n"
                "For full specifications visit https://www.leadpetroleum.com or contact us."
            )
        elif intent == Intent.COMPARE:
            text = (
                f"Here's a comparison of relevant products:\n\n{product_list}\n\n"
                "For detailed specs and a side-by-side comparison, visit our website or contact us."
            )
        elif intent == Intent.RECOMMEND:
            text = (
                f"Based on your needs, I recommend:\n\n{product_list}\n\n"
                "Visit https://www.leadpetroleum.com for more details, "
                "or contact us for personalized advice."
            )
        else:
            text = (
                f"Here are some relevant products for your query:\n\n{product_list}\n\n"
                "For more information visit https://www.leadpetroleum.com or contact us."
            )

        citations = [Citation(p["name"], p.get("page_ref", "N/A")) for p in products]

        return Answer(
            text=text,
            intent=intent,
            citations=citations,
            confidence=0.85,
        )
