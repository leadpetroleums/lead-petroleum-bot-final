"""intent.py — Classify the query to route to the right handler.

Four intents:
  - EXACT_SPEC    : "pour point of YOTTA 15000"
                    → Check manual specs CSV first. If not present, tell user
                      to check the catalogue page (no hallucination).
  - COMPARE       : "MEGA 7000 vs GIGA 10000"
                    → RAG retrieves both products, LLM compares.
  - RECOMMEND     : "oil for my diesel truck"
                    → RAG + LLM recommendation.
  - GENERAL       : "what certifications does Lead Petroleum have"
                    → RAG + LLM general answer.

Classification is rule-based first (fast, free) and falls back to LLM only
for ambiguous queries. Pure rules cover ~80% of real queries.

This module also infers the product CATEGORY the user is asking about from
vehicle/application keywords (e.g. "Mehran" -> car -> Petrol Engine Oil).
This is used by the retriever to filter ChromaDB so we never recommend
motorcycle oil for a car.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Intent(str, Enum):
    EXACT_SPEC = "exact_spec"
    COMPARE = "compare"
    RECOMMEND = "recommend"
    GENERAL = "general"


@dataclass
class IntentResult:
    intent: Intent
    products_mentioned: list[str]   # normalized product tokens, e.g. ["yotta 15000"]
    spec_mentioned: str | None      # e.g. "pour_point", "flash_point"
    inferred_category: str | None = None  # e.g. "Petrol Engine Oil"


# Known spec field names and their user-facing aliases
SPEC_KEYWORDS: dict[str, list[str]] = {
    "viscosity": ["viscosity"],
    "density": ["density"],
    "pour_point": ["pour point", "pour-point"],
    "flash_point": ["flash point", "flash-point"],
    "viscosity_index": ["viscosity index", "vi"],
    "tbn": ["tbn", "total base number", "base number"],
    "api_level": ["api level", "api spec", "api grade", "api rating"],
    "grade": ["viscosity grade", "sae grade"],
    "cold_crank": ["cold crank", "ccs"],
}

COMPARE_KEYWORDS = [
    " vs ", " versus ", "compare", "difference between",
    "which is better", "what's the difference",
]

RECOMMEND_KEYWORDS = [
    "recommend", "suggest", "which oil", "what oil",
    "best oil", "best for", "suitable for", "which product",
    "what product", "need an oil", "need oil for",
    "should i use", "what should i",
]

# Lead Petroleum product name patterns (the catalogue uses LEAD <SUBBRAND> <NUMBER>)
PRODUCT_TOKEN_PATTERN = re.compile(
    r"\b(zetta|exa|peta|tera|hecto|yotta|giga|mega|zepto|deci|deca|atto|"
    r"famto|yocto|motp|mocl|hypico|nano|cyn|centi|syn)\s*"
    r"([a-z0-9/]+(?:\s*\d+)?)",
    re.IGNORECASE,
)


# Category inference - maps vehicle/application keywords to the correct product
# category. When a query mentions any of these, we filter retrieval to that
# category only. This is the single most important guardrail against
# category-confusion errors (e.g. recommending motorcycle oil for a car).

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Petrol Engine Oil": [
        "car", "cars", "sedan", "hatchback", "petrol car", "gasoline car",
        "petrol engine", "gasoline engine", "passenger car", "passenger vehicle",
        "light commercial", "suv",
        # Common Pakistani/Asian petrol-car models
        "mehran", "cultus", "alto", "wagonr", "wagon r", "swift", "corolla",
        "city", "civic", "vitz", "prado", "land cruiser", "revo", "fortuner",
        "hilux", "bolan", "ravi", "ciaz",
    ],
    "Diesel Engine Oil": [
        "diesel", "truck", "trucks", "lorry", "bus", "buses",
        "heavy duty", "heavy-duty", "hgv", "commercial vehicle",
        "pickup", "tractor", "mining equipment", "off-road",
        "offroad", "off road", "locomotive",
    ],
    "Motorcycle Oil": [
        "motorcycle", "motorbike", "bike", "scooter", "moped",
        "4 stroke", "4-stroke", "four stroke", "4t",
        "honda cd", "honda cg", "yamaha", "suzuki gd",
        "cbr", "cc engine", "125cc", "150cc",
    ],
    "Outboard Oil": [
        "outboard", "boat engine", "jetski", "jet ski", "marine outboard",
        "2 stroke", "2-stroke", "two stroke", "2t",
    ],
    "Marine Oil": [
        "ship engine", "ship", "vessel", "marine diesel",
        "cylinder oil", "trunk piston", "medium speed engine",
        "crosshead engine", "residual fuel", "heavy fuel oil",
        "hfo", "ifo",
    ],
    "ATF Fluids": [
        "automatic transmission", "atf", "transmission fluid",
        "gearbox fluid", "automatic gearbox",
    ],
    "Gear Oil": [
        "gear oil", "gearbox oil", "differential", "manual gearbox",
        "manual transmission", "axle oil", "transfer case",
        "api gl-4", "api gl-5",
    ],
    "Industrial Gear Oil": [
        "industrial gear", "industrial gearbox", "gearbox lubricant",
        "wind turbine gear", "bevel gear", "helical gear", "worm gear",
    ],
    "Brake Fluid": [
        "brake fluid", "brakes", "brake system", "dot 3", "dot 4", "dot 5",
    ],
    "Coolant": [
        "coolant", "antifreeze", "anti-freeze", "radiator", "cooling system",
    ],
    "Grease": [
        "grease", "bearing grease", "lubricating grease", "bearings",
    ],
    "Hydraulic Oil": [
        "hydraulic", "hydraulics", "hydraulic system", "iso vg 32",
        "iso vg 46", "iso vg 68", "excavator", "forklift",
    ],
    "Turbine Oil": [
        "turbine", "steam turbine", "gas turbine",
    ],
    "Compressor Oil": [
        "compressor", "air compressor", "screw compressor",
    ],
    "Cutting Oil": [
        "cutting oil", "cutting fluid", "metalworking", "machining",
        "soluble oil",
    ],
}


def infer_category(query: str) -> str | None:
    """Return the product category most likely relevant to the query, or None
    if the query doesn't mention a vehicle/application clearly.

    When a SPECIFIC product type is mentioned (brake fluid, coolant, grease,
    hydraulic, etc.) we prefer that category over a generic vehicle mention.
    This handles queries like "brake fluid for my car" -> Brake Fluid, not
    ambiguous.
    """
    q = query.lower()
    matches: list[str] = []
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, q):
                matches.append(category)
                break

    # Exactly one category matched -> use it
    if len(matches) == 1:
        return matches[0]

    # Multiple matches: prefer specific product-type categories over
    # generic engine oil categories. This resolves "brake fluid for my car"
    # to Brake Fluid (not ambiguous).
    if len(matches) > 1:
        engine_oil_cats = {"Petrol Engine Oil", "Diesel Engine Oil", "Motorcycle Oil"}
        non_engine = [c for c in matches if c not in engine_oil_cats]
        if len(non_engine) == 1:
            return non_engine[0]

    # Ambiguous or zero matches -> let semantic search handle it
    return None


# Product mention extraction
def find_products(query: str) -> list[str]:
    """Extract product mentions from the query. Returns normalized lowercase tokens."""
    matches = PRODUCT_TOKEN_PATTERN.findall(query.lower())
    out = []
    for sub, rest in matches:
        token = f"{sub} {rest}".strip()
        out.append(token)
    return list(dict.fromkeys(out))


def find_spec_mentioned(query: str) -> str | None:
    q = query.lower()
    for spec_key, aliases in SPEC_KEYWORDS.items():
        for alias in aliases:
            if alias in q:
                return spec_key
    return None


def classify(query: str) -> IntentResult:
    q = query.lower()
    products = find_products(query)
    spec = find_spec_mentioned(query)
    category = infer_category(query)

    # Compare - strongest signal
    if any(kw in q for kw in COMPARE_KEYWORDS) and len(products) >= 1:
        return IntentResult(Intent.COMPARE, products, spec, category)

    # Exact spec - a specific product + a specific spec field
    if spec and products:
        return IntentResult(Intent.EXACT_SPEC, products, spec, category)

    # Recommend - keywords suggest asking for advice, OR they described a
    # vehicle/application (which is an implicit request for a recommendation).
    if any(kw in q for kw in RECOMMEND_KEYWORDS) or category:
        return IntentResult(Intent.RECOMMEND, products, spec, category)

    # Fallback
    return IntentResult(Intent.GENERAL, products, spec, category)
