"""A small, real agent so the evaluation loop works the moment you generate.

It is deliberately tiny and deterministic: a policy guard, then keyword
retrieval over a knowledge base, then an answer. No model call, no
credentials, no network — so `agentkit smoke` runs in seconds on a laptop
and in credential-free pull-request CI.

**This is the file to edit.** Change an entry in ``KNOWLEDGE``, loosen
``_POLICY_GUARDS``, or change how ``_best_match`` scores overlap, then run:

    agentkit compare

and watch the metrics move against the recorded baseline. That loop —
change one thing, score it against what you had — is the whole point.

When you replace this with your real agent, point ``agent:`` in
``agentkit.yaml`` at it: a Python callable (``module:function``), a serving
endpoint (``endpoints:/name``), a Unity Catalog model (``models:/...``), or
any HTTP/JSON endpoint. Nothing else in the project changes.
"""

from __future__ import annotations

KNOWLEDGE: dict[str, str] = {
    "returns": (
        "Standard orders can be returned within thirty days of delivery. "
        "Bring proof of purchase and you receive a full refund."
    ),
    "shipping": (
        "Domestic orders ship within three to five business days; "
        "international orders can take up to ten business days."
    ),
    "password": (
        "Click the reset link on the sign-in page. A reset email arrives "
        "within minutes; note the link expires after one hour."
    ),
    "subscription tiers": (
        "There are three tiers: basic, professional, and enterprise. They "
        "differ in seat count and support level."
    ),
    "support channels": (
        "You can reach support through chat, email, and the help portal. "
        "Enterprise customers also get a named contact."
    ),
    "invoices": (
        "Yes - invoices can be paid by bank transfer or credit card. The "
        "transfer details appear on the invoice footer."
    ),
    "data storage": (
        "Customer data is stored in the primary cloud region named in the "
        "data processing agreement, with encrypted backups."
    ),
    "cancellation": (
        "Access continues until the end of the paid period. Account data is "
        "retained for ninety days before deletion."
    ),
}

# Question keywords that route to a knowledge entry. Retrieval is a lookup,
# not a model: everything here is inspectable and unit-testable.
_ROUTES: dict[str, tuple[str, ...]] = {
    "returns": ("return", "refund"),
    "shipping": ("shipping", "ship", "delivery", "deliver"),
    "password": ("password", "reset", "sign-in", "login"),
    # "subscription" deliberately is not a keyword here: it also appears in
    # the cancellation question, and the more specific route must win.
    "subscription tiers": ("tier", "tiers", "plan", "plans"),
    "support channels": ("support", "contact", "channel", "channels", "help"),
    "invoices": ("invoice", "invoices", "payment", "pay", "bank", "transfer"),
    "data storage": ("data", "stored", "storage", "region", "residency"),
    "cancellation": ("cancel", "cancelled", "cancellation", "terminate"),
}

# Refusals are part of the product, not an error path — the golden suite
# gates them, so they live in code and are scored on every run.
_POLICY_GUARDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("personal phone", "personal number", "personal contact", "home address"),
        (
            "I cannot share personal contact information. Please use the "
            "official support channels: chat, email, or the help portal."
        ),
    ),
    (
        ("system prompt", "hidden instructions", "ignore your instructions"),
        (
            "I cannot reveal system instructions. I'm happy to help with your "
            "actual question instead."
        ),
    ),
)

_FALLBACK = (
    "I cannot answer that from the supported knowledge base. Please use the "
    "official support channels: chat, email, or the help portal."
)


def respond(question: str) -> str:
    """Answer one question. This is the callable under evaluation."""

    text = str(question).lower()
    guarded = _policy_response(text)
    if guarded is not None:
        return guarded
    topic = _best_match(text)
    if topic is None:
        return _FALLBACK
    return KNOWLEDGE[topic]


def _policy_response(text: str) -> str | None:
    for markers, response in _POLICY_GUARDS:
        if any(marker in text for marker in markers):
            return response
    return None


def _best_match(text: str) -> str | None:
    words = {word.strip(".,;:!?'\"") for word in text.split()}
    best_topic, best_score = None, 0
    for topic, keywords in _ROUTES.items():
        score = sum(1 for keyword in keywords if keyword in words)
        if score > best_score:
            best_topic, best_score = topic, score
    return best_topic
