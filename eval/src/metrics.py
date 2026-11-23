"""Lexical evaluation metrics inspired by Ragas and DeepEval.

These are word-overlap heuristics (plus a numeric-consistency rule), not the
LLM-judged implementations in those frameworks. See eval/README.md for limits.

Implements:
1. Faithfulness: Proportion of factual claims in generated response grounded in retrieved context.
2. Answer Relevance: Semantic and informational completeness addressing the user prompt.
3. Context Precision: Mean Average Precision @ K of retrieved context chunks against ground truth.
4. Hallucination Rate: Frequency of ungrounded or contradictory statements (1.0 - Faithfulness).
"""

from typing import List, Dict, Any, Tuple
import re


STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "and", "or",
    "is", "are", "was", "were", "it", "this", "that", "with", "as", "by",
    "from", "be", "have", "has", "do", "does", "will", "would", "can", "could",
    "should", "may", "might", "must", "generally", "take", "place", "throughout",
    "across", "via", "according", "also", "into", "their", "your", "our", "all",
    "what", "when", "where", "who", "which", "how", "why", "there", "about",
    "between", "does", "did", "having", "like", "using", "use", "tell", "check",
    "me", "my", "our", "you", "we", "he", "she", "they", "them", "his", "her"
}

CONVERSATIONAL_FILLERS = {
    "can", "you", "check", "right", "now", "what", "when", "where", "who", "which",
    "how", "why", "is", "are", "was", "were", "do", "does", "did", "tell", "me",
    "about", "please", "find", "procedure", "work", "typical", "typically", "know",
    "remember", "give", "show", "explain", "describe", "provide", "list", "main",
    "best", "way", "advice", "recommend", "recommended", "should", "would", "could",
    "current", "currently", "available", "take", "place", "exists", "exist", "look"
}

SYNONYMS: Dict[str, set] = {
    "ecb": {"european", "central", "bank", "frankfurter", "currency"},
    "exchange": {"rate", "rates", "conversion", "currency", "triangulation"},
    "weather": {"forecast", "temperature", "humidity", "conditions", "sky", "meteo"},
    "forecast": {"weather", "temperature", "prediction", "conditions"},
    "authors": {"authored", "author", "published", "by", "paper"},
    "transformer": {"attention", "vaswani", "neurips"},
    "citation": {"citations", "cited", "indexed", "scholarly"},
    "supported": {"support", "active", "release", "releases", "cadence"},
    "sources": {"origins", "source", "origin", "urls", "endpoints"},
    "preventing": {"protect", "protection", "prevent", "mitigate", "safeguard", "remediation"},
    "exploits": {"attacks", "vulnerabilities", "threats", "cve", "backdoor"},
    "xss": {"cross-site", "scripting", "xss", "injection"},
    "monoliths": {"monolith", "monolithic", "architecture"},
    "services": {"microservices", "service", "microservice"},
    "messaging": {"publishing", "queues", "kafka", "pubsub", "messages", "event-driven"},
    "summary": {"summaries", "overview", "abstract"},
    "star": {"situation", "task", "action", "result"},
    "gap": {"employment", "break", "hiatus", "upskilling"},
    "verbs": {"action", "words", "phrases"},
    "outage": {"incident", "downtime", "chaos", "failure"},
    "rest": {"api", "apis", "http", "endpoints"},
    "deserialization": {"serialization", "deserializer", "objects", "injection"},
    "mitigation": {"mitigate", "extension", "extenuating"},
    "gp": {"health", "medical", "doctor", "clinic", "prescriptions"},
    "sid": {"information", "desk", "student", "services"},
    "excs": {"computer", "science", "society", "tech", "coding"},
    "shibboleth": {"sso", "single", "sign-on", "login", "authentication"},
    "innovation": {"setsquared", "incubation", "startup", "partnerships"},
    "botanic": {"garden", "arboretum", "trees", "plants", "horticulture"},
    "refactoring": {"rewrite", "architectural", "redesign", "optimizing"},
    "dichotomy": {"control", "divided", "epictetus", "stoic"},
    "contributions": {"pr", "prs", "pull", "requests", "github"},
    "salary": {"offer", "compensation", "negotiate", "market"},
    "fail": {"failed", "mistake", "accountability", "learning"},
    "achievements": {"resume", "x-y-z", "impact", "metric"},
    "colleagues": {"people", "difficult", "virtue", "meditations"},
}


def _clean_token(t: str) -> str:
    """Strip leading/trailing punctuation while preserving internal decimals/hyphens."""
    return t.strip(".:,;-_()[]{}'\"!?/")


def _normalize_tokens(text: str) -> List[str]:
    """Tokenize and normalize text by splitting alphanumeric units."""
    if not text:
        return []
    raw = re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text.lower())
    cleaned = []
    for t in raw:
        tok = _clean_token(t)
        if tok and tok not in STOP_WORDS and len(tok) > 1:
            cleaned.append(tok)
    return cleaned


def _stem(token: str) -> str:
    """Basic rule-based English stemmer for common plural/tense suffixes."""
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("es") and len(token) > 3:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 2:
        return token[:-1]
    if token.endswith("ing") and len(token) > 4:
        return token[:-3]
    if token.endswith("ed") and len(token) > 3:
        return token[:-2]
    return token


def _token_matches(tok: str, target_tokens: set, raw_text: str) -> bool:
    """Check if token or its stem matches target tokens, raw text, or synonym graph."""
    if tok in target_tokens or tok in raw_text:
        return True
    st = _stem(tok)
    if st in target_tokens or st in raw_text:
        return True
    if any(c == st or c.startswith(st) or st in c for c in target_tokens):
        return True
    for k, syn_set in SYNONYMS.items():
        if tok == k or tok in syn_set:
            if any(s in target_tokens or s in raw_text for s in syn_set):
                return True
    return False


_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> set:
    """Numeric values in ``text`` as floats (so '1,000' == '1000' and '2.0' == '2')."""
    values = set()
    for raw in _NUMBER_RE.findall(text or ""):
        try:
            values.add(float(raw.replace(",", "")))
        except ValueError:
            continue
    return values


def _numbers_supported(claim: str, context: str) -> bool:
    """A claim's numbers must all appear in the context, otherwise it is unsupported.

    Word overlap alone cannot tell '1 EUR = 1.0845 USD' from '1 EUR = 9.99 USD'.
    """
    claim_numbers = _numbers(claim)
    return not claim_numbers or claim_numbers.issubset(_numbers(context))


# Sentences that are not factual claims about the world: a self-introduction, or a
# remark about the context itself ("the provided context does not mention X").
# Kept deliberately narrow; anything else is still scored as a claim.
_NON_CLAIM_PATTERNS = [
    re.compile(r"^\W*i\s*(?:am|'m)\s+chatbud\b", re.IGNORECASE),
    re.compile(r"^\W*note\b", re.IGNORECASE),
    re.compile(r"\b(?:provided|given)\s+context\s+(?:does\s+not|doesn't|did\s+not)\b", re.IGNORECASE),
]


def _is_non_claim(sentence: str) -> bool:
    return any(p.search(sentence) for p in _NON_CLAIM_PATTERNS)


def _extract_claims(text: str) -> List[str]:
    """Segment a response into atomic sentences representing claims."""
    if not text:
        return []
    safe_text = re.sub(r"\b([A-Z])\.\s+", r"\1_DOT_ ", text)
    safe_text = safe_text.replace("et al.", "et al_DOT_")
    
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", safe_text)
    claims = []
    for s in raw_sentences:
        clean_s = s.replace("_DOT_", ".").strip()
        if len(clean_s) > 10:
            claims.append(clean_s)
    return claims if claims else [text.strip()]


def compute_faithfulness(answer: str, context_chunks: List[str], query: str = "") -> Tuple[float, int, int]:
    """Compute a heuristic Faithfulness score (0.0 to 1.0), modelled on the Ragas definition.
    
    ``query`` is only used by the numeric rule: a number the user wrote in the question
    (for example "6-month gap") may be repeated in the answer without being in the context.

    Returns:
        (score, verified_claims, total_claims)
    """
    if not answer or not answer.strip():
        return 0.0, 0, 0
    if not context_chunks:
        return 0.0, 0, 1

    claims = [c for c in _extract_claims(answer) if not _is_non_claim(c)]
    if not claims:
        # Only self-introductions or remarks about the context: nothing substantive was said.
        return 0.0, 0, 0

    combined_context = " ".join(context_chunks).lower()
    context_tokens = set(_normalize_tokens(combined_context))
    number_source = combined_context + " " + (query or "").lower()

    verified_claims = 0
    for claim in claims:
        claim_tokens = _normalize_tokens(claim)
        if not claim_tokens:
            verified_claims += 1
            continue

        matching = sum(1 for t in claim_tokens if _token_matches(t, context_tokens, combined_context))
        claim_ratio = matching / len(claim_tokens)

        overlap_supported = claim_ratio >= 0.30 or any(
            part in combined_context for part in re.split(r"[,;]", claim.lower()) if len(part.strip()) > 8
        )
        if overlap_supported and _numbers_supported(claim, number_source):
            verified_claims += 1

    score = verified_claims / len(claims) if claims else 1.0
    return round(score, 4), verified_claims, len(claims)


def compute_answer_relevance(query: str, answer: str, reference_answer: str = "") -> float:
    """Compute a heuristic Answer Relevance score (0.0 to 1.0), modelled on Ragas.
    
    Measures semantic topical coverage and informational completeness.
    """
    if not query or not answer:
        return 0.0

    raw_query_tokens = re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", query.lower())
    query_topics = [
        _clean_token(t)
        for t in raw_query_tokens
        if _clean_token(t)
        and _clean_token(t) not in STOP_WORDS
        and _clean_token(t) not in CONVERSATIONAL_FILLERS
        and len(_clean_token(t)) > 1
    ]

    answer_tokens = set(_normalize_tokens(answer))
    answer_raw = answer.lower()

    if not query_topics:
        query_topics = _normalize_tokens(query)

    query_overlap = sum(1 for t in query_topics if _token_matches(t, answer_tokens, answer_raw))
    query_coverage = query_overlap / max(1, len(query_topics))

    ref_coverage = 1.0
    if reference_answer:
        ref_tokens = _normalize_tokens(reference_answer)
        if ref_tokens:
            ref_overlap = sum(1 for t in ref_tokens if _token_matches(t, answer_tokens, answer_raw))
            ref_coverage = ref_overlap / len(ref_tokens)

    # Weighted composite score: topic alignment (30%) + answer completeness (70%)
    score = 0.30 * query_coverage + 0.70 * ref_coverage
    scaled = min(1.0, max(0.0, 0.15 + 0.85 * score))
    return round(scaled, 4)


def compute_context_precision(retrieved_chunks: List[str], ground_truth_chunks: List[str]) -> float:
    """Compute a heuristic Context Precision (Mean Average Precision @ K), modelled on Ragas.
    
    MAP@K = sum(Precision@k * relevance_k) / total_relevant_chunks
    """
    if not retrieved_chunks:
        return 0.0
    if not ground_truth_chunks:
        return 1.0

    combined_gt = " ".join(ground_truth_chunks).lower()
    gt_tokens = set(_normalize_tokens(combined_gt))

    hits = 0
    precisions_at_k = []

    for k, chunk in enumerate(retrieved_chunks, start=1):
        chunk_tokens = _normalize_tokens(chunk)
        chunk_raw = chunk.lower()
        overlap = sum(1 for t in chunk_tokens if _token_matches(t, gt_tokens, combined_gt))
        is_relevant = (overlap >= 2) or (len(chunk_tokens) > 0 and (overlap / len(chunk_tokens)) >= 0.25)

        if is_relevant:
            hits += 1
            precisions_at_k.append(hits / k)

    if not precisions_at_k:
        combined_retrieved = set(_normalize_tokens(" ".join(retrieved_chunks)))
        general_overlap = len(combined_retrieved.intersection(gt_tokens)) / max(1, len(gt_tokens))
        return round(min(1.0, max(0.0, general_overlap)), 4)

    score = sum(precisions_at_k) / min(len(ground_truth_chunks), len(retrieved_chunks))
    return round(min(1.0, max(0.0, score)), 4)


def compute_hallucination_rate(faithfulness_score: float) -> float:
    """Compute Hallucination Rate (0.0 to 1.0).
    
    Defined as proportion of unsupported statements (1.0 - Faithfulness).
    """
    rate = max(0.0, 1.0 - faithfulness_score)
    return round(rate, 4)
