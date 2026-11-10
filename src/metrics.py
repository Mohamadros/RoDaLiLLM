"""
Metrics and baseline similarity scorers for Record Linkage / Entity Resolution.
"""

from typing import Dict, Any
import jellyfish
from Levenshtein import ratio as levenshtein_ratio
from src.models import CustomerRecord


def token_jaccard(s1: str, s2: str) -> float:
    """Calculates Jaccard similarity over word tokens."""
    tokens1 = set(s1.lower().split())
    tokens2 = set(s2.lower().split())
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union


def compute_field_similarities(rec1: CustomerRecord, rec2: CustomerRecord) -> Dict[str, float]:
    """
    Computes per-field similarity using standard string matching algorithms:
    - Names: Jaro-Winkler (sensitive to prefixes)
    - Addresses/Streets: Levenshtein ratio
    - Company/Email: Token Jaccard & Jaro-Winkler
    """
    sims = {}

    # Full Name comparison
    name1 = f"{rec1.first_name} {rec1.last_name}".strip()
    name2 = f"{rec2.first_name} {rec2.last_name}".strip()
    sims["name_jaro_winkler"] = jellyfish.jaro_winkler_similarity(name1, name2)
    sims["name_levenshtein"] = levenshtein_ratio(name1, name2)

    # Street
    sims["street_levenshtein"] = levenshtein_ratio(rec1.street, rec2.street)

    # Postal code
    sims["postal_code_exact"] = 1.0 if rec1.postal_code.strip() == rec2.postal_code.strip() else 0.0

    # City
    sims["city_jaro_winkler"] = jellyfish.jaro_winkler_similarity(rec1.city, rec2.city)

    # Company
    if rec1.company and rec2.company:
        sims["company_jaccard"] = token_jaccard(rec1.company, rec2.company)
    else:
        sims["company_jaccard"] = 1.0 if rec1.company == rec2.company else 0.0

    # Email
    if rec1.email and rec2.email:
        sims["email_jaro_winkler"] = jellyfish.jaro_winkler_similarity(rec1.email.lower(), rec2.email.lower())
    else:
        sims["email_jaro_winkler"] = 1.0 if rec1.email == rec2.email else 0.5

    # Overall full-text string metrics
    text1 = rec1.to_full_text()
    text2 = rec2.to_full_text()
    sims["full_text_levenshtein"] = levenshtein_ratio(text1, text2)
    sims["full_text_jaro_winkler"] = jellyfish.jaro_winkler_similarity(text1, text2)
    sims["full_text_jaccard"] = token_jaccard(text1, text2)

    return sims


def evaluate_linkage_decision(rec1: CustomerRecord, rec2: CustomerRecord, threshold: float = 0.75) -> Dict[str, Any]:
    """
    Simulates a baseline Record Linkage engine:
    1. Blocking check: do the blocking keys match?
    2. Composite weighted similarity: weighted combination of field matchers.
    3. Match decision: True (linked as duplicate) if composite_score >= threshold.
    """
    bk1 = rec1.blocking_key()
    bk2 = rec2.blocking_key()
    blocking_match = (bk1 == bk2)

    field_sims = compute_field_similarities(rec1, rec2)

    # Weighted score simulating enterprise matching configurations
    weights = {
        "name_jaro_winkler": 0.35,
        "street_levenshtein": 0.20,
        "postal_code_exact": 0.15,
        "city_jaro_winkler": 0.10,
        "email_jaro_winkler": 0.20,
    }
    composite_score = sum(field_sims.get(k, 0.0) * w for k, w in weights.items())

    # If blocking fails, typical engines drop the candidate pair before full comparison!
    candidate_considered = blocking_match
    is_match = (candidate_considered and composite_score >= threshold)

    # Evasion success: human can tell it's the same, but the software fails to match
    evaded = not is_match

    return {
        "blocking_key_1": bk1,
        "blocking_key_2": bk2,
        "blocking_match": blocking_match,
        "composite_score": round(composite_score, 4),
        "threshold": threshold,
        "is_match": is_match,
        "evaded": evaded,
        "field_similarities": {k: round(v, 4) for k, v in field_sims.items()},
    }
