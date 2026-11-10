"""
Adversarial heuristic perturbations simulating real human error:
1. QWERTY / QWERTZ adjacent key fat-finger typos
2. OCR optical character recognition cluster substitutions
3. Phonetic substitutions (German phonetics)
4. Domain / syntax abbreviations
"""

import random
from typing import Dict, List
from src.models import CustomerRecord

# Adjacent keys on German QWERTZ / International QWERTY keyboards
KEYBOARD_NEIGHBORS: Dict[str, str] = {
    "a": "qwsy",
    "b": "vghn",
    "c": "xdfv",
    "d": "ersfxc",
    "e": "wsdr34",
    "f": "rtgdcv",
    "g": "tyhfvb",
    "h": "yujgbn",
    "i": "ujko89",
    "j": "uikmnh",
    "k": "ijlm",
    "l": "kopö",
    "m": "njk",
    "n": "bhjm",
    "o": "iklp90",
    "p": "olü0",
    "q": "wa12",
    "r": "edft45",
    "s": "weadxz",
    "t": "rfgy56",
    "u": "zhji78",
    "v": "cfgb",
    "w": "qase23",
    "x": "zsdc",
    "y": "asxc",
    "z": "tugh67",
    "ä": "öüpl",
    "ö": "äpöl",
    "ü": "öp+",
}

# OCR scan optical confusions (single or multi-character clusters)
OCR_CONFUSIONS: List[tuple] = [
    ("rn", "m"),
    ("m", "rn"),
    ("cl", "d"),
    ("d", "cl"),
    ("vv", "w"),
    ("w", "vv"),
    ("O", "0"),
    ("0", "O"),
    ("l", "1"),
    ("1", "l"),
    ("I", "l"),
    ("B", "8"),
    ("8", "B"),
    ("S", "5"),
    ("5", "S"),
]

# Phonetic substitutions for German / English names
PHONETIC_SWAPS: List[tuple] = [
    ("sch", "sh"),
    ("sh", "sch"),
    ("ck", "k"),
    ("k", "ck"),
    ("ph", "f"),
    ("f", "ph"),
    ("dt", "t"),
    ("tz", "z"),
    ("ä", "ae"),
    ("ö", "oe"),
    ("ü", "ue"),
    ("ß", "ss"),
    ("c", "k"),
    ("v", "w"),
    ("w", "v"),
]


def apply_keyboard_typo(text: str, error_rate: float = 0.1, target_prefix: bool = True) -> str:
    """
    Substitutes characters with physical keyboard neighbors.
    Targeting the prefix (first 1-3 chars) is devastating to Jaro-Winkler!
    """
    if not text:
        return text

    chars = list(text)
    if target_prefix and len(chars) > 0 and random.random() < 0.7:
        # Attack the first character
        c_low = chars[0].lower()
        if c_low in KEYBOARD_NEIGHBORS:
            neighbor = random.choice(KEYBOARD_NEIGHBORS[c_low])
            chars[0] = neighbor.upper() if chars[0].isupper() else neighbor
            return "".join(chars)

    for i in range(len(chars)):
        c_low = chars[i].lower()
        if c_low in KEYBOARD_NEIGHBORS and random.random() < error_rate:
            neighbor = random.choice(KEYBOARD_NEIGHBORS[c_low])
            chars[i] = neighbor.upper() if chars[i].isupper() else neighbor

    return "".join(chars)


def apply_ocr_glitch(text: str) -> str:
    """
    Substitutes optical clusters (e.g. 'rn' -> 'm', 'cl' -> 'd').
    In Levenshtein, this counts as 2 edits, slashing similarity rapidly!
    """
    for src, dst in OCR_CONFUSIONS:
        if src in text and random.random() < 0.6:
            text = text.replace(src, dst, 1)
            break
    return text


def apply_phonetic_shift(text: str) -> str:
    """
    Substitutes phonetically equivalent clusters (e.g. 'Bäcker' -> 'Becker', 'Schmidt' -> 'Schmitt').
    """
    for src, dst in PHONETIC_SWAPS:
        if src in text.lower() and random.random() < 0.7:
            # Case-insensitive replace
            idx = text.lower().find(src)
            text = text[:idx] + dst + text[idx + len(src):]
            break
    return text


def perturb_record_heuristically(record: CustomerRecord, attack_type: str = "hybrid") -> CustomerRecord:
    """
    Creates an adversarial duplicate of a CustomerRecord using rule-based noise.
    attack_type: 'keyboard', 'ocr', 'phonetic', or 'hybrid'.
    """
    data = record.model_dump()

    if attack_type in ("keyboard", "hybrid"):
        # Apply keyboard slip on last name (devastating to Jaro-Winkler & blocking)
        data["last_name"] = apply_keyboard_typo(data["last_name"], error_rate=0.15, target_prefix=True)
        data["street"] = apply_keyboard_typo(data["street"], error_rate=0.08, target_prefix=False)

    if attack_type in ("ocr", "hybrid"):
        data["first_name"] = apply_ocr_glitch(data["first_name"])
        data["last_name"] = apply_ocr_glitch(data["last_name"])
        data["company"] = apply_ocr_glitch(data["company"])
        data["street"] = apply_ocr_glitch(data["street"])

    if attack_type in ("phonetic", "hybrid"):
        data["last_name"] = apply_phonetic_shift(data["last_name"])
        data["city"] = apply_phonetic_shift(data["city"])

    # Email domain variations
    if data.get("email"):
        if "@googlemail.com" in data["email"]:
            data["email"] = data["email"].replace("@googlemail.com", "@gmail.com")
        elif "@gmail.com" in data["email"]:
            data["email"] = data["email"].replace("@gmail.com", "@googlemail.com")

    return CustomerRecord(**data)
