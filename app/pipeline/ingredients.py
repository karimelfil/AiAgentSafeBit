import re
import unicodedata
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

NEG_WINDOW = 35

_ARABIC_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED\u0640]"
)

def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )

def _norm(s: str) -> str:
    if not s:
        return ""
    s = _strip_accents(s)
    s = _ARABIC_DIACRITICS.sub("", s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s.strip())
    return s

def _is_negated(full_text: str, alias: str) -> bool:
    t = _norm(full_text)
    a = _norm(alias)
    idx = t.find(a)
    if idx == -1:
        return False

    window = t[max(0, idx - NEG_WINDOW): min(len(t), idx + len(a) + NEG_WINDOW)]

    if re.search(rf"\b(no|without|free of|avoid|sans)\b.{0,15}\b{re.escape(a)}\b", window):
        return True
    if re.search(rf"(?:بدون|خالي من|خالٍ من).{{0,15}}{re.escape(a)}", window):
        return True
    if re.search(rf"\b{re.escape(a)}\b\s*-\s*free\b", window):
        return True

    if any(x in window for x in ["dairy-free", "sans lactose", "sans lait", "خالي من الحليب", "خالٍ من الحليب", "خالي من الألبان", "خالٍ من الألبان"]) and a in {"milk", "dairy", "cheese", "cream", "butter"}:
        return True
    if any(x in window for x in ["gluten-free", "sans gluten", "خالي من الغلوتين", "خالٍ من الغلوتين"]) and a in {"gluten", "wheat", "flour", "bread", "pasta", "barley", "rye"}:
        return True

    return False

def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"[a-z]+", _norm(text))

def _phrase_present_with_ocr_tolerance(full_text: str, alias: str, allow_fuzzy: bool = True) -> bool:
    normalized_alias = _norm(alias)
    if not normalized_alias:
        return False

    if re.search(rf"(?<!\w){re.escape(normalized_alias)}(?!\w)", _norm(full_text)):
        return True

    if not allow_fuzzy:
        return False

    alias_words = [word for word in normalized_alias.split() if word]
    text_words = _tokenize_words(full_text)
    if not alias_words or not text_words or len(text_words) < len(alias_words):
        return False

    window_size = len(alias_words)
    alias_joined = " ".join(alias_words)
    best_ratio = 0.0
    for idx in range(len(text_words) - window_size + 1):
        candidate = " ".join(text_words[idx : idx + window_size])
        ratio = SequenceMatcher(None, alias_joined, candidate).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
        if ratio >= 0.84:
            return True

    if len(alias_words) == 1:
        for token in text_words:
            ratio = SequenceMatcher(None, alias_words[0], token).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
            if ratio >= 0.82:
                return True

    return False

def extract_lexicon_hits(
    text: str,
    lexicon: Dict[str, List[str]],
    allow_fuzzy: bool = True,
) -> Tuple[List[str], List[str]]:

    low = _norm(text)
    found = []
    evidence = []

    for canon, aliases in lexicon.items():
        for a in aliases:
            a2 = _norm(a)
            if not a2:
                continue
            if _phrase_present_with_ocr_tolerance(low, a2, allow_fuzzy=allow_fuzzy):
                if _is_negated(text, a2):
                    continue
                found.append(canon)
                evidence.append(f"text:{a2}")
                break

    seen = set()
    out = []
    for x in found:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out, evidence

def infer_from_dish_name(dish_name: str) -> Tuple[List[str], List[str], float]:

    n = _norm(dish_name)
    inferred = []
    notes = []
    boost = 0.0

    if any(x in n for x in ["wrap", "sandwich", "burger", "bun", "bread", "toast", "pita", "tortilla"]):
        inferred.append("wheat_gluten")
        notes.append("Dish type often includes bread/wrap -> possible gluten.")
        boost = max(boost, 0.20)

    if "pasta" in n and "gluten-free" not in n:
        inferred.append("wheat_gluten")
        notes.append("Pasta is commonly wheat-based unless labeled gluten-free.")
        boost = max(boost, 0.20)

    if any(x in n for x in ["cheese", "creamy", "alfredo", "butter"]):
        inferred.append("milk")
        notes.append("Dish name suggests dairy.")
        boost = max(boost, 0.15)

    if any(x in n for x in ["mayo", "aioli"]):
        inferred.append("egg")
        notes.append("Sauces like mayo/aioli often contain egg.")
        boost = max(boost, 0.12)

    if "gluten-free" in n:
        inferred = [t for t in inferred if t != "wheat_gluten"]
        notes.append("Dish labeled gluten-free.")
        boost = max(boost, 0.12)

    if "dairy-free" in n:
        inferred = [t for t in inferred if t != "milk"]
        notes.append("Dish labeled dairy-free.")
        boost = max(boost, 0.12)

    if "no eggs" in n or "egg-free" in n:
        inferred = [t for t in inferred if t != "egg"]
        notes.append("Dish labeled no-egg.")
        boost = max(boost, 0.12)

    inferred = sorted(set(inferred))
    return inferred, notes, boost

def _estimate_slots(text: str) -> int:
    low = _norm(text)
    if not low:
        return 1
    sep = 0
    sep += low.count(",")
    sep += len(re.findall(r"\b(and|with|plus)\b", low))
    sep += len(re.findall(r"\b(et|avec)\b", low))
    sep += len(re.findall(r"\b(و|مع)\b", low))
    slots = min(8, max(1, sep + 1))
    return slots

def build_ingredients_list(
    dish_name: str,
    block: str,
    common_ingredients: Dict[str, List[str]],
    allergen_triggers: Dict[str, List[str]],
 ) -> Tuple[List[str], List[str], List[str], float, float, List[str]]:

    text = f"{dish_name}\n{block}"

    ingredients_found, ing_evidence = extract_lexicon_hits(text, common_ingredients, allow_fuzzy=True)
    triggers_found, trig_evidence = extract_lexicon_hits(text, allergen_triggers, allow_fuzzy=False)

    inferred, infer_notes, boost = infer_from_dish_name(dish_name)
    triggers_found = sorted(set(triggers_found) | set(inferred))

    evidences = []
    evidences.extend(trig_evidence[:6])
    evidences.extend(ing_evidence[:6])

    if len(ingredients_found) >= 4:
        confidence = 0.92
    elif len(ingredients_found) >= 2:
        confidence = 0.80
    elif len(ingredients_found) == 1:
        confidence = 0.65
    else:
        confidence = 0.55

    confidence = min(0.95, confidence + boost)

    slots = _estimate_slots(text)
    ingredient_coverage = min(1.0, len(ingredients_found) / max(1, slots))

    low = _norm(text)
    if "no eggs" in low or "egg-free" in low or "sans oeuf" in low or "sans œuf" in low or "بدون بيض" in low:
        triggers_found = [t for t in triggers_found if t != "egg"]
    if "no dairy" in low or "dairy-free" in low or "sans lait" in low or "sans lactose" in low or "بدون حليب" in low or "خالي من الحليب" in low or "خالٍ من الحليب" in low:
        triggers_found = [t for t in triggers_found if t != "milk"]
    if "gluten-free" in low or "sans gluten" in low or "خالي من الغلوتين" in low or "خالٍ من الغلوتين" in low:
        if not any(x in low for x in ["wheat", "bread", "flour", "barley", "rye"]):
            triggers_found = [t for t in triggers_found if t != "wheat_gluten"]

    notes = []
    notes.extend(infer_notes)

    return ingredients_found, triggers_found, evidences, confidence, ingredient_coverage, notes
