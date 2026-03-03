import re
import unicodedata
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

def extract_lexicon_hits(text: str, lexicon: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:

    low = _norm(text)
    found = []
    evidence = []

    for canon, aliases in lexicon.items():
        for a in aliases:
            a2 = _norm(a)
            if not a2:
                continue
            if re.search(rf"(?<!\w){re.escape(a2)}(?!\w)", low):
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

    ingredients_found, ing_evidence = extract_lexicon_hits(text, common_ingredients)
    triggers_found, trig_evidence = extract_lexicon_hits(text, allergen_triggers)

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
