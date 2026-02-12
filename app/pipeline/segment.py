import re
from typing import List, Dict

PRICE_ANY = re.compile(r"\b\d{1,3}([.,]\d{1,2})?\s*(\$|jd|jod|aed|sar|€|£)\b", re.IGNORECASE)
ONLY_PRICE = re.compile(r"^\s*\d{1,3}([.,]\d{1,2})?\s*(\$|jd|jod|aed|sar|€|£)\s*$", re.IGNORECASE)

BULLET_LINE = re.compile(r"^\s*[\+\-\u2022\u00bb\u00a2\u00a9\u0640]\s*", re.UNICODE)
LEADING_QUOTE = re.compile(r"^\s*[>\u00bb]\s*")

COMMON_HEADERS = {
    "menu", "dine-in menu", "dine in menu", "starters", "sides", "side", "mains", "main", "main dish", "main dishes",
    "desserts", "drinks", "beverages", "salads", "sandwiches", "burgers", "extras", "trays",
    "allergen tags", "allergen tag", "allergens", "ingredients"
}

DESC_VERBS = {
    "served", "topped", "finished", "drizzled", "brushed", "stuffed",
    "marinated", "seasoned", "simmered", "garnished"
}

DESC_STARTERS = {
    "with", "over", "finished", "topped", "drizzled", "brushed", "served"
}


def _clean(s: str) -> str:
    s = s.strip()
    s = LEADING_QUOTE.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _looks_like_header(line: str) -> bool:
    l = line.lower().strip(":|-•>»")
    if not l:
        return True
    if "menu" in l and len(l) <= 40:
        return True
    return l in COMMON_HEADERS or (l.isupper() and len(l) <= 35)


def _looks_like_dish_title(line: str) -> bool:
    if not line:
        return False

    low = line.lower().strip()

    # Never dish: bullet ingredient lines
    if BULLET_LINE.match(line):
        return False

    # Never dish: pure price
    if ONLY_PRICE.match(low):
        return False

    # Never dish: headers
    if _looks_like_header(line):
        return False
    if "detected_triggers" in low or "ingredients_found" in low:
        return False

    # Dish: contains price somewhere
    if PRICE_ANY.search(line):
        return True

    # Sentence-like lines are usually descriptions
    if (line.endswith(",") or line.endswith(".") or line.endswith("—") or line.endswith(":")) and len(line) > 15:
        return False

    # Dish-ish: short, title-like (not too long) and not a comma-heavy ingredient list
    words = [w for w in low.split() if w]
    if 2 <= len(words) <= 7 and len(line) <= 50:
        if line.count(",") >= 2:
            return False

        if any(v in low.split() for v in DESC_VERBS):
            return False

        if len(words) >= 6 and words[0] in DESC_STARTERS:
            return False

        # must contain enough letters
        if sum(c.isalpha() for c in line) >= 6:
            return True

    return False


def segment_dishes(text: str) -> List[Dict[str, str]]:
    lines = [_clean(x) for x in text.splitlines() if _clean(x)]
    dishes: List[Dict[str, str]] = []

    current_name = None
    block: List[str] = []

    def flush():
        nonlocal current_name, block
        if current_name:
            dishes.append({"dish_name": current_name, "block": " ".join(block).strip()})
        current_name = None
        block = []

    for ln in lines:
        if _looks_like_dish_title(ln):
            flush()
            # remove trailing price chunk from dish name
            name = re.sub(r"\s*[\+\-–—]?\s*\d{1,3}([.,]\d{1,2})?\s*(\$|jd|jod|aed|sar|€|£)\b.*$",
                          "", ln, flags=re.IGNORECASE).strip()
            name = re.sub(r"\s*[°º]\s*$", "", name).strip()
            current_name = name if name else ln
        else:
            # attach to current dish block (keep bullet lines as ingredients inside block)
            if current_name and not _looks_like_header(ln) and not ONLY_PRICE.match(ln):
                block.append(ln)

    flush()

    # de-dup dish names (keep first)
    seen = set()
    final = []
    for d in dishes:
        key = d["dish_name"].lower()
        if key in seen:
            continue
        seen.add(key)
        final.append(d)

    return final
