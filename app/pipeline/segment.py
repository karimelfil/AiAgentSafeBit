import re
from typing import Dict, List

#regex to remove price from dish titles
PRICE_ANY = re.compile(
    r"\b\d{1,3}([.,]\d{1,2})?\s*(\$|jd|jod|aed|sar|\u20ac|\u00a3)\b",
    re.IGNORECASE,
)
#regex to remove price to don't detect them as dish titles
ONLY_PRICE = re.compile(
    r"^\s*\d{1,3}([.,]\d{1,2})?\s*(\$|jd|jod|aed|sar|\u20ac|\u00a3)\s*$",
    re.IGNORECASE,
)

#regex to detect bullets points to dectect them as not dish titles
BULLET_LINE = re.compile(r"^\s*[\+\-\u2022\u00bb\u00a2\u00a9\u0640]\s*", re.UNICODE)

#regex to remove leading quote characters 
LEADING_QUOTE = re.compile(r"^\s*[>\u00bb]\s*")

#headers to not detect them as dish titles
COMMON_HEADERS = {
    "menu",
    "dine-in menu",
    "dine in menu",
    "breakfast",
    "lunch",
    "dinner",
    "brunch",
    "starters",
    "appetizers",
    "soups",
    "sides",
    "side",
    "mains",
    "main",
    "main dish",
    "main dishes",
    "desserts",
    "drinks",
    "beverages",
    "salads",
    "sandwiches",
    "burgers",
    "combos",
    "platters",
    "extras",
    "trays",
    "allergen tags",
    "allergen tag",
    "allergens",
    "ingredients",
}

#markers that indicate that these are ingredients
INGREDIENT_MARKERS = {
    "ingredients",
    "ingredient",
    "dish ingredients",
    "contains",
    "composition",
    "served with",
    "made with",
}

#verbs that indicate that this is a description and not a dish title
DESC_VERBS = {
    "served",
    "topped",
    "finished",
    "drizzled",
    "brushed",
    "stuffed",
    "marinated",
    "seasoned",
    "simmered",
    "garnished",
    "includes",
    "contains",
}
#starters that indicate that this is a description and not a dish title
DESC_STARTERS = {"with", "over", "finished", "topped", "drizzled", "brushed", "served", "includes", "contains"}

#terms that indicate that this is a medical profile line and not a dish title
MEDICAL_PROFILE_TERMS = {
    "allergy",
    "allergies",
    "disease",
    "diseases",
    "celiac",
    "coeliac",
    "diabetes",
    "hypertension",
    "ibs",
    "crohn",
    "pregnancy",
    "cholesterol",
    "gerd",
    "kidney",
    "lactose",
    "intolerance",
    "gout",
    "cardiovascular",
    "salt",
    "fish",
}

#regex patterns to detect and fix OCR erros related to dish and ingredient 
OCR_LINE_REPLACEMENTS = [
    (re.compile(r"^\s*dis[h]?\b[\s:.-]*", re.IGNORECASE), "Dish: "),
    (re.compile(r"^\s*ingredien\w*\b[\s:.-]*", re.IGNORECASE), "Ingredients: "),
]


#clean and normalize text before applying the process
def _clean(s: str) -> str:
    s = s.strip()
    s = LEADING_QUOTE.sub("", s)
    for pattern, replacement in OCR_LINE_REPLACEMENTS:
        s = pattern.sub(replacement, s)
    s = re.sub(r"\s+", " ", s)
    return s


#filters to detect headers
def _looks_like_header(line: str) -> bool:
    l = line.lower().strip(":|-•>»")
    if not l:
        return True #
    if "menu" in l and len(l) <= 40:
        return True
    return l in COMMON_HEADERS or (line.isupper() and len(l) <= 35)


#filters to detect ingredient lines
def _looks_like_ingredient_line(line: str) -> bool:
    low = line.lower().strip().strip(":")
    if not low:
        return False

    if low.startswith("ingredients") or low.startswith("ingredient"):
        return True

    if any(low.startswith(marker) for marker in INGREDIENT_MARKERS):
        return True

    if line.rstrip().endswith(","):
        return True

    if "," in line and any(token in low for token in [" and ", " with ", " et ", " avec ", " و ", " مع "]):
        return True

    words = re.findall(r"[^\W\d_]+", low, flags=re.UNICODE) #extract words from the line
    if words and all(w in MEDICAL_PROFILE_TERMS for w in words):
        return True

    return False


#filters to detect medical profile lines
def _looks_like_profile_line(line: str) -> bool:
    low = line.lower().strip()
    if low.startswith("selected allergies") or low.startswith("selected diseases"):
        return True
    words = re.findall(r"[^\W\d_]+", low, flags=re.UNICODE) #extract words from the line
    return len(words) >= 2 and all(w in MEDICAL_PROFILE_TERMS for w in words)


#filters to detect dish titles
def _looks_like_dish_title(line: str) -> bool:
    if not line:
        return False

    low = line.lower().strip()

    if BULLET_LINE.match(line):
        return False

    if ONLY_PRICE.match(low):
        return False

    if _looks_like_header(line):
        return False
    if _looks_like_ingredient_line(line):
        return False

    if "detected_triggers" in low or "ingredients_found" in low:
        return False

    if PRICE_ANY.search(line):
        return True

    if re.match(r"^\s*dish\s*[:\-]\s*.+", line, flags=re.IGNORECASE):
        return True

    if line.endswith(","):
        return False
    if (line.endswith(".") or line.endswith("—") or line.endswith(":")) and len(line) > 15:
        return False

    words = [w for w in low.split() if w]
    if 2 <= len(words) <= 8 and len(line) <= 60:
        if line.count(",") >= 2:
            return False

        if any(v in low.split() for v in DESC_VERBS):
            return False

        if len(words) >= 6 and words[0] in DESC_STARTERS:
            return False

        if sum(c.isalpha() for c in line) >= 6:
            return True

    return False


#detect and parse dishes that are labeled with dish Name and ingredients labels
def _parse_labeled_dishes(text: str) -> List[Dict[str, str]]:
    label_pattern = re.compile(
        r"(?is)dish\s*name\s*[:\-]\s*(?P<name>.+?)\s*(?:dish\s*)?ingredients\s*[:\-]\s*(?P<ingredients>.+?)(?=(?:\n\s*dish\s*name\s*[:\-])|(?:\n\s*selected\s*allerg(?:y|ies)\s*[:\-])|(?:\n\s*selected\s*disease(?:s)?\s*[:\-])|\Z)"
    )#regex to detect blocks of text that are labeled with "Dish Name: " and "Ingredients: " 
    out: List[Dict[str, str]] = []
    for m in label_pattern.finditer(text or ""):
        name = _clean(m.group("name"))
        ingredients = _clean(m.group("ingredients"))
        if name and ingredients:
            out.append({"dish_name": name, "block": ingredients})
    return out


#take raw ocr and return structured dishes with their names and ingredients blocks
def segment_dishes(text: str) -> List[Dict[str, str]]:

    labeled = _parse_labeled_dishes(text)
    if labeled:
        return labeled

    lines = [_clean(x) for x in text.splitlines() if _clean(x)]
    dishes: List[Dict[str, str]] = []

    #current dish name being processed
    current_name = None
    #current block of text being processed for the current dish
    block: List[str] = []

    def flush() -> None:
        nonlocal current_name, block
        if current_name:
            dishes.append({"dish_name": current_name, "block": " ".join(block).strip()})
        current_name = None
        block = []

    for ln in lines:
        if _looks_like_dish_title(ln):
            flush()

            #remove prices , dish names , trailing degree symbols
            name = re.sub(
                r"\s*[\+\-\u2013\u2014]?\s*\d{1,3}([.,]\d{1,2})?\s*(\$|jd|jod|aed|sar|\u20ac|\u00a3)\b.*$",
                "",
                ln,
                flags=re.IGNORECASE,
            ).strip()
            name = re.sub(r"^\s*dish\s*name\s*[:\-]\s*", "", name, flags=re.IGNORECASE).strip()
            name = re.sub(r"^\s*dish\s*[:\-]\s*", "", name, flags=re.IGNORECASE).strip()
            name = re.sub(r"\s*[\u00b0\u00ba]\s*$", "", name).strip()
            current_name = name if name else ln
        else:
            if (
                current_name
                and not _looks_like_header(ln)
                and not ONLY_PRICE.match(ln)
                and not _looks_like_profile_line(ln)
            ):
                block.append(re.sub(r"^\s*ingredients\s*[:\-]\s*", "", ln, flags=re.IGNORECASE))

    flush()

#remove duplicates by dish names
    seen = set()
    final = []
    for d in dishes:
        key = d["dish_name"].lower()
        if key in seen:
            continue
        seen.add(key)
        final.append(d)

    return final
