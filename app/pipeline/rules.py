from typing import Dict, List, Tuple, Optional
from app.schemas import UserProfile, Conflict

def evaluate(
    dish_name: str,
    triggers: List[str],
    evidences: List[str],
    confidence: float,
    profile: UserProfile,
    disease_rules: Dict,
    ingredients_found: Optional[List[str]] = None,
    ingredient_coverage: float = 0.0
) -> Tuple[str, List[Conflict], List[str], float]:
    conflicts: List[Conflict] = []
    notes: List[str] = []

    # normalize user allergies/diseases to comparable keys
    user_allergies = {a.strip().lower() for a in (profile.allergies or [])}
    user_diseases = {d.strip().lower() for d in (profile.diseases or [])}
    if profile.is_pregnant is True:
        user_diseases.add("pregnancy")
    ingredients_set = {i.strip().lower() for i in (ingredients_found or [])}

    # Allergy match: if user allergy name matches a trigger key or overlaps keywords
    # (In your DB you have Allergy names like "Eggs", "Fish", "Soy", "Salt"...)
    # We'll map common names -> canonical trigger keys:
    allergy_to_trigger = {
        "eggs": "egg",
        "egg": "egg",
        "milk": "milk",
        "dairy": "milk",
        "fish": "fish",
        "soy": "soy",
        "salt": "sulfites",  # (not perfect) if you keep "Salt" allergy, treat it separately in your DB
        "wheat": "wheat_gluten",
        "gluten": "wheat_gluten",
        "sesame": "sesame",
        "peanut": "peanut",
        "shellfish": "shellfish",
        # French
        "lait": "milk",
        "fromage": "milk",
        "beurre": "milk",
        "oeuf": "egg",
        "œuf": "egg",
        "oeufs": "egg",
        "poisson": "fish",
        "soja": "soy",
        "ble": "wheat_gluten",
        "blé": "wheat_gluten",
        "gluten": "wheat_gluten",
        "sesame": "sesame",
        "sésame": "sesame",
        "arachide": "peanut",
        "cacahuete": "peanut",
        "cacahuète": "peanut",
        "fruits a coque": "tree_nuts",
        "fruits à coque": "tree_nuts",
        "moutarde": "mustard",
        "celeri": "celery",
        "céleri": "celery",
        # Arabic
        "حليب": "milk",
        "لبن": "milk",
        "جبن": "milk",
        "بيض": "egg",
        "سمك": "fish",
        "صويا": "soy",
        "قمح": "wheat_gluten",
        "غلوتين": "wheat_gluten",
        "سمسم": "sesame",
        "فول سوداني": "peanut",
        "مكسرات": "tree_nuts",
        "خردل": "mustard",
        "كرفس": "celery",
        "محار": "molluscs",
        "قشريات": "shellfish"
    }

    trigger_set = set(triggers)

    for ua in user_allergies:
        mapped = allergy_to_trigger.get(ua, ua)  # fallback
        if mapped in trigger_set:
            conflicts.append(Conflict(
                type="allergy",
                trigger=mapped,
                evidence="; ".join(evidences[:3]) if evidences else "detected",
                explanation=f"Dish may contain {mapped}, which conflicts with user allergy '{ua}'."
            ))

    # Diseases (aliases supported)
    disease_alias_map = {}
    for d_name, rule in disease_rules.items():
        aliases = [d_name] + (rule.get("aliases") or [])
        for a in aliases:
            if isinstance(a, str) and a.strip():
                disease_alias_map[a.strip().lower()] = d_name

    for d_in in user_diseases:
        canonical = disease_alias_map.get(d_in, None)
        if not canonical:
            continue
        rule = disease_rules.get(canonical, {})

        avoid = set(rule.get("avoid", []))
        avoid_ing = set(rule.get("avoid_ingredients", []))

        avoid_hit = (avoid & trigger_set) | (avoid_ing & ingredients_set)
        if avoid_hit:
            for x in sorted(avoid_hit):
                conflicts.append(Conflict(
                    type="disease",
                    trigger=x,
                    evidence="; ".join(evidences[:3]) if evidences else "detected",
                    explanation=rule.get("notes", f"Conflict with disease rule for {canonical}.")
                ))

        # keyword cautions based on dish name or evidences
        ck = [k.lower() for k in rule.get("caution_keywords", [])]
        c_ing = set(i.lower() for i in rule.get("caution_ingredients", []))
        if ck:
            combined_text = (dish_name + " " + " ".join(evidences)).lower()
            if any(k in combined_text for k in ck):
                notes.append(f"Caution for {canonical}: {rule.get('notes','')}")
        if c_ing and (c_ing & ingredients_set):
            notes.append(f"Caution for {canonical}: {rule.get('notes','')}")

    # Decide final safety (never UNKNOWN)
    if conflicts:
        return "RISKY", conflicts, notes, max(confidence, 0.8)

    ambiguous = {"chef special", "chef's special", "special", "mixed", "assorted", "sauce", "surprise", "plat du jour", "يوميات", "طبق اليوم"}
    if dish_name.strip().lower() in ambiguous:
        notes.append("Dish name is ambiguous; ingredients are unclear.")
        return "CAUTION", [], notes, confidence

    # if no conflicts but low confidence or low coverage -> CAUTION (not unknown)
    if confidence < 0.5 or ingredient_coverage < 0.4:
        notes.append("Insufficient ingredient details detected. Consider asking the restaurant or checking ingredients.")
        return "CAUTION", [], notes, confidence

    return "SAFE", [], notes, confidence
