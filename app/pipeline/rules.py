from typing import Dict, List, Tuple, Optional
from app.schemas import UserProfile, Conflict

# Evaluates the safety of a dish for a user based on detected triggers, user profile, and disease rules.    
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

    user_allergies = {a.strip().lower() for a in (profile.allergies or [])}
    user_diseases = {d.strip().lower() for d in (profile.diseases or [])}
    if profile.is_pregnant is True:
        user_diseases.add("pregnancy")
    ingredients_set = {i.strip().lower() for i in (ingredients_found or [])}
    dish_name_low = dish_name.strip().lower()

    # Map common allergy terms to standardized trigger names
    allergy_to_trigger = {
        "eggs": "egg",
        "egg": "egg",
        "milk": "milk",
        "dairy": "milk",
        "fish": "fish",
        "soy": "soy",
        "salt": "sulfites",
        "wheat": "wheat_gluten",
        "gluten": "wheat_gluten",
        "sesame": "sesame",
        "peanut": "peanut",
        "peanuts": "peanut",
        "peanut allergy": "peanut",
        "peanut butter": "peanut",
        "tree nut": "tree_nuts",
        "tree nuts": "tree_nuts",
        "treenuts": "tree_nuts",
        "shellfish": "shellfish",
        "lait": "milk",
        "fromage": "milk",
        "beurre": "milk",
        "oeuf": "egg",
        "Ã…â€œuf": "egg",
        "oeufs": "egg",
        "poisson": "fish",
        "soja": "soy",
        "ble": "wheat_gluten",
        "blÃƒÂ©": "wheat_gluten",
        "sÃƒÂ©same": "sesame",
        "arachide": "peanut",
        "cacahuete": "peanut",
        "cacahuÃƒÂ¨te": "peanut",
        "fruits a coque": "tree_nuts",
        "fruits ÃƒÂ  coque": "tree_nuts",
        "moutarde": "mustard",
        "celeri": "celery",
        "cÃƒÂ©leri": "celery",
        "Ã˜Â­Ã™â€žÃ™Å Ã˜Â¨": "milk",
        "Ã™â€žÃ˜Â¨Ã™â€ ": "milk",
        "Ã˜Â¬Ã˜Â¨Ã™â€ ": "milk",
        "Ã˜Â¨Ã™Å Ã˜Â¶": "egg",
        "Ã˜Â³Ã™â€¦Ã™Æ’": "fish",
        "Ã˜ÂµÃ™Ë†Ã™Å Ã˜Â§": "soy",
        "Ã™â€šÃ™â€¦Ã˜Â­": "wheat_gluten",
        "Ã˜ÂºÃ™â€žÃ™Ë†Ã˜ÂªÃ™Å Ã™â€ ": "wheat_gluten",
        "Ã˜Â³Ã™â€¦Ã˜Â³Ã™â€¦": "sesame",
        "Ã™ÂÃ™Ë†Ã™â€ž Ã˜Â³Ã™Ë†Ã˜Â¯Ã˜Â§Ã™â€ Ã™Å ": "peanut",
        "Ã™â€¦Ã™Æ’Ã˜Â³Ã˜Â±Ã˜Â§Ã˜Âª": "tree_nuts",
        "Ã˜Â®Ã˜Â±Ã˜Â¯Ã™â€ž": "mustard",
        "Ã™Æ’Ã˜Â±Ã™ÂÃ˜Â³": "celery",
        "Ã™â€¦Ã˜Â­Ã˜Â§Ã˜Â±": "molluscs",
        "Ã™â€šÃ˜Â´Ã˜Â±Ã™Å Ã˜Â§Ã˜Âª": "shellfish"
    }

    allergy_ingredient_hints = {
        "peanut": {"peanut", "peanuts", "peanut_butter"},
        "tree_nuts": {"nuts", "almond", "walnut", "cashew", "pistachio", "pecan", "hazelnut"},
        "green bean": {"beans", "green beans", "green_beans"},
        "green beans": {"beans", "green beans", "green_beans"},
    }

    disease_alias_overrides = {
        "hypercholesterolemia": "High Cholesterol",
        "high cholesterol": "High Cholesterol",
        "heart failure": "Hypertension",
        "congestive heart failure": "Hypertension",
        "chf": "Hypertension",
    }

    trigger_set = {t.strip().lower() for t in triggers}

    for ua in user_allergies:
        # Map user allergy to standardized trigger if possible, otherwise use the original allergy term
        mapped = allergy_to_trigger.get(ua, ua)
        ingredient_hints = allergy_ingredient_hints.get(ua, set()) | allergy_ingredient_hints.get(mapped, set())
        # Check if the mapped trigger is in the detected triggers for the dish
        if mapped in trigger_set or bool(ingredient_hints & ingredients_set) or ua in dish_name_low:
            conflicts.append(Conflict(
                type="allergy",
                trigger=mapped,
                evidence="; ".join(evidences[:3]) if evidences else "detected",
                explanation=f"Dish may contain {mapped}, which conflicts with user allergy '{ua}'."
            ))

    disease_alias_map = {}
    for d_name, rule in disease_rules.items():
        aliases = [d_name] + (rule.get("aliases") or [])
        for a in aliases:
            if isinstance(a, str) and a.strip():
                disease_alias_map[a.strip().lower()] = d_name

    for d_in in user_diseases:
        canonical = disease_alias_map.get(d_in, disease_alias_overrides.get(d_in))
        if not canonical:
            continue

        # Retrieve the disease rule using the canonical name
        rule = disease_rules.get(canonical, {})

        avoid = {item.lower() for item in rule.get("avoid", [])}
        avoid_ing = {item.lower() for item in rule.get("avoid_ingredients", [])}

        # Check if any of the triggers or detected ingredients conflict with the disease rule
        avoid_hit = (avoid & trigger_set) | (avoid_ing & ingredients_set)
        # If there's a conflict, add it to the list of conflicts with an explanation
        if avoid_hit:
            for x in sorted(avoid_hit):
                conflicts.append(Conflict(
                    type="disease",
                    trigger=x,
                    evidence="; ".join(evidences[:3]) if evidences else "detected",
                    explanation=rule.get("notes", f"Conflict with disease rule for {canonical}.")
                ))
        # Check for caution keywords or ingredients and add notes if found
        ck = [k.lower() for k in rule.get("caution_keywords", [])]
        c_ing = {i.lower() for i in rule.get("caution_ingredients", [])}
        # If caution keywords are present in the dish name or evidences, or if caution ingredients are detected, add a note for the user
        if ck:
            combined_text = (dish_name + " " + " ".join(evidences)).lower()
            if any(k in combined_text for k in ck):
                notes.append(f"Caution for {canonical}: {rule.get('notes','')}")
        if c_ing and (c_ing & ingredients_set):
            notes.append(f"Caution for {canonical}: {rule.get('notes','')}")

    if conflicts:
        return "unsafe", conflicts, notes, max(confidence, 0.8)

    ambiguous = {"chef special", "chef's special", "special", "mixed", "assorted", "sauce", "surprise", "plat du jour", "Ã™Å Ã™Ë†Ã™â€¦Ã™Å Ã˜Â§Ã˜Âª", "Ã˜Â·Ã˜Â¨Ã™â€š Ã˜Â§Ã™â€žÃ™Å Ã™Ë†Ã™â€¦"}
    if dish_name.strip().lower() in ambiguous:
        notes.append("Dish name is ambiguous; ingredients are unclear.")
        return "risky", [], notes, confidence

    if confidence < 0.5 or ingredient_coverage < 0.4:
        notes.append("Insufficient ingredient details detected. Consider asking the restaurant or checking ingredients.")
        return "risky", [], notes, confidence

    if notes:
        return "risky", [], notes, confidence

    return "safe", [], notes, confidence
