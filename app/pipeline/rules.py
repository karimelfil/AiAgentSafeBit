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

    user_allergies = {a.strip().lower() for a in (profile.allergies or [])}
    user_diseases = {d.strip().lower() for d in (profile.diseases or [])}
    if profile.is_pregnant is True:
        user_diseases.add("pregnancy")
    ingredients_set = {i.strip().lower() for i in (ingredients_found or [])}

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
        "shellfish": "shellfish",
        "lait": "milk",
        "fromage": "milk",
        "beurre": "milk",
        "oeuf": "egg",
        "Å“uf": "egg",
        "oeufs": "egg",
        "poisson": "fish",
        "soja": "soy",
        "ble": "wheat_gluten",
        "blÃ©": "wheat_gluten",
        "gluten": "wheat_gluten",
        "sesame": "sesame",
        "sÃ©same": "sesame",
        "arachide": "peanut",
        "cacahuete": "peanut",
        "cacahuÃ¨te": "peanut",
        "fruits a coque": "tree_nuts",
        "fruits Ã  coque": "tree_nuts",
        "moutarde": "mustard",
        "celeri": "celery",
        "cÃ©leri": "celery",
        "Ø­Ù„ÙŠØ¨": "milk",
        "Ù„Ø¨Ù†": "milk",
        "Ø¬Ø¨Ù†": "milk",
        "Ø¨ÙŠØ¶": "egg",
        "Ø³Ù…Ùƒ": "fish",
        "ØµÙˆÙŠØ§": "soy",
        "Ù‚Ù…Ø­": "wheat_gluten",
        "ØºÙ„ÙˆØªÙŠÙ†": "wheat_gluten",
        "Ø³Ù…Ø³Ù…": "sesame",
        "ÙÙˆÙ„ Ø³ÙˆØ¯Ø§Ù†ÙŠ": "peanut",
        "Ù…ÙƒØ³Ø±Ø§Øª": "tree_nuts",
        "Ø®Ø±Ø¯Ù„": "mustard",
        "ÙƒØ±ÙØ³": "celery",
        "Ù…Ø­Ø§Ø±": "molluscs",
        "Ù‚Ø´Ø±ÙŠØ§Øª": "shellfish"
    }

    trigger_set = set(triggers)

    for ua in user_allergies:
        mapped = allergy_to_trigger.get(ua, ua)
        if mapped in trigger_set:
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

        ck = [k.lower() for k in rule.get("caution_keywords", [])]
        c_ing = set(i.lower() for i in rule.get("caution_ingredients", []))
        if ck:
            combined_text = (dish_name + " " + " ".join(evidences)).lower()
            if any(k in combined_text for k in ck):
                notes.append(f"Caution for {canonical}: {rule.get('notes','')}")
        if c_ing and (c_ing & ingredients_set):
            notes.append(f"Caution for {canonical}: {rule.get('notes','')}")

    if conflicts:
        return "RISKY", conflicts, notes, max(confidence, 0.8)

    ambiguous = {"chef special", "chef's special", "special", "mixed", "assorted", "sauce", "surprise", "plat du jour", "ÙŠÙˆÙ…ÙŠØ§Øª", "Ø·Ø¨Ù‚ Ø§Ù„ÙŠÙˆÙ…"}
    if dish_name.strip().lower() in ambiguous:
        notes.append("Dish name is ambiguous; ingredients are unclear.")
        return "CAUTION", [], notes, confidence

    if confidence < 0.5 or ingredient_coverage < 0.4:
        notes.append("Insufficient ingredient details detected. Consider asking the restaurant or checking ingredients.")
        return "CAUTION", [], notes, confidence

    return "SAFE", [], notes, confidence
