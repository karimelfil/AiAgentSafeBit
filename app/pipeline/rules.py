from typing import Dict, List, Tuple, Optional
from app.schemas import UserProfile, Conflict


def _display_term(value: str) -> str:
    cleaned = str(value or "").replace("_", " ").strip()
    if not cleaned:
        return ""
    return " ".join(
        part.capitalize() if part.lower() not in {"and", "or", "of"} else part.lower()
        for part in cleaned.split()
    )


def _condition_label(value: str) -> str:
    low = str(value or "").strip().lower()
    labels = {
        "high cholesterol": "high cholesterol",
        "hypertension": "high blood pressure",
        "pregnancy": "pregnancy",
        "cardiovascular disease": "heart health needs",
        "lactose intolerance": "lactose intolerance",
    }
    return labels.get(low, str(value or "").strip() or "your health profile")


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
        "egg white": "egg",
        "egg yolk": "egg",
        "milk": "milk",
        "dairy": "milk",
        "casein": "milk",
        "whey": "milk",
        "butter": "milk",
        "cheese": "milk",
        "yogurt": "milk",
        "cream": "milk",
        "ghee": "milk",
        "fish": "fish",
        "salmon": "fish",
        "tuna": "fish",
        "cod": "fish",
        "sardines": "fish",
        "anchovies": "fish",
        "soy": "soy",
        "salt": "sulfites",
        "wheat": "wheat_gluten",
        "gluten": "wheat_gluten",
        "barley": "wheat_gluten",
        "rye": "wheat_gluten",
        "oats": "wheat_gluten",
        "spelt": "wheat_gluten",
        "kamut": "wheat_gluten",
        "malt": "wheat_gluten",
        "sesame": "sesame",
        "peanut": "peanut",
        "peanuts": "peanut",
        "peanut allergy": "peanut",
        "peanut butter": "peanut",
        "tree nut": "tree_nuts",
        "tree nuts": "tree_nuts",
        "treenuts": "tree_nuts",
        "almonds": "tree_nuts",
        "walnuts": "tree_nuts",
        "cashews": "tree_nuts",
        "pistachios": "tree_nuts",
        "hazelnuts": "tree_nuts",
        "pecans": "tree_nuts",
        "brazil nuts": "tree_nuts",
        "macadamia nuts": "tree_nuts",
        "pine nuts": "tree_nuts",
        "chestnuts": "tree_nuts",
        "shellfish": "shellfish",
        "crustaceans": "shellfish",
        "shrimp": "shellfish",
        "crab": "shellfish",
        "lobster": "shellfish",
        "mollusks": "molluscs",
        "molluscs": "molluscs",
        "scallops": "molluscs",
        "oysters": "molluscs",
        "clams": "molluscs",
        "mussels": "molluscs",
        "squid": "molluscs",
        "octopus": "molluscs",
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
        "sésame": "sesame",
        "arachide": "peanut",
        "cacahuete": "peanut",
        "cacahuète": "peanut",
        "fruits a coque": "tree_nuts",
        "fruits à coque": "tree_nuts",
        "moutarde": "mustard",
        "celeri": "celery",
        "céléri": "celery",
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
        "قشريات": "shellfish",
    }

    allergy_ingredient_hints = {
        "peanut": {"peanut", "peanuts", "peanut_butter"},
        "tree_nuts": {"nuts", "almond", "walnut", "cashew", "pistachio", "pecan", "hazelnut"},
        "almonds": {"almond", "almonds"},
        "walnuts": {"walnut", "walnuts"},
        "cashews": {"cashew", "cashews"},
        "pistachios": {"pistachio", "pistachios"},
        "hazelnuts": {"hazelnut", "hazelnuts"},
        "pecans": {"pecan", "pecans"},
        "brazil nuts": {"brazil nut", "brazil nuts"},
        "macadamia nuts": {"macadamia nut", "macadamia nuts"},
        "pine nuts": {"pine nut", "pine nuts"},
        "chestnuts": {"chestnut", "chestnuts"},
        "milk": {"milk", "cheese", "yogurt", "cream", "butter", "ghee", "casein", "whey"},
        "fish": {"fish", "salmon", "tuna", "cod", "sardine", "anchovy"},
        "shellfish": {"shrimp", "prawn", "crab", "lobster", "scampi"},
        "molluscs": {"mussel", "mussels", "oyster", "oysters", "clam", "clams", "squid", "octopus", "calamari", "scallop", "scallops"},
        "wheat_gluten": {"wheat", "barley", "rye", "oats", "spelt", "kamut", "malt", "flour", "bread", "pasta"},
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
            allergy_label = _display_term(ua)
            trigger_label = _display_term(mapped)
            conflicts.append(Conflict(
                type="allergy",
                trigger=mapped,
                evidence="; ".join(evidences[:3]) if evidences else "detected",
                explanation=f"This dish may contain {trigger_label}, which does not match your {allergy_label} allergy."
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
                notes.append(f"For your {_condition_label(canonical)}, this dish may need caution. {rule.get('notes','')}")
        if c_ing and (c_ing & ingredients_set):
            notes.append(f"For your {_condition_label(canonical)}, this dish may need caution. {rule.get('notes','')}")

    if conflicts:
        return "unsafe", conflicts, notes, max(confidence, 0.8)

    ambiguous = {"chef special", "chef's special", "special", "mixed", "assorted", "sauce", "surprise", "plat du jour"}
    if dish_name.strip().lower() in ambiguous:
        notes.append("The dish name is too general to identify the ingredients with confidence.")
        return "risky", [], notes, confidence

    if confidence < 0.5 or ingredient_coverage < 0.4:
        notes.append("There is not enough ingredient detail to give a fully confident recommendation. Please confirm the ingredients with the restaurant.")
        return "risky", [], notes, confidence

    if notes:
        return "risky", [], notes, confidence

    return "safe", [], notes, confidence
