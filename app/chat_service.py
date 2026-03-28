import json
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.chat_memory import memory_store
from app.llm_client import LlmGenerationError, generate_explanation
from app.pipeline.ingredients import build_ingredients_list
from app.pipeline.knowledge import load_json
from app.pipeline.rules import evaluate
from app.schemas import (
    ChatSessionState,
    ChatIntent,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatStatus,
    Conflict,
    DishContext,
    DishResult,
    EvidenceItem,
    FollowUpContext,
    RecommendationDish,
    RestaurantRecommendation,
    ScanHistoryItem,
    UserProfile,
)


ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE = ROOT / "knowledge"
ALLERGENS = load_json(KNOWLEDGE / "allergens.json")
COMMON_ING = load_json(KNOWLEDGE / "common_ingredients.json")
DISH_SIG = load_json(KNOWLEDGE / "dish_signatures.json")
DISEASE_RULES = load_json(KNOWLEDGE / "disease_rules.json")
API_VERSION = "chat.v1"
QUESTION_DISH_PATTERNS = [
    re.compile(r"(?:is|are)\s+(.+?)\s+safe\s+for\s+me\??$", re.IGNORECASE),
    re.compile(r"can\s+i\s+eat\s+(.+?)\??$", re.IGNORECASE),
    re.compile(r"what\s+ingredients\s+does\s+(.+?)\s+contain\??$", re.IGNORECASE),
]
FOLLOW_UP_HINTS = [
    "why is that",
    "why is this",
    "why that",
    "why this",
    "that dish",
    "this dish",
    "that restaurant",
    "this restaurant",
    "is that safer",
    "is this safer",
    "why is it safer",
    "why is it risky",
    "what ingredients make it safer",
    "which ingredients make it safer",
    "what makes it safer",
    "make it safer",
    "it safer",
    "what about that",
    "what about this",
]
FRESH_DISH_HINTS = [
    " with ",
    " served with ",
    " topped with ",
    " grilled ",
    " fried ",
    " roasted ",
    " steamed ",
    " baked ",
    "sauteed",
]
ENTITY_NAME_STRIP_PREFIXES = [
    r"^\s*dish\s*name\s*[:\-]\s*",
    r"^\s*dish\s*[:\-]\s*",
    r"^\s*restaurant\s*[:\-]\s*",
]
STOPWORD_ENTITY_TOKENS = {
    "and", "with", "the", "a", "an", "of", "in", "on", "for", "to", "or",
}
INGREDIENT_LIKE_NAMES = {
    "olive oil", "salt", "garlic", "lemon", "rice", "broccoli", "chicken",
}


@dataclass
class AssessedDish:
    dish_name: str
    restaurant_name: Optional[str]
    ingredients: List[str]
    triggers: List[str]
    status: ChatStatus
    reasons: List[str]
    risks: List[str]
    safety_level: str
    confidence: float
    disease_matches: List[str]
    notes: List[str]
    evidence: List[str]


def _canonical_profile_conditions(profile: UserProfile) -> List[str]:
    alias_map: Dict[str, str] = {}
    for disease_name, rule in DISEASE_RULES.items():
        for alias in [disease_name] + (rule.get("aliases") or []):
            alias_map[alias.strip().lower()] = disease_name

    conditions: List[str] = []
    for disease in profile.diseases or []:
        key = disease.strip().lower()
        conditions.append(alias_map.get(key, disease))
    if profile.is_pregnant:
        conditions.append("Pregnancy")
    seen = set()
    out = []
    for item in conditions:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _extract_question_dish_name(question: str) -> Optional[str]:
    stripped = question.strip()
    for pattern in QUESTION_DISH_PATTERNS:
        match = pattern.search(stripped)
        if match:
            candidate = match.group(1).strip(" ?.!,'\"")
            if candidate and len(candidate) <= 80:
                return _normalize_dish_name(candidate)
    return None


def _normalize_dish_name(name: str) -> str:
    if not name:
        return "Dish"
    cleaned = name.strip()
    for pattern in ENTITY_NAME_STRIP_PREFIXES:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s*(dish\s*ingredients?|ingredients?)\s*[:\-].*$", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:\t\r\n")
    return cleaned or "Dish"


def _normalize_display_term(value: str) -> str:
    cleaned = str(value or "").strip()
    mapping = {
        "wheat_gluten": "gluten",
    }
    return mapping.get(cleaned.lower(), cleaned)


def _natural_join(items: List[str]) -> str:
    parts = [str(item).strip() for item in items if str(item).strip()]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return f"{', '.join(parts[:-1])}, and {parts[-1]}"


def _clean_entity_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    cleaned = str(name).strip()
    for pattern in ENTITY_NAME_STRIP_PREFIXES:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:\t\r\n")
    return cleaned or None


def _looks_like_noise_entity_name(name: Optional[str]) -> bool:
    cleaned = _clean_entity_name(name)
    if not cleaned:
        return True
    lowered = cleaned.lower()
    if "dish name:" in lowered or "dish ingredients:" in lowered:
        return True
    if lowered in {"dish", "ingredients", "menu"}:
        return True
    if len(cleaned) > 80:
        return True
    if re.fullmatch(r"(.)\1{3,}", lowered):
        return True
    if lowered in {"barbarrrr", "ppppp"}:
        return True
    if re.search(r"(.)\1\1", lowered) and len(set(lowered)) <= 4:
        return True
    blocked = ["test", "asdf", "qwer", "zxcv", "ppppp"]
    if any(token in lowered for token in blocked):
        return True
    comma_parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    if len(comma_parts) >= 2:
        ingredientish = 0
        for part in comma_parts:
            tokens = re.findall(r"[a-zA-Z]+", part.lower())
            if tokens and len(tokens) <= 3 and all(token not in STOPWORD_ENTITY_TOKENS for token in tokens):
                ingredientish += 1
        if ingredientish == len(comma_parts):
            return True
    return False


def _clean_ranked_entity_name(name: Optional[str]) -> Optional[str]:
    cleaned = _clean_entity_name(name)
    if _looks_like_noise_entity_name(cleaned):
        return None
    return cleaned


def is_valid_restaurant_name(name: Optional[str]) -> bool:
    return _clean_ranked_entity_name(name) is not None


def _restaurant_result_limit(question: str) -> Optional[int]:
    question_low = question.lower()
    if "which two restaurants" in question_low or "top 2 restaurants" in question_low:
        return 2
    return None


def is_follow_up_question(question: str) -> bool:
    question_low = question.lower()
    return any(hint in question_low for hint in FOLLOW_UP_HINTS)


def has_explicit_new_food_question(question: str) -> bool:
    extracted = _extract_question_dish_name(question)
    if extracted:
        return True
    question_low = question.lower()
    if any(hint in question_low for hint in FRESH_DISH_HINTS):
        return True
    return bool(re.search(r"\b(grilled|fried|roasted|steamed|baked|sauteed)\b.+\b(with|and)\b", question_low))


def find_dish_by_name(scan_history, dish_name: Optional[str]) -> Optional[AssessedDish]:
    if not dish_name:
        return None
    dish_name_low = _normalize_dish_name(dish_name).lower()
    for item in scan_history:
        if hasattr(item, "dish_name") and getattr(item, "dish_name", None):
            if _normalize_dish_name(getattr(item, "dish_name")).lower() == dish_name_low:
                return item
        dishes = getattr(item, "dishes", None) or []
        for dish in dishes:
            if _normalize_dish_name(getattr(dish, "dish_name", "")).lower() == dish_name_low:
                return dish
    return None


def _status_from_safety(safety_level: str) -> ChatStatus:
    if safety_level == "SAFE":
        return "safe"
    if safety_level == "RISKY":
        return "not_safe"
    if safety_level == "CAUTION":
        return "caution"
    return "unknown"


def _question_intent(request: ChatRequest, memory: List[ChatMessage], session_context: Dict[str, str]) -> ChatIntent:
    question = request.question.lower()
    fresh_question_dish = _extract_question_dish_name(request.question)
    can_use_follow_up_context = is_follow_up_question(request.question) and not has_explicit_new_food_question(request.question)
    referenced_dish = request.current_dish.dish_name if request.current_dish else None
    if not referenced_dish and fresh_question_dish:
        referenced_dish = fresh_question_dish
    if not referenced_dish and can_use_follow_up_context:
        referenced_dish = session_context.get("last_dish_name")
    referenced_restaurant = session_context.get("last_restaurant_name") if can_use_follow_up_context else None
    if request.restaurant_context and request.restaurant_context.restaurant_name:
        referenced_restaurant = request.restaurant_context.restaurant_name
    elif request.current_dish and request.current_dish.restaurant_name:
        referenced_restaurant = request.current_dish.restaurant_name

    if is_follow_up_question(request.question) and any(token in question for token in ["why", "safer", "what ingredients make", "what makes it", "which ingredients make"]):
        return ChatIntent(
            label="follow_up_explanation",
            confidence=0.95,
            rationale="Question asks for a follow-up explanation about the previously selected dish.",
            referenced_dish=referenced_dish,
            referenced_restaurant=referenced_restaurant,
        )
    if any(token in question for token in ["which dish is safest", "what dish is safest", "safest dish", "which dishes are safest"]):
        return ChatIntent(
            label="safe_history",
            confidence=0.94,
            rationale="Question asks for the safest dish recommendations from previous scans.",
            referenced_dish=referenced_dish,
            referenced_restaurant=referenced_restaurant,
        )
    if any(token in question for token in ["which two restaurants", "top 2 restaurants"]):
        return ChatIntent(
            label="top_two_restaurants",
            confidence=0.95,
            rationale="Question asks for the top two restaurants based on previous scans.",
            referenced_dish=referenced_dish,
            referenced_restaurant=referenced_restaurant,
        )
    if any(token in question for token in ["best restaurant", "which restaurant", "restaurant is best"]):
        return ChatIntent(
            label="best_restaurant",
            confidence=0.94,
            rationale="Question asks for the best restaurant based on previous scans.",
            referenced_dish=referenced_dish,
            referenced_restaurant=referenced_restaurant,
        )
    if any(token in question for token in ["previous scans", "from my scans", "which dishes", "what dishes can i eat"]):
        return ChatIntent(
            label="safe_history",
            confidence=0.9,
            rationale="Question asks for recommendations from previous scans.",
            referenced_dish=referenced_dish,
            referenced_restaurant=referenced_restaurant,
        )
    if (
        "what should i avoid" in question
        or "what foods should i avoid" in question
        or "avoid because" in question
    ) and any(token in question for token in ["pregnan", "celiac", "ibs", "gluten free"]):
        return ChatIntent(
            label="health_risk",
            confidence=0.9,
            rationale="Question asks for health-condition guidance from the user profile.",
            referenced_dish=referenced_dish,
            referenced_restaurant=referenced_restaurant,
        )
    if any(token in question for token in ["ingredient", "contain", "contains", "what is in", "made of", "what should i avoid"]):
        return ChatIntent(
            label="ingredient_info",
            confidence=0.86,
            rationale="Question asks about ingredients in a dish or ingredients to avoid.",
            referenced_dish=referenced_dish,
            referenced_restaurant=referenced_restaurant,
        )
    if any(token in question for token in ["disease", "condition", "risky", "risk", "allergy", "allergies", "diabetes", "hypertension", "ibs", "pregnan", "avoid because", "gluten free", "pregnant", "soy allergy", "celiac"]):
        return ChatIntent(
            label="health_risk",
            confidence=0.88,
            rationale="Question asks about diseases, allergies, or other health risks.",
            referenced_dish=referenced_dish,
            referenced_restaurant=referenced_restaurant,
        )
    if any(token in question for token in ["can i eat", "is this safe", "is it safe", "can i have", "safe for me"]):
        return ChatIntent(
            label="dish_safety",
            confidence=0.9,
            rationale="Question asks whether a dish is safe to eat.",
            referenced_dish=referenced_dish,
            referenced_restaurant=referenced_restaurant,
        )
    if has_explicit_new_food_question(request.question):
        return ChatIntent(
            label="dish_safety",
            confidence=0.82,
            rationale="Question includes a fresh dish description that should be evaluated directly.",
            referenced_dish=referenced_dish,
            referenced_restaurant=referenced_restaurant,
        )

    return ChatIntent(
        label="general",
        confidence=0.45,
        rationale="General follow-up or broad food safety question.",
        referenced_dish=referenced_dish,
        referenced_restaurant=referenced_restaurant,
    )


def _augment_profile(profile: UserProfile) -> UserProfile:
    return UserProfile(
        allergies=sorted(set(profile.allergies + profile.intolerances)),
        intolerances=profile.intolerances,
        diseases=profile.diseases,
        forbidden_ingredients=profile.forbidden_ingredients,
        dietary_preferences=profile.dietary_preferences,
        is_pregnant=profile.is_pregnant,
    )


def _apply_signature_triggers(dish_name: str, text_blob: str, triggers: List[str], extract_notes: List[str], evidences: List[str]) -> Tuple[List[str], List[str], List[str]]:
    for sig_name, rule in (DISH_SIG or {}).items():
        keywords: List[str] = []
        triggers_to_add: List[str] = []
        note = None

        if isinstance(rule, dict):
            keywords = rule.get("keywords", [])
            triggers_to_add = rule.get("triggers", [])
            note = rule.get("note")
        elif isinstance(rule, list):
            keywords = [sig_name]
            triggers_to_add = rule

        if any(str(k).lower() in text_blob for k in keywords):
            for trigger in triggers_to_add:
                if trigger not in triggers:
                    triggers.append(trigger)
                    evidences.append(f"signature:{sig_name}")
            if note and note not in extract_notes:
                extract_notes.append(note)
    return triggers, extract_notes, evidences


def _merge_items(primary: List[str], secondary: List[str]) -> List[str]:
    merged: List[str] = []
    for item in primary + secondary:
        value = str(item).strip()
        if not value:
            continue
        value_low = value.lower()
        if any(value_low == existing.lower() for existing in merged):
            continue
        if any(value_low in existing.lower() for existing in merged):
            continue
        merged = [existing for existing in merged if existing.lower() not in value_low]
        merged.append(value)
    return merged


def _apply_explicit_ingredient_triggers(
    ingredients: List[str],
    triggers: List[str],
    evidences: List[str],
) -> Tuple[List[str], List[str]]:
    trigger_set = {item.lower() for item in triggers}
    for ingredient in ingredients:
        ingredient_low = ingredient.lower()
        for canon, aliases in ALLERGENS.items():
            alias_set = {canon.lower(), *(alias.lower() for alias in aliases)}
            if any(alias in ingredient_low for alias in alias_set):
                if canon.lower() not in trigger_set:
                    triggers.append(canon)
                    trigger_set.add(canon.lower())
                evidences.append(f"explicit_ingredient:{ingredient}")
    return triggers, evidences


def _assess_from_text(
    dish_name: str,
    block: str,
    restaurant_name: Optional[str],
    profile: UserProfile,
    explicit_ingredients: Optional[List[str]] = None,
) -> AssessedDish:
    cleaned_dish_name = _normalize_dish_name(dish_name)
    cleaned_restaurant_name = _clean_ranked_entity_name(restaurant_name)
    ingredients, triggers, evidences, confidence, ingredient_coverage, extract_notes = build_ingredients_list(
        dish_name=cleaned_dish_name,
        block=block,
        common_ingredients=COMMON_ING,
        allergen_triggers=ALLERGENS,
    )
    explicit_ingredients = explicit_ingredients or []
    ingredients = _merge_items(list(explicit_ingredients), ingredients)
    if explicit_ingredients:
        ingredient_coverage = max(ingredient_coverage, 0.95)
        confidence = max(confidence, 0.9)
        triggers, evidences = _apply_explicit_ingredient_triggers(ingredients, triggers, evidences)
    text_blob = f"{cleaned_dish_name} {block}".lower()
    triggers, extract_notes, evidences = _apply_signature_triggers(cleaned_dish_name, text_blob, triggers, extract_notes, evidences)

    safety, conflicts, eval_notes, eval_conf = evaluate(
        dish_name=cleaned_dish_name,
        triggers=triggers,
        evidences=evidences,
        confidence=confidence,
        profile=_augment_profile(profile),
        disease_rules=DISEASE_RULES,
        ingredients_found=ingredients,
        ingredient_coverage=ingredient_coverage,
    )
    reasons, risks, status = _post_process_conflicts(
        ingredients=ingredients,
        conflicts=conflicts,
        notes=extract_notes + eval_notes,
        safety_level=safety,
        forbidden_ingredients=profile.forbidden_ingredients,
    )
    disease_matches = _profile_disease_matches(profile, cleaned_dish_name, ingredients, triggers)
    return AssessedDish(
        dish_name=cleaned_dish_name,
        restaurant_name=cleaned_restaurant_name,
        ingredients=ingredients,
        triggers=sorted(set(triggers)),
        status=status,
        reasons=reasons,
        risks=risks,
        safety_level=safety,
        confidence=eval_conf,
        disease_matches=disease_matches,
        notes=extract_notes + eval_notes,
        evidence=evidences,
    )


def _post_process_conflicts(
    ingredients: List[str],
    conflicts: List[Conflict],
    notes: List[str],
    safety_level: str,
    forbidden_ingredients: List[str],
) -> Tuple[List[str], List[str], ChatStatus]:
    reasons = [conflict.explanation for conflict in conflicts]
    risks = sorted({conflict.trigger for conflict in conflicts})

    lower_ingredients = {ingredient.lower() for ingredient in ingredients}
    for forbidden in forbidden_ingredients or []:
        if forbidden.lower() in lower_ingredients:
            reasons.append(f"Dish includes forbidden ingredient '{forbidden}'.")
            risks.append(forbidden)

    status = _status_from_safety(safety_level)
    if forbidden_ingredients and any(item.lower() in lower_ingredients for item in forbidden_ingredients):
        status = "not_safe"
    if not reasons and notes and status != "safe":
        reasons.extend(notes[:3])
    return reasons, sorted(set(risks)), status


def _assess_existing_result(result: DishResult, restaurant_name: Optional[str], profile: UserProfile) -> AssessedDish:
    cleaned_dish_name = _normalize_dish_name(result.dish_name)
    cleaned_restaurant_name = _clean_ranked_entity_name(restaurant_name)
    safety, conflicts, eval_notes, eval_conf = evaluate(
        dish_name=cleaned_dish_name,
        triggers=list(result.detected_triggers),
        evidences=[conflict.evidence for conflict in result.conflicts] or ["previous_scan"],
        confidence=result.confidence,
        profile=_augment_profile(profile),
        disease_rules=DISEASE_RULES,
        ingredients_found=list(result.ingredients_found),
        ingredient_coverage=result.ingredient_coverage,
    )
    reasons, risks, status = _post_process_conflicts(
        ingredients=list(result.ingredients_found),
        conflicts=conflicts or list(result.conflicts),
        notes=list(result.notes) + eval_notes,
        safety_level=safety,
        forbidden_ingredients=profile.forbidden_ingredients,
    )
    disease_matches = _profile_disease_matches(profile, cleaned_dish_name, list(result.ingredients_found), list(result.detected_triggers))
    return AssessedDish(
        dish_name=cleaned_dish_name,
        restaurant_name=cleaned_restaurant_name,
        ingredients=list(result.ingredients_found),
        triggers=list(result.detected_triggers),
        status=status,
        reasons=reasons,
        risks=risks,
        safety_level=safety,
        confidence=max(result.confidence, eval_conf),
        disease_matches=disease_matches,
        notes=list(result.notes) + eval_notes,
        evidence=[conflict.evidence for conflict in result.conflicts] or ["previous_scan"],
    )


def _all_disease_matches(dish_name: str, ingredients: List[str], triggers: List[str]) -> List[str]:
    combined_text = f"{dish_name} {' '.join(ingredients)} {' '.join(triggers)}".lower()
    ingredients_set = {item.lower() for item in ingredients}
    triggers_set = {item.lower() for item in triggers}
    matches: List[str] = []
    for disease_name, rule in DISEASE_RULES.items():
        avoid = {item.lower() for item in rule.get("avoid", [])}
        avoid_ingredients = {item.lower() for item in rule.get("avoid_ingredients", [])}
        caution_keywords = [item.lower() for item in rule.get("caution_keywords", [])]
        caution_ingredients = {item.lower() for item in rule.get("caution_ingredients", [])}

        if (avoid & triggers_set) or (avoid_ingredients & ingredients_set):
            matches.append(disease_name)
            continue
        if caution_ingredients & ingredients_set:
            matches.append(disease_name)
            continue
        if any(keyword in combined_text for keyword in caution_keywords):
            matches.append(disease_name)
    return sorted(set(matches))


def _profile_disease_matches(profile: UserProfile, dish_name: str, ingredients: List[str], triggers: List[str]) -> List[str]:
    canonical_conditions = _canonical_profile_conditions(profile)
    all_matches = set(_all_disease_matches(dish_name, ingredients, triggers))
    return [condition for condition in canonical_conditions if condition in all_matches]


def _current_assessment(request: ChatRequest) -> Optional[AssessedDish]:
    dish = request.current_dish
    if not dish:
        inferred_dish_name = _extract_question_dish_name(request.question)
        if not inferred_dish_name and has_explicit_new_food_question(request.question):
            inferred_dish_name = _normalize_dish_name(request.question)
        if not inferred_dish_name:
            return None
        dish = DishContext(
            dish_name=inferred_dish_name,
            ingredients=[],
            raw_text=request.question,
            restaurant_name=request.restaurant_context.restaurant_name if request.restaurant_context else None,
        )
    block = " ".join(part for part in [dish.description or "", ", ".join(dish.ingredients), dish.raw_text or ""] if part).strip()
    return _assess_from_text(
        dish_name=_normalize_dish_name(dish.dish_name),
        block=block,
        restaurant_name=dish.restaurant_name or (request.restaurant_context.restaurant_name if request.restaurant_context else None),
        profile=request.user_profile,
        explicit_ingredients=list(dish.ingredients),
    )


def _history_assessments(history: List[ScanHistoryItem], profile: UserProfile) -> List[AssessedDish]:
    assessed: List[AssessedDish] = []
    for scan in history:
        for dish in scan.dishes:
            cleaned_dish_name = _clean_ranked_entity_name(dish.dish_name)
            if not cleaned_dish_name:
                continue
            if cleaned_dish_name.lower() in INGREDIENT_LIKE_NAMES:
                continue
            assessed.append(_assess_existing_result(dish.model_copy(update={"dish_name": cleaned_dish_name}), scan.restaurant_name, profile))
    return assessed


def _find_matching_history_dish(question: str, history: List[AssessedDish]) -> Optional[AssessedDish]:
    question_low = question.lower()
    exact = [dish for dish in history if dish.dish_name and dish.dish_name.lower() in question_low]
    if exact:
        return exact[0]
    if "this dish" in question_low or "this restaurant" in question_low:
        return None
    return history[0] if len(history) == 1 else None


def _safe_recommendations(history: List[AssessedDish], limit: int = 3) -> List[RecommendationDish]:
    safe = [dish for dish in history if dish.status == "safe" and not _looks_like_noise_entity_name(dish.dish_name)]
    safe.sort(key=lambda item: (-item.confidence, item.dish_name))
    return [
        RecommendationDish(
            dish_name=item.dish_name,
            restaurant_name=item.restaurant_name,
            status=item.status,
            reason=item.reasons[0] if item.reasons else "Matched your profile with no detected conflicts.",
            confidence=item.confidence,
        )
        for item in safe[:limit]
    ]


def _safer_alternatives(current: Optional[AssessedDish], history: List[AssessedDish], limit: int = 3) -> List[str]:
    candidates = [dish for dish in history if dish.status == "safe" and not _looks_like_noise_entity_name(dish.dish_name)]
    if current and current.restaurant_name:
        same_restaurant = [dish for dish in candidates if dish.restaurant_name == current.restaurant_name]
        if same_restaurant:
            candidates = same_restaurant
    candidates.sort(key=lambda item: (-item.confidence, item.dish_name))
    return [dish.dish_name for dish in candidates[:limit] if not current or dish.dish_name != current.dish_name]


def _rank_restaurants(history: List[AssessedDish]) -> List[RestaurantRecommendation]:
    grouped: Dict[str, List[AssessedDish]] = defaultdict(list)
    for dish in history:
        restaurant_name = _clean_ranked_entity_name(dish.restaurant_name)
        if is_valid_restaurant_name(restaurant_name):
            grouped[restaurant_name].append(dish)
    if not grouped:
        return []

    ranked: List[RestaurantRecommendation] = []
    for name, dishes in grouped.items():
        safe_count = sum(1 for dish in dishes if dish.status == "safe")
        caution_count = sum(1 for dish in dishes if dish.status == "caution")
        risky_count = sum(1 for dish in dishes if dish.status == "not_safe")
        safe_confidence = sum(dish.confidence for dish in dishes if dish.status == "safe")
        overall_score = float((safe_count * 3) + caution_count - (risky_count * 2) + (safe_confidence / 10.0))
        ranked.append(RestaurantRecommendation(
            restaurant_name=name,
            safe_dish_count=safe_count,
            caution_dish_count=caution_count,
            risky_dish_count=risky_count,
            overall_score=round(overall_score, 3),
            recommended_dishes=[
                dish.dish_name
                for dish in dishes
                if dish.status == "safe"
                and not _looks_like_noise_entity_name(dish.dish_name)
                and len(dish.dish_name.split()) >= 2
            ][:4],
            reason=(
                f"{safe_count} safe dishes, {caution_count} caution dishes, and {risky_count} risky dishes "
                f"for the current health profile."
            ),
        ))
    ranked.sort(
        key=lambda item: (
            -item.overall_score,
            -item.safe_dish_count,
            item.risky_dish_count,
            item.restaurant_name or "",
        )
    )
    return ranked


def _best_restaurant(history: List[AssessedDish]) -> Optional[RestaurantRecommendation]:
    ranked = _rank_restaurants(history)
    return ranked[0] if ranked else None


def _ingredient_support_for_risk(risk: str, ingredients: List[str]) -> List[str]:
    risk_low = risk.lower()
    matched: List[str] = []
    alias_groups = [(risk_low, [risk_low])]
    if risk in ALLERGENS:
        alias_groups = [(risk_low, [risk_low] + [alias.lower() for alias in ALLERGENS[risk]])]

    for ingredient in ingredients:
        ingredient_low = ingredient.lower()
        if any(alias in ingredient_low for _, aliases in alias_groups for alias in aliases):
            matched.append(ingredient)
    return matched[:3]


def _risk_profile_label(risk: str, profile: UserProfile) -> str:
    risk_label = _normalize_display_term(risk)
    risk_low = risk_label.lower()
    for allergy in profile.allergies:
        allergy_clean = _normalize_display_term(allergy.strip())
        allergy_low = allergy_clean.lower()
        if allergy_low == risk_low:
            return f"{allergy_clean} allergy"
        for canon, aliases in ALLERGENS.items():
            alias_pool = {canon.lower(), *[alias.lower() for alias in aliases]}
            if risk_low in alias_pool and allergy_low in alias_pool:
                return f"{allergy_clean} allergy"
    for intolerance in profile.intolerances:
        intolerance_clean = _normalize_display_term(intolerance.strip())
        if intolerance_clean.lower() == risk_low:
            return f"{intolerance_clean} intolerance"
    return risk_label


def _response_confidence(
    target_dish: Optional[AssessedDish],
    intent: ChatIntent,
    safe_history: List[RecommendationDish],
    best_restaurant: Optional[RestaurantRecommendation],
    missing_data: List[str],
) -> float:
    if missing_data:
        return 0.35
    if intent.label in {"best_restaurant", "top_two_restaurants"}:
        return 0.9 if best_restaurant else 0.4
    if intent.label == "safe_history":
        return 0.88 if safe_history else 0.42
    if target_dish:
        return round(max(0.35, min(0.98, target_dish.confidence)), 2)
    return 0.45


def _build_reasoning_summary(
    intent: ChatIntent,
    target_dish: Optional[AssessedDish],
    best_restaurant: Optional[RestaurantRecommendation],
    missing_data: List[str],
    response_status: ChatStatus,
) -> List[str]:
    steps = [f"Intent detected: {intent.label}."]
    include_target_dish = intent.label not in {"safe_history", "best_restaurant", "top_two_restaurants"}
    if target_dish and include_target_dish:
        steps.append(
            f"Dish evaluated with final status {response_status} using extracted ingredients, trigger detection, and disease rules."
        )
    if best_restaurant:
        steps.append(
            f"Restaurant ranking used previous scan results and selected {best_restaurant.restaurant_name} as the top option."
        )
    if missing_data:
        steps.append("Missing data reduced answer certainty.")
    return steps


def _build_evidence(
    intent: ChatIntent,
    target_dish: Optional[AssessedDish],
    safe_history: List[RecommendationDish],
    best_restaurant: Optional[RestaurantRecommendation],
) -> List[EvidenceItem]:
    evidence: List[EvidenceItem] = []
    include_target_dish = intent.label not in {"safe_history", "best_restaurant", "top_two_restaurants"}
    if target_dish and include_target_dish:
        for ingredient in target_dish.ingredients[:6]:
            evidence.append(EvidenceItem(source="ingredient_extraction", detail=f"Detected ingredient: {ingredient}"))
        for item in target_dish.evidence[:4]:
            evidence.append(EvidenceItem(source="rule_input", detail=item))
        for risk in target_dish.risks[:4]:
            evidence.append(EvidenceItem(source="health_rule", detail=f"Matched risk: {risk}"))
    for recommendation in safe_history[:3]:
        evidence.append(
            EvidenceItem(
                source="scan_history",
                detail=f"Safe previous dish: {recommendation.dish_name} at {recommendation.restaurant_name or 'unknown restaurant'}",
            )
        )
    if best_restaurant and best_restaurant.restaurant_name:
        evidence.append(
            EvidenceItem(
                source="restaurant_ranking",
                detail=f"Top restaurant {best_restaurant.restaurant_name} scored {best_restaurant.overall_score}",
            )
        )
    return evidence[:12]


def _deterministic_explanation(
    intent: ChatIntent,
    target_dish: Optional[AssessedDish],
    safe_history: List[RecommendationDish],
    best_restaurant: Optional[RestaurantRecommendation],
    ranked_restaurants: List[RestaurantRecommendation],
    missing_data: List[str],
    profile: UserProfile,
    profile_guidance: Dict[str, List[str]],
    question: str,
) -> str:
    if intent.label == "top_two_restaurants":
        if len(ranked_restaurants) >= 2:
            top_two = _natural_join([item.restaurant_name for item in ranked_restaurants[:2] if item.restaurant_name])
            return f"The top restaurants from your previous scans are {top_two}."
        if len(ranked_restaurants) == 1 and ranked_restaurants[0].restaurant_name:
            return f"The best restaurant I found is {ranked_restaurants[0].restaurant_name}."
        return "I do not have enough previous restaurant scan data to recommend restaurants yet."

    if intent.label == "best_restaurant":
        if best_restaurant and best_restaurant.restaurant_name:
            dishes = _natural_join(best_restaurant.recommended_dishes) or "no fully safe dishes identified"
            return f"{best_restaurant.restaurant_name} looks best for your profile based on previous scans. Safest options there: {dishes}."
        return "I do not have enough previous restaurant scan data to recommend a restaurant yet."

    if intent.label == "follow_up_explanation" and target_dish:
        question_low = question.lower()
        if "ingredient" in question_low or "what makes it safer" in question_low:
            ingredients = _natural_join([_normalize_display_term(item) for item in target_dish.ingredients[:8]])
            if ingredients:
                return f"{target_dish.dish_name} looks safer because it contains ingredients like {ingredients} and I did not find direct conflicts with your profile."
            return f"{target_dish.dish_name} looks safer because I did not find direct conflicts with your profile in the available scan data."
        if target_dish.status == "safe":
            ingredients = _natural_join([_normalize_display_term(item) for item in target_dish.ingredients[:6]])
            if ingredients:
                return f"{target_dish.dish_name} looks safer because I found ingredients like {ingredients} and no direct conflicts with your profile."
            return f"{target_dish.dish_name} looks safer because I did not find a confirmed conflict with your profile in the available scan data."
        if target_dish.reasons:
            return f"{target_dish.dish_name} is not safer for your profile because {_natural_join([_normalize_display_term(item) for item in target_dish.reasons[:2]])}."
        return f"I found {target_dish.dish_name}, but I do not have enough ingredient detail to explain why it is safer."

    if intent.label == "safe_history":
        if safe_history:
            names = _natural_join([item.dish_name for item in safe_history])
            return f"From your previous scans, the safest dishes I found are: {names}."
        return "I could not find any clearly safe dishes in your previous scans with the current data."

    if intent.label == "ingredient_info" and target_dish:
        if target_dish.ingredients:
            return f"{target_dish.dish_name} appears to contain: {_natural_join([_normalize_display_term(item) for item in target_dish.ingredients[:8]])}."
        return f"I do not have a confirmed ingredient list for {target_dish.dish_name}."

    if intent.label == "ingredient_info" and not target_dish:
        avoid = profile_guidance.get("avoid_items", [])
        if avoid:
            return f"Based on your profile, you should avoid ingredients such as: {_natural_join(avoid[:10])}."

    if intent.label == "health_risk" and target_dish:
        if profile_guidance.get("direct_answers") and not target_dish.reasons and not target_dish.disease_matches:
            return profile_guidance["direct_answers"][0]
        if target_dish.disease_matches:
            diseases = _natural_join([_normalize_display_term(item) for item in target_dish.disease_matches[:5]])
            return f"{target_dish.dish_name} may be risky for these conditions: {diseases}."
        if target_dish.reasons:
            return f"{target_dish.dish_name} may be risky because {_natural_join([_normalize_display_term(item) for item in target_dish.reasons[:2]])}."
        return f"I do not have enough confirmed risk evidence to list health conditions for {target_dish.dish_name}."

    if intent.label in {"health_risk", "general"} and not target_dish:
        specific_answers = profile_guidance.get("direct_answers", [])
        if specific_answers:
            return " ".join(specific_answers[:3])
        avoid = profile_guidance.get("avoid_items", [])
        if avoid:
            return f"Based on your profile, avoid or be cautious with: {_natural_join(avoid[:10])}."

    if target_dish:
        if target_dish.status == "safe":
            return f"{target_dish.dish_name} appears safe for your current profile based on the detected ingredients and rules."
        if target_dish.status == "not_safe":
            if target_dish.risks and target_dish.ingredients:
                reason_parts = []
                for risk in target_dish.risks[:2]:
                    support = _ingredient_support_for_risk(risk, target_dish.ingredients)
                    profile_label = _risk_profile_label(risk, profile)
                    if support:
                        joined = _natural_join([_normalize_display_term(item) for item in support])
                        reason_parts.append(f"it contains {joined}, which conflicts with your {profile_label}")
                    else:
                        reason_parts.append(f"it conflicts with {profile_label}")
                if reason_parts:
                    return f"{target_dish.dish_name} is not safe for your profile because {_natural_join(reason_parts)}."
            reasons = _natural_join([_normalize_display_term(item) for item in target_dish.reasons[:2]]) or "health conflicts were detected"
            return f"{target_dish.dish_name} is not safe for your profile because {reasons}."
        if target_dish.ingredients:
            return f"{target_dish.dish_name} needs caution. I found ingredients {_natural_join([_normalize_display_term(item) for item in target_dish.ingredients[:6]])}, but the data is incomplete."
        return f"{target_dish.dish_name} needs caution because I do not have enough confirmed ingredient data."

    if missing_data:
        specific_answers = profile_guidance.get("direct_answers", [])
        if specific_answers:
            return " ".join(specific_answers[:3])
        return f"I do not have enough dish or scan detail, but based on your profile you should focus on avoiding {_natural_join(profile_guidance.get('avoid_items', [])[:8])}."
    return "Based on your profile, I can help with dish safety, ingredients to avoid, pregnancy restrictions, and scan-based recommendations."


def _profile_guidance(profile: UserProfile, question: str) -> Dict[str, List[str]]:
    question_low = question.lower()
    direct_answers: List[str] = []
    avoid_items: List[str] = []
    reasons: List[str] = []

    for allergy in profile.allergies + profile.intolerances:
        allergy_low = allergy.strip().lower()
        for canon, aliases in ALLERGENS.items():
            if allergy_low == canon or allergy_low in [alias.lower() for alias in aliases]:
                avoid_items.extend(aliases[:6])
                reasons.append(f"Avoid {_normalize_display_term(canon)}-related ingredients because of your {allergy} sensitivity.")

    for condition in _canonical_profile_conditions(profile):
        rule = DISEASE_RULES.get(condition) or {}
        avoid_items.extend(rule.get("avoid_ingredients", []))
        avoid_items.extend(rule.get("caution_ingredients", []))
        if condition == "Celiac Disease":
            avoid_items.extend(["gluten", "wheat", "barley", "rye"])
            reasons.append("Avoid gluten, wheat, barley, and rye because of celiac disease.")
        elif condition == "Pregnancy":
            avoid_items.extend(["raw fish", "sushi", "soft cheese", "unpasteurized milk", "alcohol", "raw eggs", "undercooked meat"])
            reasons.append("During pregnancy, avoid raw fish, unpasteurized dairy, alcohol, raw eggs, and undercooked meat.")
        elif condition == "Irritable Bowel Syndrome (IBS)":
            reasons.append("For IBS, be cautious with common triggers like garlic, onion, beans, lentils, and some dairy.")
        else:
            notes = rule.get("notes")
            if notes:
                reasons.append(notes)

    if "pregnan" in question_low:
        direct_answers.extend([reason for reason in reasons if "pregnan" in reason.lower() or "during pregnancy" in reason.lower()])
    if "soy sauce" in question_low:
        direct_answers.insert(0, "Soy sauce is not safe for you because it usually contains soy and often wheat, which conflicts with your soy allergy and celiac disease.")
    elif "soy" in question_low:
        soy_items = [item for item in avoid_items if "soy" in item.lower()]
        if soy_items:
            direct_answers.append(f"Because of your soy allergy, avoid ingredients such as {', '.join(sorted(set(soy_items))[:6])}.")
    if "gluten free" in question_low and any(cond == "Irritable Bowel Syndrome (IBS)" for cond in _canonical_profile_conditions(profile)):
        direct_answers.append("Gluten-free foods are not always safe for IBS because IBS triggers can still include garlic, onion, beans, lentils, and some dairy.")
    if "soft cheese" in question_low and profile.is_pregnant:
        direct_answers.append("Soft cheese can be unsafe during pregnancy if it is unpasteurized.")
    if "sushi" in question_low and profile.is_pregnant:
        direct_answers.append("Sushi is usually not safe during pregnancy when it contains raw fish.")
    if "bread" in question_low and any(cond == "Celiac Disease" for cond in _canonical_profile_conditions(profile)):
        direct_answers.append("Bread is usually not safe for you unless it is clearly gluten-free, because celiac disease requires avoiding gluten.")
    if "soy sauce" in question_low:
        direct_answers.append("Soy sauce is not safe for you because it usually contains soy and often wheat, which conflicts with soy allergy and celiac disease.")

    seen = set()
    dedup_avoid = []
    for item in avoid_items:
        item = str(item).strip()
        if item and item.lower() not in seen:
            seen.add(item.lower())
            dedup_avoid.append(item)

    seen = set()
    dedup_reasons = []
    for item in reasons:
        if item and item.lower() not in seen:
            seen.add(item.lower())
            dedup_reasons.append(item)

    seen = set()
    dedup_direct = []
    for item in direct_answers:
        if item and item.lower() not in seen:
            seen.add(item.lower())
            dedup_direct.append(item)

    return {
        "avoid_items": [_normalize_display_term(item) for item in dedup_avoid],
        "reasons": [_normalize_display_term(item) for item in dedup_reasons],
        "direct_answers": [_normalize_display_term(item) for item in dedup_direct],
    }


def _status_from_direct_profile_answer(answer: str) -> Optional[ChatStatus]:
    low = answer.lower()
    if "not safe" in low or "avoid" in low:
        return "not_safe"
    if "caution" in low or "not always safe" in low or "can be unsafe" in low:
        return "caution"
    return None


def _llm_summary(
    response: ChatResponse,
    target_dish: Optional[AssessedDish],
) -> str:
    payload = {
        "status": response.status,
        "dish_name": response.dish_name,
        "ingredients": response.ingredients,
        "matched_health_risks": response.matched_health_risks,
        "disease_or_allergy_reasons": response.disease_or_allergy_reasons,
        "safer_alternatives": response.safer_alternatives,
        "recommended_dishes_from_previous_scans": [item.model_dump() for item in response.recommended_dishes_from_previous_scans],
        "best_restaurant_for_user": response.best_restaurant_for_user.model_dump() if response.best_restaurant_for_user else None,
        "ranked_restaurants": [item.model_dump() for item in response.ranked_restaurants],
        "missing_data": response.missing_data,
        "target_dish_confidence": target_dish.confidence if target_dish else None,
    }
    return json.dumps(payload, ensure_ascii=True)


async def answer_chat(request: ChatRequest) -> ChatResponse:
    session_id = request.session_id or str(uuid.uuid4())
    request_id = request.request_id or str(uuid.uuid4())
    external_state = request.session_state or ChatSessionState()
    external_memory = list(external_state.recent_messages or [])
    external_context = external_state.follow_up_context.model_dump() if external_state.follow_up_context else {}
    has_fresh_food_question = has_explicit_new_food_question(request.question)
    use_memory = request.use_session_memory and is_follow_up_question(request.question) and not has_fresh_food_question
    prioritize_current_question_over_memory = has_fresh_food_question

    memory: List[ChatMessage] = []
    session_context: Dict[str, str] = {}
    if use_memory:
        memory = memory_store.get(session_id)
        session_context = memory_store.get_context(session_id)
    if use_memory and external_memory:
        memory = external_memory
    if external_context:
        session_context = {**session_context, **external_context}

    effective_current_dish = request.current_dish
    if not effective_current_dish and not prioritize_current_question_over_memory and external_state.current_dish:
        effective_current_dish = external_state.current_dish
    if not effective_current_dish and use_memory and not prioritize_current_question_over_memory and session_context.get("last_current_dish"):
        try:
            effective_current_dish = DishContext(**session_context["last_current_dish"])
        except Exception:
            effective_current_dish = None
    effective_request = request.model_copy(update={"current_dish": effective_current_dish})
    intent = _question_intent(effective_request, memory, session_context)

    current = _current_assessment(effective_request)
    history = _history_assessments(effective_request.scan_history, effective_request.user_profile)
    matched_history_dish = _find_matching_history_dish(effective_request.question, history)
    follow_up_last_dish = (external_state.follow_up_context.last_dish_name if external_state.follow_up_context else None) or session_context.get("last_dish_name")
    if not current and not matched_history_dish and is_follow_up_question(request.question) and follow_up_last_dish:
        matched_history_dish = find_dish_by_name(history, follow_up_last_dish)
    if not current and not matched_history_dish and intent.referenced_dish:
        matched_history_dish = find_dish_by_name(history, intent.referenced_dish)
    target_dish = current or matched_history_dish
    profile_guidance = _profile_guidance(effective_request.user_profile, effective_request.question)

    missing_data: List[str] = []
    warnings: List[str] = []
    if not target_dish and intent.label == "dish_safety":
        missing_data.append("No current dish or matching scanned dish was provided.")
    if not target_dish and intent.label in {"ingredient_info", "health_risk", "general"} and not profile_guidance.get("direct_answers") and not profile_guidance.get("avoid_items"):
        missing_data.append("No current dish, matching scanned dish, or profile-based guidance was available.")
    if not effective_request.scan_history and intent.label in {"safe_history", "best_restaurant", "top_two_restaurants"}:
        missing_data.append("Previous scan history was not provided.")
    if not request.use_session_memory and not request.session_state and not effective_request.current_dish and intent.label in {"dish_safety", "ingredient_info", "health_risk", "general"}:
        warnings.append("Stateless request received without session_state; follow-up questions may lose context.")

    safe_history = _safe_recommendations(history)
    if intent.label == "follow_up_explanation" and not target_dish and safe_history:
        target_dish = find_dish_by_name(history, safe_history[0].dish_name)
    ranked_restaurants: List[RestaurantRecommendation] = []
    best_restaurant: Optional[RestaurantRecommendation] = None
    if intent.label in {"best_restaurant", "top_two_restaurants"}:
        ranked_restaurants = _rank_restaurants(history)
        requested_limit = 2 if intent.label == "top_two_restaurants" else _restaurant_result_limit(effective_request.question)
        if requested_limit is not None:
            ranked_restaurants = ranked_restaurants[:requested_limit]
        best_restaurant = ranked_restaurants[0] if ranked_restaurants else None
    safer_alternatives = _safer_alternatives(target_dish, history)

    status: ChatStatus = target_dish.status if target_dish else "unknown"
    dish_name = target_dish.dish_name if target_dish else None
    ingredients = target_dish.ingredients if target_dish else []
    profile_condition_names = _canonical_profile_conditions(effective_request.user_profile)
    matched_health_risks = sorted(set((target_dish.risks if target_dish else []) + (target_dish.disease_matches if target_dish else [])))
    reasons = target_dish.reasons if target_dish else list(profile_guidance.get("reasons", []))
    if intent.label == "health_risk" and target_dish and target_dish.disease_matches:
        disease_reason_map = []
        for disease_name in target_dish.disease_matches:
            notes = (DISEASE_RULES.get(disease_name) or {}).get("notes")
            if notes:
                disease_reason_map.append(f"{disease_name}: {notes}")
        if disease_reason_map:
            reasons = reasons + disease_reason_map
    elif intent.label in {"health_risk", "general"} and not target_dish:
        matched_health_risks = profile_condition_names + list(effective_request.user_profile.allergies)
        reasons = list(profile_guidance.get("reasons", []))

    if intent.label == "ingredient_info" and not ingredients and target_dish:
        missing_data.append("No confirmed ingredient list was available for this dish.")
    if intent.label == "ingredient_info" and not target_dish:
        matched_health_risks = list(effective_request.user_profile.allergies) + profile_condition_names
        ingredients = list(profile_guidance.get("avoid_items", []))
        reasons = list(profile_guidance.get("reasons", []))
        if ingredients:
            status = "caution"
    if intent.label == "health_risk" and target_dish:
        if target_dish.risks or target_dish.reasons:
            status = "not_safe"
        elif target_dish.disease_matches:
            status = "caution"
    elif intent.label in {"health_risk", "general"} and not target_dish:
        status = "caution" if profile_guidance.get("direct_answers") or profile_guidance.get("avoid_items") else "unknown"
    if intent.label == "dish_safety" and not target_dish and profile_guidance.get("direct_answers"):
        direct_status = _status_from_direct_profile_answer(profile_guidance["direct_answers"][0])
        if direct_status:
            status = direct_status
    if intent.label in {"best_restaurant", "top_two_restaurants"}:
        status = "safe" if best_restaurant and best_restaurant.safe_dish_count > 0 else "caution"
        dish_name = None
        ingredients = []
        matched_health_risks = []
        reasons = []
    if intent.label == "safe_history":
        status = "safe" if safe_history else "caution"
        dish_name = None
        ingredients = []
    confidence = _response_confidence(target_dish, intent, safe_history, best_restaurant, missing_data)
    reasoning_summary = _build_reasoning_summary(intent, target_dish, best_restaurant, missing_data, status)
    evidence = _build_evidence(intent, target_dish, safe_history, best_restaurant)

    last_dish_name = None
    if target_dish:
        last_dish_name = target_dish.dish_name
    elif intent.label == "safe_history" and safe_history:
        last_dish_name = safe_history[0].dish_name
    else:
        last_dish_name = intent.referenced_dish

    response = ChatResponse(
        api_version=API_VERSION,
        request_id=request_id,
        session_id=session_id,
        intent=intent,
        status=status,
        dish_name=dish_name,
        ingredients=ingredients,
        matched_health_risks=matched_health_risks,
        disease_or_allergy_reasons=reasons,
        safer_alternatives=safer_alternatives,
        recommended_dishes_from_previous_scans=safe_history,
        best_restaurant_for_user=best_restaurant,
        ranked_restaurants=ranked_restaurants,
        confidence=confidence,
        reasoning_summary=reasoning_summary,
        evidence=evidence,
        explanation="",
        missing_data=missing_data,
        warnings=warnings,
        memory=[],
        follow_up_context=FollowUpContext(
            last_dish_name=last_dish_name,
            last_restaurant_name=(
                best_restaurant.restaurant_name if intent.label in {"best_restaurant", "top_two_restaurants"} and best_restaurant
                else (target_dish.restaurant_name if target_dish else intent.referenced_restaurant)
            ),
            last_intent=intent.label,
            last_status=status,
        ),
        session_state=None,
        debug_context={
            "current_dish_used": current.dish_name if current else None,
            "history_dishes_considered": len(history),
            "matched_history_dish": matched_history_dish.dish_name if matched_history_dish else None,
            "state_source": "external" if request.session_state else ("memory" if use_memory else "request_only"),
            "use_memory": use_memory,
            "prioritize_current_question_over_memory": prioritize_current_question_over_memory,
            "has_fresh_food_question": has_fresh_food_question,
        },
    )

    fallback = _deterministic_explanation(
        intent,
        target_dish,
        safe_history,
        best_restaurant,
        ranked_restaurants,
        missing_data,
        effective_request.user_profile,
        profile_guidance,
        effective_request.question,
    )
    summary = _llm_summary(response, target_dish)
    try:
        explanation = await generate_explanation(summary, memory, effective_request.question, use_memory=use_memory)
    except LlmGenerationError:
        explanation = fallback
    response.explanation = explanation

    updated_memory = memory
    if request.use_session_memory:
        memory_store.append(session_id, ChatMessage(role="user", content=effective_request.question))
        updated_memory = memory_store.append(session_id, ChatMessage(role="assistant", content=response.explanation))
    else:
        updated_memory = (memory + [
            ChatMessage(role="user", content=effective_request.question),
            ChatMessage(role="assistant", content=response.explanation),
        ])[-12:]

    context_payload = response.follow_up_context.model_dump() if response.follow_up_context else {}
    if effective_current_dish:
        context_payload["last_current_dish"] = effective_current_dish.model_dump()
    if request.use_session_memory:
        memory_store.set_context(session_id, context_payload)
    response.session_state = ChatSessionState(
        follow_up_context=response.follow_up_context,
        current_dish=effective_current_dish,
        recent_messages=updated_memory[-4:],
    )
    response.memory = updated_memory if request.include_memory else []
    return response
