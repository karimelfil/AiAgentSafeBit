from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import json
import uuid
import re
from pathlib import Path

from app.chat_service import answer_chat
from app.schemas import UserProfile, AnalyzeMenuResponse, ChatRequest, ChatResponse, DishResult
from app.pipeline.knowledge import load_json
from app.pipeline.ocr import extract_text
from app.pipeline.segment import segment_dishes
from app.pipeline.ingredients import build_ingredients_list
from app.pipeline.rules import evaluate

app = FastAPI(title="SafeBite AI Agent", version="1.0")


ROOT = Path(__file__).resolve().parent.parent


KNOWLEDGE = ROOT / "knowledge"

ALLERGENS = load_json(KNOWLEDGE / "allergens.json")
COMMON_ING = load_json(KNOWLEDGE / "common_ingredients.json")
DISH_SIG = load_json(KNOWLEDGE / "dish_signatures.json")
DISEASE_RULES = load_json(KNOWLEDGE / "disease_rules.json")


# This function cleans and normalizes dish names by removing common prefixes, suffixes, and extraneous information.
def _normalize_dish_name(name: str) -> str:
    if not name:
        return "Dish"
    cleaned = name.strip()
    cleaned = re.sub(r"^\s*dis[h]?\s*[:\-]\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^\s*dish\s*name\s*[:\-]\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s*(dish\s*ingredients?|ingredients?)\s*[:\-].*$", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:\t\r\n")
    return cleaned or "Dish"

# Endpoint to analyze an uploaded menu image and extract dish information, ingredients, and potential allergen triggers based on the user's profile.
@app.post("/analyze-menu", response_model=AnalyzeMenuResponse)
async def analyze_menu(
    file: UploadFile = File(...),
    user_profile_json: str = Form(...)
):

    menu_upload_id = str(uuid.uuid4())



    try:
        profile = UserProfile(**json.loads(user_profile_json))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


    file_bytes = await file.read()
    text = extract_text(file_bytes, file.content_type or "", file.filename)

    items = segment_dishes(text) or [
        {"dish_name": "Menu", "block": text[:2500]}
    ]

    results = []

    for it in items:
        dish_name = _normalize_dish_name(it.get("dish_name") or "Dish")
        block = (it.get("block") or "").strip()


        ingredients, triggers, evidences, confidence, ingredient_coverage, extract_notes = build_ingredients_list(
            dish_name=dish_name,
            block=block,
            common_ingredients=COMMON_ING,
            allergen_triggers=ALLERGENS
        )

        #combine dish name and block into one lowercase string
        text_blob = f"{dish_name} {block}".lower()

    #check for dish signatures and add triggers and notes accordingly
        for sig_name, rule in (DISH_SIG or {}).items():

            keywords = []
            triggers_to_add = []
            note = None

            
            if isinstance(rule, dict):
                keywords = rule.get("keywords", [])
                triggers_to_add = rule.get("triggers", [])
                note = rule.get("note")

            elif isinstance(rule, list):
                keywords = [sig_name]
                triggers_to_add = rule

            else:
                continue

            #if any of the keywords are found in the text blob, add the corresponding triggers and notes
            if any(str(k).lower() in text_blob for k in keywords):
                for t in triggers_to_add:
                    if t not in triggers:
                        triggers.append(t)
                        evidences.append(f"signature:{sig_name}")

                if note and note not in extract_notes:
                    extract_notes.append(note)
        safety, conflicts, eval_notes, eval_conf = evaluate(
            dish_name=dish_name,
            triggers=triggers,
            evidences=evidences,
            confidence=confidence,
            profile=profile,
            disease_rules=DISEASE_RULES,
            ingredients_found=ingredients,
            ingredient_coverage=ingredient_coverage
        )
        #confirmation needed if confidence is low or ingredient coverage is low
        needs_confirm = eval_conf < 0.5 or ingredient_coverage < 0.35
        if needs_confirm:
            eval_notes.append("The ingredient details are limited, so please confirm them with the restaurant or upload a clearer photo for a more reliable result.")

        results.append(DishResult(
            dish_name=dish_name,
            detected_triggers=sorted(set(triggers)),
            ingredients_found=ingredients,
            safety_level=safety,
            confidence=eval_conf,
            ingredient_coverage=ingredient_coverage,
            needs_user_confirmation=needs_confirm,
            conflicts=conflicts,
            notes=list(set(extract_notes + eval_notes))
        ))

    return AnalyzeMenuResponse(
        menu_upload_id=menu_upload_id,
        extracted_text_preview=text[:400],
        dishes=results
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question is required.")
    if (
        not request.use_session_memory
        and not request.session_state
        and not request.current_dish
        and not request.scan_history
        and any(token in request.question.lower() for token in ["this dish", "this restaurant", "that dish", "that restaurant"])
    ):
        raise HTTPException(
            status_code=400,
            detail="Stateless follow-up requires session_state, current_dish, or scan_history context.",
        )
    return await answer_chat(request)
