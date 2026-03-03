from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import json
import uuid
from pathlib import Path

from app.schemas import UserProfile, AnalyzeMenuResponse, DishResult
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
        dish_name = (it.get("dish_name") or "Dish").strip()
        block = (it.get("block") or "").strip()


        ingredients, triggers, evidences, confidence, ingredient_coverage, extract_notes = build_ingredients_list(
            dish_name=dish_name,
            block=block,
            common_ingredients=COMMON_ING,
            allergen_triggers=ALLERGENS
        )

        text_blob = f"{dish_name} {block}".lower()

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
        needs_confirm = eval_conf < 0.5 or ingredient_coverage < 0.35
        if needs_confirm:
            eval_notes.append("Low extraction confidence. Ask the user to retake the photo or confirm ingredients.")

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
