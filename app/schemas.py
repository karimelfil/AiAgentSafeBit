from pydantic import BaseModel
from typing import List, Optional, Literal

SafetyLevel = Literal["SAFE", "RISKY", "CAUTION"]

class UserProfile(BaseModel):
    allergies: List[str] = []
    diseases: List[str] = []
    is_pregnant: Optional[bool] = None


class Conflict(BaseModel):
    type: str
    trigger: str
    evidence: str
    explanation: str


class DishResult(BaseModel):
    dish_name: str
    detected_triggers: List[str] = []
    ingredients_found: List[str] = []
    safety_level: SafetyLevel = "CAUTION"
    confidence: float = 0.0
    ingredient_coverage: float = 0.0
    needs_user_confirmation: bool = False
    conflicts: List[Conflict] = []
    notes: List[str] = []


class AnalyzeMenuResponse(BaseModel):
    menu_upload_id: str
    extracted_text_preview: str
    dishes: List[DishResult]
