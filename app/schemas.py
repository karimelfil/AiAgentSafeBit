from typing import List, Literal, Optional

from pydantic import BaseModel, Field

#Define Safetylevel for each dish
SafetyLevel = Literal["safe", "risky", "unsafe"]

#Define the user health profile 
class UserProfile(BaseModel):
    allergies: List[str] = Field(default_factory=list)
    intolerances: List[str] = Field(default_factory=list)
    diseases: List[str] = Field(default_factory=list)
    forbidden_ingredients: List[str] = Field(default_factory=list)
    dietary_preferences: List[str] = Field(default_factory=list)
    is_pregnant: Optional[bool] = None

# Define the structure for conflicts detected in dish analysis
class Conflict(BaseModel):
    type: str
    trigger: str
    evidence: str
    explanation: str

# Define the structure for dish analysis results
class DishResult(BaseModel):
    dish_name: str
    detected_triggers: List[str] = Field(default_factory=list)
    ingredients_found: List[str] = Field(default_factory=list)
    safety_level: SafetyLevel = "risky"
    confidence: float = 0.0
    ingredient_coverage: float = 0.0
    needs_user_confirmation: bool = False
    conflicts: List[Conflict] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)

# Define the structure for the response of menu analysis
class AnalyzeMenuResponse(BaseModel):
    menu_upload_id: str
    extracted_text_preview: str
    dishes: List[DishResult]
