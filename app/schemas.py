from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

#Define Safetylevel for each dish
SafetyLevel = Literal["safe", "risky", "unsafe"]

#chat
ChatStatus = Literal["safe", "caution", "not_safe", "unknown"]

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



#chat related schemas
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class EvidenceItem(BaseModel):
    source: str
    detail: str


class ChatIntent(BaseModel):
    label: Literal[
        "dish_safety",
        "ingredient_info",
        "health_risk",
        "safe_history",
        "best_restaurant",
        "top_two_restaurants",
        "follow_up_explanation",
        "general"
    ]
    confidence: float = 0.0
    rationale: str = ""
    referenced_dish: Optional[str] = None
    referenced_restaurant: Optional[str] = None


class RestaurantContext(BaseModel):
    restaurant_name: Optional[str] = None
    menu_upload_id: Optional[str] = None


class DishContext(BaseModel):
    dish_name: str
    description: Optional[str] = None
    ingredients: List[str] = Field(default_factory=list)
    raw_text: Optional[str] = None
    restaurant_name: Optional[str] = None


class ScanHistoryItem(BaseModel):
    menu_upload_id: Optional[str] = None
    restaurant_name: Optional[str] = None
    scanned_at: Optional[str] = None
    extracted_text_preview: Optional[str] = None
    dishes: List[DishResult] = Field(default_factory=list)


class RecommendationDish(BaseModel):
    dish_name: str
    restaurant_name: Optional[str] = None
    status: ChatStatus = "unknown"
    reason: str = ""
    confidence: float = 0.0


class RestaurantRecommendation(BaseModel):
    restaurant_name: Optional[str] = None
    safe_dish_count: int = 0
    caution_dish_count: int = 0
    risky_dish_count: int = 0
    overall_score: float = 0.0
    recommended_dishes: List[str] = Field(default_factory=list)
    reason: str = ""


class FollowUpContext(BaseModel):
    last_dish_name: Optional[str] = None
    last_restaurant_name: Optional[str] = None
    last_intent: Optional[str] = None
    last_status: Optional[ChatStatus] = None


class ChatSessionState(BaseModel):
    follow_up_context: Optional[FollowUpContext] = None
    current_dish: Optional[DishContext] = None
    recent_messages: List[ChatMessage] = Field(default_factory=list)


class ChatRequest(BaseModel):
    question: str
    user_profile: UserProfile
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    scan_history: List[ScanHistoryItem] = Field(default_factory=list)
    current_dish: Optional[DishContext] = None
    restaurant_context: Optional[RestaurantContext] = None
    session_state: Optional[ChatSessionState] = None
    use_session_memory: bool = True
    include_memory: bool = False


class ChatResponse(BaseModel):
    api_version: str = "chat.v1"
    request_id: Optional[str] = None
    session_id: str
    intent: ChatIntent
    status: ChatStatus = "unknown"
    dish_name: Optional[str] = None
    ingredients: List[str] = Field(default_factory=list)
    matched_health_risks: List[str] = Field(default_factory=list)
    disease_or_allergy_reasons: List[str] = Field(default_factory=list)
    safer_alternatives: List[str] = Field(default_factory=list)
    recommended_dishes_from_previous_scans: List[RecommendationDish] = Field(default_factory=list)
    best_restaurant_for_user: Optional[RestaurantRecommendation] = None
    ranked_restaurants: List[RestaurantRecommendation] = Field(default_factory=list)
    confidence: float = 0.0
    reasoning_summary: List[str] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    explanation: str
    missing_data: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    memory: List[ChatMessage] = Field(default_factory=list)
    follow_up_context: Optional[FollowUpContext] = None
    session_state: Optional[ChatSessionState] = None
    debug_context: Dict[str, Any] = Field(default_factory=dict)
