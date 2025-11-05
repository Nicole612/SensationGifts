"""
Base AI Client - Abstract Foundation für alle AI-Model Integrations

DESIGN PRINCIPLES:
- Abstract Base Class für einheitliche API
- Error Handling & Rate Limiting integriert
- Performance & Cost Tracking
- Structured Output Validation
- Easy Extensibility für neue Models

Clean Architecture: Alle AI-Clients implementieren diese Interface
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, TypeVar, Generic
from datetime import datetime, timedelta
from dataclasses import dataclass
import time
from enum import Enum
from pydantic import BaseModel, Field


# === ENUMS & TYPES ===

class AIModelType(Enum):
    """Verfügbare AI-Model Typen"""
    OPENAI_GPT4 = "openai_gpt4"
    OPENAI_GPT35 = "openai_gpt35"
    GROQ_MIXTRAL = "groq_mixtral"  
    GROQ_LLAMA = "groq_llama"
    GEMINI_PRO = "gemini_pro"
    GEMINI_FLASH = "gemini_flash"
    ANTHROPIC_CLAUDE = "anthropic_claude"


class ResponseFormat(Enum):
    """AI Response Formate"""
    JSON = "json"
    TEXT = "text"
    STRUCTURED = "structured"


class ModelCapability(Enum):
    """Was kann ein Model?"""
    TEXT_GENERATION = "text_generation"
    JSON_OUTPUT = "json_output" 
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    FAST_RESPONSE = "fast_response"
    HIGH_QUALITY = "high_quality"
    COST_EFFICIENT = "cost_efficient"
    COMPLEX_REASONING = "complex_reasoning"
    ETHICAL_REASONING = "ethical_reasoning"  
    CULTURAL_SENSITIVITY = "cultural_sensitivity"
    MULTIMODAL = "multimodal"

# === DATA CLASSES ===

@dataclass
class ModelMetrics:
    """Performance & Cost Metrics für AI-Models"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_response_time: float = 0.0
    last_request_time: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def cost_per_request(self) -> float:
        if self.successful_requests == 0:
            return 0.0
        return self.total_cost / self.successful_requests


@dataclass 
class AIRequest:
    """Standardisierte AI-Request Structure"""
    prompt: str
    system_prompt: Optional[str] = None
    response_format: ResponseFormat = ResponseFormat.TEXT
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    top_p: float = 1.0
    json_schema: Optional[Dict] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AIResponse:
    """Standardisierte AI-Response Structure"""
    content: str
    model_type: AIModelType
    tokens_used: int
    cost: float
    response_time: float
    timestamp: datetime
    success: bool
    error: Optional[str] = None
    raw_response: Optional[Dict] = None
    parsed_json: Optional[Dict] = None
    
    @property
    def cost_per_token(self) -> float:
        if self.tokens_used == 0:
            return 0.0
        return self.cost / self.tokens_used


# === PYDANTIC SCHEMAS FÜR STRUCTURED OUTPUTS ===

class GiftRecommendationSchema(BaseModel):
    """Schema für AI-generierte Geschenkempfehlungen"""
    gift_name: str = Field(description="Name des empfohlenen Geschenks")
    reasoning: str = Field(description="Warum passt dieses Geschenk zur Person?")
    match_score: float = Field(ge=0, le=1, description="Wie gut passt es? (0-1)")
    emotional_appeal: str = Field(description="Welche Emotion wird angesprochen?")
    personalization_ideas: List[str] = Field(description="Wie kann es personalisiert werden?")
    price_range: str = Field(description="Empfohlene Preisklasse")
    alternative_gifts: List[str] = Field(description="Alternative Geschenk-Ideen")
    confidence: float = Field(ge=0, le=1, description="AI Confidence in der Empfehlung")


class PersonalityInsightSchema(BaseModel):
    """Schema für AI-Persönlichkeitsanalyse"""
    personality_summary: str = Field(description="Kurze Persönlichkeits-Zusammenfassung")
    dominant_traits: List[str] = Field(description="Top 3 Persönlichkeitseigenschaften")
    emotional_triggers: List[str] = Field(description="Was macht die Person glücklich?")
    gift_preferences: Dict[str, float] = Field(description="Präferenzen mit Scores")
    avoid_categories: List[str] = Field(description="Was sollte vermieden werden?")
    relationship_context: str = Field(description="Wie beeinflusst die Beziehung die Geschenkwahl?")


# === BASE AI CLIENT (ABSTRACT) ===

class BaseAIClient(ABC):
    """
    Abstract Base Class für alle AI-Model Clients
    
    Jeder AI-Provider (OpenAI, Groq, etc.) erbt von dieser Klasse
    und implementiert die abstract methods.
    
    Features:
    - Einheitliche API für alle Models
    - Automatic Rate Limiting  
    - Cost & Performance Tracking
    - Error Handling mit Retries
    - Structured Output Validation
    """
    
    def __init__(self, 
                 api_key: str,
                 model_type: AIModelType,
                 rate_limit_per_minute: int = 60,
                 max_retries: int = 3):
        self.api_key = api_key
        self.model_type = model_type
        self.rate_limit_per_minute = rate_limit_per_minute
        self.max_retries = max_retries
        
        # Metrics & Tracking
        self.metrics = ModelMetrics()
        self._request_timestamps: List[datetime] = []
        
        # Model-specific configuration
        self.capabilities = self._get_model_capabilities()
        self.pricing = self._get_model_pricing()
        
    # === ABSTRACT METHODS (müssen implementiert werden) ===
    
    @abstractmethod
    def _make_api_call(self, request: AIRequest) -> AIResponse:
        """
        Macht den tatsächlichen API-Call zum AI-Provider
        Muss von jedem Client implementiert werden
        """
        pass
    
    @abstractmethod
    def _get_model_capabilities(self) -> List[ModelCapability]:
        """Definiert was dieses Model kann"""
        pass
    
    @abstractmethod
    def _get_model_pricing(self) -> Dict[str, float]:
        """Definiert Pricing pro Token/Request"""
        pass
    
    # === PUBLIC API (einheitlich für alle Clients) ===
    
    def generate_text(self, 
                     prompt: str,
                     system_prompt: Optional[str] = None,
                     temperature: float = 0.7,
                     max_tokens: Optional[int] = None) -> AIResponse:
        """
        Generiert Text-Response von AI
        
        Args:
            prompt: Main user prompt
            system_prompt: System instructions (optional)
            temperature: Creativity level (0.0-1.0)
            max_tokens: Maximum response length
            
        Returns:
            AIResponse mit generiertem Text
        """
        request = AIRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            response_format=ResponseFormat.TEXT,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return self._execute_request(request)
    
    def generate_json(self,
                     prompt: str,
                     json_schema: Dict,
                     system_prompt: Optional[str] = None,
                     temperature: float = 0.3) -> AIResponse:
        """
        Generiert strukturierten JSON-Response
        
        Args:
            prompt: Main user prompt
            json_schema: Expected JSON structure
            system_prompt: System instructions (optional)
            temperature: Lower für strukturierte Outputs
            
        Returns:
            AIResponse mit validiertem JSON
        """
        if ModelCapability.JSON_OUTPUT not in self.capabilities:
            raise NotImplementedError(f"{self.model_type} unterstützt kein JSON Output")
        
        request = AIRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            response_format=ResponseFormat.JSON,
            json_schema=json_schema,
            temperature=temperature
        )
        
        return self._execute_request(request)
    
    def recommend_gift(self, 
                      personality_profile: Dict,
                      occasion: str,
                      budget_range: str,
                      relationship: str) -> GiftRecommendationSchema:
        """
        High-level Gift Recommendation API
        
        Args:
            personality_profile: Persönlichkeits-Daten
            occasion: Anlass (Geburtstag, etc.)
            budget_range: Budget als String "50-150"
            relationship: Beziehung (Partner, Freund, etc.)
            
        Returns:
            Validierte Geschenk-Empfehlung
        """
        prompt = self._build_gift_recommendation_prompt(
            personality_profile, occasion, budget_range, relationship
        )
        
        schema = GiftRecommendationSchema.model_json_schema()
        
        response = self.generate_json(
            prompt=prompt,
            json_schema=schema,
            system_prompt=self._get_gift_expert_system_prompt(),
            temperature=0.4  # Etwas strukturierter für Empfehlungen
        )
        
        if not response.success:
            raise Exception(f"Gift recommendation failed: {response.error}")
        
        # Validiere & Parse JSON
        try:
            return GiftRecommendationSchema.model_validate(response.parsed_json)
        except Exception as e:
            raise Exception(f"Invalid AI response format: {e}")
    
    # === INTERNAL METHODS ===
    
    def _execute_request(self, request: AIRequest) -> AIResponse:
        """
        Führt AI-Request mit Error Handling & Retries aus
        """
        # Rate Limiting Check
        if not self._check_rate_limit():
            time.sleep(self._get_rate_limit_delay())
        
        # Execute with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                # Update metrics
                self.metrics.total_requests += 1
                start_time = time.time()
                
                # Make actual API call
                response = self._make_api_call(request)
                
                # Update success metrics
                response_time = time.time() - start_time
                response.response_time = response_time
                
                if response.success:
                    self.metrics.successful_requests += 1
                    self.metrics.total_tokens += response.tokens_used
                    self.metrics.total_cost += response.cost
                    self._update_avg_response_time(response_time)
                else:
                    self.metrics.failed_requests += 1
                
                self.metrics.last_request_time = datetime.now()
                self._request_timestamps.append(datetime.now())
                
                return response
                
            except Exception as e:
                last_error = e
                self.metrics.failed_requests += 1
                
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    time.sleep(wait_time)
                    continue
        
        # All retries failed
        return AIResponse(
            content="",
            model_type=self.model_type,
            tokens_used=0,
            cost=0.0,
            response_time=0.0,
            timestamp=datetime.now(),
            success=False,
            error=f"Failed after {self.max_retries} retries: {last_error}"
        )
    
    def _check_rate_limit(self) -> bool:
        """Prüft ob Rate Limit eingehalten wird"""
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)
        
        # Entferne alte Timestamps
        self._request_timestamps = [
            ts for ts in self._request_timestamps if ts > cutoff
        ]
        
        return len(self._request_timestamps) < self.rate_limit_per_minute
    
    def _get_rate_limit_delay(self) -> float:
        """Berechnet notwendige Wartezeit für Rate Limiting"""
        if not self._request_timestamps:
            return 0.0
        
        oldest_request = min(self._request_timestamps)
        wait_until = oldest_request + timedelta(minutes=1)
        now = datetime.now()
        
        if wait_until > now:
            return (wait_until - now).total_seconds()
        return 0.0
    
    def _update_avg_response_time(self, response_time: float):
        """Aktualisiert durchschnittliche Response Time"""
        if self.metrics.avg_response_time == 0:
            self.metrics.avg_response_time = response_time
        else:
            # Exponential moving average
            self.metrics.avg_response_time = (
                0.9 * self.metrics.avg_response_time + 0.1 * response_time
            )
    
    def _build_gift_recommendation_prompt(self,
                                        personality_profile: Dict,
                                        occasion: str,
                                        budget_range: str,
                                        relationship: str) -> str:
        """Baut Prompt für Geschenk-Empfehlung (hier bleibt als einfache Fallback-Implementierung, in Child Class OpenAICLient in _build_advanced_gift_prompt ist erweitert und in AnthropicClient sollte eine eigene Methode bekommen)"""
        return f"""
        Als Geschenk-Experte empfiehl ein perfektes Geschenk für:
        
        Personality: {personality_profile}
        Anlass: {occasion}
        Budget: {budget_range}€
        Beziehung: {relationship}
        
        Berücksichtige die Persönlichkeit und gib eine durchdachte Empfehlung.
        """
    
    def _get_gift_expert_system_prompt(self) -> str:
        """System Prompt für Gift Recommendations (wird später erweitert)"""
        return """
        Du bist ein Experte für personalisierte Geschenke mit 20 Jahren Erfahrung.
        Du verstehst Persönlichkeitspsychologie und emotionale Trigger.
        Deine Empfehlungen sind immer durchdacht, kreativ und passend.
        Antworte ausschließlich im angegebenen JSON-Format.
        """
    
    # === UTILITY METHODS ===
    
    def get_metrics(self) -> ModelMetrics:
        """Gibt aktuelle Performance-Metriken zurück"""
        return self.metrics
    
    def reset_metrics(self):
        """Setzt Metriken zurück (für Testing)"""
        self.metrics = ModelMetrics()
        self._request_timestamps = []
    
    def can_handle_request(self, request_type: str) -> bool:
        """Prüft ob Model einen bestimmten Request-Typ verarbeiten kann"""
        capability_map = {
            "json": ModelCapability.JSON_OUTPUT,
            "vision": ModelCapability.VISION,
            "functions": ModelCapability.FUNCTION_CALLING
        }
        
        required_capability = capability_map.get(request_type)
        return required_capability in self.capabilities if required_capability else True
    
    def estimate_cost(self, prompt_length: int, max_tokens: int = 500) -> float:
        """Schätzt Kosten für einen Request"""
        input_tokens = prompt_length // 4  # Rough estimation
        output_tokens = max_tokens
        
        input_cost = input_tokens * self.pricing.get("input_per_token", 0.0)
        output_cost = output_tokens * self.pricing.get("output_per_token", 0.0)
        
        return input_cost + output_cost
    
    def __repr__(self):
        return f"<{self.__class__.__name__} {self.model_type.value} (Success: {self.metrics.success_rate:.2%})>"


# === UTILITY FUNCTIONS ===

def get_available_models() -> List[AIModelType]:
    """Gibt Liste aller verfügbaren AI-Models zurück"""
    return list(AIModelType)


def get_best_model_for_task(task_type: str, 
                           priority: str = "quality") -> Optional[AIModelType]:
    """
    Empfiehlt bestes Model für spezifische Aufgabe
    
    Args:
        task_type: "text", "json", "fast", "creative"
        priority: "quality", "speed", "cost"
        
    Returns:
        Empfohlenes AIModelType oder None
    """
    recommendations = {
        ("text", "quality"): AIModelType.OPENAI_GPT4,
        ("text", "speed"): AIModelType.GROQ_LLAMA,
        ("text", "cost"): AIModelType.GEMINI_FLASH,
        ("json", "quality"): AIModelType.OPENAI_GPT4,
        ("json", "speed"): AIModelType.GROQ_MIXTRAL,
        ("creative", "quality"): AIModelType.ANTHROPIC_CLAUDE,
        ("creative", "speed"): AIModelType.GROQ_LLAMA,
    }
    
    return recommendations.get((task_type, priority))