"""
AI Engine Output Schemas - Emotional Gift Recommendation Responses (v2.0)
========================================================================

🎁 EMOTIONAL INTELLIGENCE: AI-Outputs die echte menschliche Verbindungen schaffen
🚀 ENTERPRISE FEATURES: Performance tracking, quality scoring, multi-model responses
✨ PERSONALIZATION: Jede Response individuell auf Persönlichkeit + Emotion optimiert

Diese Schemas definieren was aus der AI rauskommt - strukturiert, validiert, emotional intelligent.
Fokus: Schenken als Erlebnis, nicht als Produktkauf.
"""

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator
from typing import Optional, List, Dict, Any, Union, Literal, Annotated
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
import json
from collections import defaultdict


# =============================================================================
# EMOTIONAL INTELLIGENCE ENUMS: Was macht Geschenke besonders?
# =============================================================================

class ConfidenceLevel(str, Enum):
    """Confidence-Level für AI-Empfehlungen (EMOTIONAL CONTEXT AWARE)"""
    VERY_LOW = "very_low"       # 0.0-0.3: Unsicher, mehr Input nötig
    LOW = "low"                 # 0.3-0.5: Begrenzte Konfidenz
    MEDIUM = "medium"           # 0.5-0.7: Solide Empfehlung  
    HIGH = "high"               # 0.7-0.9: Sehr konfident
    VERY_HIGH = "very_high"     # 0.9-1.0: Außergewöhnlich sicher
    EMOTIONALLY_PERFECT = "emotionally_perfect"  # 🚀 NEW: Perfect emotional match
    AI_OPTIMIZED = "ai_optimized"  # 🚀 NEW: AI-optimized recommendation
    ENSEMBLE_VALIDATED = "ensemble_validated"  # 🚀 NEW: Multiple AI models agree


class GiftCategory(str, Enum):
    """Gift Categories mit emotionalem Focus (ENHANCED für Emotional Connection)"""
    # ✅ TRADITIONAL: Bewährte Kategorien
    EXPERIENCES = "experiences"              # Erlebnisse statt Dinge
    ART_CRAFTS = "art_crafts"               # Kreative Ausdrücke
    BOOKS_MEDIA = "books_media"             # Wissen und Inspiration
    TECHNOLOGY = "technology"               # Innovation und Effizienz
    HOME_LIVING = "home_living"             # Komfort und Geborgenheit
    FASHION_STYLE = "fashion_style"         # Persönlicher Ausdruck
    HEALTH_WELLNESS = "health_wellness"     # Wohlbefinden und Selbstfürsorge
    FOOD_DRINKS = "food_drinks"             # Genuss und Verbindung
    SPORTS_OUTDOOR = "sports_outdoor"       # Aktivität und Abenteuer
    LUXURY_PREMIUM = "luxury_premium"       # Exklusivität und Status
    
    # 🚀 EMOTIONAL: Kategorien die emotionale Verbindungen schaffen
    MEMORY_MAKERS = "memory_makers"         # Schaffen unvergessliche Momente
    EMOTIONAL_BONDS = "emotional_bonds"     # Stärken menschliche Verbindungen
    PERSONAL_GROWTH = "personal_growth"     # Fördern Entwicklung und Träume
    COMFORT_CARE = "comfort_care"           # Spenden Trost und Geborgenheit
    SURPRISE_DELIGHT = "surprise_delight"   # Überraschung und pure Freude
    SHARED_ADVENTURES = "shared_adventures" # Gemeinsame Erlebnisse
    MINDFUL_MOMENTS = "mindful_moments"     # Achtsamkeit und Entschleunigung
    CREATIVE_EXPRESSION = "creative_expression" # Künstlerischer Selbstausdruck
    ACHIEVEMENT_CELEBRATION = "achievement_celebration" # Erfolge feiern
    UNCONDITIONAL_LOVE = "unconditional_love" # Bedingungslose Zuneigung zeigen


class OptimizationObjective(str, Enum):
    """Optimization objectives for AI engine (ENHANCED)"""
    QUALITY = "quality"                    # Maximize recommendation quality
    SPEED = "speed"                       # Minimize response time
    COST = "cost"                        # Minimize cost
    PERSONALIZATION = "personalization"   # Maximize personalization
    EMOTIONAL_IMPACT = "emotional_impact" # Maximize emotional resonance
    BALANCED = "balanced"                 # Balanced optimization
    EXPERIMENTAL = "experimental"         # Try new approaches


class ModelType(str, Enum):
    """AI Model types for tracking and optimization"""
    OPENAI_GPT4 = "openai_gpt4"
    OPENAI_GPT35 = "openai_gpt35"
    GROQ_MIXTRAL = "groq_mixtral"
    GROQ_LLAMA = "groq_llama"
    ANTHROPIC_CLAUDE = "anthropic_claude"
    GOOGLE_GEMINI = "google_gemini"
    AUTO_SELECT = "auto_select"


class AdvancedTechnique(str, Enum):
    """Advanced prompt engineering techniques"""
    META_PROMPTING = "meta_prompting"
    SELF_CORRECTION = "self_correction"
    ENSEMBLE_PROMPTING = "ensemble_prompting"
    CONSTITUTIONAL_AI = "constitutional_ai"
    ADAPTIVE_LEARNING = "adaptive_learning"
    MULTI_STEP_REASONING = "multi_step_reasoning"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    FEW_SHOT_LEARNING = "few_shot_learning"


class RecommendationReason(str, Enum):
    """Warum diese Empfehlung? (EMOTIONAL REASONING)"""
    # ✅ PERSONALITY-BASED: Big Five + Limbic Reasoning
    HIGH_OPENNESS_MATCH = "high_openness_match"         # Kreativität + Neugier
    CONSCIENTIOUSNESS_FIT = "conscientiousness_fit"     # Organisation + Qualität
    EXTRAVERSION_APPEAL = "extraversion_appeal"         # Soziale Energie
    AGREEABLENESS_HARMONY = "agreeableness_harmony"     # Harmonie + Kooperation
    EMOTIONAL_STABILITY = "emotional_stability"         # Innere Ruhe
    STIMULANZ_EXCITEMENT = "stimulanz_excitement"       # Aufregung + Adrenalin
    DOMINANZ_STATUS = "dominanz_status"                 # Status + Kontrolle
    BALANCE_HARMONY = "balance_harmony"                 # Ausgeglichenheit
    
    # 🚀 EMOTIONAL: Gefühlsbasierte Begründungen
    DEEPENS_CONNECTION = "deepens_connection"           # Vertieft Beziehung
    EXPRESSES_GRATITUDE = "expresses_gratitude"         # Zeigt Dankbarkeit
    CELEBRATES_UNIQUENESS = "celebrates_uniqueness"     # Feiert Einzigartigkeit
    SUPPORTS_DREAMS = "supports_dreams"                 # Unterstützt Träume
    PROVIDES_COMFORT = "provides_comfort"               # Spendet Trost
    SPARKS_JOY = "sparks_joy"                          # Entfacht pure Freude
    HONORS_TRADITION = "honors_tradition"               # Ehrt Traditionen
    EMBRACES_FUTURE = "embraces_future"                 # Umarmt Zukunft
    SHARED_MEMORY = "shared_memory"                     # Gemeinsame Erinnerung
    SELFLESS_GIVING = "selfless_giving"                 # Selbstlose Gabe


# =============================================================================
# 🎁 CORE GIFT RECOMMENDATION: Herzstück der emotionalen Empfehlungen
# =============================================================================

class GiftRecommendation(BaseModel):
    """
    Einzelne Geschenkempfehlung mit emotionaler Intelligenz (ENHANCED V2)
    
    🎁 EMOTIONAL FOCUS: Jede Empfehlung erzählt eine Geschichte
    🧠 AI-POWERED: Big Five + Limbic + Kontext = perfekte Matches
    ✨ EXPERIENCE: Fokus auf Gefühle, nicht nur Produkteigenschaften
    """
    
    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "extra": "forbid",
        "use_enum_values": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v)
        }
    }
    
    # === GESCHENK-IDENTIFIKATION ===
    title: Annotated[
        str,
        Field(
            min_length=5, max_length=200,
            description="Prägnanter, emotionaler Titel des Geschenks"
        )
    ]
    
    description: Annotated[
        str,
        Field(
            min_length=20, max_length=1000,
            description="Detaillierte, emotionale Beschreibung die Begeisterung weckt"
        )
    ]
    
    category: GiftCategory = Field(
        ...,
        description="Kategorie mit emotionalem Focus"
    )
    
    # === PREIS UND VERFÜGBARKEIT ===
    price_range: Annotated[
        str,
        Field(
            pattern=r"^€\d+(-€\d+)?$",
            description="Preisbereich (z.B. '€25-€45' oder '€120')"
        )
    ]
    
    estimated_price: Optional[Decimal] = Field(
        None, ge=0,
        description="Geschätzter Durchschnittspreis"
    )
    
    availability: str = Field(
        ..., max_length=100,
        description="Verfügbarkeit und Lieferzeit"
    )
    
    where_to_find: List[str] = Field(
        default_factory=list,
        description="Wo kann man es kaufen/finden"
    )
    
    # === EMOTIONALE INTELLIGENZ ===
    emotional_impact: Annotated[
        str,
        Field(
            min_length=10, max_length=500,
            description="Welche Emotionen wird dieses Geschenk auslösen?"
        )
    ]
    
    personal_connection: Annotated[
        str,
        Field(
            min_length=10, max_length=500,
            description="Warum passt es perfekt zu dieser Person?"
        )
    ]
    
    relationship_benefit: Annotated[
        str,
        Field(
            min_length=10, max_length=400,
            description="Wie stärkt es die Beziehung?"
        )
    ]
    
    # === HELDENREISE EMOTIONAL STORY ===
    emotional_story: Optional[str] = Field(
        None, max_length=800,
        description="Die emotionale Geschichte hinter dem Geschenk (Heldenreise-Katalog)"
    )
    
    # === AI-REASONING ===
    personality_match: Annotated[
        str,
        Field(
            min_length=15, max_length=600,
            description="Detaillierte Big Five + Limbic Persönlichkeits-Begründung"
        )
    ]
    
    primary_reason: RecommendationReason = Field(
        ...,
        description="Hauptgrund für diese Empfehlung"
    )
    
    supporting_reasons: List[RecommendationReason] = Field(
        default_factory=list,
        description="Zusätzliche unterstützende Gründe"
    )
    
    # === QUALITÄT UND VERTRAUEN ===
    confidence_score: Annotated[
        float,
        Field(
            ge=0.0, le=1.0,
            description="AI-Konfidenz in diese Empfehlung"
        )
    ]
    
    confidence_level: ConfidenceLevel = Field(
        ...,
        description="Menschenlesbare Konfidenz-Einschätzung"
    )
    
    uniqueness_score: Annotated[
        float,
        Field(
            ge=0.0, le=1.0,
            description="Wie einzigartig/überraschend ist diese Empfehlung?"
        )
    ]
    
    # === ZUSÄTZLICHE INSIGHTS ===
    presentation_tips: Optional[str] = Field(
        None, max_length=400,
        description="Wie sollte das Geschenk präsentiert werden?"
    )
    
    timing_advice: Optional[str] = Field(
        None, max_length=300,
        description="Wann ist der beste Zeitpunkt für die Übergabe?"
    )
    
    personalization_ideas: List[str] = Field(
        default_factory=list,
        description="Ideen zur weiteren Personalisierung"
    )
    
    potential_concerns: List[str] = Field(
        default_factory=list,
        description="Mögliche Bedenken oder Risiken"
    )
    
    alternatives: List[str] = Field(
        default_factory=list,
        description="Alternative Optionen falls das Hauptgeschenk nicht passt"
    )
    
    # === AI-METADATEN ===
    ai_model_used: Optional[str] = Field(
        None,
        description="Welches AI-Model hat diese Empfehlung generiert"
    )
    
    generation_time_ms: Optional[int] = Field(
        None, ge=0,
        description="Wie lange hat die Generierung gedauert"
    )
    
    personalization_depth: Annotated[
        float,
        Field(
            default=0.7, ge=0.0, le=1.0,
            description="Tiefe der Personalisierung (0=generisch, 1=hochpersonalisiert)"
        )
    ]
    
    # === ENHANCED AI METADATA ===
    optimization_techniques_used: List[AdvancedTechnique] = Field(
        default_factory=list,
        description="Advanced techniques used in generation"
    )
    
    model_confidence: Annotated[
        float,
        Field(
            default=0.8, ge=0.0, le=1.0,
            description="Model's confidence in this specific recommendation"
        )
    ]
    
    ensemble_score: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Score from ensemble validation (if used)"
    )
    
    cost_estimate: Optional[Decimal] = Field(
        None, ge=0,
        description="Estimated cost for this recommendation"
    )
    
    token_efficiency: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Token efficiency score (quality per token)"
    )
    
    # 🚀 EMOTIONAL ANALYTICS
    emotional_tags: List[str] = Field(
        default_factory=list,
        description="Emotionale Tags für besseres Verständnis"
    )
    
    memory_potential: Annotated[
        float,
        Field(
            default=0.5, ge=0.0, le=1.0,
            description="Potenzial für unvergessliche Erinnerungen"
        )
    ]
    
    surprise_factor: Annotated[
        float,
        Field(
            default=0.5, ge=0.0, le=1.0,
            description="Überraschungsfaktor (0=erwartet, 1=völlig unerwartet)"
        )
    ]
    
    # === COMPUTED PROPERTIES ===
    @computed_field
    @property
    def overall_score(self) -> float:
        """Kombinierter Score aus Konfidenz, Einzigartigkeit und emotionalem Potenzial (ENHANCED)"""
        base_score = (
            self.confidence_score * 0.3 +
            self.uniqueness_score * 0.25 +
            self.memory_potential * 0.2 +
            self.surprise_factor * 0.1 +
            self.personalization_depth * 0.1 +
            self.model_confidence * 0.05
        )
        
        # Boost for ensemble validation
        if self.ensemble_score and self.ensemble_score > 0.8:
            base_score += 0.05
        
        # Boost for advanced techniques
        if self.optimization_techniques_used:
            technique_bonus = min(0.1, len(self.optimization_techniques_used) * 0.02)
            base_score += technique_bonus
        
        return min(1.0, base_score)
    
    @computed_field
    @property
    def emotional_profile(self) -> Dict[str, Any]:
        """Emotionales Profil dieser Empfehlung"""
        return {
            "emotional_depth": (self.memory_potential + self.personalization_depth) / 2,
            "relationship_impact": self.confidence_score * 0.8,  # High confidence = good relationship fit
            "surprise_and_delight": (self.surprise_factor + self.uniqueness_score) / 2,
            "overall_emotional_value": self.overall_score
        }
    
    @computed_field
    @property
    def recommendation_strength(self) -> str:
        """Textuelle Einschätzung der Empfehlungsstärke"""
        score = self.overall_score
        
        if score >= 0.9:
            return "Außergewöhnlich perfekt"
        elif score >= 0.8:
            return "Sehr stark empfohlen"
        elif score >= 0.7:
            return "Empfehlenswert"
        elif score >= 0.6:
            return "Solide Option"
        elif score >= 0.5:
            return "Überlegenswert"
        else:
            return "Mit Vorbehalt"
    
    # === FIELD VALIDATORS ===
    @field_validator('confidence_level')
    @classmethod
    def sync_confidence_level_with_score(cls, v, info):
        """Synchronisiert confidence_level mit confidence_score"""
        score = info.data.get('confidence_score')
        
        if score is not None:
            if score >= 0.95:
                return ConfidenceLevel.EMOTIONALLY_PERFECT
            elif score >= 0.9:
                return ConfidenceLevel.VERY_HIGH
            elif score >= 0.7:
                return ConfidenceLevel.HIGH
            elif score >= 0.5:
                return ConfidenceLevel.MEDIUM
            elif score >= 0.3:
                return ConfidenceLevel.LOW
            else:
                return ConfidenceLevel.VERY_LOW
        
        return v
    
    @field_validator('emotional_tags')
    @classmethod
    def ensure_emotional_tags_quality(cls, v):
        """Stellt sicher dass emotionale Tags hochwertig sind"""
        if not v:
            return ["thoughtful", "personal"]  # Default emotional tags
        
        # Filter out generic tags, keep emotional ones
        emotional_keywords = [
            "love", "joy", "surprise", "comfort", "inspiration", "gratitude",
            "celebration", "connection", "warmth", "excitement", "peace",
            "adventure", "care", "appreciation", "wonder", "delight"
        ]
        
        quality_tags = []
        for tag in v:
            if any(keyword in tag.lower() for keyword in emotional_keywords):
                quality_tags.append(tag)
            elif len(tag) > 3:  # Keep non-emotional but descriptive tags
                quality_tags.append(tag)
        
        return quality_tags[:8]  # Limit to 8 most relevant tags
    
    @model_validator(mode='after')
    def ensure_emotional_coherence(self):
        """Stellt sicher dass alle emotionalen Aspekte kohärent sind"""
        
        # High memory potential should correlate with higher confidence
        if self.memory_potential > 0.8 and self.confidence_score < 0.6:
            # Boost confidence for high memory potential gifts
            self.confidence_score = min(1.0, self.confidence_score + 0.1)
        
        # Ensure emotional impact aligns with confidence
        if self.confidence_score > 0.8 and len(self.emotional_impact) < 50:
            # High confidence recommendations should have rich emotional descriptions
            pass  # Could enhance emotional_impact here
        
        return self
    
    # === PRESENTATION METHODS ===
    def to_story_format(self) -> str:
        """
        Konvertiert Empfehlung zu einer emotionalen Geschichte
        
        🎁 INNOVATION: Geschenke als Geschichten erzählen
        """
        story = f"**{self.title}**\n\n"
        story += f"{self.description}\n\n"
        story += f"**Warum es perfekt ist:** {self.personal_connection}\n\n"
        story += f"**Emotionale Wirkung:** {self.emotional_impact}\n\n"
        story += f"**Für eure Beziehung:** {self.relationship_benefit}\n\n"
        
        if self.presentation_tips:
            story += f"**Präsentations-Tipp:** {self.presentation_tips}\n\n"
        
        story += f"**Preis:** {self.price_range} | **Verfügbarkeit:** {self.availability}"
        
        return story
    
    def get_key_highlights(self) -> List[str]:
        """Wichtigste Highlights als Bullet Points"""
        highlights = []
        
        highlights.append(f"✨ {self.recommendation_strength}")
        highlights.append(f"🎯 {self.category.value.replace('_', ' ').title()}")
        highlights.append(f"💝 {self.primary_reason.value.replace('_', ' ').title()}")
        highlights.append(f"💰 {self.price_range}")
        
        if self.uniqueness_score > 0.7:
            highlights.append(f"🌟 Außergewöhnlich einzigartig")
        
        if self.surprise_factor > 0.7:
            highlights.append(f"🎉 Hoher Überraschungsfaktor")
        
        if self.memory_potential > 0.8:
            highlights.append(f"💭 Unvergesslich-Potenzial")
        
        return highlights


# =============================================================================
# 🧠 PERSONALITY ANALYSIS RESULT: Big Five + Limbic Insights
# =============================================================================

class PersonalityAnalysisResult(BaseModel):
    """
    Ergebnis der Big Five + Limbic Persönlichkeitsanalyse (ENHANCED V2)
    
    🧠 SCIENTIFIC: Fundierte Persönlichkeitspsychologie
    ❤️ EMOTIONAL: Emotionale Trigger und Bedürfnisse verstehen
    🎁 ACTIONABLE: Direkt umsetzbare Geschenk-Insights
    """
    
    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "extra": "forbid",
        "use_enum_values": True
    }
    
    # === PERSÖNLICHKEITS-ÜBERSICHT ===
    personality_summary: Annotated[
        str,
        Field(
            min_length=50, max_length=800,
            description="Zusammenfassende Persönlichkeitsbeschreibung"
        )
    ]
    
    personality_archetype: str = Field(
        ..., max_length=100,
        description="Persönlichkeits-Archetyp (z.B. 'Kreativer Harmonisierer')"
    )
    
    dominant_traits: List[str] = Field(
        ..., min_items=1, max_items=5,
        description="Dominante Persönlichkeitseigenschaften"
    )
    
    # === BIG FIVE ANALYSIS ===
    big_five_insights: Dict[str, str] = Field(
        ...,
        description="Detaillierte Insights zu jeder Big Five Dimension"
    )
    
    big_five_gift_implications: Dict[str, List[str]] = Field(
        ...,
        description="Was jede Big Five Dimension für Geschenke bedeutet"
    )
    
    # === LIMBIC SYSTEM ANALYSIS ===
    limbic_type: str = Field(
        ..., max_length=50,
        description="Limbic Persönlichkeitstyp"
    )
    
    emotional_drivers: List[str] = Field(
        ..., min_items=1, max_items=6,
        description="Hauptemotionale Antriebe"
    )
    
    purchase_motivations: List[str] = Field(
        ..., min_items=1, max_items=8,
        description="Was motiviert zum 'Kauf' (emotionale Trigger)"
    )
    
    limbic_insights: Dict[str, str] = Field(
        ...,
        description="Detaillierte Limbic System Insights"
    )
    
    # === GIFT STRATEGY ===
    recommended_gift_categories: List[str] = Field(
        ..., min_items=2, max_items=10,
        description="Empfohlene Geschenk-Kategorien, priorisiert"
    )
    
    gift_dos: List[str] = Field(
        ..., min_items=3, max_items=8,
        description="Was bei Geschenken beachten (DOs)"
    )
    
    gift_donts: List[str] = Field(
        ..., min_items=2, max_items=6,
        description="Was bei Geschenken vermeiden (DON'Ts)"
    )
    
    emotional_appeal_strategies: List[str] = Field(
        ..., min_items=2, max_items=6,
        description="Wie man emotional ansprechen sollte"
    )
    
    # === PERSONALIZATION INSIGHTS ===
    personalization_opportunities: List[str] = Field(
        default_factory=list,
        description="Spezifische Personalisierungsmöglichkeiten"
    )
    
    relationship_considerations: Dict[str, str] = Field(
        default_factory=dict,
        description="Überlegungen je nach Beziehungstyp"
    )
    
    cultural_sensitivities: List[str] = Field(
        default_factory=list,
        description="Kulturelle Sensitivitäten zu beachten"
    )
    
    # === QUALITY & CONFIDENCE ===
    analysis_confidence: Annotated[
        float,
        Field(
            ge=0.0, le=1.0,
            description="Vertrauen in die Persönlichkeitsanalyse"
        )
    ]
    
    analysis_depth: Annotated[
        float,
        Field(
            ge=0.0, le=1.0,
            description="Tiefe der Analyse (0=oberflächlich, 1=sehr tiefgreifend)"
        )
    ]
    
    data_completeness: Annotated[
        float,
        Field(
            ge=0.0, le=1.0,
            description="Vollständigkeit der zugrundeliegenden Daten"
        )
    ]
    
    # === AI METADATA ===
    ai_model_used: Optional[str] = Field(None)
    analysis_time_ms: Optional[int] = Field(None, ge=0)
    advanced_techniques_used: List[str] = Field(default_factory=list)
    
    # === COMPUTED INSIGHTS ===
    @computed_field
    @property
    def overall_reliability(self) -> float:
        """Gesamte Verlässlichkeit der Analyse"""
        return (
            self.analysis_confidence * 0.5 +
            self.data_completeness * 0.3 +
            self.analysis_depth * 0.2
        )
    
    @computed_field
    @property
    def gift_strategy_summary(self) -> str:
        """Kurze Zusammenfassung der Geschenk-Strategie"""
        top_categories = self.recommended_gift_categories[:3]
        main_appeal = self.emotional_appeal_strategies[0] if self.emotional_appeal_strategies else "thoughtful personalization"
        
        return f"Fokus auf {', '.join(top_categories)} mit {main_appeal}"
    
    @computed_field
    @property
    def personality_complexity(self) -> str:
        """Einschätzung der Persönlichkeits-Komplexität"""
        trait_count = len(self.dominant_traits)
        driver_count = len(self.emotional_drivers)
        total_complexity = trait_count + driver_count
        
        if total_complexity >= 8:
            return "Hochkomplex"
        elif total_complexity >= 6:
            return "Komplex"
        elif total_complexity >= 4:
            return "Moderat"
        else:
            return "Einfach"
    
    def get_gift_guidance_for_relationship(self, relationship_type: str) -> Dict[str, Any]:
        """Spezifische Guidance für einen Beziehungstyp"""
        base_guidance = {
            "recommended_categories": self.recommended_gift_categories[:5],
            "emotional_approach": self.emotional_appeal_strategies[:3],
            "dos": self.gift_dos[:5],
            "donts": self.gift_donts[:3]
        }
        
        # Relationship-specific considerations
        if relationship_type in self.relationship_considerations:
            base_guidance["special_considerations"] = self.relationship_considerations[relationship_type]
        
        return base_guidance


# =============================================================================
# 🎯 GIFT RECOMMENDATION RESPONSE: Vollständige AI-Antwort
# =============================================================================

class GiftRecommendationResponse(BaseModel):
    """
    Vollständige AI-Response mit Geschenkempfehlungen (ENHANCED V2)
    
    🎁 COMPREHENSIVE: Alles was für perfekte Geschenke nötig ist
    🧠 INTELLIGENT: KI-gestützte Insights und Begründungen
    ❤️ EMOTIONAL: Fokus auf Gefühle und menschliche Verbindungen
    """
    
    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "extra": "forbid",
        "use_enum_values": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat()
        }
    }
    
    # === MAIN RECOMMENDATIONS ===
    recommendations: List[GiftRecommendation] = Field(
        ..., min_items=1, max_items=10,
        description="Hauptgeschenkempfehlungen, nach Qualität sortiert"
    )
    
    # === PERSONALITY CONTEXT ===
    personality_analysis: PersonalityAnalysisResult = Field(
        ...,
        description="Detaillierte Persönlichkeitsanalyse die zur Empfehlung führte"
    )
    
    # === RESPONSE INSIGHTS ===
    overall_strategy: Annotated[
        str,
        Field(
            min_length=30, max_length=600,
            description="Übergeordnete Strategie für diese Geschenkauswahl"
        )
    ]
    
    key_considerations: List[str] = Field(
        ..., min_items=2, max_items=8,
        description="Wichtigste Überlegungen die in die Auswahl eingeflossen sind"
    )
    
    emotional_themes: List[str] = Field(
        ..., min_items=1, max_items=6,
        description="Zentrale emotionale Themen dieser Empfehlungen"
    )
    
    # === ALTERNATIVE OPTIONS ===
    budget_alternatives: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Alternative Empfehlungen für verschiedene Budgets"
    )
    
    last_minute_options: List[str] = Field(
        default_factory=list,
        description="Last-Minute Optionen falls die Zeit knapp wird"
    )
    
    experience_vs_material: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Aufschlüsselung in Erlebnisse vs. materielle Geschenke"
    )
    
    # === CONTEXT & OCCASION ===
    occasion_specific_advice: Optional[str] = Field(
        None, max_length=500,
        description="Spezifische Ratschläge für den Anlass"
    )
    
    relationship_guidance: Optional[str] = Field(
        None, max_length=500,
        description="Guidance spezifisch für diese Beziehung"
    )
    
    cultural_considerations: List[str] = Field(
        default_factory=list,
        description="Kulturelle Aspekte zu beachten"
    )
    
    timing_recommendations: Optional[str] = Field(
        None, max_length=300,
        description="Wann und wie übergeben"
    )
    
    # === QUALITY METRICS ===
    overall_confidence: Annotated[
        float,
        Field(
            ge=0.0, le=1.0,
            description="Gesamtvertrauen in die Empfehlungen"
        )
    ]
    
    personalization_score: Annotated[
        float,
        Field(
            ge=0.0, le=1.0,
            description="Wie personalisiert sind die Empfehlungen"
        )
    ]
    
    novelty_score: Annotated[
        float,
        Field(
            ge=0.0, le=1.0,
            description="Wie neuartig/überraschend sind die Empfehlungen"
        )
    ]
    
    emotional_resonance: Annotated[
        float,
        Field(
            ge=0.0, le=1.0,
            description="Erwartete emotionale Resonanz beim Empfänger"
        )
    ]
    
    # === AI METADATA ===
    ai_model_used: str = Field(...)
    processing_time_ms: int = Field(..., ge=0)
    prompt_strategy: str = Field(...)
    optimization_goal: OptimizationObjective = Field(...)
    
    advanced_techniques_applied: List[AdvancedTechnique] = Field(default_factory=list)
    fallback_used: bool = Field(default=False)
    consensus_validation: Optional[bool] = Field(None)
    
    # === ENHANCED OPTIMIZATION METADATA ===
    model_selection_reason: Optional[str] = Field(
        None, max_length=200,
        description="Why this specific model was selected"
    )
    
    optimization_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Detailed optimization metrics"
    )
    
    cost_breakdown: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Cost breakdown by component"
    )
    
    performance_insights: List[str] = Field(
        default_factory=list,
        description="Performance insights and recommendations"
    )
    
    next_optimization_suggestions: List[str] = Field(
        default_factory=list,
        description="Suggestions for next optimization cycle"
    )
    
    # === RESPONSE METADATA ===
    response_id: Optional[str] = Field(None)
    generated_at: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = Field(None)
    
    # === FEEDBACK & LEARNING ===
    improvement_suggestions: List[str] = Field(
        default_factory=list,
        description="Wie könnte die nächste Empfehlung noch besser werden"
    )
    
    feedback_request: Optional[str] = Field(
        None, max_length=200,
        description="Spezifische Feedback-Bitte an den User"
    )
    
    # === COMPUTED PROPERTIES ===
    @computed_field
    @property
    def response_quality_score(self) -> float:
        """Kombinierter Response-Qualitätsscore"""
        # Average recommendation quality
        avg_rec_quality = sum(rec.overall_score for rec in self.recommendations) / len(self.recommendations)
        
        return (
            avg_rec_quality * 0.4 +
            self.overall_confidence * 0.3 +
            self.personalization_score * 0.2 +
            self.emotional_resonance * 0.1
        )
    
    @computed_field
    @property
    def response_summary(self) -> str:
        """Kurze Zusammenfassung der Response"""
        rec_count = len(self.recommendations)
        top_category = self.recommendations[0].category.value.replace('_', ' ').title()
        confidence_desc = "sehr confident" if self.overall_confidence > 0.8 else "confident" if self.overall_confidence > 0.6 else "moderat confident"
        
        return f"{rec_count} {confidence_desc}e Empfehlungen, Fokus: {top_category}"
    
    @computed_field
    @property
    def emotional_summary(self) -> Dict[str, Any]:
        """Emotionale Zusammenfassung der Empfehlungen"""
        all_emotional_tags = []
        for rec in self.recommendations:
            all_emotional_tags.extend(rec.emotional_tags)
        
        # Count emotion frequency
        emotion_counts = defaultdict(int)
        for tag in all_emotional_tags:
            emotion_counts[tag] += 1
        
        top_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "dominant_emotions": [emotion for emotion, _ in top_emotions],
            "emotional_diversity": len(set(all_emotional_tags)),
            "average_memory_potential": sum(rec.memory_potential for rec in self.recommendations) / len(self.recommendations),
            "average_surprise_factor": sum(rec.surprise_factor for rec in self.recommendations) / len(self.recommendations),
            "emotional_coherence": self.emotional_resonance
        }
    
    def get_executive_summary(self) -> str:
        """Executive Summary für schnelle Übersicht"""
        summary = f"🎁 **Geschenkempfehlungen für {self.personality_analysis.personality_archetype}**\n\n"
        summary += f"**Strategie:** {self.overall_strategy}\n\n"
        summary += f"**Top-Empfehlung:** {self.recommendations[0].title}\n"
        summary += f"**Emotionaler Fokus:** {', '.join(self.emotional_themes[:3])}\n"
        summary += f"**Qualität:** {self.response_summary}\n\n"
        
        if self.occasion_specific_advice:
            summary += f"**Anlass-Tipp:** {self.occasion_specific_advice}\n\n"
        
        return summary
    
    def get_recommendations_by_budget(self, max_budget: float) -> List[GiftRecommendation]:
        """Filtert Empfehlungen nach Budget"""
        filtered = []
        
        for rec in self.recommendations:
            # Extract max price from price_range
            import re
            price_match = re.findall(r'€(\d+)', rec.price_range)
            if price_match:
                max_price = float(price_match[-1])  # Take highest price
                if max_price <= max_budget:
                    filtered.append(rec)
        
        return filtered
    
    @field_validator('recommendations')
    @classmethod
    def sort_recommendations_by_quality(cls, v):
        """Sortiert Empfehlungen nach Overall-Score"""
        return sorted(v, key=lambda x: x.overall_score, reverse=True)
    
    @model_validator(mode='after')
    def ensure_response_coherence(self):
        """Stellt Kohärenz der gesamten Response sicher"""
        
        # Ensure emotional themes align with recommendations
        rec_emotions = []
        for rec in self.recommendations:
            rec_emotions.extend(rec.emotional_tags)
        
        # Update emotional themes if they don't align
        if self.emotional_themes:
            theme_alignment = any(
                theme.lower() in ' '.join(rec_emotions).lower() 
                for theme in self.emotional_themes
            )
            
            if not theme_alignment and rec_emotions:
                # Add most common emotion from recommendations
                emotion_counts = defaultdict(int)
                for emotion in rec_emotions:
                    emotion_counts[emotion] += 1
                
                if emotion_counts:
                    most_common = max(emotion_counts.items(), key=lambda x: x[1])[0]
                    if most_common not in self.emotional_themes:
                        self.emotional_themes.append(most_common)
        
        return self


# =============================================================================
# ⚡ QUICK RECOMMENDATION RESPONSE: Für Speed-optimierte Anfragen
# =============================================================================

class QuickRecommendationResponse(BaseModel):
    """
    Schnelle Geschenkempfehlungen für eilige Situationen (ENHANCED V2)
    
    ⚡ SPEED: Optimiert für Groq und schnelle Entscheidungen
    🎯 FOCUSED: Weniger Details, aber trotzdem emotional intelligent
    🚀 PRACTICAL: Sofort umsetzbare Empfehlungen
    """
    
    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "extra": "forbid",
        "use_enum_values": True
    }
    
    # === QUICK RECOMMENDATIONS ===
    quick_picks: List[Dict[str, Any]] = Field(
        ..., min_items=3, max_items=6,
        description="Schnelle Top-Picks mit essentiellen Infos"
    )
    
    # === CONDENSED INSIGHTS ===
    personality_snapshot: Annotated[
        str,
        Field(
            min_length=20, max_length=200,
            description="Kompakte Persönlichkeits-Übersicht"
        )
    ]
    
    key_insight: Annotated[
        str,
        Field(
            min_length=15, max_length=150,
            description="Wichtigste Erkenntnis für Geschenkauswahl"
        )
    ]
    
    emotional_focus: str = Field(
        ..., max_length=100,
        description="Emotionaler Hauptfokus"
    )
    
    # === QUICK GUIDANCE ===
    dos: List[str] = Field(
        ..., min_items=2, max_items=4,
        description="Wichtigste DOs"
    )
    
    donts: List[str] = Field(
        ..., min_items=1, max_items=3,
        description="Wichtigste DON'Ts"
    )
    
    # === QUICK METADATA ===
    confidence: Annotated[
        float,
        Field(ge=0.0, le=1.0)
    ]
    
    processing_time_ms: int = Field(..., ge=0)
    ai_model_used: str = Field(...)
    
    @computed_field
    @property
    def quick_summary(self) -> str:
        """Ein-Satz Zusammenfassung"""
        top_pick = self.quick_picks[0]['title'] if self.quick_picks else "Personalisierte Empfehlung"
        return f"Top-Tipp: {top_pick} - {self.emotional_focus}"


# =============================================================================
# 📊 PERFORMANCE & ANALYTICS: AI-Model Performance Tracking
# =============================================================================

class AIModelPerformanceMetrics(BaseModel):
    """
    Performance-Metriken für AI-Model Tracking (ENHANCED V2)
    
    📊 ANALYTICS: Verfolge Model-Performance über Zeit
    🚀 OPTIMIZATION: Basis für kontinuierliche Verbesserung
    🎯 QUALITY: Messe was wirklich zählt - User-Zufriedenheit
    """
    
    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "extra": "forbid",
        "use_enum_values": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v)
        }
    }
    
    # === MODEL IDENTIFICATION ===
    model_name: str = Field(...)
    model_version: Optional[str] = Field(None)
    request_id: Optional[str] = Field(None)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # === CORE PERFORMANCE METRICS ===
    response_time_ms: int = Field(..., ge=0)
    tokens_used: Optional[int] = Field(None, ge=0)
    cost_estimate: Optional[Decimal] = Field(None, ge=0)
    
    # === QUALITY METRICS ===
    output_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    user_satisfaction_predicted: Optional[float] = Field(None, ge=0.0, le=1.0)
    personalization_quality: Optional[float] = Field(None, ge=0.0, le=1.0)
    emotional_resonance_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # === SUCCESS METRICS ===
    request_successful: bool = Field(...)
    had_errors: bool = Field(default=False)
    required_fallback: bool = Field(default=False)
    required_retry: bool = Field(default=False)
    
    # === CONTEXT METRICS ===
    request_type: str = Field(...)
    complexity_level: str = Field(...)
    optimization_goal: str = Field(...)
    
    # === BUSINESS METRICS ===
    user_interaction_quality: Optional[float] = Field(None, ge=0.0, le=1.0)
    recommendation_acceptance_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    gift_purchase_likelihood: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # === EMOTIONAL INTELLIGENCE METRICS ===
    emotional_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    empathy_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    cultural_sensitivity: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    @computed_field
    @property
    def overall_performance_score(self) -> float:
        """Kombinierter Performance-Score"""
        scores = []
        
        if self.output_quality_score:
            scores.append(self.output_quality_score * 0.3)
        
        if self.user_satisfaction_predicted:
            scores.append(self.user_satisfaction_predicted * 0.25)
        
        if self.emotional_resonance_score:
            scores.append(self.emotional_resonance_score * 0.2)
        
        if self.personalization_quality:
            scores.append(self.personalization_quality * 0.15)
        
        # Success penalty
        success_score = 1.0 if self.request_successful and not self.had_errors else 0.5
        scores.append(success_score * 0.1)
        
        return sum(scores) / len(scores) if scores else 0.5


# =============================================================================
# ❌ ERROR RESPONSE: Wenn etwas schiefgeht
# =============================================================================

class ErrorResponse(BaseModel):
    """
    Error Response mit hilfreichen Informationen (ENHANCED V2)
    
    ❌ TRANSPARENT: Klare Fehlerkommunikation
    🔧 ACTIONABLE: Was kann der User tun?
    🤝 EMPATHETIC: Auch Fehler können menschlich kommuniziert werden
    """
    
    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "extra": "forbid",
        "use_enum_values": True
    }
    
    # === ERROR DETAILS ===
    error_type: str = Field(...)
    error_message: Annotated[
        str,
        Field(min_length=10, max_length=500)
    ]
    
    user_friendly_message: Annotated[
        str,
        Field(
            min_length=20, max_length=300,
            description="Benutzerfreundliche Fehlererklärung"
        )
    ]
    
    # === RECOVERY SUGGESTIONS ===
    suggested_actions: List[str] = Field(
        ..., min_items=1, max_items=5,
        description="Was kann der User tun um das Problem zu lösen"
    )
    
    fallback_recommendations: List[str] = Field(
        default_factory=list,
        description="Fallback-Empfehlungen falls verfügbar"
    )
    
    # === TECHNICAL DETAILS ===
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = Field(None)
    ai_model_attempted: Optional[str] = Field(None)
    processing_stage: Optional[str] = Field(None)
    
    # === RETRY INFORMATION ===
    is_retryable: bool = Field(default=True)
    retry_suggestions: List[str] = Field(default_factory=list)
    estimated_retry_success: Optional[float] = Field(None, ge=0.0, le=1.0)


# =============================================================================
# 🏆 SPECIALIZED QUIZ RESULT: Für Persönlichkeits-Quiz Integration
# =============================================================================

class PersonalityQuizResult(BaseModel):
    """
    Ergebnis eines Persönlichkeits-Quiz mit Geschenk-Integration (ENHANCED V2)
    
    🧩 INTERACTIVE: Ergebnisse von interaktiven Persönlichkeits-Tests
    🎁 ACTIONABLE: Direkt zu Geschenkempfehlungen verknüpft
    🎯 ENGAGING: Macht Persönlichkeitsanalyse zum Erlebnis
    """
    
    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "extra": "forbid",
        "use_enum_values": True
    }
    
    # === QUIZ METADATA ===
    quiz_type: str = Field(...)
    quiz_version: str = Field(...)
    completion_time_seconds: int = Field(..., ge=0)
    
    # === QUIZ RESULTS ===
    raw_scores: Dict[str, float] = Field(...)
    personality_type: str = Field(...)
    type_description: Annotated[
        str,
        Field(min_length=50, max_length=800)
    ]
    
    # === GIFT CONNECTIONS ===
    gift_personality_match: Annotated[
        str,
        Field(
            min_length=30, max_length=500,
            description="Wie die Quiz-Ergebnisse zu Geschenk-Präferenzen führen"
        )
    ]
    
    immediate_gift_suggestions: List[str] = Field(
        ..., min_items=3, max_items=8,
        description="Sofortige Geschenk-Ideen basierend auf Quiz"
    )
    
    # === ENGAGEMENT ===
    fun_facts: List[str] = Field(
        default_factory=list,
        description="Interessante Facts über diesen Persönlichkeitstyp"
    )
    
    famous_people_same_type: List[str] = Field(
        default_factory=list,
        description="Berühmte Personen mit ähnlicher Persönlichkeit"
    )
    
    strengths: List[str] = Field(..., min_items=2, max_items=6)
    potential_blind_spots: List[str] = Field(default_factory=list)
    
    @computed_field
    @property
    def quiz_summary(self) -> str:
        """Kurze Quiz-Zusammenfassung"""
        return f"Du bist ein {self.personality_type} - {self.type_description[:100]}..."


# =============================================================================
# EXPORTS: Clean Interface für andere Module
# =============================================================================

__all__ = [
    # === ENUMS ===
    'ConfidenceLevel', 'GiftCategory', 'RecommendationReason', 'OptimizationObjective',
    'ModelType', 'AdvancedTechnique',
    
    # === CORE MODELS ===
    'GiftRecommendation', 'PersonalityAnalysisResult', 'GiftRecommendationResponse',
    
    # === SPECIALIZED MODELS ===
    'QuickRecommendationResponse', 'PersonalityQuizResult',
    'AIModelPerformanceMetrics', 'ErrorResponse'
]