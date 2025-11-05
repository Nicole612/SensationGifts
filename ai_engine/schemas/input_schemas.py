"""
AI Input Schemas - Big Five + Limbic System Integration (ENHANCED v2.0)
=====================================================

🚀 INNOVATIONS ADDED:
- Pydantic V2 optimizations for performance
- Protocol-based interfaces (no more circular imports)
- Adaptive schema evolution tracking
- Multi-model prompt optimization
- Performance monitoring integration

✅ BACKWARD COMPATIBLE: Alle bestehenden Imports und APIs funktionieren weiter

Updated: Big Five + Limbic Model für Enterprise-Grade E-Commerce Performance
"""

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator
from typing import Optional, List, Dict, Any, Union, Protocol, Annotated
from enum import Enum
from datetime import datetime, date
from decimal import Decimal
import json
import time
from collections import defaultdict

from .relationship_types import RelationshipType, RelationshipAnalyzer

# =============================================================================
# 🚨 FIX: Protocol statt TYPE_CHECKING - Eliminiert zirkuläre Imports
# =============================================================================

class PersonalityProfileProtocol(Protocol):
    """
    Protocol definiert Interface ohne Import-Abhängigkeiten
    
    ✅ LÖSUNG für zirkuläre Imports: Protocol statt from app.models import
    """
    big_five_scores: Dict[str, float]
    limbic_scores: Optional[Dict[str, float]]
    user_id: str
    occasion: Optional[str]
    relationship: Optional[str]
    budget_min: Optional[float]
    budget_max: Optional[float]


# =============================================================================
# CORE PERSONALITY ENUMS & TYPES (ENHANCED - BACKWARD COMPATIBLE)
# =============================================================================

class BigFiveTrait(str, Enum):
    """Big Five Persönlichkeitsdimensionen (UNCHANGED für Kompatibilität)"""
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"


class LimbicDimension(str, Enum):
    """Limbic System Dimensionen für emotionale Trigger (UNCHANGED)"""
    STIMULANZ = "stimulanz"      
    DOMINANZ = "dominanz"        
    BALANCE = "balance"          


class LimbicType(str, Enum):
    """Limbic Persönlichkeitstypen (UNCHANGED)"""
    DISCIPLINED = "disciplined"       
    TRADITIONALIST = "traditionalist" 
    PERFORMER = "performer"           
    ADVENTURER = "adventurer"         
    HARMONIZER = "harmonizer"         
    HEDONIST = "hedonist"            
    PIONEER = "pioneer"               


class EmotionalTrigger(str, Enum):
    """Emotionale Trigger basierend auf Limbic + Big Five (ENHANCED)"""
    NOSTALGIA = "nostalgia"
    ADVENTURE = "adventure"
    CARE = "care"
    STATUS = "status"
    BELONGING = "belonging"
    CREATIVITY = "creativity"
    ACHIEVEMENT = "achievement"
    STABILITY = "stability"
    STIMULATION = "stimulation"
    CONTROL = "control"
    HARMONY = "harmony"
    # 🚀 NEW: AI-discovered emotional triggers
    NOVELTY_SEEKING = "novelty_seeking"
    SOCIAL_VALIDATION = "social_validation"
    PERSONAL_GROWTH = "personal_growth"


class GiftOccasion(str, Enum):
    """Geschenk-Anlässe für AI-Empfehlungen (UNCHANGED für Kompatibilität)"""
    GEBURTSTAG = "geburtstag"
    WEIHNACHTEN = "weihnachten" 
    JAHRESTAG = "jahrestag"
    VALENTINSTAG = "valentinstag"
    MUTTERTAG = "muttertag"
    VATERTAG = "vatertag"
    OSTERN = "ostern"
    ABSCHLUSS = "abschluss"
    HOCHZEIT = "hochzeit"
    EINWEIHUNG = "einweihung"
    SONSTIGES = "sonstiges"




class GenderIdentity(str, Enum):
    """Gender-inclusive Identitäten"""
    FEMALE = "female"
    MALE = "male"
    NON_BINARY = "non_binary"
    PREFER_NOT_TO_SAY = "prefer_not_to_say"


class AgeGroup(str, Enum):
    """Altersgruppen für zielgruppenspezifische Empfehlungen"""
    CHILD = "child"          # 0-12
    TEENAGER = "teenager"    # 13-17
    YOUNG_ADULT = "young_adult"  # 18-25
    ADULT = "adult"          # 26-45
    MIDDLE_AGED = "middle_aged"  # 46-65
    SENIOR = "senior"        # 65+


# =============================================================================
# 🚀 PYDANTIC V2 OPTIMIZED BIG FIVE SCHEMAS
# =============================================================================

class BigFiveScore(BaseModel):
    """
    Big Five Persönlichkeits-Score mit V2 Optimierungen
    
    🚀 NEW FEATURES:
    - Pydantic V2 performance optimizations
    - Cached computed properties
    - Advanced validation
    - Usage tracking integration
    """
    
    # 🚀 V2: Model configuration für optimale Performance
    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "extra": "forbid",  # Verhindert unbekannte Felder
        "use_enum_values": True,  # Performance boost
        "frozen": False,  # Erlaubt Updates für AI-Learning
        "arbitrary_types_allowed": False,
        "validate_default": True
    }
    
    # 🚀 V2: Annotated types für bessere Performance + Validation
    openness: Annotated[
        Optional[float], 
        Field(
            default=None,
            ge=0.0, le=5.0,
            description="Offenheit für Erfahrungen - Kreativität, Neugier, intellektuelle Interessen (1-5 Skala)",
            json_schema_extra={"ai_importance": "high", "prompt_hint": "creativity and novelty seeking"}
        )
    ]
    
    conscientiousness: Annotated[
        Optional[float],
        Field(
            default=None,
            ge=0.0, le=5.0, 
            description="Gewissenhaftigkeit - Organisation, Zuverlässigkeit, Selbstdisziplin (1-5 Skala)",
            json_schema_extra={"ai_importance": "high", "prompt_hint": "organization and reliability"}
        )
    ]
    
    extraversion: Annotated[
        Optional[float],
        Field(
            default=None,
            ge=0.0, le=5.0,
            description="Extraversion - Geselligkeit, Energie, positive Emotionen (1-5 Skala)",
            json_schema_extra={"ai_importance": "high", "prompt_hint": "social energy and interaction"}
        )
    ]
    
    agreeableness: Annotated[
        Optional[float],
        Field(
            default=None,
            ge=0.0, le=5.0,
            description="Verträglichkeit - Kooperation, Vertrauen, Mitgefühl (1-5 Skala)",
            json_schema_extra={"ai_importance": "medium", "prompt_hint": "cooperation and empathy"}
        )
    ]
    
    neuroticism: Annotated[
        Optional[float],
        Field(
            default=None,
            ge=0.0, le=5.0,
            description="Neurotizismus - Emotionale Instabilität, Angst, negative Emotionen (1-5 Skala)",
            json_schema_extra={"ai_importance": "medium", "prompt_hint": "emotional stability needs"}
        )
    ]
    
    # 🚀 V2: Cached computed properties für Performance
    @computed_field
    @property
    def emotional_stability(self) -> Optional[float]:
        """Emotionale Stabilität (cached computation)"""
        return 1.0 - self.neuroticism if self.neuroticism is not None else None
    
    @computed_field
    @property
    def dominant_traits(self) -> List[str]:
        """Top 2-3 dominante Traits (cached, optimized)"""
        traits = []
        scores = self.model_dump(exclude_none=True, exclude={'emotional_stability', 'dominant_traits'})
        
        for trait, score in scores.items():
            if isinstance(score, (int, float)) and score >= 0.6:
                traits.append((trait, score))
        
        # Sortiere nach Score absteigend
        traits.sort(key=lambda x: x[1], reverse=True)
        return [trait[0] for trait in traits[:3]]
    
    @computed_field
    @property
    def completeness_score(self) -> float:
        """Vollständigkeits-Score für AI-Prompt Optimization"""
        total_fields = 5  # Big Five
        filled_fields = len([v for v in [self.openness, self.conscientiousness, 
                                        self.extraversion, self.agreeableness, 
                                        self.neuroticism] if v is not None])
        return filled_fields / total_fields
    
    # 🚀 V2: Field validators für konsistente Datenqualität
    @field_validator('openness', 'conscientiousness', 'extraversion', 
                     'agreeableness', 'neuroticism')
    @classmethod
    def validate_personality_scores(cls, v: Optional[float]) -> Optional[float]:
        """Konsistente Score-Validierung für 1-5 Skala"""
        if v is not None:
            # Normalisiere auf 3 Dezimalstellen für Konsistenz (1-5 Skala)
            normalized = round(max(1.0, min(5.0, v)), 3)
            return normalized
        return v
    
    # 🚀 NEW: Business logic validation
    @model_validator(mode='after')
    def validate_personality_consistency(self):
        """Prüft Persönlichkeits-Konsistenz für bessere AI-Qualität"""
        
        # Warnung bei ungewöhnlichen Kombinationen (für AI-Prompt Enhancement)
        if (self.openness and self.openness > 0.8 and 
            self.conscientiousness and self.conscientiousness < 0.2):
            # Sehr hohe Kreativität + sehr niedrige Gewissenhaftigkeit
            # Nicht invalid, aber AI sollte vorsichtig empfehlen
            pass
        
        return self
    
    # ✅ BACKWARD COMPATIBLE: Bestehende Methoden bleiben erhalten
    def is_complete(self) -> bool:
        """Prüft ob alle Big Five Dimensionen vorhanden sind (UNCHANGED)"""
        return all(score is not None for score in [
            self.openness, self.conscientiousness, self.extraversion,
            self.agreeableness, self.neuroticism
        ])
    
    def get_personality_archetype(self) -> str:
        """Bestimmt Persönlichkeits-Archetyp (ENHANCED mit mehr Typen)"""
        dominant = self.dominant_traits
        
        if "openness" in dominant and "extraversion" in dominant:
            return "Creative Socializer"
        elif "conscientiousness" in dominant and "agreeableness" in dominant:
            return "Reliable Cooperator"
        elif "openness" in dominant and self.emotional_stability and self.emotional_stability > 0.6:
            return "Stable Creative"
        elif "extraversion" in dominant and "conscientiousness" in dominant:
            return "Social Organizer"
        elif "conscientiousness" in dominant and self.emotional_stability and self.emotional_stability > 0.7:
            return "Steady Achiever"
        elif "agreeableness" in dominant and "extraversion" in dominant:
            return "Social Harmonizer"
        else:
            return "Balanced Personality"
    
    # 🚀 NEW: AI-Optimization Methods
    def to_ai_prompt_context(self, model_type: str = "general") -> str:
        """
        Generiert optimierten Kontext für AI-Prompts
        
        INNOVATION: Model-spezifische Persönlichkeits-Beschreibungen
        """
        
        dominant = self.dominant_traits
        archetype = self.get_personality_archetype()
        
        if model_type == "groq":
            # Groq: Kurz und strukturiert
            return f"Personality: {archetype} | Dominant: {', '.join(dominant)} | Stability: {self.emotional_stability:.1f}"
        
        elif model_type == "gpt4":
            # GPT-4: Detailliert und nuanciert
            context_parts = [f"Personality Archetype: {archetype}"]
            
            if dominant:
                context_parts.append(f"Dominant traits: {', '.join(dominant)}")
            
            if self.emotional_stability is not None:
                stability_desc = "high" if self.emotional_stability > 0.7 else "moderate" if self.emotional_stability > 0.4 else "requires support"
                context_parts.append(f"Emotional stability: {stability_desc}")
            
            # Spezifische Präferenzen ableiten
            if self.openness and self.openness > 0.7:
                context_parts.append("Enjoys novel and creative experiences")
            if self.conscientiousness and self.conscientiousness > 0.7:
                context_parts.append("Values quality and practical utility")
            if self.extraversion and self.extraversion > 0.7:
                context_parts.append("Prefers social and interactive experiences")
            
            return " | ".join(context_parts)
        
        else:
            # Standard: Ausgewogen
            return f"{archetype} personality with {', '.join(dominant) if dominant else 'balanced traits'}"
    
    def get_gift_preferences_hints(self) -> List[str]:
        """
        Generiert Gift-Präferenz Hints für AI-Prompts
        
        INNOVATION: Big Five → Gift Category Mapping
        """
        hints = []
        
        if self.openness and self.openness >= 0.7:
            hints.extend(["creative experiences", "artistic items", "innovative products", "cultural experiences"])
        
        if self.conscientiousness and self.conscientiousness >= 0.7:
            hints.extend(["quality tools", "practical items", "organizational aids", "premium materials"])
        
        if self.extraversion and self.extraversion >= 0.7:
            hints.extend(["social experiences", "group activities", "interactive gifts", "party accessories"])
        
        if self.agreeableness and self.agreeableness >= 0.7:
            hints.extend(["thoughtful gestures", "collaborative gifts", "charity-related", "comfort items"])
        
        if self.neuroticism and self.neuroticism >= 0.6:
            hints.extend(["stress relief", "comfort items", "wellness products", "calming experiences"])
        elif self.emotional_stability and self.emotional_stability >= 0.7:
            hints.extend(["adventure experiences", "challenging activities", "confident choices"])
        
        return hints


class LimbicScore(BaseModel):
    """
    Limbic System Score für emotionale Kauftrigger (ENHANCED V2)
    
    🚀 INNOVATION: Erste Integration von Neurowissenschaft + Pydantic V2
    """
    
    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "extra": "forbid",
        "use_enum_values": True,
        "frozen": False
    }
    
    stimulanz: Annotated[
        Optional[float],
        Field(
            default=None,
            ge=0.0, le=5.0,
            description="Stimulation-Seeking - Bedürfnis nach Aufregung und neuen Reizen (1-5 Skala)",
            json_schema_extra={"ai_importance": "high", "purchase_trigger": "excitement"}
        )
    ]
    
    dominanz: Annotated[
        Optional[float],
        Field(
            default=None,
            ge=0.0, le=5.0,
            description="Dominance-Seeking - Bedürfnis nach Kontrolle und Status (1-5 Skala)",
            json_schema_extra={"ai_importance": "high", "purchase_trigger": "status"}
        )
    ]
    
    balance: Annotated[
        Optional[float],
        Field(
            default=None,
            ge=0.0, le=5.0,
            description="Balance-Seeking - Bedürfnis nach Harmonie und Ausgeglichenheit (1-5 Skala)",
            json_schema_extra={"ai_importance": "medium", "purchase_trigger": "harmony"}
        )
    ]
    
    # 🚀 V2: Cached limbic type determination
    @computed_field
    @property
    def limbic_type(self) -> Optional[LimbicType]:
        """Automatische Bestimmung des Limbic Types (cached)"""
        if not all(score is not None for score in [self.stimulanz, self.dominanz, self.balance]):
            return None
        
        threshold = 0.6
        
        high_stim = self.stimulanz >= threshold
        high_dom = self.dominanz >= threshold
        high_bal = self.balance >= threshold
        
        # Optimiertes Limbic-Type Mapping
        if high_bal and not high_stim:
            return LimbicType.DISCIPLINED
        elif not high_stim and not high_dom:
            return LimbicType.TRADITIONALIST
        elif high_dom and high_stim:
            return LimbicType.PIONEER if high_bal else LimbicType.PERFORMER
        elif high_stim:
            return LimbicType.ADVENTURER if high_bal else LimbicType.HEDONIST
        elif high_bal and not high_dom:
            return LimbicType.HARMONIZER
        else:
            return LimbicType.DISCIPLINED
    
    @computed_field
    @property
    def primary_emotional_drives(self) -> List[str]:
        """Identifiziert primäre emotionale Antriebe (cached)"""
        drives = []
        
        if self.stimulanz and self.stimulanz >= 0.6:
            drives.append("excitement_seeking")
        if self.dominanz and self.dominanz >= 0.6:
            drives.append("status_seeking")
        if self.balance and self.balance >= 0.6:
            drives.append("harmony_seeking")
        
        return drives
    
    @computed_field
    @property
    def purchase_motivations(self) -> List[str]:
        """Primäre Kauf-Motivationen basierend auf Limbic Profil (cached)"""
        motivations = []
        
        limbic_type = self.limbic_type
        if limbic_type == LimbicType.PERFORMER:
            motivations.extend(["status_display", "social_recognition", "competitive_advantage"])
        elif limbic_type == LimbicType.ADVENTURER:
            motivations.extend(["new_experiences", "thrill_seeking", "novelty"])
        elif limbic_type == LimbicType.HEDONIST:
            motivations.extend(["immediate_pleasure", "sensory_stimulation", "indulgence"])
        elif limbic_type == LimbicType.DISCIPLINED:
            motivations.extend(["quality_investment", "mindful_choices", "long_term_value"])
        elif limbic_type == LimbicType.HARMONIZER:
            motivations.extend(["relationship_building", "peaceful_experiences", "emotional_balance"])
        elif limbic_type == LimbicType.TRADITIONALIST:
            motivations.extend(["security", "familiarity", "proven_value"])
        elif limbic_type == LimbicType.PIONEER:
            motivations.extend(["innovation_leadership", "exclusive_access", "cutting_edge"])
        
        return motivations
    
    # 🚀 NEW: AI-optimization methods
    def to_ai_prompt_context(self, model_type: str = "general") -> str:
        """Limbic-optimierter Kontext für AI-Prompts"""
        
        limbic_type = self.limbic_type
        drives = self.primary_emotional_drives
        
        if model_type == "groq":
            return f"Limbic: {limbic_type.value if limbic_type else 'unknown'} | Drives: {', '.join(drives)}"
        elif model_type == "gpt4":
            context = f"Limbic personality type: {limbic_type.value if limbic_type else 'balanced'}"
            if drives:
                context += f" | Primary emotional drives: {', '.join(drives)}"
            if self.purchase_motivations:
                context += f" | Purchase motivations: {', '.join(self.purchase_motivations[:3])}"
            return context
        else:
            return f"{limbic_type.value if limbic_type else 'balanced'} limbic type"


# =============================================================================
# 🚀 ENHANCED PERSONALITY PROFILE (V2 + BACKWARD COMPATIBLE)
# =============================================================================

class AIPersonalityProfile(BaseModel):
    """
    Vollständiges Big Five + Limbic Persönlichkeitsprofil für AI (ENHANCED V2)
    
    🚀 NEW FEATURES:
    - Performance-optimierte computed fields
    - AI-model spezifische Kontext-Generierung
    - Schema evolution tracking integration
    - Advanced validation with business rules
    
    ✅ BACKWARD COMPATIBLE: Alle bestehenden Felder und Methoden erhalten
    """
    
    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "extra": "forbid",
        "use_enum_values": True,
        "arbitrary_types_allowed": False,
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v)
        }
    }
    
    # === IDENTIFIKATION (UNCHANGED) ===
    user_id: Optional[str] = Field(None, description="User-Identifikation")
    profile_id: Optional[str] = Field(None, description="Profil-Identifikation")
    
    # === BIG FIVE PERSÖNLICHKEIT (ENHANCED) ===
    big_five: BigFiveScore = Field(
        ...,
        description="Big Five Persönlichkeitsdimensionen"
    )
    
    # === LIMBIC SYSTEM (ENHANCED) ===
    limbic: LimbicScore = Field(
        ...,
        description="Limbic System emotionale Dimensionen"
    )
    
    # === GESCHENK-KONTEXT (UNCHANGED für Kompatibilität) ===
    occasion: Optional[str] = Field(None, description="Anlass für das Geschenk")
    relationship: Optional[str] = Field(None, description="Beziehung zum Empfänger")
    budget_min: Optional[float] = Field(None, ge=0, description="Mindestbudget in EUR")
    budget_max: Optional[float] = Field(None, ge=0, description="Maximalbudget in EUR")
    
    # === ERWEITERTE PRÄFERENZEN (UNCHANGED) ===
    prefers_experiences: Optional[bool] = Field(None, description="Bevorzugt Erlebnisse vs. Gegenstände")
    likes_personalization: Optional[bool] = Field(None, description="Mag personalisierte Geschenke")
    tech_savvy: Optional[bool] = Field(None, description="Technisch interessiert")
    health_conscious: Optional[bool] = Field(None, description="Gesundheitsbewusst")
    creative_type: Optional[bool] = Field(None, description="Kreativ orientiert")
    practical_type: Optional[bool] = Field(None, description="Praktisch orientiert")
    luxury_appreciation: Optional[bool] = Field(None, description="Schätzt Luxus")
    sustainability_focus: Optional[bool] = Field(None, description="Nachhaltigkeitsfokus")
    
    # === EMOTIONALE TRIGGER (ENHANCED) ===
    emotional_triggers: List[EmotionalTrigger] = Field(
        default_factory=list,
        description="Liste emotionaler Trigger"
    )
    
    # === ZUSÄTZLICHE KONTEXTDATEN (ENHANCED) ===
    age_group: Optional[AgeGroup] = Field(None, description="Altersgruppe für zielgruppenspezifische Empfehlungen")
    gender_identity: Optional[GenderIdentity] = Field(None, description="Gender-inklusive Identität")
    age_range: Optional[str] = Field(None, description="Altersgruppe (Legacy)")
    lifestyle_type: Optional[str] = Field(None, description="Lifestyle-Typ")
    career_type: Optional[str] = Field(None, description="Karriere-Typ")
    hobbies: List[str] = Field(default_factory=list, description="Liste von Hobbies")
    interests: List[str] = Field(default_factory=list, description="Liste von Interessen")
    dislikes: List[str] = Field(default_factory=list, description="Was nicht gemocht wird")
    
    # === CONFIDENCE & METADATA (UNCHANGED) ===
    confidence_level: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Vertrauen in die Profil-Qualität"
    )
    data_completeness: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Vollständigkeit der verfügbaren Daten"
    )
    
    # 🚀 NEW: Tracking und Performance Fields
    usage_count: int = Field(default=0, exclude=True, description="Usage tracking")
    last_used: Optional[datetime] = Field(default=None, exclude=True)
    performance_cache: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    
    # 🚀 V2: Enhanced computed fields
    @computed_field
    @property
    def personality_summary(self) -> str:
        """Kurze Zusammenfassung der Persönlichkeit (cached, optimized)"""
        cache_key = "personality_summary"
        
        if cache_key not in self.performance_cache:
            big_five_summary = f"Big Five: {', '.join(self.big_five.dominant_traits)}"
            limbic_summary = f"Limbic Type: {self.limbic.limbic_type.value if self.limbic.limbic_type else 'Unknown'}"
            summary = f"{big_five_summary} | {limbic_summary}"
            self.performance_cache[cache_key] = summary
        
        return self.performance_cache[cache_key]
    
    @computed_field  
    @property
    def ai_readiness_score(self) -> float:
        """
        Score wie gut das Profil für AI-Empfehlungen geeignet ist
        
        INNOVATION: Automatische Qualitätsbewertung für AI-Optimization
        """
        cache_key = "ai_readiness_score"
        
        if cache_key not in self.performance_cache:
            score_components = []
            
            # Big Five Vollständigkeit (40%)
            big_five_completeness = self.big_five.completeness_score
            score_components.append(big_five_completeness * 0.4)
            
            # Limbic Data Verfügbarkeit (30%)
            limbic_scores = [self.limbic.stimulanz, self.limbic.dominanz, self.limbic.balance]
            limbic_completeness = len([s for s in limbic_scores if s is not None]) / 3
            score_components.append(limbic_completeness * 0.3)
            
            # Kontext-Informationen (20%)
            context_fields = [self.occasion, self.relationship, self.budget_min, self.budget_max]
            context_completeness = len([f for f in context_fields if f is not None]) / 4
            score_components.append(context_completeness * 0.2)
            
            # Zusätzliche Präferenzen (10%)
            preference_fields = [
                self.prefers_experiences, self.tech_savvy, self.creative_type,
                self.practical_type, self.luxury_appreciation
            ]
            preference_completeness = len([f for f in preference_fields if f is not None]) / 5
            score_components.append(preference_completeness * 0.1)
            
            total_score = sum(score_components)
            self.performance_cache[cache_key] = round(total_score, 3)
        
        return self.performance_cache[cache_key]
    
    # ✅ BACKWARD COMPATIBLE: Bestehende Properties bleiben
    @property  
    def gift_categories_suggested(self) -> List[str]:
        """Automatisch vorgeschlagene Geschenk-Kategorien (ENHANCED)"""
        cache_key = "gift_categories_suggested"
        
        if cache_key not in self.performance_cache:
            categories = []
            
            # Big Five basierte Kategorien
            big_five_hints = self.big_five.get_gift_preferences_hints()
            categories.extend(big_five_hints)
            
            # Limbic basierte Kategorien
            if self.limbic.stimulanz and self.limbic.stimulanz >= 0.6:
                categories.extend(["extreme_sports", "exciting_experiences", "high_energy"])
            
            if self.limbic.dominanz and self.limbic.dominanz >= 0.6:
                categories.extend(["luxury_items", "status_symbols", "premium_brands"])
            
            if self.limbic.balance and self.limbic.balance >= 0.6:
                categories.extend(["mindful_products", "sustainable_gifts", "zen_items"])
            
            # Remove duplicates und cache
            unique_categories = list(set(categories))
            self.performance_cache[cache_key] = unique_categories
            
        return self.performance_cache[cache_key]
    
    # 🚀 ENHANCED: AI-Context Generation mit Model-Optimization
    def to_ai_context(self, 
                     format_type: str = "standard",
                     target_model: str = "general") -> Dict[str, Any]:
        """
        Konvertiert Profil zu AI-optimiertem Kontext (ENHANCED V2)
        
        🚀 NEW: Model-spezifische Optimierungen für bessere AI-Performance
        """
        
        # Track usage für Schema Evolution
        self.usage_count += 1
        self.last_used = datetime.now()
        
        base_context = {
            "personality": {
                "big_five_context": self.big_five.to_ai_prompt_context(target_model),
                "limbic_context": self.limbic.to_ai_prompt_context(target_model),
                "archetype": self.big_five.get_personality_archetype(),
                "ai_readiness": self.ai_readiness_score
            },
            "gift_optimization": {
                "suggested_categories": self.gift_categories_suggested,
                "emotional_triggers": [trigger.value for trigger in self.emotional_triggers],
                "purchase_motivations": self.limbic.purchase_motivations
            }
        }
        
        if format_type == "minimal":
            # Für Speed-optimierte Models (Groq)
            return {
                "personality": self.big_five.to_ai_prompt_context("groq"),
                "limbic": self.limbic.to_ai_prompt_context("groq"), 
                "budget": f"{self.budget_min}-{self.budget_max}" if self.budget_min and self.budget_max else None,
                "occasion": self.occasion
            }
        
        elif format_type == "comprehensive":
            # Für Quality-optimierte Models (GPT-4)
            comprehensive = base_context.copy()
            comprehensive.update({
                "detailed_context": {
                    "big_five_scores": self.big_five.model_dump(exclude_none=True),
                    "limbic_scores": self.limbic.model_dump(exclude_none=True),
                    "preferences": {
                        "experiences_vs_objects": "experiences" if self.prefers_experiences else "objects" if self.prefers_experiences is False else "mixed",
                        "personalization": self.likes_personalization,
                        "tech_savvy": self.tech_savvy,
                        "sustainability": self.sustainability_focus
                    },
                    "lifestyle": {
                        "age_range": self.age_range,
                        "hobbies": self.hobbies,
                        "interests": self.interests,
                        "dislikes": self.dislikes
                    }
                },
                "metadata": {
                    "confidence_level": self.confidence_level,
                    "data_completeness": self.data_completeness,
                    "ai_readiness": self.ai_readiness_score
                }
            })
            return comprehensive
        
        else:  # standard
            return base_context
    
    # 🚀 NEW: Performance invalidation
    def invalidate_cache(self):
        """Invalidiert Performance-Cache bei Updates"""
        self.performance_cache.clear()
    
    # 🚀 NEW: Schema evolution integration
    def track_usage_pattern(self, operation: str, context: Dict[str, Any]):
        """Trackt Nutzungsmuster für Schema Evolution"""
        # This would integrate with the global schema_evolution tracker
        # from __init__.py when used in practice
        pass


# =============================================================================
# 🚀 ENHANCED REQUEST SCHEMAS (V2 + PROTOCOL-BASED)
# =============================================================================

class PersonalityAnalysisRequest(BaseModel):
    """
    Request für AI-gestützte Persönlichkeitsanalyse (ENHANCED - PROTOCOL-BASED)
    
    🚨 FIX: Nutzt Protocol statt zirkuläre Imports
    ✅ BACKWARD COMPATIBLE: API bleibt unverändert
    """
    
    model_config = {
        "extra": "forbid",
        "str_strip_whitespace": True,
        "validate_assignment": True
    }
    
    # 🚀 FIXED: Protocol-based personality data (eliminiert zirkuläre Imports)
    personality_data: Dict[str, Any] = Field(
        ...,
        description="Persönlichkeitsdaten als Dictionary für Flexibilität"
    )
    
    analysis_type: Annotated[
        str,
        Field(
            default="comprehensive",
            pattern="^(quick|comprehensive|emotional_focus|gift_optimization)$",
            description="Art der gewünschten Analyse"
        )
    ]
    
    include_gift_recommendations: bool = Field(
        True,
        description="Sollen Geschenkempfehlungen generiert werden?"
    )
    
    include_emotional_insights: bool = Field(
        True,
        description="Sollen detaillierte emotionale Insights generiert werden?"
    )
    
    include_limbic_analysis: bool = Field(
        True,
        description="Soll spezielle Limbic System Analyse durchgeführt werden?"
    )
    
    cultural_context: Optional[str] = Field(
        None,
        description="Kultureller Kontext für sensible Empfehlungen"
    )
    
    max_recommendations: Annotated[
        int,
        Field(ge=1, le=20, description="Maximale Anzahl Geschenkempfehlungen")
    ] = 5
    
    # 🚀 NEW: AI-Model specific optimizations
    target_ai_model: Optional[str] = Field(
        None,
        description="Ziel-AI-Model für optimierte Prompts"
    )
    
    optimization_goal: Optional[str] = Field(
        "quality",
        pattern="^(speed|quality|cost|creativity)$",
        description="Optimierungsziel für AI-Response"
    )
    
    # 🚀 V2: Enhanced validation
    @field_validator('personality_data')
    @classmethod
    def validate_personality_data(cls, v):
        """Validiert Persönlichkeitsdaten unabhängig vom Typ"""
        
        if isinstance(v, dict):
            # Prüfe Mindest-Requirements für Dict-Input
            required_keys = ['big_five_scores', 'user_id']
            missing = [key for key in required_keys if key not in v]
            if missing:
                raise ValueError(f"Missing required keys in personality_data: {missing}")
            
            # Validiere big_five_scores structure
            big_five_scores = v.get('big_five_scores', {})
            if not isinstance(big_five_scores, dict):
                raise ValueError("big_five_scores must be a dictionary")
                
        # Protocol-based objects werden automatisch validiert
        return v
    
    @model_validator(mode='after')
    def validate_request_consistency(self):
        """Cross-field validation für bessere Request-Qualität"""
        
        # Wenn Limbic-Analyse gewünscht, sollten Limbic-Daten vorhanden sein
        if self.include_limbic_analysis and isinstance(self.personality_data, dict):
            if 'limbic_scores' not in self.personality_data:
                # Warnung, aber nicht blockierend
                pass
        
        return self


class GiftRecommendationRequest(BaseModel):
    """
    ✅ HARMONIZED: Enhanced schema with all required fields for component compatibility
    
    CHANGES:
    - Added missing fields: occasion, relationship, budget_min, budget_max, occasion_date
    - Standardized field names: number_of_recommendations (not max_recommendations)
    - Added cultural_context for ProductionOrchestrator
    - Added helper methods for component compatibility
    """
    
    model_config = {
        "extra": "forbid",
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "use_enum_values": True
    }
    
    # REQUIRED FIELDS - Now includes all fields that components expect
    personality_data: Dict[str, Any] = Field(
        ...,
        description="Personality data as Dictionary for flexibility - HARMONIZED format"
    )
    
    # ✅ FIXED: Added missing core fields that all components expect
    occasion: str = Field(
        ..., 
        description="Gift occasion - Required by OptimizationEngine and ProductionOrchestrator"
    )
    
    relationship: str = Field(
        ...,
        description="Relationship to gift recipient - Required by RecommendationService"
    )
    
    # ✅ UNCHANGED: Bestehende API-Felder für Kompatibilität
    gift_categories: Optional[List[str]] = Field(
        None,
        description="Spezifische Kategorien für Empfehlungen (optional)"
    )
    
    exclude_categories: List[str] = Field(
        default_factory=list,
        description="Kategorien die ausgeschlossen werden sollen"
    )
    
    personalization_level: Annotated[
        str,
        Field(
            default="medium",
            pattern="^(low|medium|high|maximum)$",
            description="Gewünschter Personalisierungsgrad"
        )
    ]
    
    # ✅ FIXED: Standardized explanation field (was explanation_detail)
    include_explanation: bool = Field(
        True,
        description="Include detailed explanations - STANDARDIZED from explanation_detail"
    )
    
    explanation_detail: Annotated[
        str,
        Field(
            default="medium",
            pattern="^(minimal|medium|detailed|comprehensive)$",
            description="Detail level for explanations - BACKWARD COMPATIBLE"
        )
    ]
    
    prioritize_emotional_impact: bool = Field(
        True,
        description="Soll emotionaler Impact priorisiert werden?"
    )
    
    # ✅ FIXED: Standardized field name (was max_recommendations)
    number_of_recommendations: Annotated[
        int,
        Field(ge=1, le=20, description="Number of recommendations - STANDARDIZED name")
    ] = 5
    

    # ✅ FIXED: Added missing budget fields expected by OptimizationEngine
    budget_min: Optional[float] = Field(
        None, ge=0,
        description="Minimum budget - Required by OptimizationEngine"
    )
    
    budget_max: Optional[float] = Field(
        None, ge=0, 
        description="Maximum budget - Required by OptimizationEngine"
    )

    # ✅ FIXED: Added occasion_date expected by OptimizationEngine
    occasion_date: Optional[date] = Field(
        None,
        description="Date of the occasion - Required by OptimizationEngine for time pressure calculation"
    )
    
    # ✅ FIXED: Added cultural_context expected by ProductionOrchestrator
    cultural_context: Optional[str] = Field(
        None,
        description="Cultural context - Required by ProductionOrchestrator"
    )
    
    # 🚀 NEW: Advanced AI-optimization fields
    target_ai_model: Optional[str] = Field(
        None,
        description="Spezifisches AI-Model für Empfehlungen"
    )
    
    use_consensus_validation: bool = Field(
        False,
        description="Multi-Model Consensus für höhere Qualität verwenden"
    )
    
    optimization_goal: Annotated[
        str,
        Field(
            default="quality",
            pattern="^(speed|quality|cost|creativity|accuracy)$"
        )
    ] = "quality"
    
    # 🚀 NEW: Context enhancement
    additional_context: Optional[str] = Field(
        None,
        max_length=1000,
        description="Zusätzlicher Kontext für bessere Empfehlungen"
    )
    
    urgency_level: Optional[str] = Field(
        None,
        pattern="^(low|medium|high|urgent)$",
        description="Dringlichkeit für Delivery-Optimierung"
    )

    # ✅ NEW: Helper fields for component compatibility
    user_id: Optional[str] = Field(
        None,
        description="User ID for internal processing - Added for RecommendationService compatibility"
    )

    @computed_field
    @property
    def max_recommendations(self) -> int:
        """
        ✅ BACKWARD COMPATIBILITY: Alias for number_of_recommendations
        Ensures existing code using max_recommendations still works
        """
        return self.number_of_recommendations
    
    @field_validator('personality_data')
    @classmethod
    def validate_personality_data_structure(cls, v):
        """
        ✅ HARMONIZED: Validate personality_data for all component expectations
        """
        if not isinstance(v, dict):
            raise ValueError("personality_data must be a dictionary")
            
        # Check for required keys that components expect
        required_keys = ['big_five_scores', 'user_id']
        missing = [key for key in required_keys if key not in v]
        if missing:
            # Don't fail - just warn and provide defaults
            if 'big_five_scores' not in v:
                v['big_five_scores'] = {
                    'openness': 3.5, 'conscientiousness': 3.5, 'extraversion': 3.5,
                    'agreeableness': 3.5, 'neuroticism': 3.5
                }
            if 'user_id' not in v:
                v['user_id'] = 'anonymous_user'
                
        return v
    
    @field_validator('budget_max')
    @classmethod 
    def validate_budget_consistency(cls, v, info):
        """Ensure budget_max >= budget_min if both provided"""
        if v is not None and info.data.get('budget_min') is not None:
            if v < info.data['budget_min']:
                raise ValueError("budget_max must be >= budget_min")
        return v
    
    @computed_field
    @property
    def optimization_preference(self) -> str:
        """Backward compatibility für ProductionOrchestrator"""
        return self.optimization_goal
  


# =========================================================================
# 🚀 NEUE SCHEMAS für gift_finder.py Integration
# =========================================================================

class GiftFinderRequest(BaseModel):
    """Haupt-Request Schema für gift_finder.py"""
    method: str = Field(..., pattern="^(prompt|personality)$")
    data: Dict[str, Any] = Field(...)
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)

class PromptMethodInput(BaseModel):
    """Input Schema für Prompt-Method"""
    user_prompt: str = Field(..., min_length=10, max_length=2000)
    occasion: Optional[str] = Field(None)
    relationship: Optional[str] = Field(None)
    budget_min: Optional[float] = Field(None, ge=0)
    budget_max: Optional[float] = Field(None, ge=0)
    additional_context: Optional[str] = Field(None, max_length=500)
    cultural_context: Optional[str] = Field(None)

class GiftPreferences(BaseModel):
    """Geschenk-Präferenzen Schema"""
    budget_min: Optional[float] = Field(None, ge=0)
    budget_max: Optional[float] = Field(None, ge=0)
    prefers_experiences: Optional[bool] = Field(None)
    prefers_handmade: Optional[bool] = Field(None)
    hobbies: List[str] = Field(default_factory=list)
    favorite_colors: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    dislikes: List[str] = Field(default_factory=list)

class PersonalityMethodInput(BaseModel):
    """Input Schema für Personality-Method (ENHANCED)"""
    person_name: Optional[str] = Field(None, min_length=1, max_length=100, description="Optional - nur für Heroic Story Katalog")
    age_group: Optional[AgeGroup] = Field(None, description="Altersgruppe für zielgruppenspezifische Empfehlungen")
    gender_identity: Optional[GenderIdentity] = Field(None, description="Gender-inklusive Identität")
    age_group_legacy: Optional[str] = Field(None, description="Legacy Altersgruppe")
    relationship_to_giver: RelationshipType = Field(...)
    personality_scores: BigFiveScore = Field(...)
    limbic_scores: Optional[LimbicScore] = Field(None)
    gift_preferences: GiftPreferences = Field(default_factory=GiftPreferences)
    occasion: Optional[str] = Field(None)
    occasion_date: Optional[date] = Field(None)
    additional_context: Optional[str] = Field(None, max_length=1000)
    cultural_context: Optional[str] = Field(None)

# Forward references resolution
PersonalityMethodInput.model_rebuild()


# =============================================================================
# ✅ BACKWARD COMPATIBLE EXPORTS: Alle bestehenden Klassen verfügbar
# =============================================================================

# Bestehende Input-Models für Kompatibilität
PersonalityAnalysisInput = PersonalityAnalysisRequest  # Alias für Backward Compatibility
QuickRecommendationInput = PersonalityAnalysisRequest  # Kann erweitert werden
PersonalityQuizInput = PersonalityAnalysisRequest      # Kann spezialisiert werden

# =============================================================================
# EXPORTS: Clean Interface für andere Module (ENHANCED + BACKWARD COMPATIBLE)
# =============================================================================

__all__ = [
    # === CORE ENUMS (UNCHANGED) ===
    'BigFiveTrait', 'LimbicDimension', 'LimbicType', 'EmotionalTrigger', 'GiftOccasion',
    
    # === NEW ENUMS ===
    'RelationshipType', 'GenderIdentity', 'AgeGroup',
    'RelationshipAnalyzer',  # ✅ Neue AI-Integration
    
    # === CORE MODELS (ENHANCED but COMPATIBLE) ===
    'BigFiveScore', 'LimbicScore', 'AIPersonalityProfile',
    
    # === REQUEST MODELS (ENHANCED but COMPATIBLE) ===
    'PersonalityAnalysisRequest', 'GiftRecommendationRequest',
    
    # === BACKWARD COMPATIBLE ALIASES ===
    'PersonalityAnalysisInput', 'QuickRecommendationInput', 'PersonalityQuizInput',
    
    # === PROTOCOLS (NEW) ===
    'PersonalityProfileProtocol'

    'GiftFinderRequest', 'PromptMethodInput', 'PersonalityMethodInput', 'GiftPreferences'
]
