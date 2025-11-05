"""
RelationshipType - Complete Implementation für SensationGifts AI Engine
=====================================================================

🎁 GIFT-OPTIMIZED: Speziell für emotionale Geschenkempfehlungen entwickelt
🧠 AI-INTELLIGENT: Integriert mit Big Five + Limbic-System
❤️ RELATIONSHIP-AWARE: Berücksichtigt emotionale Nähe und soziale Dynamiken

Diese Implementierung löst alle Import-Probleme und bietet eine vollständige
Relationship-Type Klassifikation für das SensationGifts-System.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field, computed_field
from datetime import datetime


# =============================================================================
# 🎯 CORE RELATIONSHIP TYPE ENUM - Erweitert und Gift-optimiert
# =============================================================================

class RelationshipType(str, Enum):
    """
    Beziehungstypen für Geschenkempfehlungen mit emotionaler Intelligenz
    
    🎯 FOKUS: Emotionale Nähe, soziale Dynamiken und Gift-Appropriateness
    🚀 ENHANCED: Mehr Granularität als Standard-Relationship-Enums
    ✅ COMPATIBLE: Funktioniert mit bestehenden AI-Engine Schemas
    """
    
    # === INTIMATE RELATIONSHIPS ===
    PARTNER = "partner"                        # Romantic partner, sehr nahe Beziehung
    SPOUSE = "spouse"                          # Ehepartner/in
    FIANCE = "fiance"                          # Verlobt/e
    
    # === FAMILY RELATIONSHIPS ===
    FAMILY_PARENT = "family_parent"            # Mutter oder Vater
    FAMILY_SIBLING = "family_sibling"          # Bruder oder Schwester
    FAMILY_CHILD = "family_child"              # Sohn oder Tochter
    FAMILY_GRANDPARENT = "family_grandparent"  # Großeltern
    FAMILY_GRANDCHILD = "family_grandchild"    # Enkelkind
    FAMILY_EXTENDED = "family_extended"        # Onkel, Tante, Cousin, etc.
    FAMILY_IN_LAW = "family_in_law"           # Schwiegereltern, etc.
    
    # === FRIENDSHIP LEVELS ===
    FRIEND_BEST = "friend_best"                # Beste/r Freund/in, sehr enge Bindung
    FRIEND_CLOSE = "friend_close"              # Enge/r Freund/in, persönliche Verbindung
    FRIEND_GOOD = "friend_good"                # Gute/r Freund/in, regelmäßiger Kontakt
    FRIEND_CASUAL = "friend_casual"            # Lockere/r Freund/in, gelegentlicher Kontakt
    FRIEND_ACQUAINTANCE = "friend_acquaintance" # Bekannte/r
    
    # === PROFESSIONAL RELATIONSHIPS ===
    COLLEAGUE = "colleague"                    # Arbeitskolleg/in, gleiche Ebene
    BOSS = "boss"                             # Vorgesetzte/r
    EMPLOYEE = "employee"                     # Untergebene/r
    BUSINESS_PARTNER = "business_partner"     # Geschäftspartner/in
    CLIENT = "client"                         # Kunde/Kundin
    MENTOR = "mentor"                         # Mentor/in
    MENTEE = "mentee"                         # Mentee/Schützling
    
    # === EDUCATIONAL RELATIONSHIPS ===
    TEACHER = "teacher"                       # Lehrer/in oder Professor/in
    STUDENT = "student"                       # Schüler/in oder Student/in
    CLASSMATE = "classmate"                   # Klassenkamerad/in
    
    # === SPECIAL RELATIONSHIPS ===
    NEIGHBOR = "neighbor"                     # Nachbar/in
    ROOMMATE = "roommate"                     # Mitbewohner/in
    LANDLORD = "landlord"                     # Vermieter/in
    TENANT = "tenant"                         # Mieter/in
    
    # === DATING & ROMANTIC SPECTRUM ===
    DATING_NEW = "dating_new"                 # Neue Dating-Beziehung
    DATING_SERIOUS = "dating_serious"         # Ernsthafte Dating-Beziehung
    EX_PARTNER_FRIENDLY = "ex_partner_friendly" # Ex-Partner/in (freundschaftlich)
    CRUSH = "crush"                           # Schwärmerei/Heimliche Liebe
    
    # === SERVICE & PROFESSIONAL SERVICES ===
    DOCTOR = "doctor"                         # Arzt/Ärztin
    THERAPIST = "therapist"                   # Therapeut/in
    HAIRDRESSER = "hairdresser"              # Friseur/in
    PERSONAL_TRAINER = "personal_trainer"     # Personal Trainer/in
    
    # === GENERAL & UNKNOWN ===
    ACQUAINTANCE = "acquaintance"             # Allgemeine Bekanntschaft
    STRANGER_FRIENDLY = "stranger_friendly"   # Freundliche/r Fremde/r (z.B. Nachbar)
    OTHER = "other"                           # Andere Beziehung
    UNKNOWN = "unknown"                       # Beziehungstyp unbekannt
    
    # === COMPUTED PROPERTIES für AI-Optimization ===
    
    @property
    def intimacy_level(self) -> float:
        """
        Emotionale Nähe/Intimität von 0.0 (distanziert) bis 1.0 (sehr intim)
        
        🎁 GIFT IMPACT: Bestimmt wie persönlich/intim Geschenke sein können
        """
        intimacy_map = {
            # Sehr intim (0.9-1.0)
            self.SPOUSE: 1.0,
            self.PARTNER: 1.0,
            self.FIANCE: 0.95,
            
            # Sehr nah (0.8-0.9)
            self.FRIEND_BEST: 0.9,
            self.FAMILY_CHILD: 0.9,
            self.FAMILY_PARENT: 0.85,
            self.FAMILY_SIBLING: 0.8,
            
            # Nah (0.6-0.8)
            self.DATING_SERIOUS: 0.75,
            self.FRIEND_CLOSE: 0.7,
            self.FAMILY_GRANDPARENT: 0.65,
            self.FAMILY_GRANDCHILD: 0.65,
            
            # Moderat (0.4-0.6)
            self.FRIEND_GOOD: 0.6,
            self.FAMILY_EXTENDED: 0.55,
            self.ROOMMATE: 0.5,
            self.DATING_NEW: 0.45,
            
            # Professionell nah (0.3-0.5)
            self.MENTOR: 0.45,
            self.MENTEE: 0.4,
            self.BUSINESS_PARTNER: 0.35,
            
            # Professionell (0.2-0.3)
            self.COLLEAGUE: 0.3,
            self.CLASSMATE: 0.25,
            self.TEACHER: 0.25,
            self.STUDENT: 0.25,
            
            # Distanziert professionell (0.1-0.2)
            self.BOSS: 0.2,
            self.EMPLOYEE: 0.2,
            self.CLIENT: 0.15,
            
            # Schwach/Services (0.05-0.15)
            self.NEIGHBOR: 0.15,
            self.FRIEND_CASUAL: 0.3,  # Etwas höher als andere casual
            self.FRIEND_ACQUAINTANCE: 0.2,
            self.ACQUAINTANCE: 0.1,
            self.DOCTOR: 0.1,
            self.HAIRDRESSER: 0.1,
            
            # Minimal (0.0-0.1)
            self.STRANGER_FRIENDLY: 0.05,
            self.OTHER: 0.1,
            self.UNKNOWN: 0.1,
        }
        return intimacy_map.get(self, 0.1)
    
    @property
    def formality_level(self) -> float:
        """
        Formalitäts-Level von 0.0 (informell) bis 1.0 (sehr formell)
        
        🎁 GIFT IMPACT: Bestimmt wie formell/professionell Geschenke sein sollten
        """
        formality_map = {
            # Sehr formell (0.8-1.0)
            self.BOSS: 1.0,
            self.CLIENT: 0.9,
            self.DOCTOR: 0.85,
            self.TEACHER: 0.8,
            
            # Formell (0.6-0.8)
            self.BUSINESS_PARTNER: 0.75,
            self.COLLEAGUE: 0.7,
            self.MENTOR: 0.65,
            
            # Semi-formell (0.4-0.6)
            self.EMPLOYEE: 0.5,
            self.STUDENT: 0.45,
            self.NEIGHBOR: 0.4,
            
            # Gemischt (0.2-0.4)
            self.FRIEND_ACQUAINTANCE: 0.35,
            self.ACQUAINTANCE: 0.3,
            self.CLASSMATE: 0.25,
            
            # Informell (0.0-0.2)
            self.PARTNER: 0.0,
            self.SPOUSE: 0.0,
            self.FIANCE: 0.0,
            self.FRIEND_BEST: 0.0,
            self.FRIEND_CLOSE: 0.1,
            self.FAMILY_SIBLING: 0.1,
            self.FAMILY_CHILD: 0.0,
            self.FAMILY_PARENT: 0.05,
            self.ROOMMATE: 0.15,
            self.DATING_SERIOUS: 0.05,
            self.DATING_NEW: 0.2,
        }
        return formality_map.get(self, 0.3)
    
    @property
    def gift_budget_expectation(self) -> Tuple[float, float]:
        """
        Erwartete Budget-Range (min, max) in EUR für angemessene Geschenke
        
        🎁 GIFT IMPACT: Hilft bei Budget-Guidance für AI-Empfehlungen
        """
        budget_map = {
            # Premium/Intimate Relationships
            self.SPOUSE: (80, 500),
            self.PARTNER: (60, 400),
            self.FIANCE: (100, 600),
            
            # Family - varies by direction
            self.FAMILY_PARENT: (50, 250),
            self.FAMILY_CHILD: (30, 200),
            self.FAMILY_SIBLING: (25, 150),
            self.FAMILY_GRANDPARENT: (40, 180),
            self.FAMILY_GRANDCHILD: (20, 100),
            
            # Close Friends
            self.FRIEND_BEST: (40, 200),
            self.FRIEND_CLOSE: (25, 120),
            self.FRIEND_GOOD: (15, 80),
            self.FRIEND_CASUAL: (10, 50),
            
            # Professional - Conservative
            self.BOSS: (30, 100),
            self.COLLEAGUE: (15, 60),
            self.EMPLOYEE: (20, 80),
            self.CLIENT: (25, 150),
            self.BUSINESS_PARTNER: (40, 200),
            
            # Dating
            self.DATING_SERIOUS: (40, 180),
            self.DATING_NEW: (15, 60),
            
            # Services/Casual
            self.NEIGHBOR: (10, 40),
            self.TEACHER: (15, 50),
            self.DOCTOR: (20, 60),
            self.HAIRDRESSER: (15, 50),
        }
        return budget_map.get(self, (10, 50))
    
    @property
    def gift_category_preferences(self) -> List[str]:
        """
        Bevorzugte Geschenk-Kategorien für diese Beziehung
        
        🎁 GIFT IMPACT: Direkte Category-Suggestions für AI
        """
        category_map = {
            # Intimate - Personal, emotional, experiences
            self.SPOUSE: ["experiences", "jewelry", "personalized", "luxury", "romantic"],
            self.PARTNER: ["experiences", "romantic", "personalized", "jewelry", "memories"],
            self.FIANCE: ["romantic", "luxury", "personalized", "experiences", "future_oriented"],
            
            # Family - Varies by relationship
            self.FAMILY_PARENT: ["practical", "comfort", "experiences", "health_wellness", "memories"],
            self.FAMILY_CHILD: ["toys", "educational", "experiences", "creative", "age_appropriate"],
            self.FAMILY_SIBLING: ["fun", "shared_interests", "personalized", "experiences", "memories"],
            
            # Friends - Social, fun, shared interests
            self.FRIEND_BEST: ["experiences", "personalized", "shared_interests", "fun", "memories"],
            self.FRIEND_CLOSE: ["experiences", "shared_interests", "personalized", "fun"],
            self.FRIEND_GOOD: ["shared_interests", "experiences", "fun", "practical"],
            self.FRIEND_CASUAL: ["universal", "consumable", "small_tokens"],
            
            # Professional - Safe, appropriate, universal
            self.BOSS: ["professional", "universal", "quality", "conservative"],
            self.COLLEAGUE: ["professional", "universal", "consumable", "small_tokens"],
            self.CLIENT: ["professional", "branded", "quality", "appreciative"],
            
            # Dating - Appropriate to stage
            self.DATING_SERIOUS: ["romantic", "experiences", "personalized", "thoughtful"],
            self.DATING_NEW: ["thoughtful", "safe", "experiences", "universal"],
            
            # Service relationships
            self.TEACHER: ["appreciative", "professional", "consumable"],
            self.DOCTOR: ["appreciative", "professional", "universal"],
        }
        return category_map.get(self, ["universal", "safe", "consumable"])
    
    @property
    def emotional_safety_level(self) -> float:
        """
        Emotionale Sicherheit - wie "sicher" ist es emotionale Geschenke zu geben?
        0.0 = Risiko emotional unangemessen, 1.0 = Emotional sehr sicher
        
        🎁 GIFT IMPACT: Bestimmt ob emotionale/sentimentale Geschenke angemessen sind
        """
        safety_map = {
            # Emotional sehr sicher
            self.SPOUSE: 1.0,
            self.PARTNER: 1.0,
            self.FAMILY_CHILD: 1.0,
            self.FAMILY_PARENT: 0.95,
            
            # Emotional sicher
            self.FRIEND_BEST: 0.9,
            self.FAMILY_SIBLING: 0.85,
            self.FIANCE: 0.95,
            
            # Moderat sicher
            self.FRIEND_CLOSE: 0.7,
            self.DATING_SERIOUS: 0.75,
            self.FAMILY_EXTENDED: 0.6,
            
            # Vorsichtig
            self.FRIEND_GOOD: 0.5,
            self.DATING_NEW: 0.4,
            self.COLLEAGUE: 0.3,
            
            # Emotional riskant
            self.BOSS: 0.2,
            self.CLIENT: 0.25,
            self.ACQUAINTANCE: 0.2,
            
            # Sehr riskant
            self.STRANGER_FRIENDLY: 0.1,
            self.UNKNOWN: 0.1,
        }
        return safety_map.get(self, 0.3)
    
    @property
    def reciprocity_expectation(self) -> float:
        """
        Erwartung von Gegenseitigkeit - 0.0 = keine Erwartung, 1.0 = hohe Erwartung
        
        🎁 GIFT IMPACT: Beeinflusst ob Geschenk "Verpflichtung" schafft
        """
        reciprocity_map = {
            # Hohe Reziprozität
            self.SPOUSE: 0.8,
            self.PARTNER: 0.8,
            self.FRIEND_BEST: 0.7,
            
            # Moderate Reziprozität
            self.FAMILY_SIBLING: 0.6,
            self.FRIEND_CLOSE: 0.6,
            self.COLLEAGUE: 0.5,
            
            # Niedrige Reziprozität
            self.FAMILY_PARENT: 0.3,  # Eltern müssen nicht zurückschenken
            self.FAMILY_CHILD: 0.2,   # Kinder müssen nicht zurückschenken
            self.TEACHER: 0.1,        # Service-Geschenk
            self.DOCTOR: 0.1,
            
            # Keine Erwartung
            self.BOSS: 0.9,           # Heikel - könnte als Bestechung wirken!
            self.CLIENT: 0.4,
        }
        return reciprocity_map.get(self, 0.4)


# =============================================================================
# 🚀 RELATIONSHIP ANALYZER - AI-Integration Class
# =============================================================================

class RelationshipAnalyzer(BaseModel):
    """
    Intelligente Relationship-Analyse für AI-Geschenkempfehlungen
    
    🧠 AI-POWERED: Kombiniert Relationship-Typ mit Persönlichkeits-Daten
    🎁 GIFT-FOCUSED: Optimiert für Geschenkempfehlungs-Algorithmen
    ❤️ EMOTIONALLY-AWARE: Berücksichtigt emotionale Nuancen
    """
    
    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "extra": "forbid",
        "use_enum_values": False
    }
    
    # === CORE RELATIONSHIP DATA ===
    relationship_type: RelationshipType = Field(
        ...,
        description="Primary relationship type"
    )
    
    def __init__(self, **data):
        # Ensure relationship_type is a RelationshipType enum
        if 'relationship_type' in data and isinstance(data['relationship_type'], str):
            try:
                data['relationship_type'] = RelationshipType(data['relationship_type'])
            except ValueError:
                # Fallback to PARTNER if invalid
                data['relationship_type'] = RelationshipType.PARTNER
        
        super().__init__(**data)
    
    relationship_duration: Optional[str] = Field(
        None,
        description="How long the relationship has existed (e.g., '2 years', '6 months')"
    )
    
    relationship_quality: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Quality/strength of the relationship (0.0=weak, 1.0=very strong)"
    )
    
    # === CONTEXT MODIFIERS ===
    is_long_distance: bool = Field(
        default=False,
        description="Is this a long-distance relationship?"
    )
    
    shared_interests: List[str] = Field(
        default_factory=list,
        description="Shared interests or activities"
    )
    
    cultural_background_similar: Optional[bool] = Field(
        None,
        description="Similar cultural backgrounds?"
    )
    
    # === GIFT-SPECIFIC CONTEXT ===
    has_given_gifts_before: bool = Field(
        default=False,
        description="Have you given gifts to this person before?"
    )
    
    previous_gift_reactions: Optional[str] = Field(
        None,
        description="How did they react to previous gifts? (positive/neutral/negative)"
    )
    
    special_occasions_celebrated: List[str] = Field(
        default_factory=list,
        description="Which occasions do you typically celebrate together?"
    )
    
    # === COMPUTED ANALYSIS ===
    
    @computed_field
    @property
    def gift_appropriateness_score(self) -> float:
        """
        Kombinierter Score für Gift-Angemessenheit (0.0-1.0)
        
        Berücksichtigt:
        - Relationship intimacy
        - Emotional safety
        - Cultural appropriateness
        - Previous gift history
        """
        base_score = (self.relationship_type.intimacy_level + 
                     self.relationship_type.emotional_safety_level) / 2
        
        # Quality modifier
        if self.relationship_quality is not None:
            quality_modifier = (self.relationship_quality - 0.5) * 0.2  # ±0.1
            base_score += quality_modifier
        
        # Experience modifier
        if self.has_given_gifts_before:
            if self.previous_gift_reactions == "positive":
                base_score += 0.1
            elif self.previous_gift_reactions == "negative":
                base_score -= 0.15
        
        # Long distance consideration
        if self.is_long_distance:
            base_score -= 0.05  # Slightly more challenging
        
        return max(0.0, min(1.0, base_score))
    
    @computed_field
    @property
    def recommended_budget_range(self) -> Tuple[float, float]:
        """
        AI-optimierte Budget-Empfehlung basierend auf Beziehungsanalyse
        """
        base_min, base_max = self.relationship_type.gift_budget_expectation
        
        # Quality adjustments
        if self.relationship_quality is not None:
            quality_factor = 0.5 + (self.relationship_quality * 0.5)  # 0.5 to 1.0
            base_min *= quality_factor
            base_max *= quality_factor
        
        # Duration adjustments
        if self.relationship_duration:
            if "year" in self.relationship_duration.lower():
                # Longer relationships = potentially higher budgets
                duration_factor = 1.2
            elif "month" in self.relationship_duration.lower():
                duration_factor = 1.0
            else:
                duration_factor = 0.8  # New relationships
            
            base_max *= duration_factor
        
        return (round(base_min, 2), round(base_max, 2))
    
    @computed_field
    @property
    def gift_strategy_recommendations(self) -> Dict[str, Any]:
        """
        Strategische Empfehlungen für Gift-Giving mit dieser Person
        
        🎁 DIRECT AI INPUT: Diese Daten gehen direkt in AI-Prompts
        """
        strategy = {
            "primary_approach": self._get_primary_approach(),
            "categories_to_focus": self.relationship_type.gift_category_preferences,
            "categories_to_avoid": self._get_categories_to_avoid(),
            "personalization_level": self._get_personalization_level(),
            "emotional_tone": self._get_emotional_tone(),
            "presentation_style": self._get_presentation_style(),
            "timing_considerations": self._get_timing_considerations()
        }
        
        return strategy
    
    def _get_primary_approach(self) -> str:
        """Hauptansatz für Gift-Giving"""
        intimacy = self.relationship_type.intimacy_level
        formality = self.relationship_type.formality_level
        
        if intimacy >= 0.8:
            return "deeply_personal_and_meaningful"
        elif formality >= 0.7:
            return "respectful_and_professional"
        elif intimacy >= 0.5:
            return "thoughtful_and_personal"
        else:
            return "safe_and_universal"
    
    def _get_categories_to_avoid(self) -> List[str]:
        """Kategorien die vermieden werden sollten"""
        avoid = []
        
        if self.relationship_type.formality_level >= 0.7:
            avoid.extend(["romantic", "intimate", "personal_care"])
        
        if self.relationship_type.intimacy_level <= 0.3:
            avoid.extend(["romantic", "deeply_personal", "intimate"])
        
        if self.relationship_type in [RelationshipType.BOSS, RelationshipType.CLIENT]:
            avoid.extend(["romantic", "intimate", "political", "religious"])
        
        return avoid
    
    def _get_personalization_level(self) -> str:
        """Empfohlenes Personalisierungs-Level"""
        intimacy = self.relationship_type.intimacy_level
        
        if intimacy >= 0.8:
            return "highly_personalized"
        elif intimacy >= 0.5:
            return "moderately_personalized"
        elif intimacy >= 0.3:
            return "lightly_personalized"
        else:
            return "generic_but_thoughtful"
    
    def _get_emotional_tone(self) -> str:
        """Empfohlener emotionaler Ton"""
        safety = self.relationship_type.emotional_safety_level
        formality = self.relationship_type.formality_level
        
        if safety >= 0.8 and formality <= 0.3:
            return "warm_and_affectionate"
        elif formality >= 0.7:
            return "respectful_and_appreciative"
        elif safety >= 0.5:
            return "friendly_and_caring"
        else:
            return "polite_and_considerate"
    
    def _get_presentation_style(self) -> str:
        """Empfohlener Präsentations-Stil"""
        formality = self.relationship_type.formality_level
        
        if formality >= 0.8:
            return "elegant_and_formal"
        elif formality >= 0.5:
            return "neat_and_professional"
        else:
            return "creative_and_personal"
    
    def _get_timing_considerations(self) -> List[str]:
        """Timing-Überlegungen"""
        considerations = []
        
        if self.relationship_type.reciprocity_expectation >= 0.7:
            considerations.append("consider_reciprocal_timing")
        
        if self.relationship_type.formality_level >= 0.7:
            considerations.append("respect_professional_boundaries")
        
        if self.is_long_distance:
            considerations.append("plan_for_shipping_time")
        
        return considerations
    
    # === AI-PROMPT INTEGRATION ===
    
    def to_ai_context(self) -> str:
        """
        Konvertiert Relationship-Analyse zu AI-Prompt-Kontext
        
        🚀 DIRECT AI INPUT: Optimiert für AI-Model Consumption
        """
        context_parts = []
        
        # Basic relationship info
        if hasattr(self.relationship_type, 'value'):
            relationship_name = self.relationship_type.value
        else:
            relationship_name = str(self.relationship_type)
        context_parts.append(f"Relationship: {relationship_name}")
        
        # Key metrics
        intimacy = self.relationship_type.intimacy_level
        formality = self.relationship_type.formality_level
        safety = self.relationship_type.emotional_safety_level
        
        context_parts.append(f"Intimacy: {intimacy:.1f}/1.0 (emotional closeness)")
        context_parts.append(f"Formality: {formality:.1f}/1.0 (professional appropriateness)")
        context_parts.append(f"Emotional Safety: {safety:.1f}/1.0 (sentimental gift appropriateness)")
        
        # Budget guidance
        budget_min, budget_max = self.recommended_budget_range
        context_parts.append(f"Recommended Budget: €{budget_min:.0f}-€{budget_max:.0f}")
        
        # Strategy summary
        strategy = self.gift_strategy_recommendations
        context_parts.append(f"Primary Approach: {strategy['primary_approach']}")
        context_parts.append(f"Personalization Level: {strategy['personalization_level']}")
        context_parts.append(f"Emotional Tone: {strategy['emotional_tone']}")
        
        # Categories
        focus_categories = ", ".join(strategy['categories_to_focus'][:3])
        context_parts.append(f"Focus Categories: {focus_categories}")
        
        if strategy['categories_to_avoid']:
            avoid_categories = ", ".join(strategy['categories_to_avoid'])
            context_parts.append(f"Avoid Categories: {avoid_categories}")
        
        return " | ".join(context_parts)
    
    def to_detailed_ai_context(self) -> Dict[str, Any]:
        """
        Detaillierter Kontext für Advanced AI Models (GPT-4, Claude)
        """
        return {
            "relationship_metrics": {
                "type": self.relationship_type.value,
                "intimacy_level": self.relationship_type.intimacy_level,
                "formality_level": self.relationship_type.formality_level,
                "emotional_safety": self.relationship_type.emotional_safety_level,
                "reciprocity_expectation": self.relationship_type.reciprocity_expectation,
                "appropriateness_score": self.gift_appropriateness_score
            },
            "budget_guidance": {
                "min_amount": self.recommended_budget_range[0],
                "max_amount": self.recommended_budget_range[1],
                "expectation_level": "high" if self.relationship_type.reciprocity_expectation > 0.6 else "low"
            },
            "strategy": self.gift_strategy_recommendations,
            "context_modifiers": {
                "relationship_duration": self.relationship_duration,
                "relationship_quality": self.relationship_quality,
                "is_long_distance": self.is_long_distance,
                "has_gift_history": self.has_given_gifts_before,
                "previous_reactions": self.previous_gift_reactions
            }
        }


# =============================================================================
# 🎁 GIFT-SPECIFIC RELATIONSHIP HELPERS
# =============================================================================

class RelationshipGiftGuide:
    """
    Statische Helper-Methoden für Gift-Recommendations basierend auf Relationships
    
    🎯 UTILITY CLASS: Praktische Funktionen für AI-Integration
    """
    
    @staticmethod
    def create_relationship_analyzer(
        relationship_type: RelationshipType,
        duration: str = None,
        quality: float = None,
        **kwargs
    ) -> RelationshipAnalyzer:
        """Quick factory für RelationshipAnalyzer"""
        return RelationshipAnalyzer(
            relationship_type=relationship_type,
            relationship_duration=duration,
            relationship_quality=quality,
            **kwargs
        )
    
    @staticmethod
    def get_safe_relationship_types() -> List[RelationshipType]:
        """Beziehungstypen die für Gift-Giving 'sicher' sind"""
        return [
            RelationshipType.FAMILY_PARENT,
            RelationshipType.FAMILY_CHILD,
            RelationshipType.FAMILY_SIBLING,
            RelationshipType.FRIEND_CLOSE,
            RelationshipType.FRIEND_GOOD,
            RelationshipType.PARTNER,
            RelationshipType.SPOUSE
        ]
    
    @staticmethod
    def get_professional_appropriate_types() -> List[RelationshipType]:
        """Beziehungstypen wo professionelle Geschenke angemessen sind"""
        return [
            RelationshipType.COLLEAGUE,
            RelationshipType.BOSS,
            RelationshipType.CLIENT,
            RelationshipType.BUSINESS_PARTNER,
            RelationshipType.TEACHER,
            RelationshipType.MENTOR
        ]
    
    @staticmethod
    def get_romantic_appropriate_types() -> List[RelationshipType]:
        """Beziehungstypen wo romantische Geschenke angemessen sind"""
        return [
            RelationshipType.PARTNER,
            RelationshipType.SPOUSE,
            RelationshipType.FIANCE,
            RelationshipType.DATING_SERIOUS,
            RelationshipType.DATING_NEW  # Mit Vorsicht
        ]
    
    @staticmethod
    def relationship_compatibility_matrix() -> Dict[RelationshipType, Dict[str, List[str]]]:
        """
        Compatibility Matrix für Cross-Reference
        
        🎁 AI UTILITY: Hilft AI bei der Kategorie-Auswahl
        """
        return {
            RelationshipType.SPOUSE: {
                "excellent": ["romantic", "luxury", "experiences", "personalized", "jewelry"],
                "good": ["home_decor", "technology", "books", "wellness"],
                "avoid": ["professional_only", "impersonal"]
            },
            RelationshipType.BOSS: {
                "excellent": ["professional", "consumable", "universal"],
                "good": ["books", "quality_items"],
                "avoid": ["romantic", "intimate", "too_expensive", "personal"]
            },
            RelationshipType.FRIEND_CLOSE: {
                "excellent": ["experiences", "shared_interests", "personalized", "fun"],
                "good": ["books", "creative", "technology"],
                "avoid": ["romantic", "too_intimate"]
            }
            # ... Weitere Beziehungstypen können hier hinzugefügt werden
        }


# =============================================================================
# 🔧 INTEGRATION UTILITIES für SensationGifts AI-Engine
# =============================================================================

def create_relationship_context_for_ai(
    relationship_type: RelationshipType,
    additional_context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Quick Utility für AI-Prompt Context Generation
    
    Args:
        relationship_type: Der Beziehungstyp
        additional_context: Zusätzliche Kontext-Informationen
    
    Returns:
        Dictionary mit AI-optimiertem Relationship-Kontext
    """
    
    # Ensure relationship_type is a RelationshipType enum
    if isinstance(relationship_type, str):
        try:
            relationship_type = RelationshipType(relationship_type)
        except ValueError:
            # Fallback to PARTNER if invalid
            relationship_type = RelationshipType.PARTNER
    
    analyzer = RelationshipAnalyzer(relationship_type=relationship_type)
    
    if additional_context:
        # Update analyzer mit additional context
        for key, value in additional_context.items():
            if hasattr(analyzer, key):
                setattr(analyzer, key, value)
    
    return {
        "relationship_summary": analyzer.to_ai_context(),
        "detailed_analysis": analyzer.to_detailed_ai_context(),
        "gift_appropriateness": analyzer.gift_appropriateness_score,
        "strategy": analyzer.gift_strategy_recommendations
    }


def get_relationship_budget_guidance(relationship_type: RelationshipType) -> Dict[str, Any]:
    """
    Schnelle Budget-Guidance für ein Relationship Type
    
    Returns:
        Dictionary mit Budget-Empfehlungen
    """
    budget_min, budget_max = relationship_type.gift_budget_expectation
    
    return {
        "min_budget": budget_min,
        "max_budget": budget_max,
        "sweet_spot": (budget_min + budget_max) / 2,
        "formality_level": relationship_type.formality_level,
        "intimacy_level": relationship_type.intimacy_level,
        "safety_note": "high" if relationship_type.emotional_safety_level > 0.7 else "moderate" if relationship_type.emotional_safety_level > 0.4 else "low"
    }


# =============================================================================
# 🚀 AI-ENGINE INTEGRATION FUNCTIONS
# =============================================================================

def integrate_relationship_with_personality(
    relationship_type: RelationshipType,
    big_five_scores: Dict[str, float],
    limbic_scores: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Intelligente Integration von Relationship + Personality für AI-Empfehlungen
    
    🧠 CORE AI FUNCTION: Kombiniert Relationship-Dynamik mit Persönlichkeits-Profiling
    
    Args:
        relationship_type: Der Beziehungstyp
        big_five_scores: Big Five Persönlichkeits-Scores
        limbic_scores: Limbic System Scores (optional)
    
    Returns:
        Integrierter Kontext für AI-Model Consumption
    """
    
    # Create relationship analyzer
    analyzer = RelationshipAnalyzer(relationship_type=relationship_type)
    
    # Personality-based relationship adjustments
    adjustments = {}
    
    # High agreeableness + close relationship = mehr sentimentale Geschenke
    if big_five_scores.get('agreeableness', 0.5) > 0.7 and analyzer.relationship_type.intimacy_level > 0.6:
        adjustments['sentimental_boost'] = True
        adjustments['emotional_safety_boost'] = 0.1
    
    # High extraversion + professional relationship = mehr soziale Geschenke
    if big_five_scores.get('extraversion', 0.5) > 0.7 and analyzer.relationship_type.formality_level > 0.5:
        adjustments['social_professional_gifts'] = True
    
    # High conscientiousness + any relationship = mehr qualitätsorientierte Geschenke
    if big_five_scores.get('conscientiousness', 0.5) > 0.7:
        adjustments['quality_focus'] = True
        adjustments['practical_preference'] = True
    
    # Limbic adjustments
    if limbic_scores:
        # High stimulanz + close relationship = aufregendere Geschenke
        if limbic_scores.get('stimulanz', 0.5) > 0.7 and analyzer.relationship_type.intimacy_level > 0.5:
            adjustments['excitement_boost'] = True
        
        # High dominanz + professional = status-bewusste Geschenke
        if limbic_scores.get('dominanz', 0.5) > 0.7 and analyzer.relationship_type.formality_level > 0.4:
            adjustments['status_preference'] = True
    
    return {
        "relationship_analysis": analyzer.to_detailed_ai_context(),
        "personality_relationship_fit": adjustments,
        "integrated_strategy": _create_integrated_strategy(analyzer, big_five_scores, limbic_scores),
        "ai_prompt_context": _generate_integrated_prompt_context(analyzer, big_five_scores, adjustments)
    }


def _create_integrated_strategy(
    analyzer: RelationshipAnalyzer,
    big_five_scores: Dict[str, float],
    limbic_scores: Dict[str, float] = None
) -> Dict[str, Any]:
    """Erstellt integrierte Gift-Strategie basierend auf Relationship + Personality"""
    
    base_strategy = analyzer.gift_strategy_recommendations
    
    # Personality-based category modifications
    enhanced_categories = base_strategy['categories_to_focus'].copy()
    
    # Add personality-based categories
    if big_five_scores.get('openness', 0.5) > 0.7:
        enhanced_categories.extend(['creative', 'artistic', 'experiential'])
    
    if big_five_scores.get('conscientiousness', 0.5) > 0.7:
        enhanced_categories.extend(['practical', 'quality', 'organizational'])
    
    if big_five_scores.get('extraversion', 0.5) > 0.7:
        enhanced_categories.extend(['social', 'group_experiences'])
    
    # Limbic-based enhancements
    if limbic_scores:
        if limbic_scores.get('stimulanz', 0.5) > 0.7:
            enhanced_categories.extend(['exciting', 'adventure', 'novel'])
        
        if limbic_scores.get('dominanz', 0.5) > 0.7:
            enhanced_categories.extend(['luxury', 'status', 'exclusive'])
        
        if limbic_scores.get('balance', 0.5) > 0.7:
            enhanced_categories.extend(['wellness', 'mindful', 'sustainable'])
    
    return {
        "enhanced_categories": list(set(enhanced_categories))[:8],  # Limit to top 8
        "personality_weight": 0.6,  # 60% personality, 40% relationship
        "relationship_weight": 0.4,
        "confidence_adjustment": min(1.0, analyzer.gift_appropriateness_score + 0.1)
    }


def _generate_integrated_prompt_context(
    analyzer: RelationshipAnalyzer,
    big_five_scores: Dict[str, float],
    adjustments: Dict[str, Any]
) -> str:
    """Generiert integrierten Prompt-Kontext für AI-Models"""
    
    context_parts = []
    
    # Base relationship context
    context_parts.append(analyzer.to_ai_context())
    
    # Personality adjustments
    if adjustments.get('sentimental_boost'):
        context_parts.append("PERSONALITY BOOST: High agreeableness + close relationship = prioritize sentimental gifts")
    
    if adjustments.get('quality_focus'):
        context_parts.append("PERSONALITY BOOST: High conscientiousness = prioritize quality and practical value")
    
    if adjustments.get('excitement_boost'):
        context_parts.append("LIMBIC BOOST: High stimulation-seeking + close relationship = prioritize exciting experiences")
    
    if adjustments.get('status_preference'):
        context_parts.append("LIMBIC BOOST: High dominance + professional context = consider status-appropriate gifts")
    
    return " | ".join(context_parts)


# =============================================================================
# ✅ EXPORTS - Saubere Integration mit SensationGifts
# =============================================================================

__all__ = [
    # === CORE ENUM ===
    'RelationshipType',
    
    # === ANALYSIS CLASSES ===
    'RelationshipAnalyzer',
    'RelationshipGiftGuide', 
    
    # === UTILITY FUNCTIONS ===
    'create_relationship_context_for_ai',
    'get_relationship_budget_guidance',
    'integrate_relationship_with_personality',
]


# =============================================================================
# 📋 INTEGRATION CHECKLIST für dein SensationGifts Projekt
# =============================================================================

"""
INTEGRATION CHECKLIST:
======================

✅ 1. In ai_engine/schemas/input_schemas.py:
   - RelationshipType enum hinzufügen (ist schon da aber erweitern)
   - Import: from .relationship_types import RelationshipType

✅ 2. In ai_engine/schemas/__init__.py:
   - Export hinzufügen: 'RelationshipType', 'RelationshipAnalyzer'

✅ 3. In ai_engine/prompts/gift_prompts.py:
   - Import ergänzen: from ..schemas import RelationshipType
   - Verwendung in ContextualPromptBuilder

✅ 4. In app/models/personality.py oder user.py:
   - RelationshipAnalyzer integration für Database-Modelle

✅ 5. In app/services/recommendation_service.py:
   - integrate_relationship_with_personality() verwenden

USAGE EXAMPLE:
==============

```python
# Basic usage
relationship = RelationshipType.FRIEND_CLOSE
analyzer = RelationshipAnalyzer(relationship_type=relationship)
print(analyzer.to_ai_context())

# Advanced integration
big_five = {'openness': 0.8, 'extraversion': 0.7, 'conscientiousness': 0.6}
limbic = {'stimulanz': 0.8, 'dominanz': 0.3, 'balance': 0.6}

integrated_context = integrate_relationship_with_personality(
    RelationshipType.PARTNER,
    big_five,
    limbic
)

# Use in AI prompts
ai_context = integrated_context['ai_prompt_context']
```

Diese Implementation löst alle Import-Probleme und bietet eine vollständige
Relationship-Type-Lösung für dein SensationGifts AI-Engine System!
"""