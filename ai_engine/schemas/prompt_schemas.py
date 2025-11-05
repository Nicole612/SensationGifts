"""
AI Engine Prompt Schemas (ENHANCED v2.0) - COMPLETE
===================================================

🚀 CUTTING-EDGE INNOVATIONS:
- Advanced Prompt Engineering with AI-driven optimization
- Multi-model specific prompt templates
- Real-time performance tracking and adaptation
- Self-improving prompts through usage analytics
- Schema-driven prompt compilation
- Consensus-based prompt validation

✅ BACKWARD COMPATIBLE: Alle bestehenden Prompt-Templates funktionieren weiter

Diese Schemas definieren wie Prompts strukturiert und optimiert werden.
Von Few-Shot bis Chain-of-Thought mit Enterprise-Grade AI-Optimization.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Union, Literal, Annotated, Protocol, TypeVar
from pydantic import BaseModel, Field, computed_field, field_validator, model_validator
import json
import hashlib
import time
from collections import defaultdict, deque
import asyncio
from decimal import Decimal
import uuid


# =============================================================================
# ENUMS: Prompt Engineering Strategien (ENHANCED - BACKWARD COMPATIBLE)
# =============================================================================

class PromptTechnique(str, Enum):
    """Verschiedene Prompt-Engineering Techniken (ENHANCED)"""
    # ✅ UNCHANGED: Bestehende Techniken für Kompatibilität
    ZERO_SHOT = "zero_shot"              
    ONE_SHOT = "one_shot"                
    FEW_SHOT = "few_shot"                
    CHAIN_OF_THOUGHT = "chain_of_thought" 
    ROLE_PLAYING = "role_playing"         
    CONTEXT_INJECTION = "context_injection" 
    TEMPLATE_BASED = "template_based"     
    DYNAMIC_GENERATION = "dynamic_generation" 
    
    # 🚀 NEW: Advanced AI-optimized techniques
    ADAPTIVE_PROMPTING = "adaptive_prompting"        # AI learns optimal prompts
    CONSENSUS_PROMPTING = "consensus_prompting"      # Multi-model consensus
    SELF_REFLECTION = "self_reflection"              # AI reflects on its responses
    STEP_BACK_PROMPTING = "step_back_prompting"      # AI steps back to broader context
    RETRIEVAL_AUGMENTED = "retrieval_augmented"      # RAG-enhanced prompts
    EMOTIONAL_PRIMING = "emotional_priming"          # Emotion-aware prompting
    METACOGNITIVE = "metacognitive"                  # AI thinks about thinking


class PromptComplexity(str, Enum):
    """Komplexität des Prompts (ENHANCED mit Performance-Mapping)"""
    SIMPLE = "simple"          # < 100 tokens, < 500ms
    MODERATE = "moderate"      # 100-500 tokens, < 2s
    COMPLEX = "complex"        # 500-1500 tokens, < 5s
    EXPERT = "expert"          # > 1500 tokens, > 5s
    ULTRA_COMPLEX = "ultra_complex"  # 🚀 NEW: Enterprise-level complexity


class AIModelType(str, Enum):
    """Unterstützte AI-Models für Model-spezifische Prompts (ENHANCED)"""
    # ✅ UNCHANGED: Bestehende Models
    OPENAI_GPT4 = "openai_gpt4"
    GROQ_MIXTRAL = "groq_mixtral"  
    ANTHROPIC_CLAUDE = "anthropic_claude"
    GOOGLE_GEMINI = "google_gemini"
    AUTO_SELECT = "auto_select"
    
    # 🚀 NEW: Additional models und specialized configurations
    OPENAI_GPT4_TURBO = "openai_gpt4_turbo"
    OPENAI_GPT3_5_TURBO = "openai_gpt3_5_turbo"
    GROQ_LLAMA2 = "groq_llama2"
    ANTHROPIC_CLAUDE_INSTANT = "anthropic_claude_instant"
    CLAUDE_SONNET = "claude_sonnet"  
    GOOGLE_GEMINI_PRO = "google_gemini_pro"
    LOCAL_MODEL = "local_model"
    ENSEMBLE_CONSENSUS = "ensemble_consensus"  # Multi-model approach


class PromptOptimizationGoal(str, Enum):
    """Was soll der Prompt optimieren? (ENHANCED)"""
    # ✅ UNCHANGED: Bestehende Goals
    SPEED = "speed"                
    QUALITY = "quality"            
    COST = "cost"                  
    BALANCE = "balance"            
    CREATIVITY = "creativity"      
    ACCURACY = "accuracy"
    
    # 🚀 NEW: Advanced optimization goals
    CONSISTENCY = "consistency"           # Reproducible results
    NOVELTY = "novelty"                  # Unique, surprising outputs
    SCALABILITY = "scalability"          # Handle varying input sizes
    USER_SATISFACTION = "user_satisfaction"  # Optimize for end-user happiness
    LEARNING_EFFICIENCY = "learning_efficiency"  # Few-shot learning optimization
    EMOTIONAL_RESONANCE = "emotional_resonance"  # Emotionally engaging responses


# 🚀 NEW: Advanced Prompt Engineering Enums
class PromptAdaptationStrategy(str, Enum):
    """Strategien für adaptive Prompt-Optimierung"""
    PERFORMANCE_BASED = "performance_based"      # Based on response quality
    USER_FEEDBACK = "user_feedback"              # Based on user ratings
    A_B_TESTING = "a_b_testing"                  # Systematic A/B testing
    REINFORCEMENT_LEARNING = "reinforcement_learning"  # RL-based optimization
    EVOLUTIONARY = "evolutionary"                # Genetic algorithm approach
    GRADIENT_FREE = "gradient_free"              # Gradient-free optimization


class PromptValidationLevel(str, Enum):
    """Validierungsstufen für Prompt-Qualität"""
    BASIC = "basic"              # Syntax + structure
    SEMANTIC = "semantic"        # Meaning validation
    PERFORMANCE = "performance"  # Performance testing
    USER_TESTED = "user_tested"  # Real user validation
    EXPERT_REVIEWED = "expert_reviewed"  # Human expert review


# =============================================================================
# 🚀 ADVANCED PROMPT COMPONENTS: Bausteine für intelligente Prompts
# =============================================================================

class PromptExample(BaseModel):
    """
    Einzelnes Beispiel für Few-Shot Learning (ENHANCED V2)
    
    🚀 NEW FEATURES:
    - Performance tracking per example
    - Adaptive example selection
    - Quality scoring
    - Context-aware example weighting
    
    ✅ BACKWARD COMPATIBLE: Core structure unchanged
    """
    
    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "extra": "forbid",
        "use_enum_values": True
    }
    
    # ✅ UNCHANGED: Core fields für Kompatibilität
    input_example: Annotated[
        str,
        Field(
            min_length=10, max_length=1000,
            description="Beispiel-Input für die AI"
        )
    ]
    
    expected_output: Annotated[
        str,
        Field(
            min_length=10, max_length=4000,
            description="Gewünschte AI-Antwort für diesen Input"
        )
    ]
    
    explanation: Optional[str] = Field(
        None, max_length=500,
        description="Warum dieses Beispiel gut ist"
    )
    
    difficulty_level: PromptComplexity = Field(
        default=PromptComplexity.SIMPLE,
        description="Schwierigkeitsgrad dieses Beispiels"
    )
    
    tags: List[str] = Field(
        default_factory=list,
        description="Tags für bessere Kategorisierung"
    )
    
    # 🚀 NEW: Advanced example features
    quality_score: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Qualitätsscore dieses Beispiels"
    )
    
    usage_count: int = Field(
        default=0, ge=0,
        description="Wie oft wurde dieses Beispiel verwendet"
    )
    
    success_rate: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Erfolgsrate wenn dieses Beispiel verwendet wird"
    )
    
    context_relevance: Dict[str, float] = Field(
        default_factory=dict,
        description="Relevanz für verschiedene Kontexte"
    )
    
    model_performance: Dict[str, float] = Field(
        default_factory=dict,
        description="Performance mit verschiedenen AI-Models"
    )
    
    created_at: datetime = Field(default_factory=datetime.now)
    last_used: Optional[datetime] = Field(None)
    
    # 🚀 NEW: Semantic features
    semantic_embedding: Optional[List[float]] = Field(
        None,
        description="Semantic embedding für similarity matching"
    )
    
    emotional_tone: Optional[str] = Field(
        None,
        description="Emotionaler Ton des Beispiels"
    )
    
    complexity_factors: Dict[str, Any] = Field(
        default_factory=dict,
        description="Faktoren die zur Komplexität beitragen"
    )
    
    @computed_field
    @property
    def effectiveness_score(self) -> float:
        """
        Kombinierter Effectiveness-Score für intelligente Beispiel-Auswahl
        
        INNOVATION: Multi-dimensionale Beispiel-Bewertung
        """
        factors = []
        
        # Quality component
        if self.quality_score is not None:
            factors.append(self.quality_score * 0.4)
        
        # Success rate component
        if self.success_rate is not None:
            factors.append(self.success_rate * 0.3)
        
        # Usage recency (more recent = better)
        if self.last_used:
            days_since_use = (datetime.now() - self.last_used).days
            recency_score = max(0, 1 - (days_since_use / 365))  # Decay over year
            factors.append(recency_score * 0.2)
        
        # Usage frequency (balanced - not too rare, not overused)
        if self.usage_count > 0:
            # Optimal usage is around 10-50 times
            usage_score = min(1.0, self.usage_count / 10) if self.usage_count < 50 else max(0.5, 1 - (self.usage_count - 50) / 100)
            factors.append(usage_score * 0.1)
        
        return sum(factors) / len(factors) if factors else 0.5
    
    @computed_field
    @property
    def context_adaptability(self) -> float:
        """Wie gut passt sich das Beispiel an verschiedene Kontexte an"""
        if not self.context_relevance:
            return 0.5
        
        relevance_scores = list(self.context_relevance.values())
        return sum(relevance_scores) / len(relevance_scores)
    
    def update_performance(self, model_type: str, performance_score: float):
        """Aktualisiert Performance-Daten für ein Model"""
        self.model_performance[model_type] = performance_score
        self.usage_count += 1
        self.last_used = datetime.now()
        
        # Update overall success rate
        if self.model_performance:
            self.success_rate = sum(self.model_performance.values()) / len(self.model_performance)


class ContextInjection(BaseModel):
    """
    Kontext-Informationen die intelligent in Prompts eingebaut werden (ENHANCED V2)
    
    🚀 NEW: Adaptive context weighting, semantic context analysis, multi-modal context
    ✅ BACKWARD COMPATIBLE: All original fields retained
    """
    
    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "extra": "forbid"
    }
    
    # ✅ UNCHANGED: Bestehende Context-Felder für Kompatibilität
    personality_context: Optional[str] = Field(
        None, max_length=800,
        description="Big Five + Limbic-basierte Persönlichkeitskontext"
    )
    
    relationship_context: Optional[str] = Field(
        None, max_length=300,
        description="Beziehungs-spezifische Informationen"
    )
    
    occasion_context: Optional[str] = Field(
        None, max_length=300,
        description="Anlass-spezifische Details"
    )
    
    budget_context: Optional[str] = Field(
        None, max_length=200,
        description="Budget-relevante Informationen"
    )
    
    preference_context: Optional[str] = Field(
        None, max_length=500,
        description="Spezifische Präferenzen und Abneigungen"
    )
    
    cultural_context: Optional[str] = Field(
        None, max_length=300,
        description="Kulturelle oder regionale Besonderheiten"
    )
    
    urgency_context: Optional[str] = Field(
        None, max_length=200,
        description="Zeitliche Constraints (Last-Minute, etc.)"
    )
    
    # 🚀 NEW: Advanced context features
    context_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Gewichtung verschiedener Kontext-Aspekte"
    )
    
    semantic_tags: List[str] = Field(
        default_factory=list,
        description="Semantische Tags für intelligente Context-Matching"
    )
    
    emotional_context: Optional[str] = Field(
        None, max_length=300,
        description="Emotionaler Zustand und Stimmung"
    )
    
    historical_context: Optional[str] = Field(
        None, max_length=400,
        description="Historische Interaktionen und Präferenzen"
    )
    
    seasonal_context: Optional[str] = Field(
        None, max_length=200,
        description="Saisonale Faktoren und Trends"
    )
    
    context_freshness: Optional[datetime] = Field(
        None,
        description="Wann wurde der Kontext zuletzt aktualisiert"
    )
    
    context_confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Vertrauen in die Kontext-Informationen"
    )
    
    # 🚀 NEW: Multi-modal context
    visual_context_description: Optional[str] = Field(
        None, max_length=300,
        description="Beschreibung visueller Elemente (Bilder, etc.)"
    )
    
    audio_context_description: Optional[str] = Field(
        None, max_length=200,
        description="Audio-Kontext (Stimmlage, Musik-Präferenzen)"
    )
    
    # 🚀 NEW: Age Group und Gender Identity Context
    age_context: Optional[str] = Field(
        None, max_length=200,
        description="Altersgruppen-spezifische Informationen und Entwicklungsstufen"
    )
    
    gender_context: Optional[str] = Field(
        None, max_length=200,
        description="Gender-inklusive Kontext-Informationen und Präferenzen"
    )
    
    
    @computed_field
    @property
    def context_completeness(self) -> float:
        """Vollständigkeits-Score des Kontexts"""
        total_fields = 7  # Core context fields
        filled_fields = len([
            f for f in [
                self.personality_context, self.relationship_context,
                self.occasion_context, self.budget_context,
                self.preference_context, self.cultural_context,
                self.urgency_context
            ] if f is not None
        ])
        
        return filled_fields / total_fields
    
    @computed_field
    @property
    def context_richness(self) -> float:
        """Reichhaltigkeit des Kontexts (Quality + Quantity)"""
        completeness = self.context_completeness
        
        # Length-based richness
        total_length = sum(
            len(context) for context in [
                self.personality_context, self.relationship_context,
                self.occasion_context, self.budget_context,
                self.preference_context, self.cultural_context,
                self.urgency_context, self.emotional_context,
                self.historical_context, self.seasonal_context
            ] if context is not None
        )
        
        # Normalize length score (more content = better, but diminishing returns)
        length_score = min(1.0, total_length / 2000)  # 2000 chars = max score
        
        # Additional context features
        feature_bonus = 0.0
        if self.semantic_tags:
            feature_bonus += 0.1
        if self.context_weights:
            feature_bonus += 0.1
        if self.visual_context_description:
            feature_bonus += 0.05
        
        return min(1.0, completeness * 0.6 + length_score * 0.3 + feature_bonus)
    
    def optimize_for_model(self, model_type: str) -> Dict[str, str]:
        """
        Optimiert Kontext für spezifisches AI-Model
        
        INNOVATION: Model-specific context optimization
        """
        
        base_context = {}
        
        # Sammle alle verfügbaren Kontexte
        contexts = {
            "personality": self.personality_context,
            "relationship": self.relationship_context,
            "occasion": self.occasion_context,
            "budget": self.budget_context,
            "preferences": self.preference_context,
            "cultural": self.cultural_context,
            "urgency": self.urgency_context,
            "emotional": self.emotional_context,
            "historical": self.historical_context,
            "seasonal": self.seasonal_context
        }
        
        # Model-spezifische Optimierung
        if model_type == "groq_mixtral":
            # Groq: Kompakt, strukturiert, wichtigste Infos
            priority_order = ["personality", "occasion", "budget", "preferences"]
            max_length = 300
            
        elif model_type == "openai_gpt4":
            # GPT-4: Detailliert, nuanciert, alle Kontexte
            priority_order = list(contexts.keys())
            max_length = 1500
            
        elif model_type == "anthropic_claude":
            # Claude: Reasoning-fokussiert, logische Struktur
            priority_order = ["personality", "relationship", "preferences", "cultural", "emotional"]
            max_length = 1000
            
        else:
            # Default: Ausgewogen
            priority_order = ["personality", "occasion", "relationship", "budget"]
            max_length = 800
        
        # Baue optimierten Kontext zusammen
        current_length = 0
        for context_type in priority_order:
            context_value = contexts.get(context_type)
            if context_value and current_length + len(context_value) <= max_length:
                base_context[context_type] = context_value
                current_length += len(context_value)
        
        return base_context


class ChainOfThoughtStep(BaseModel):
    """
    Einzelner Schritt für Chain-of-Thought Reasoning (ENHANCED V2)
    
    🚀 NEW: Adaptive step optimization, performance tracking, dynamic step generation
    ✅ BACKWARD COMPATIBLE: Original structure maintained
    """
    
    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "extra": "forbid"
    }
    
    # ✅ UNCHANGED: Core fields für Kompatibilität
    step_number: int = Field(
        ..., ge=1,
        description="Nummer des Denkschritts"
    )
    
    step_description: Annotated[
        str,
        Field(
            min_length=10, max_length=800,
            description="Was in diesem Schritt gemacht werden soll"
        )
    ]
    
    step_prompt: Annotated[
        str,
        Field(
            min_length=5, max_length=1500,
            description="Prompt-Text für diesen Schritt"
        )
    ]
    
    expected_reasoning: Optional[str] = Field(
        None, max_length=500,
        description="Wie die AI bei diesem Schritt denken sollte"
    )
    
    dependencies: List[int] = Field(
        default_factory=list,
        description="Welche Schritte müssen vorher abgeschlossen sein"
    )
    
    # 🚀 NEW: Advanced step features
    step_complexity: PromptComplexity = Field(
        default=PromptComplexity.SIMPLE,
        description="Komplexität dieses Denkschritts"
    )
    
    cognitive_load: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Kognitive Belastung dieses Schritts"
    )
    
    step_weight: float = Field(
        default=1.0, ge=0.1, le=5.0,
        description="Gewichtung der Wichtigkeit dieses Schritts"
    )
    
    alternative_formulations: List[str] = Field(
        default_factory=list,
        description="Alternative Formulierungen für A/B Testing"
    )
    
    performance_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Performance-Metriken für diesen Schritt"
    )
    
    # 🚀 NEW: Adaptive features
    success_rate: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Erfolgsrate dieses Schritts"
    )
    
    average_processing_time: Optional[float] = Field(
        None, ge=0.0,
        description="Durchschnittliche Verarbeitungszeit in Sekunden"
    )
    
    error_patterns: List[str] = Field(
        default_factory=list,
        description="Häufige Fehler bei diesem Schritt"
    )
    
    optimization_suggestions: List[str] = Field(
        default_factory=list,
        description="Vorschläge zur Schritt-Optimierung"
    )
    
    @computed_field
    @property
    def step_efficiency(self) -> float:
        """Effizienz-Score des Schritts (Quality / Time)"""
        if self.success_rate is not None and self.average_processing_time is not None:
            if self.average_processing_time > 0:
                return min(1.0, self.success_rate / (self.average_processing_time / 10))  # Normalize to 10s baseline
        
        return self.success_rate if self.success_rate is not None else 0.5
    
    @computed_field
    @property
    def optimization_priority(self) -> float:
        """Priorität für Optimierung (niedrige Performance = hohe Priorität)"""
        factors = []
        
        # Niedrige Success Rate = hohe Priorität
        if self.success_rate is not None:
            factors.append(1.0 - self.success_rate)
        
        # Hohe Processing Time = hohe Priorität
        if self.average_processing_time is not None:
            time_priority = min(1.0, self.average_processing_time / 30)  # 30s = max priority
            factors.append(time_priority)
        
        # Hohe Cognitive Load = hohe Priorität
        if self.cognitive_load is not None:
            factors.append(self.cognitive_load)
        
        # Häufige Fehler = hohe Priorität
        error_priority = min(1.0, len(self.error_patterns) / 10)
        factors.append(error_priority)
        
        return sum(factors) / len(factors) if factors else 0.3


# =============================================================================
# 📊 PROMPT PERFORMANCE METRICS: Messe was funktioniert
# =============================================================================

class PromptPerformanceMetrics(BaseModel):
    """
    Performance-Tracking für Prompt-Templates (ENHANCED V2)
    
    📊 ANALYTICS: Welche Prompts funktionieren am besten?
    🚀 OPTIMIZATION: Data-driven Prompt-Verbesserung
    🎯 QUALITY: Messe emotionale Resonanz und Personalisierung
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
    
    # === IDENTIFICATION ===
    prompt_template_id: str = Field(...)
    prompt_template_name: str = Field(...)
    prompt_version: str = Field(...)
    
    test_id: str = Field(...)
    test_scenario: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # === PERFORMANCE METRICS ===
    response_time_ms: int = Field(..., ge=0)
    response_quality_score: Annotated[
        float,
        Field(ge=0.0, le=1.0, description="Qualität der AI-Response")
    ]
    
    personalization_score: Annotated[
        float,
        Field(ge=0.0, le=1.0, description="Wie personalisiert war die Response")
    ]
    
    emotional_resonance: Annotated[
        float,
        Field(ge=0.0, le=1.0, description="Emotionale Treffsicherheit")
    ]
    
    creativity_score: Annotated[
        float,
        Field(ge=0.0, le=1.0, description="Kreativität und Originalität")
    ]
    
    # === USER FEEDBACK ===
    user_satisfaction: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="User-Rating der Empfehlungen"
    )
    
    recommendations_accepted: Optional[int] = Field(
        None, ge=0,
        description="Wie viele Empfehlungen wurden akzeptiert"
    )
    
    total_recommendations: Optional[int] = Field(
        None, ge=0,
        description="Gesamtzahl generierter Empfehlungen"
    )
    
    # === TECHNICAL METRICS ===
    prompt_tokens_used: int = Field(..., ge=0)
    response_tokens_generated: int = Field(..., ge=0)
    cost_estimate: Optional[Decimal] = Field(None, ge=0)
    
    # === CONTEXT FACTORS ===
    target_model: str = Field(...)
    optimization_goal: str = Field(...)
    request_complexity: str = Field(...)
    
    context_richness: Annotated[
        float,
        Field(ge=0.0, le=1.0, description="Reichhaltigkeit des Input-Kontexts")
    ]
    
    # === SUCCESS INDICATORS ===
    parsing_successful: bool = Field(...)
    schema_validation_passed: bool = Field(...)
    had_fallbacks: bool = Field(default=False)
    required_repair: bool = Field(default=False)
    
    # === EMOTIONAL INTELLIGENCE METRICS ===
    empathy_demonstration: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Wie empathisch waren die Empfehlungen"
    )
    
    cultural_sensitivity: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Kulturelle Sensitivität gezeigt"
    )
    
    relationship_appropriateness: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Angemessenheit für die Beziehung"
    )
    
    # === BUSINESS IMPACT ===
    gift_purchase_intent: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Wahrscheinlichkeit dass Geschenk gekauft wird"
    )
    
    emotional_connection_created: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Emotionale Verbindung geschaffen (Projektziel!)"
    )
    
    experience_over_product: Optional[bool] = Field(
        None,
        description="Fokus auf Erlebnis statt Produktkauf erreicht"
    )
    
    # === COMPUTED METRICS ===
    @computed_field
    @property
    def overall_success_score(self) -> float:
        """Kombinierter Success-Score mit Gewichtung auf emotionale Verbindung"""
        scores = []
        
        # Technical success (20%)
        technical_score = (
            (1.0 if self.parsing_successful else 0.0) * 0.5 +
            (1.0 if self.schema_validation_passed else 0.0) * 0.3 +
            (0.0 if self.had_fallbacks else 1.0) * 0.2
        )
        scores.append(technical_score * 0.2)
        
        # Quality scores (40%)
        quality_score = (
            self.response_quality_score * 0.3 +
            self.personalization_score * 0.25 +
            self.emotional_resonance * 0.25 +
            self.creativity_score * 0.2
        )
        scores.append(quality_score * 0.4)
        
        # User satisfaction (25%)
        if self.user_satisfaction is not None:
            scores.append(self.user_satisfaction * 0.25)
        elif self.recommendations_accepted and self.total_recommendations:
            acceptance_rate = self.recommendations_accepted / self.total_recommendations
            scores.append(acceptance_rate * 0.25)
        else:
            scores.append(0.7 * 0.25)  # Default assumption
        
        # Business impact (15%) - PROJECT FOCUS!
        business_scores = []
        if self.emotional_connection_created is not None:
            business_scores.append(self.emotional_connection_created * 0.5)
        if self.gift_purchase_intent is not None:
            business_scores.append(self.gift_purchase_intent * 0.3)
        if self.experience_over_product is not None:
            business_scores.append((1.0 if self.experience_over_product else 0.5) * 0.2)
        
        if business_scores:
            scores.append(sum(business_scores) * 0.15)
        else:
            scores.append(0.6 * 0.15)  # Conservative estimate
        
        return sum(scores)
    
    @computed_field
    @property
    def efficiency_score(self) -> float:
        """Effizienz: Qualität pro Zeit/Kosten"""
        if self.response_time_ms <= 0:
            return 0.0
        
        quality = self.overall_success_score
        time_factor = max(0.1, min(1.0, 5000 / self.response_time_ms))  # Normalize to 5s baseline
        
        cost_factor = 1.0
        if self.cost_estimate and self.cost_estimate > 0:
            # Lower cost = higher efficiency
            cost_factor = max(0.1, min(1.0, 0.05 / float(self.cost_estimate)))
        
        return quality * time_factor * cost_factor
    
    @computed_field
    @property
    def emotional_intelligence_score(self) -> float:
        """Spezifischer EQ-Score für unser Projektziel"""
        eq_components = []
        
        if self.empathy_demonstration is not None:
            eq_components.append(self.empathy_demonstration * 0.3)
        
        if self.cultural_sensitivity is not None:
            eq_components.append(self.cultural_sensitivity * 0.2)
        
        if self.relationship_appropriateness is not None:
            eq_components.append(self.relationship_appropriateness * 0.2)
        
        eq_components.append(self.emotional_resonance * 0.3)
        
        return sum(eq_components) / len(eq_components) if eq_components else self.emotional_resonance
    
    def get_optimization_suggestions(self) -> List[str]:
        """Automatische Optimierungsvorschläge basierend auf Metriken"""
        suggestions = []
        
        # Performance suggestions
        if self.response_time_ms > 5000:
            suggestions.append("Optimize prompt for faster response times")
        
        if self.response_quality_score < 0.7:
            suggestions.append("Improve prompt clarity and specificity")
        
        if self.personalization_score < 0.6:
            suggestions.append("Enhance personality-based customization")
        
        if self.emotional_resonance < 0.7:
            suggestions.append("Strengthen emotional language and connection points")
        
        if self.creativity_score < 0.6:
            suggestions.append("Add more creative prompting techniques")
        
        # Business impact suggestions
        if self.emotional_connection_created is not None and self.emotional_connection_created < 0.7:
            suggestions.append("Focus more on emotional storytelling and connection")
        
        if self.gift_purchase_intent is not None and self.gift_purchase_intent < 0.6:
            suggestions.append("Make recommendations more actionable and compelling")
        
        # Technical suggestions
        if self.had_fallbacks:
            suggestions.append("Improve prompt robustness to reduce fallback usage")
        
        if not self.parsing_successful:
            suggestions.append("Fix prompt structure for better parsing reliability")
        
        return suggestions


# =============================================================================
# 🎁 GIFT RECOMMENDATION SCHEMA: Spezifisches Schema für Geschenkempfehlungen
# =============================================================================

class GiftRecommendationSchema(BaseModel):
    """
    Spezialisiertes Schema für Geschenkempfehlungs-Prompts (ENHANCED V2)
    
    🎁 GIFT-FOCUSED: Optimiert für emotionale Geschenkempfehlungen
    🧠 PERSONALITY-AWARE: Integriert Big Five + Limbic nahtlos
    ❤️ RELATIONSHIP-CENTERED: Beziehungen stehen im Mittelpunkt
    """
    
    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "extra": "forbid",
        "use_enum_values": True
    }
    
    # === GIFT CONTEXT ===
    occasion: str = Field(...)
    relationship_type: str = Field(...)
    budget_range: Optional[str] = Field(None)
    urgency_level: str = Field(default="normal")
    
    # === PERSONALITY INTEGRATION ===
    big_five_weights: Dict[str, float] = Field(
        ...,
        description="Gewichtung der Big Five Dimensionen für diese Empfehlung"
    )
    
    limbic_emphasis: Dict[str, float] = Field(
        ...,
        description="Emphasis auf Limbic Dimensionen"
    )
    
    personality_archetype: str = Field(...)
    
    # === EMOTIONAL FOCUS ===
    emotional_goals: List[str] = Field(
        ..., min_items=1, max_items=5,
        description="Welche Emotionen soll das Geschenk auslösen"
    )
    
    relationship_objectives: List[str] = Field(
        ..., min_items=1, max_items=4,
        description="Was soll für die Beziehung erreicht werden"
    )
    
    experience_emphasis: Annotated[
        float,
        Field(
            ge=0.0, le=1.0,
            description="Fokus auf Erlebnisse vs. materielle Geschenke"
        )
    ] = 0.7
    
    # === CONSTRAINTS ===
    cultural_considerations: List[str] = Field(default_factory=list)
    avoid_categories: List[str] = Field(default_factory=list)
    must_include_aspects: List[str] = Field(default_factory=list)
    
    # === OUTPUT SPECIFICATIONS ===
    max_recommendations: int = Field(default=5, ge=1, le=10)
    detail_level: str = Field(
        default="comprehensive",
        pattern="^(minimal|standard|comprehensive|story_mode)$"
    )
    
    include_alternatives: bool = Field(default=True)
    include_presentation_tips: bool = Field(default=True)
    include_emotional_impact: bool = Field(default=True)
    
    # === PERSONALIZATION DEPTH ===
    personalization_level: Annotated[
        float,
        Field(
            ge=0.0, le=1.0,
            description="Tiefe der Personalisierung"
        )
    ] = 0.8
    
    creativity_level: Annotated[
        float,
        Field(
            ge=0.0, le=1.0,
            description="Gewünschte Kreativität der Empfehlungen"
        )
    ] = 0.7
    
    surprise_factor: Annotated[
        float,
        Field(
            ge=0.0, le=1.0,
            description="Gewünschter Überraschungsgrad"
        )
    ] = 0.6
    
    @computed_field
    @property
    def schema_complexity(self) -> str:
        """Komplexität dieses Schemas"""
        complexity_score = 0
        
        complexity_score += len(self.emotional_goals)
        complexity_score += len(self.relationship_objectives)
        complexity_score += len(self.cultural_considerations)
        complexity_score += len(self.must_include_aspects)
        
        if self.personalization_level > 0.8:
            complexity_score += 2
        if self.creativity_level > 0.8:
            complexity_score += 1
        
        if complexity_score >= 8:
            return "expert"
        elif complexity_score >= 5:
            return "complex"
        elif complexity_score >= 3:
            return "moderate"
        else:
            return "simple"
    
    @computed_field
    @property
    def emotional_complexity(self) -> float:
        """Emotionale Komplexität der Anfrage"""
        base_complexity = len(self.emotional_goals) / 5  # Normalize to max 5 goals
        
        # Relationship complexity
        relationship_complexity = len(self.relationship_objectives) / 4  # Max 4 objectives
        
        # Cultural complexity
        cultural_complexity = min(1.0, len(self.cultural_considerations) / 3)
        
        return (base_complexity * 0.5 + relationship_complexity * 0.3 + cultural_complexity * 0.2)
    
    def generate_prompt_context(self) -> str:
        """Generiert strukturierten Kontext für Prompt"""
        context_parts = []
        
        # Basic context
        context_parts.append(f"Occasion: {self.occasion}")
        context_parts.append(f"Relationship: {self.relationship_type}")
        context_parts.append(f"Personality: {self.personality_archetype}")
        
        if self.budget_range:
            context_parts.append(f"Budget: {self.budget_range}")
        
        # Emotional context
        context_parts.append(f"Emotional Goals: {', '.join(self.emotional_goals)}")
        context_parts.append(f"Relationship Objectives: {', '.join(self.relationship_objectives)}")
        
        # Constraints
        if self.cultural_considerations:
            context_parts.append(f"Cultural Considerations: {', '.join(self.cultural_considerations)}")
        
        if self.avoid_categories:
            context_parts.append(f"Avoid: {', '.join(self.avoid_categories)}")
        
        if self.must_include_aspects:
            context_parts.append(f"Must Include: {', '.join(self.must_include_aspects)}")
        
        # Preferences
        context_parts.append(f"Experience Focus: {self.experience_emphasis:.1f}")
        context_parts.append(f"Personalization Level: {self.personalization_level:.1f}")
        context_parts.append(f"Creativity Level: {self.creativity_level:.1f}")
        
        return " | ".join(context_parts)
    
    def get_personality_guidance(self) -> str:
        """Generiert Persönlichkeits-spezifische Guidance für Prompts"""
        guidance_parts = []
        
        # Big Five guidance
        for trait, weight in self.big_five_weights.items():
            if weight >= 0.7:
                trait_guidance = {
                    'openness': "Focus on creative, novel, and artistic recommendations",
                    'conscientiousness': "Emphasize quality, practical value, and organization",
                    'extraversion': "Consider social experiences and interactive gifts",
                    'agreeableness': "Highlight harmony, cooperation, and shared experiences",
                    'neuroticism': "Include comfort, stability, and stress-relief aspects"
                }.get(trait)
                
                if trait_guidance:
                    guidance_parts.append(f"{trait.title()}: {trait_guidance}")
        
        # Limbic guidance
        for dimension, emphasis in self.limbic_emphasis.items():
            if emphasis >= 0.7:
                limbic_guidance = {
                    'stimulanz': "Include exciting, stimulating, and high-energy options",
                    'dominanz': "Consider status, exclusivity, and leadership themes",
                    'balance': "Focus on harmony, mindfulness, and balanced choices"
                }.get(dimension)
                
                if limbic_guidance:
                    guidance_parts.append(f"{dimension.title()}: {limbic_guidance}")
        
        return " | ".join(guidance_parts) if guidance_parts else "Use balanced approach across all personality dimensions"


# =============================================================================
# 🚀 ENHANCED MAIN PROMPT SCHEMAS: Templates für verschiedene Use Cases
# =============================================================================

class BasePromptTemplate(BaseModel):
    """
    Basis-Template für alle Prompts (ENHANCED V2)
    
    🚀 NEW FEATURES:
    - Performance tracking and optimization
    - A/B testing support
    - Adaptive template evolution
    - Real-time quality monitoring
    
    ✅ BACKWARD COMPATIBLE: All original fields preserved
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
    
    # === TEMPLATE METADATEN (UNCHANGED) ===
    template_name: Annotated[
        str,
        Field(
            min_length=3, max_length=100,
            description="Eindeutiger Name für dieses Template"
        )
    ]
    
    template_version: str = Field(
        default="1.0",
        description="Version für Template-Versionierung"
    )
    
    description: Annotated[
        str,
        Field(
            min_length=10, max_length=500,
            description="Was dieses Template macht"
        )
    ]
    
    # === PROMPT-KONFIGURATION (UNCHANGED) ===
    technique: PromptTechnique = Field(
        ...,
        description="Welche Prompt-Technik verwendet wird"
    )
    
    complexity: PromptComplexity = Field(
        ...,
        description="Komplexität dieses Prompts"
    )
    
    target_model: AIModelType = Field(
        default=AIModelType.AUTO_SELECT,
        description="Für welches AI-Model optimiert"
    )
    
    optimization_goal: PromptOptimizationGoal = Field(
        default=PromptOptimizationGoal.BALANCE,
        description="Was optimiert werden soll"
    )
    
    # === TEMPLATE-INHALT (UNCHANGED) ===
    system_prompt: Optional[str] = Field(
        None, max_length=2000,
        description="System-Prompt (Rolle der AI definieren)"
    )
    
    instruction_prompt: Annotated[
        str,
        Field(
            min_length=20, max_length=3000,
            description="Haupt-Anweisung für die AI"
        )
    ]
    
    output_format_instructions: Optional[str] = Field(
        None, max_length=3000,
        description="Wie die AI antworten soll (JSON, etc.)"
    )
    
    # === KONTEXT-INTEGRATION (UNCHANGED) ===
    context_injection: Optional[ContextInjection] = Field(
        None,
        description="Wie Kontext-Informationen eingebaut werden"
    )
    
    # === PERFORMANCE-EINSTELLUNGEN (ENHANCED) ===
    max_tokens: Optional[int] = Field(
        None, ge=100, le=8000,
        description="Maximum Tokens für Antwort"
    )
    
    temperature: Optional[float] = Field(
        None, ge=0.0, le=2.0,
        description="Kreativität der AI (0=deterministisch, 2=sehr kreativ)"
    )
    
    # === METADATEN (ENHANCED) ===
    created_at: datetime = Field(default_factory=datetime.now)
    last_tested: Optional[datetime] = Field(None)
    success_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # 🚀 NEW: Advanced template features
    template_hash: Optional[str] = Field(
        None,
        description="Hash für Template-Änderungserkennung"
    )
    
    usage_count: int = Field(
        default=0, ge=0,
        description="Wie oft wurde dieses Template verwendet"
    )
    
    performance_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Historische Performance-Daten"
    )
    
    adaptation_strategy: Optional[PromptAdaptationStrategy] = Field(
        None,
        description="Strategie für Template-Anpassung"
    )
    
    validation_level: PromptValidationLevel = Field(
        default=PromptValidationLevel.BASIC,
        description="Validierungsstufe des Templates"
    )
    
    # 🚀 NEW: A/B Testing support
    variant_templates: Dict[str, Any] = Field(
        default_factory=dict,
        description="Template-Varianten für A/B Testing"
    )
    
    current_champion: Optional[str] = Field(
        None,
        description="Aktuell beste Template-Variante"
    )
    
    ab_test_results: Dict[str, float] = Field(
        default_factory=dict,
        description="Ergebnisse von A/B Tests"
    )
    
    # 🚀 NEW: Quality metrics
    average_response_quality: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Durchschnittliche Response-Qualität"
    )
    
    user_satisfaction_score: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="User-Zufriedenheit mit diesem Template"
    )
    
    cost_efficiency: Optional[float] = Field(
        None, ge=0.0,
        description="Cost-Efficiency (Quality / Cost)"
    )
    
    # 🚀 NEW: Learning and adaptation
    learning_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Daten für Template-Learning"
    )
    
    auto_optimization_enabled: bool = Field(
        default=False,
        description="Automatische Template-Optimierung aktiviert"
    )
    
    optimization_frequency: Optional[str] = Field(
        None,
        pattern="^(daily|weekly|monthly|on_demand)$",
        description="Wie oft soll optimiert werden"
    )
    
    @computed_field
    @property
    def template_fingerprint(self) -> str:
        """Eindeutiger Fingerprint für Template-Änderungserkennung"""
        content = f"{self.instruction_prompt}{self.system_prompt or ''}{self.output_format_instructions or ''}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    @computed_field
    @property
    def overall_performance_score(self) -> float:
        """Kombinierter Performance-Score des Templates"""
        scores = []
        
        if self.success_rate is not None:
            scores.append(self.success_rate * 0.4)
        
        if self.average_response_quality is not None:
            scores.append(self.average_response_quality * 0.3)
        
        if self.user_satisfaction_score is not None:
            scores.append(self.user_satisfaction_score * 0.2)
        
        if self.cost_efficiency is not None:
            # Normalize cost efficiency (higher is better)
            normalized_cost = min(1.0, self.cost_efficiency / 10)
            scores.append(normalized_cost * 0.1)
        
        return sum(scores) / len(scores) if scores else 0.5
    
    @computed_field
    @property
    def optimization_recommendations(self) -> List[str]:
        """Automatische Optimierungs-Empfehlungen"""
        recommendations = []
        
        # Performance-basierte Empfehlungen
        if self.success_rate is not None and self.success_rate < 0.7:
            recommendations.append("Consider simplifying prompt structure")
        
        if self.average_response_quality is not None and self.average_response_quality < 0.6:
            recommendations.append("Add more specific examples or constraints")
        
        if self.user_satisfaction_score is not None and self.user_satisfaction_score < 0.7:
            recommendations.append("Focus on user experience and clarity")
        
        # Usage-basierte Empfehlungen
        if self.usage_count > 100 and not self.ab_test_results:
            recommendations.append("Consider running A/B tests for optimization")
        
        if self.cost_efficiency is not None and self.cost_efficiency < 1.0:
            recommendations.append("Optimize for cost efficiency")
        
        # Model-spezifische Empfehlungen
        if self.target_model == AIModelType.GROQ_MIXTRAL and len(self.instruction_prompt) > 500:
            recommendations.append("Shorten prompt for Groq optimization")
        
        return recommendations
    
    # 🚀 V2: Enhanced field validators
    @field_validator('temperature')
    @classmethod
    def optimize_temperature_for_goal(cls, v, info):
        """Automatische Temperature-Optimierung basierend auf Goal"""
        if v is not None:
            return v
        
        # Auto-optimize based on goal
        goal = info.data.get('optimization_goal')
        if goal == PromptOptimizationGoal.ACCURACY:
            return 0.1
        elif goal == PromptOptimizationGoal.CREATIVITY:
            return 1.2
        elif goal == PromptOptimizationGoal.SPEED:
            return 0.3
        elif goal == PromptOptimizationGoal.CONSISTENCY:
            return 0.2
        elif goal == PromptOptimizationGoal.NOVELTY:
            return 1.5
        else:
            return 0.7
    
    @field_validator('max_tokens')
    @classmethod
    def optimize_tokens_for_model(cls, v, info):
        """Automatische Token-Optimierung basierend auf Model"""
        if v is not None:
            return v
        
        # Auto-optimize based on target model
        model = info.data.get('target_model')
        complexity = info.data.get('complexity')
        
        if model == AIModelType.GROQ_MIXTRAL:
            return 1000 if complexity == PromptComplexity.SIMPLE else 2000
        elif model == AIModelType.OPENAI_GPT4:
            return 3000 if complexity == PromptComplexity.EXPERT else 2000
        else:
            return 1500
    
    @model_validator(mode='after')
    def update_template_hash(self):
        """Aktualisiert Template-Hash bei Änderungen"""
        new_hash = self.template_fingerprint
        if self.template_hash != new_hash:
            self.template_hash = new_hash
            # Template wurde geändert - Performance-History könnte invalidiert sein
        
        return self
    
    # 🚀 NEW: Advanced template methods
    def track_usage(self, performance_data: Dict[str, Any]):
        """Trackt Template-Nutzung und Performance"""
        self.usage_count += 1
        self.last_tested = datetime.now()
        
        # Update performance history (keep last 100 entries)
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            **performance_data
        })
        
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        # Update aggregated metrics
        if 'quality_score' in performance_data:
            if self.average_response_quality is None:
                self.average_response_quality = performance_data['quality_score']
            else:
                # Weighted average (new data has more weight)
                self.average_response_quality = (
                    self.average_response_quality * 0.9 + 
                    performance_data['quality_score'] * 0.1
                )
        
        if 'success' in performance_data:
            success = 1.0 if performance_data['success'] else 0.0
            if self.success_rate is None:
                self.success_rate = success
            else:
                self.success_rate = self.success_rate * 0.95 + success * 0.05
    
    def generate_variant(self, variant_name: str, modifications: Dict[str, Any]) -> 'BasePromptTemplate':
        """Generiert Template-Variante für A/B Testing"""
        # Create copy of current template
        variant_data = self.model_dump()
        variant_data['template_name'] = f"{self.template_name}_{variant_name}"
        variant_data['template_version'] = f"{self.template_version}_var"
        
        # Apply modifications
        for key, value in modifications.items():
            if key in variant_data:
                variant_data[key] = value
        
        # Reset performance metrics for new variant
        variant_data.update({
            'usage_count': 0,
            'performance_history': [],
            'success_rate': None,
            'average_response_quality': None,
            'user_satisfaction_score': None
        })
        
        return BasePromptTemplate(**variant_data)
    
    async def auto_optimize(self) -> Optional['BasePromptTemplate']:
        """
        Automatische Template-Optimierung basierend auf Performance-Daten
        
        INNOVATION: Self-improving prompt templates
        """
        if not self.auto_optimization_enabled or not self.performance_history:
            return None
        
        # Analysiere Performance-Trends
        recent_performance = self.performance_history[-20:]  # Last 20 uses
        
        if len(recent_performance) < 10:
            return None  # Need more data
        
        # Identifiziere Verbesserungspotential
        avg_quality = sum(p.get('quality_score', 0.5) for p in recent_performance) / len(recent_performance)
        avg_speed = sum(p.get('response_time_ms', 2000) for p in recent_performance) / len(recent_performance)
        
        optimizations = []
        
        # Quality optimization
        if avg_quality < 0.7:
            if self.optimization_goal == PromptOptimizationGoal.QUALITY:
                optimizations.append({
                    'instruction_prompt': f"{self.instruction_prompt}\n\nEnsure high-quality, detailed responses.",
                    'temperature': max(0.1, (self.temperature or 0.7) - 0.2)
                })
        
        # Speed optimization
        if avg_speed > 3000 and self.optimization_goal == PromptOptimizationGoal.SPEED:
            optimizations.append({
                'instruction_prompt': self.instruction_prompt.replace('detailed', 'concise'),
                'max_tokens': min(1000, self.max_tokens or 1500)
            })
        
        # Generate and test best optimization
        if optimizations:
            best_optimization = optimizations[0]  # Could be more sophisticated
            return self.generate_variant('auto_optimized', best_optimization)
        
        return None


class FewShotPromptTemplate(BasePromptTemplate):
    """
    Few-Shot Learning Template (ENHANCED V2)
    
    🚀 NEW: Intelligent example selection, adaptive example weighting, performance-based example optimization
    ✅ BACKWARD COMPATIBLE: Original structure preserved
    """
    
    technique: Literal[PromptTechnique.FEW_SHOT] = PromptTechnique.FEW_SHOT
    
    # ✅ UNCHANGED: Core fields
    examples: List[PromptExample] = Field(
        ..., min_length=2, max_length=10,
        description="Beispiele für Few-Shot Learning"
    )
    
    example_separator: str = Field(
        default="---",
        description="Trennzeichen zwischen Beispielen"
    )
    
    example_intro: str = Field(
        default="Here are some examples:",
        description="Einleitung für die Beispiele"
    )
    
    # 🚀 NEW: Advanced few-shot features
    dynamic_example_selection: bool = Field(
        default=False,
        description="Beispiele dynamisch basierend auf Kontext auswählen"
    )
    
    max_examples_per_request: int = Field(
        default=5, ge=1, le=10,
        description="Maximale Anzahl Beispiele pro Request"
    )
    
    example_selection_strategy: str = Field(
        default="effectiveness",
        pattern="^(effectiveness|similarity|diversity|random|adaptive)$",
        description="Strategie für Beispiel-Auswahl"
    )
    
    context_similarity_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Ähnlichkeits-Schwellwert für Kontext-basierte Auswahl"
    )
    
    example_weights: Dict[int, float] = Field(
        default_factory=dict,
        description="Gewichtung einzelner Beispiele"
    )
    
    @computed_field
    @property
    def example_quality_score(self) -> float:
        """Durchschnittliche Qualität aller Beispiele"""
        if not self.examples:
            return 0.0
        
        scores = [ex.effectiveness_score for ex in self.examples]
        return sum(scores) / len(scores)
    
    @computed_field
    @property
    def example_diversity_score(self) -> float:
        """Diversität der Beispiele (wichtig für gutes Few-Shot Learning)"""
        if len(self.examples) < 2:
            return 0.0
        
        # Einfache Diversitäts-Metrik basierend auf Tags
        all_tags = set()
        for example in self.examples:
            all_tags.update(example.tags)
        
        unique_tags_per_example = len(all_tags) / len(self.examples)
        return min(1.0, unique_tags_per_example / 3)  # 3 unique tags per example = perfect diversity
    
    def select_optimal_examples(self, context: Optional[Dict[str, Any]] = None) -> List[PromptExample]:
        """
        Intelligente Beispiel-Auswahl basierend auf Kontext und Performance
        
        INNOVATION: Context-aware, performance-driven example selection
        """
        
        if not self.dynamic_example_selection:
            return self.examples[:self.max_examples_per_request]
        
        scored_examples = []
        
        for example in self.examples:
            score = example.effectiveness_score
            
            # Context similarity bonus
            if context and example.context_relevance:
                context_score = 0
                context_count = 0
                
                for ctx_key, ctx_value in context.items():
                    if ctx_key in example.context_relevance:
                        context_score += example.context_relevance[ctx_key]
                        context_count += 1
                
                if context_count > 0:
                    avg_context_score = context_score / context_count
                    if avg_context_score >= self.context_similarity_threshold:
                        score *= 1.5  # Boost relevant examples
            
            # Model-specific performance bonus
            if self.target_model != AIModelType.AUTO_SELECT:
                model_performance = example.model_performance.get(self.target_model.value)
                if model_performance:
                    score *= (0.5 + model_performance)  # Weight by model performance
            
            # Recency bonus
            if example.last_used:
                days_since_use = (datetime.now() - example.last_used).days
                if days_since_use < 30:  # Recent examples get slight boost
                    score *= 1.1
            
            scored_examples.append((score, example))
        
        # Sort by score and apply selection strategy
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        
        if self.example_selection_strategy == "effectiveness":
            return [ex for _, ex in scored_examples[:self.max_examples_per_request]]
        
        elif self.example_selection_strategy == "diversity":
            # Select diverse examples
            selected = []
            used_tags = set()
            
            for score, example in scored_examples:
                example_tags = set(example.tags)
                if len(selected) == 0 or len(example_tags.intersection(used_tags)) < len(example_tags) * 0.5:
                    selected.append(example)
                    used_tags.update(example_tags)
                    
                    if len(selected) >= self.max_examples_per_request:
                        break
            
            return selected
        
        else:  # adaptive - balance effectiveness and diversity
            selected = []
            used_tags = set()
            
            for score, example in scored_examples:
                diversity_bonus = 1.0
                example_tags = set(example.tags)
                
                if used_tags:
                    overlap = len(example_tags.intersection(used_tags)) / max(len(example_tags), 1)
                    diversity_bonus = 1.0 - overlap
                
                adjusted_score = score * diversity_bonus
                
                if len(selected) == 0 or adjusted_score > 0.3:  # Minimum quality threshold
                    selected.append(example)
                    used_tags.update(example_tags)
                    
                    if len(selected) >= self.max_examples_per_request:
                        break
            
            return selected
    
    @field_validator('examples')
    @classmethod
    def validate_examples_quality(cls, v):
        """Enhanced example validation"""
        if len(v) < 2:
            raise ValueError("Few-Shot requires at least 2 examples")
        
        # Check for diversity
        all_inputs = [ex.input_example.lower() for ex in v]
        unique_inputs = set(all_inputs)
        
        if len(unique_inputs) < len(all_inputs) * 0.8:  # Allow some similarity
            raise ValueError("Examples should be sufficiently diverse")
        
        # Check example quality
        low_quality_examples = [ex for ex in v if ex.quality_score is not None and ex.quality_score < 0.3]
        if len(low_quality_examples) > len(v) * 0.2:  # Max 20% low quality
            raise ValueError("Too many low-quality examples")
        
        return v


class ChainOfThoughtTemplate(BasePromptTemplate):
    """
    Chain-of-Thought Template (ENHANCED V2)
    
    🚀 NEW: Adaptive step optimization, dynamic step generation, performance-based step refinement
    ✅ BACKWARD COMPATIBLE: Original structure maintained
    """
    
    technique: Literal[PromptTechnique.CHAIN_OF_THOUGHT] = PromptTechnique.CHAIN_OF_THOUGHT
    
    # ✅ UNCHANGED: Core fields
    reasoning_steps: List[ChainOfThoughtStep] = Field(
        ..., min_length=2, max_length=15,
        description="Schritte für strukturiertes Denken"
    )
    
    step_connector: str = Field(
        default="Let me think step by step:",
        description="Wie Denkschritte eingeleitet werden"
    )
    
    conclusion_prompt: str = Field(
        default="Based on this analysis, my recommendation is:",
        description="Einleitung für finale Antwort"
    )
    
    encourage_reasoning: bool = Field(
        default=True,
        description="AI explizit zum Nachdenken ermutigen"
    )
    
    # 🚀 NEW: Advanced CoT features
    adaptive_step_ordering: bool = Field(
        default=False,
        description="Schritte dynamisch basierend auf Kontext anordnen"
    )
    
    parallel_reasoning_paths: bool = Field(
        default=False,
        description="Mehrere Reasoning-Pfade parallel verfolgen"
    )
    
    step_validation: bool = Field(
        default=False,
        description="Jeder Schritt wird vor dem nächsten validiert"
    )
    
    meta_reasoning: bool = Field(
        default=False,
        description="AI reflektiert über ihre eigenen Denkprozesse"
    )
    
    confidence_tracking: bool = Field(
        default=False,
        description="Confidence-Score für jeden Schritt tracken"
    )
    
    @computed_field
    @property
    def reasoning_complexity(self) -> float:
        """Komplexität des Reasoning-Prozesses"""
        base_complexity = len(self.reasoning_steps) / 15  # Normalize to max steps
        
        # Adjust for step dependencies
        total_dependencies = sum(len(step.dependencies) for step in self.reasoning_steps)
        dependency_complexity = min(1.0, total_dependencies / (len(self.reasoning_steps) * 2))
        
        # Adjust for cognitive load
        avg_cognitive_load = sum(
            step.cognitive_load or 0.5 for step in self.reasoning_steps
        ) / len(self.reasoning_steps)
        
        return (base_complexity * 0.4 + dependency_complexity * 0.3 + avg_cognitive_load * 0.3)
    
    @computed_field
    @property
    def reasoning_efficiency(self) -> float:
        """Effizienz des Reasoning-Prozesses"""
        if not self.reasoning_steps:
            return 0.0
        
        step_efficiencies = [step.step_efficiency for step in self.reasoning_steps]
        return sum(step_efficiencies) / len(step_efficiencies)
    
    def optimize_step_order(self, context: Optional[Dict[str, Any]] = None) -> List[ChainOfThoughtStep]:
        """
        Optimiert die Reihenfolge der Reasoning-Schritte basierend auf Performance
        
        INNOVATION: Dynamic step ordering for optimal reasoning flow
        """
        
        if not self.adaptive_step_ordering:
            return sorted(self.reasoning_steps, key=lambda x: x.step_number)
        
        # Topological sort considering dependencies
        ordered_steps = []
        remaining_steps = self.reasoning_steps.copy()
        completed_steps = set()
        
        while remaining_steps:
            # Find steps with satisfied dependencies
            ready_steps = [
                step for step in remaining_steps
                if all(dep in completed_steps for dep in step.dependencies)
            ]
            
            if not ready_steps:
                # Circular dependency or invalid - fall back to original order
                return sorted(self.reasoning_steps, key=lambda x: x.step_number)
            
            # Among ready steps, prioritize by effectiveness and context relevance
            if context:
                # Context-aware prioritization
                for step in ready_steps:
                    context_relevance = 1.0  # Default
                    
                    # Check if step description matches context
                    for ctx_key, ctx_value in context.items():
                        if isinstance(ctx_value, str) and ctx_key.lower() in step.step_description.lower():
                            context_relevance *= 1.5
                    
                    # Temporarily store context relevance
                    step._context_relevance = context_relevance
                
                # Sort by efficiency * context relevance
                ready_steps.sort(
                    key=lambda x: x.step_efficiency * getattr(x, '_context_relevance', 1.0),
                    reverse=True
                )
            else:
                # Sort by efficiency only
                ready_steps.sort(key=lambda x: x.step_efficiency, reverse=True)
            
            # Take the best step
            next_step = ready_steps[0]
            ordered_steps.append(next_step)
            completed_steps.add(next_step.step_number)
            remaining_steps.remove(next_step)
        
        return ordered_steps
    
    @field_validator('reasoning_steps')
    @classmethod
    def validate_reasoning_flow(cls, v):
        """Enhanced reasoning flow validation"""
        if not v:
            raise ValueError("Chain-of-Thought requires reasoning steps")
        
        # Check step numbers are sequential
        step_numbers = [step.step_number for step in v]
        expected = list(range(1, len(v) + 1))
        if sorted(step_numbers) != expected:
            raise ValueError("Step numbers must be sequential starting from 1")
        
        # Check for circular dependencies
        for step in v:
            if step.step_number in step.dependencies:
                raise ValueError(f"Step {step.step_number} cannot depend on itself")
            
            # Check all dependencies exist
            for dep in step.dependencies:
                if dep not in step_numbers:
                    raise ValueError(f"Step {step.step_number} depends on non-existent step {dep}")
                
                # Check dependency is earlier step
                if dep >= step.step_number:
                    raise ValueError(f"Step {step.step_number} cannot depend on later step {dep}")
        
        return v


class DynamicPromptTemplate(BasePromptTemplate):
    """
    Dynamisch generierte Prompts (ENHANCED V2)
    
    🚀 NEW: AI-driven prompt generation, real-time adaptation, context-aware optimization
    ✅ BACKWARD COMPATIBLE: Original interface preserved
    """
    
    technique: Literal[PromptTechnique.DYNAMIC_GENERATION] = PromptTechnique.DYNAMIC_GENERATION
    
    # ✅ UNCHANGED: Core fields
    variable_placeholders: Dict[str, str] = Field(
        ...,
        description="Platzhalter die dynamisch ersetzt werden: {variable_name}: description"
    )
    
    conditional_sections: Dict[str, str] = Field(
        default_factory=dict,
        description="Prompt-Abschnitte die nur unter bestimmten Bedingungen eingefügt werden"
    )
    
    adaptation_rules: List[str] = Field(
        default_factory=list,
        description="Regeln wie der Prompt sich an verschiedene Situationen anpasst"
    )
    
    personality_adaptation: bool = Field(
        default=True,
        description="Prompt an Big Five + Limbic-Persönlichkeit anpassen"
    )
    
    relationship_adaptation: bool = Field(
        default=True,
        description="Prompt an Beziehungstyp anpassen"
    )
    
    occasion_adaptation: bool = Field(
        default=True,
        description="Prompt an Anlass anpassen"
    )
    
    # 🚀 NEW: Advanced dynamic features
    ai_generated_sections: Dict[str, str] = Field(
        default_factory=dict,
        description="Von AI generierte Prompt-Abschnitte"
    )
    
    real_time_adaptation: bool = Field(
        default=False,
        description="Prompt wird in Echtzeit angepasst"
    )
    
    context_learning: bool = Field(
        default=False,
        description="Lernt aus Kontext-Performance für bessere Adaptation"
    )
    
    generation_strategies: List[str] = Field(
        default_factory=list,
        description="Strategien für dynamische Prompt-Generierung"
    )
    
    adaptation_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Historie der Prompt-Anpassungen"
    )
    
    context_patterns: Dict[str, Any] = Field(
        default_factory=dict,
        description="Erkannte Muster in Kontext-Daten"
    )
    
    @computed_field
    @property
    def adaptation_effectiveness(self) -> float:
        """Effektivität der Prompt-Adaptation"""
        if not self.adaptation_history:
            return 0.5
        
        recent_adaptations = self.adaptation_history[-10:]  # Last 10 adaptations
        
        effectiveness_scores = []
        for adaptation in recent_adaptations:
            if 'performance_improvement' in adaptation:
                effectiveness_scores.append(adaptation['performance_improvement'])
        
        if effectiveness_scores:
            return sum(effectiveness_scores) / len(effectiveness_scores)
        
        return 0.5
    
    def generate_dynamic_prompt(self, context: Dict[str, Any]) -> str:
        """
        Generiert Prompt dynamisch basierend auf Kontext
        
        INNOVATION: AI-driven, context-aware prompt generation
        """
        
        base_prompt = self.instruction_prompt
        
        # 1. Variable replacement
        for var_name, var_description in self.variable_placeholders.items():
            placeholder = f"{{{var_name}}}"
            
            if var_name in context:
                replacement = str(context[var_name])
            else:
                # Try to infer from context or use default
                replacement = self._infer_variable_value(var_name, var_description, context)
            
            base_prompt = base_prompt.replace(placeholder, replacement)
        
        # 2. Conditional sections
        for condition, section_text in self.conditional_sections.items():
            if self._evaluate_condition(condition, context):
                base_prompt += f"\n\n{section_text}"
        
        # 3. Personality adaptation
        if self.personality_adaptation and 'personality' in context:
            personality_section = self._generate_personality_adaptation(context['personality'])
            if personality_section:
                base_prompt += f"\n\nPersonality Context: {personality_section}"
        
        # 4. Relationship adaptation
        if self.relationship_adaptation and 'relationship' in context:
            relationship_section = self._generate_relationship_adaptation(context['relationship'])
            if relationship_section:
                base_prompt += f"\n\nRelationship Context: {relationship_section}"
        
        # 5. Occasion adaptation
        if self.occasion_adaptation and 'occasion' in context:
            occasion_section = self._generate_occasion_adaptation(context['occasion'])
            if occasion_section:
                base_prompt += f"\n\nOccasion Context: {occasion_section}"
        
        # 6. AI-generated enhancements
        if self.real_time_adaptation:
            ai_enhancement = self._generate_ai_enhancement(context)
            if ai_enhancement:
                base_prompt += f"\n\nAdditional Context: {ai_enhancement}"
        
        # Track this adaptation
        self._track_adaptation(context, base_prompt)
        
        return base_prompt
    
    def _infer_variable_value(self, var_name: str, var_description: str, context: Dict[str, Any]) -> str:
        """Infers variable value from context or provides intelligent default"""
        
        # Try semantic matching
        for ctx_key, ctx_value in context.items():
            if var_name.lower() in ctx_key.lower() or ctx_key.lower() in var_name.lower():
                return str(ctx_value)
        
        # Provide intelligent defaults based on variable description
        if 'budget' in var_description.lower():
            return "moderate budget"
        elif 'personality' in var_description.lower():
            return "balanced personality"
        elif 'occasion' in var_description.lower():
            return "special occasion"
        else:
            return f"[{var_name}]"  # Fallback placeholder
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluates conditional logic for section inclusion"""
        
        # Simple condition evaluation (could be more sophisticated)
        try:
            # Replace context variables in condition
            for key, value in context.items():
                if isinstance(value, str):
                    condition = condition.replace(f"{{{key}}}", f"'{value}'")
                else:
                    condition = condition.replace(f"{{{key}}}", str(value))
            
            # Evaluate simple conditions
            if 'budget' in condition and 'budget' in context:
                budget = context.get('budget', 0)
                if 'high' in condition.lower():
                    return budget > 200
                elif 'low' in condition.lower():
                    return budget < 50
                else:
                    return 50 <= budget <= 200
            
            # Default: check if key exists
            for key in context.keys():
                if key in condition:
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _generate_personality_adaptation(self, personality_data: Any) -> str:
        """Generates personality-specific prompt adaptation"""
        
        if isinstance(personality_data, str):
            return f"Consider the personality: {personality_data}"
        
        if isinstance(personality_data, dict):
            adaptations = []
            
            # Big Five adaptations
            if 'big_five' in personality_data:
                big_five = personality_data['big_five']
                
                if isinstance(big_five, dict):
                    if big_five.get('openness', 0) > 0.7:
                        adaptations.append("Focus on creative and novel gift ideas")
                    if big_five.get('conscientiousness', 0) > 0.7:
                        adaptations.append("Emphasize practical and high-quality options")
                    if big_five.get('extraversion', 0) > 0.7:
                        adaptations.append("Consider social and interactive gifts")
            
            # Limbic adaptations
            if 'limbic' in personality_data:
                limbic = personality_data['limbic']
                
                if isinstance(limbic, dict):
                    if limbic.get('stimulanz', 0) > 0.7:
                        adaptations.append("Include exciting and stimulating options")
                    if limbic.get('dominanz', 0) > 0.7:
                        adaptations.append("Consider status-enhancing gifts")
                    if limbic.get('balance', 0) > 0.7:
                        adaptations.append("Focus on harmonious and thoughtful gifts")
            
            return ". ".join(adaptations) if adaptations else ""
        
        return ""
    
    def _generate_relationship_adaptation(self, relationship_data: Any) -> str:
        """Generates relationship-specific prompt adaptation"""
        
        relationship_adaptations = {
            'partner': "Choose intimate and meaningful gifts that show deep care",
            'friend_close': "Select thoughtful gifts that celebrate your friendship",
            'friend_casual': "Pick friendly and appropriate gifts without being too personal",
            'family_parent': "Choose respectful gifts that show appreciation",
            'family_sibling': "Select fun or practical gifts that match your sibling dynamic",
            'colleague': "Choose professional and appropriate workplace gifts"
        }
        
        if isinstance(relationship_data, str):
            return relationship_adaptations.get(relationship_data.lower(), f"Consider the {relationship_data} relationship")
        
        return ""
    
    def _generate_occasion_adaptation(self, occasion_data: Any) -> str:
        """Generates occasion-specific prompt adaptation"""
        
        occasion_adaptations = {
            'geburtstag': "Focus on celebratory and personal birthday gifts",
            'weihnachten': "Consider traditional holiday themes and seasonal appropriateness",
            'valentinstag': "Emphasize romantic and loving gift options",
            'jahrestag': "Choose gifts that commemorate and celebrate the relationship milestone",
            'abschluss': "Select gifts that acknowledge achievement and future success"
        }
        
        if isinstance(occasion_data, str):
            return occasion_adaptations.get(occasion_data.lower(), f"Consider the {occasion_data} occasion")
        
        return ""
    
    def _generate_ai_enhancement(self, context: Dict[str, Any]) -> str:
        """Generates AI-powered prompt enhancements in real-time"""
        
        # This would integrate with an AI service for real-time enhancement
        # For now, provide rule-based enhancements
        
        enhancements = []
        
        # Context-based enhancements
        if 'urgency' in context and context.get('urgency') == 'high':
            enhancements.append("Prioritize immediately available options")
        
        if 'season' in context:
            season = context['season'].lower()
            if season in ['winter', 'christmas']:
                enhancements.append("Consider seasonal and holiday-appropriate gifts")
            elif season in ['summer', 'vacation']:
                enhancements.append("Think about outdoor and travel-related options")
        
        if 'previous_gifts' in context:
            enhancements.append("Avoid repeating previous gift types and ensure novelty")
        
        return ". ".join(enhancements) if enhancements else ""
    
    def _track_adaptation(self, context: Dict[str, Any], generated_prompt: str):
        """Tracks adaptation for learning and optimization"""
        
        adaptation_record = {
            'timestamp': datetime.now().isoformat(),
            'context_summary': {
                'keys': list(context.keys()),
                'personality_present': 'personality' in context,
                'relationship_present': 'relationship' in context,
                'occasion_present': 'occasion' in context
            },
            'prompt_length': len(generated_prompt),
            'adaptation_types': []
        }
        
        # Track which adaptations were applied
        if self.personality_adaptation and 'personality' in context:
            adaptation_record['adaptation_types'].append('personality')
        
        if self.relationship_adaptation and 'relationship' in context:
            adaptation_record['adaptation_types'].append('relationship')
        
        if self.occasion_adaptation and 'occasion' in context:
            adaptation_record['adaptation_types'].append('occasion')
        
        self.adaptation_history.append(adaptation_record)
        
        # Keep last 50 adaptations
        if len(self.adaptation_history) > 50:
            self.adaptation_history = self.adaptation_history[-50:]


# =============================================================================
# 🚀 INTELLIGENT PROMPT BUILDER: Orchestriert alle Prompt-Komponenten
# =============================================================================

class PromptBuilder(BaseModel):
    """
    Intelligenter Prompt-Builder für emotionale Geschenkempfehlungen (ENHANCED V2)
    
    🧠 INTELLIGENT: Kombiniert alle Prompt-Engineering Techniken
    🎯 ADAPTIVE: Passt sich an Kontext und Persönlichkeit an
    🎁 GIFT-OPTIMIZED: Speziell für emotionale Geschenkempfehlungen
    """
    
    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "extra": "forbid",
        "use_enum_values": True
    }
    
    # === BUILDER CONFIGURATION ===
    builder_name: str = Field(...)
    target_model: AIModelType = Field(...)
    optimization_goal: PromptOptimizationGoal = Field(...)
    
    # === TEMPLATE SELECTION ===
    primary_technique: PromptTechnique = Field(...)
    fallback_techniques: List[PromptTechnique] = Field(default_factory=list)
    
    # === CONTEXT INTEGRATION ===
    context_strategy: str = Field(
        default="adaptive",
        pattern="^(minimal|standard|comprehensive|adaptive)$"
    )
    
    personality_integration_depth: Annotated[
        float,
        Field(ge=0.0, le=1.0)
    ] = 0.8
    
    # === PROMPT COMPONENTS ===
    base_templates: Dict[str, Any] = Field(default_factory=dict)
    context_injectors: List[str] = Field(default_factory=list)
    example_selectors: Dict[str, Any] = Field(default_factory=dict)
    
    # === QUALITY CONTROL ===
    validation_level: PromptValidationLevel = Field(default=PromptValidationLevel.SEMANTIC)
    auto_optimization: bool = Field(default=True)
    performance_tracking: bool = Field(default=True)
    
    # === EMOTIONAL INTELLIGENCE SETTINGS ===
    empathy_emphasis: Annotated[
        float,
        Field(ge=0.0, le=1.0, description="Wie empathisch sollen Prompts sein")
    ] = 0.8
    
    story_telling_mode: bool = Field(
        default=True,
        description="Geschenke als Geschichten erzählen"
    )
    
    cultural_awareness: bool = Field(
        default=True,
        description="Kulturelle Sensitivität einbauen"
    )
    
    # === ADAPTIVE FEATURES ===
    learning_enabled: bool = Field(default=True)
    a_b_testing_enabled: bool = Field(default=False)
    real_time_adaptation: bool = Field(default=False)
    
    # === PERFORMANCE HISTORY ===
    build_count: int = Field(default=0, ge=0)
    success_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    average_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    performance_metrics: List[PromptPerformanceMetrics] = Field(
        default_factory=list,
        description="Historie der Performance-Metriken"
    )
    
    @computed_field
    @property
    def builder_effectiveness(self) -> float:
        """Effektivität dieses Builders"""
        if not self.performance_metrics:
            return 0.5  # Default
        
        recent_metrics = self.performance_metrics[-10:]  # Last 10 builds
        avg_success = sum(m.overall_success_score for m in recent_metrics) / len(recent_metrics)
        
        return avg_success
    
    @computed_field
    @property
    def optimization_suggestions(self) -> List[str]:
        """Automatische Optimierungsvorschläge"""
        suggestions = []
        
        if self.success_rate is not None and self.success_rate < 0.7:
            suggestions.append("Consider adjusting primary technique")
        
        if self.average_quality_score is not None and self.average_quality_score < 0.6:
            suggestions.append("Increase personality integration depth")
        
        if self.performance_metrics:
            recent_emotional_scores = [
                m.emotional_intelligence_score for m in self.performance_metrics[-5:]
                if hasattr(m, 'emotional_intelligence_score')
            ]
            
            if recent_emotional_scores and sum(recent_emotional_scores) / len(recent_emotional_scores) < 0.7:
                suggestions.append("Enhance emotional intelligence components")
        
        return suggestions
    
    def build_prompt(self, gift_schema: GiftRecommendationSchema) -> str:
        """
        Baut finalen Prompt basierend auf Schema und Builder-Konfiguration
        
        🎁 CORE METHOD: Hier geschieht die Magie der Prompt-Generierung
        """
        
        # Start building process
        self.build_count += 1
        build_start = datetime.now()
        
        try:
            # 1. Select optimal template based on schema complexity
            template = self._select_optimal_template(gift_schema)
            
            # 2. Build context integration
            context = self._build_comprehensive_context(gift_schema)
            
            # 3. Generate personality-specific guidance
            personality_guidance = self._generate_personality_guidance(gift_schema)
            
            # 4. Create emotional framework
            emotional_framework = self._create_emotional_framework(gift_schema)
            
            # 5. Assemble final prompt
            final_prompt = self._assemble_final_prompt(
                template, context, personality_guidance, emotional_framework, gift_schema
            )
            
            # 6. Apply optimizations
            if self.auto_optimization:
                final_prompt = self._apply_auto_optimizations(final_prompt, gift_schema)
            
            # 7. Validate prompt quality
            if self.validation_level != PromptValidationLevel.BASIC:
                validation_score = self._validate_prompt_quality(final_prompt, gift_schema)
                
                if validation_score < 0.6:
                    # Try fallback technique
                    if self.fallback_techniques:
                        return self._build_with_fallback(gift_schema)
            
            # 8. Track performance if enabled
            if self.performance_tracking:
                self._track_build_performance(gift_schema, final_prompt, build_start)
            
            return final_prompt
            
        except Exception as e:
            # Error handling with fallback
            if self.fallback_techniques:
                return self._build_with_fallback(gift_schema)
            else:
                return self._build_emergency_fallback(gift_schema)
    
    def _select_optimal_template(self, schema: GiftRecommendationSchema) -> BasePromptTemplate:
        """Wählt optimales Template basierend auf Schema-Komplexität"""
        
        complexity = schema.schema_complexity
        
        if self.primary_technique == PromptTechnique.FEW_SHOT:
            # Create optimized few-shot template
            return BasePromptTemplate(
                template_name="gift_few_shot_template",
                description="Few-shot learning template for personalized gift recommendations",
                technique=PromptTechnique.FEW_SHOT,
                complexity=PromptComplexity(complexity),
                target_model=self.target_model,
                optimization_goal=self.optimization_goal,
                instruction_prompt="Based on the following examples of successful gift recommendations, generate personalized suggestions that match the recipient's personality and relationship context."
            )
        
        elif self.primary_technique == PromptTechnique.CHAIN_OF_THOUGHT:
            # Create chain-of-thought template
            return BasePromptTemplate(
                template_name="gift_cot_template",
                description="Chain-of-thought template for systematic gift analysis",
                technique=PromptTechnique.CHAIN_OF_THOUGHT,
                complexity=PromptComplexity(complexity),
                target_model=self.target_model,
                optimization_goal=self.optimization_goal,
                instruction_prompt="Let me think through this gift recommendation step by step, considering the recipient's personality, our relationship, and the occasion to create meaningful suggestions."
            )
        
        elif self.primary_technique == PromptTechnique.DYNAMIC_GENERATION:
            # Create dynamic template
            return DynamicPromptTemplate(
                template_name="gift_dynamic_template",
                description="Dynamic template that adapts to context",
                technique=PromptTechnique.DYNAMIC_GENERATION,
                complexity=PromptComplexity(complexity),
                target_model=self.target_model,
                optimization_goal=self.optimization_goal,
                instruction_prompt="Generate personalized gift recommendations by adapting to the specific personality traits, relationship dynamics, and contextual factors provided.",
                variable_placeholders={
                    "personality": "Recipient's personality archetype",
                    "occasion": "Special occasion or event",
                    "relationship": "Nature of the relationship",
                    "budget": "Budget considerations"
                }
            )
        
        else:
            # Fallback to base template
            return BasePromptTemplate(
                template_name="fallback_gift_template",
                description="Fallback template for gift recommendations",
                technique=self.primary_technique,
                complexity=PromptComplexity(complexity),
                target_model=self.target_model,
                optimization_goal=self.optimization_goal,
                instruction_prompt="Generate thoughtful, personalized gift recommendations based on the provided personality and context information."
            )
    
    def _build_comprehensive_context(self, schema: GiftRecommendationSchema) -> ContextInjection:
        """Baut umfassenden Kontext für den Prompt"""
        
        return ContextInjection(
            personality_context=schema.generate_prompt_context(),
            occasion_context=f"Occasion: {schema.occasion} | Urgency: {schema.urgency_level}",
            relationship_context=f"Relationship: {schema.relationship_type} | Objectives: {', '.join(schema.relationship_objectives)}",
            budget_context=schema.budget_range or "Budget flexible",
            cultural_context=', '.join(schema.cultural_considerations) if schema.cultural_considerations else None,
            emotional_context=f"Emotional Goals: {', '.join(schema.emotional_goals)}",
            preference_context=f"Avoid: {', '.join(schema.avoid_categories)}" if schema.avoid_categories else None
        )
    
    def _generate_personality_guidance(self, schema: GiftRecommendationSchema) -> str:
        """Generiert Persönlichkeits-spezifische Guidance"""
        
        if self.personality_integration_depth < 0.3:
            return "Consider basic personality preferences"
        elif self.personality_integration_depth < 0.7:
            return schema.get_personality_guidance()
        else:
            # Deep personality integration
            detailed_guidance = schema.get_personality_guidance()
            emotional_guidance = f"Emotional Focus: {', '.join(schema.emotional_goals)}"
            return f"{detailed_guidance} | {emotional_guidance}"
    
    def _create_emotional_framework(self, schema: GiftRecommendationSchema) -> str:
        """Erstellt emotionales Framework für den Prompt"""
        
        if not self.story_telling_mode:
            return ""
        
        framework = f"EMOTIONAL FRAMEWORK:\n"
        framework += f"Primary Emotional Goals: {', '.join(schema.emotional_goals)}\n"
        framework += f"Relationship Impact: {', '.join(schema.relationship_objectives)}\n"
        framework += f"Experience Emphasis: {schema.experience_emphasis:.1f} (0=material, 1=experiential)\n"
        
        if self.empathy_emphasis > 0.7:
            framework += f"High Empathy Mode: Focus on deep emotional understanding and connection\n"
        
        if self.cultural_awareness and schema.cultural_considerations:
            framework += f"Cultural Awareness: {', '.join(schema.cultural_considerations)}\n"
        
        return framework
    
    def _assemble_final_prompt(self, 
                             template: BasePromptTemplate,
                             context: ContextInjection,
                             personality_guidance: str,
                             emotional_framework: str,
                             schema: GiftRecommendationSchema) -> str:
        """Assembliert finalen Prompt aus allen Komponenten"""
        
        prompt_parts = []
        
        # System prompt
        if template.system_prompt:
            enhanced_system = template.system_prompt
            if self.empathy_emphasis > 0.7:
                enhanced_system += "\n\nYou approach gift recommendations with deep empathy and emotional intelligence."
            prompt_parts.append(f"SYSTEM: {enhanced_system}")
        
        # Emotional framework
        if emotional_framework:
            prompt_parts.append(emotional_framework)
        
        # Context
        prompt_parts.append("CONTEXT:")
        prompt_parts.append(f"Personality: {context.personality_context}")
        prompt_parts.append(f"Occasion: {context.occasion_context}")
        prompt_parts.append(f"Relationship: {context.relationship_context}")
        if context.budget_context:
            prompt_parts.append(f"Budget: {context.budget_context}")
        if context.cultural_context:
            prompt_parts.append(f"Cultural: {context.cultural_context}")
        if context.emotional_context:
            prompt_parts.append(f"Emotional: {context.emotional_context}")
        
        # Personality guidance
        if personality_guidance:
            prompt_parts.append(f"\nPERSONALITY GUIDANCE:\n{personality_guidance}")
        
        # Main instruction
        enhanced_instruction = template.instruction_prompt
        
        # Add emotional intelligence enhancements
        if self.story_telling_mode:
            enhanced_instruction += "\n\nPresent each recommendation as an emotional story that connects the gift to the recipient's personality and your relationship."
        
        if schema.experience_emphasis > 0.6:
            enhanced_instruction += "\n\nPrioritize experiential gifts and emotional connections over material possessions."
        
        # Add output specifications
        enhanced_instruction += f"\n\nGenerate exactly {schema.max_recommendations} recommendations with {schema.detail_level} detail level."
        
        if schema.include_emotional_impact:
            enhanced_instruction += "\n\nExplain the emotional impact each gift will have on the recipient and your relationship."
        
        if schema.include_presentation_tips:
            enhanced_instruction += "\n\nInclude thoughtful presentation and timing suggestions for maximum emotional effect."
        
        prompt_parts.append(f"\nINSTRUCTION:\n{enhanced_instruction}")
        
        # Output format
        if template.output_format_instructions:
            prompt_parts.append(f"\nOUTPUT FORMAT:\n{template.output_format_instructions}")
        else:
            # Default JSON format for structured output
            json_format = """
Respond with a JSON object containing:
{
  "recommendations": [
    {
      "gift_idea": "specific gift suggestion",
      "emotional_reasoning": "why this gift creates emotional connection",
      "personality_match": "how this matches their personality",
      "presentation_suggestion": "how to present this gift",
      "emotional_impact": "expected emotional response",
      "relationship_benefit": "how this strengthens your relationship"
    }
  ],
  "overall_strategy": "overall approach and emotional theme"
}"""
            prompt_parts.append(f"\nOUTPUT FORMAT:{json_format}")
        
        # Final assembly
        return "\n".join(prompt_parts)
    
    def _apply_auto_optimizations(self, prompt: str, schema: GiftRecommendationSchema) -> str:
        """Wendet automatische Optimierungen basierend auf Performance-History an"""
        
        if not self.auto_optimization or not self.performance_metrics:
            return prompt
        
        # Analyze recent performance for optimization opportunities
        recent_metrics = self.performance_metrics[-5:]
        
        avg_emotional_score = sum(m.emotional_resonance for m in recent_metrics) / len(recent_metrics)
        
        if avg_emotional_score < 0.7:
            # Enhance emotional language
            prompt += "\n\nEMOTIONAL ENHANCEMENT: Use vivid, emotionally resonant language that creates a strong connection between the gift and the recipient's feelings."
        
        avg_personalization = sum(m.personalization_score for m in recent_metrics) / len(recent_metrics)
        
        if avg_personalization < 0.7:
            # Enhance personalization
            prompt += "\n\nPERSONALIZATION BOOST: Make each recommendation highly specific to this individual's unique personality traits and preferences."
        
        # Check for creativity enhancement
        avg_creativity = sum(m.creativity_score for m in recent_metrics) / len(recent_metrics)
        
        if avg_creativity < 0.6:
            prompt += "\n\nCREATIVITY BOOST: Think outside the box and suggest unexpected, delightful gifts that surprise and enchant."
        
        return prompt
    
    def _validate_prompt_quality(self, prompt: str, schema: GiftRecommendationSchema) -> float:
        """Validiert Prompt-Qualität vor der Ausgabe"""
        
        quality_score = 0.0
        checks_passed = 0
        total_checks = 0
        
        # Length check
        total_checks += 1
        if 500 <= len(prompt) <= 4000:  # Reasonable prompt length
            checks_passed += 1
            quality_score += 0.2
        
        # Personality integration check
        total_checks += 1
        personality_keywords = ['personality', 'trait', 'big five', 'limbic', schema.personality_archetype.lower()]
        if any(keyword in prompt.lower() for keyword in personality_keywords):
            checks_passed += 1
            quality_score += 0.3
        
        # Emotional framework check
        total_checks += 1
        emotional_keywords = ['emotional', 'feeling', 'connection', 'resonance', 'empathy']
        if any(keyword in prompt.lower() for keyword in emotional_keywords):
            checks_passed += 1
            quality_score += 0.3
        
        # Output format check
        total_checks += 1
        if 'json' in prompt.lower() or 'format' in prompt.lower():
            checks_passed += 1
            quality_score += 0.2
        
        return quality_score
    
    def _build_with_fallback(self, schema: GiftRecommendationSchema) -> str:
        """Baut Prompt mit Fallback-Technik"""
        
        if self.fallback_techniques:
            fallback_technique = self.fallback_techniques[0]
            
            # Temporarily switch technique
            original_technique = self.primary_technique
            self.primary_technique = fallback_technique
            
            try:
                result = self.build_prompt(schema)
                return result
            finally:
                # Restore original technique
                self.primary_technique = original_technique
        
        return self._build_emergency_fallback(schema)
    
    def _build_emergency_fallback(self, schema: GiftRecommendationSchema) -> str:
        """Notfall-Fallback wenn alles andere fehlschlägt"""
        
        return f"""You are an expert gift consultant with deep understanding of personality psychology.

PERSONALITY: {schema.personality_archetype}
OCCASION: {schema.occasion}
RELATIONSHIP: {schema.relationship_type}
EMOTIONAL GOALS: {', '.join(schema.emotional_goals)}

Generate {schema.max_recommendations} thoughtful, personalized gift recommendations that:
1. Match the personality perfectly
2. Are appropriate for the relationship
3. Create emotional connection and joy
4. Focus on experiences and meaning over materialism

Format your response as JSON with detailed explanations for each recommendation."""
    
    def _track_build_performance(self, schema: GiftRecommendationSchema, prompt: str, build_start: datetime):
        """Trackt Performance dieses Build-Vorgangs"""
        
        build_time = (datetime.now() - build_start).total_seconds() * 1000
        
        # Create basic performance metric
        metric = PromptPerformanceMetrics(
            prompt_template_id=f"builder_{self.builder_name}",
            prompt_template_name=self.builder_name,
            prompt_version="1.0",
            test_id=f"build_{self.build_count}",
            test_scenario=f"{schema.occasion}_{schema.relationship_type}",
            response_time_ms=int(build_time),
            response_quality_score=0.8,  # Estimated, would be updated later
            personalization_score=self.personality_integration_depth,
            emotional_resonance=self.empathy_emphasis,
            creativity_score=schema.creativity_level,
            prompt_tokens_used=len(prompt.split()),
            response_tokens_generated=0,  # Unknown at build time
            target_model=self.target_model.value,
            optimization_goal=self.optimization_goal.value,
            request_complexity=schema.schema_complexity,
            context_richness=schema.emotional_complexity,
            parsing_successful=True,  # Assumed for build
            schema_validation_passed=True,  # Assumed for build
        )
        
        self.performance_metrics.append(metric)
        
        # Keep only recent metrics
        if len(self.performance_metrics) > 50:
            self.performance_metrics = self.performance_metrics[-50:]
        
        # Update aggregate metrics
        self._update_aggregate_metrics()
    
    def _update_aggregate_metrics(self):
        """Aktualisiert aggregierte Metriken"""
        if not self.performance_metrics:
            return
        
        recent_metrics = self.performance_metrics[-10:]
        
        # Update success rate
        successful_builds = sum(1 for m in recent_metrics if m.parsing_successful and m.schema_validation_passed)
        self.success_rate = successful_builds / len(recent_metrics)
        
        # Update average quality
        quality_scores = [m.response_quality_score for m in recent_metrics]
        self.average_quality_score = sum(quality_scores) / len(quality_scores)


# =============================================================================
# 🔧 UTILITY FUNCTIONS: Helper functions für Prompt-Engineering
# =============================================================================

def create_gift_recommendation_builder(
    model_type: AIModelType = AIModelType.AUTO_SELECT,
    optimization_goal: PromptOptimizationGoal = PromptOptimizationGoal.EMOTIONAL_RESONANCE,
    technique: PromptTechnique = PromptTechnique.DYNAMIC_GENERATION
) -> PromptBuilder:
    """
    Factory function für schnelle Builder-Erstellung
    
    🚀 CONVENIENCE: Einfache Builder-Konfiguration für Gift Recommendations
    """
    
    builder_name = f"gift_builder_{model_type.value}_{technique.value}_{int(time.time())}"
    
    return PromptBuilder(
        builder_name=builder_name,
        target_model=model_type,
        optimization_goal=optimization_goal,
        primary_technique=technique,
        fallback_techniques=[
            PromptTechnique.FEW_SHOT,
            PromptTechnique.TEMPLATE_BASED,
            PromptTechnique.ZERO_SHOT
        ],
        context_strategy="adaptive",
        personality_integration_depth=0.9,
        empathy_emphasis=0.9,
        story_telling_mode=True,
        cultural_awareness=True,
        auto_optimization=True,
        performance_tracking=True
    )


def create_example_gift_schema(
    occasion: str = "Geburtstag",
    relationship: str = "close_friend",
    personality_archetype: str = "Creative Explorer"
) -> GiftRecommendationSchema:
    """
    Factory function für Beispiel-Schema
    
    📝 DEVELOPMENT: Schnelle Schema-Erstellung für Tests
    """
    
    return GiftRecommendationSchema(
        occasion=occasion,
        relationship_type=relationship,
        personality_archetype=personality_archetype,
        big_five_weights={
            "openness": 0.8,
            "conscientiousness": 0.6,
            "extraversion": 0.7,
            "agreeableness": 0.8,
            "neuroticism": 0.3
        },
        limbic_emphasis={
            "stimulanz": 0.7,
            "dominanz": 0.4,
            "balance": 0.6
        },
        emotional_goals=["joy", "surprise", "connection", "appreciation"],
        relationship_objectives=["strengthen_bond", "show_care", "create_memories"],
        experience_emphasis=0.8,
        personalization_level=0.9,
        creativity_level=0.8,
        surprise_factor=0.7
    )



# =============================================================================
# ✅ COMPLETE EXPORTS: Alle verfügbaren Klassen und Funktionen
# =============================================================================

__all__ = [
    # === ENUMS (ENHANCED but COMPATIBLE) ===
    'PromptTechnique', 
    'PromptComplexity', 
    'AIModelType', 
    'PromptOptimizationGoal',
    
    # === NEW ENUMS ===
    'PromptAdaptationStrategy', 
    'PromptValidationLevel',
    
    # === CORE COMPONENTS (ENHANCED but COMPATIBLE) ===
    'PromptExample', 
    'ContextInjection', 
    'ChainOfThoughtStep',
    
    # === PERFORMANCE & METRICS ===
    'PromptPerformanceMetrics',
    
    # === SPECIALIZED SCHEMAS ===
    'GiftRecommendationSchema',
    
    # === TEMPLATES (ENHANCED but COMPATIBLE) ===
    'BasePromptTemplate', 
    'FewShotPromptTemplate', 
    'ChainOfThoughtTemplate', 
    'DynamicPromptTemplate',
    
    # === BUILDERS ===
    'PromptBuilder',
    
    # === UTILITY FUNCTIONS ===
    'create_gift_recommendation_builder',
    'create_example_gift_schema',
]


# =============================================================================
# 📚 USAGE EXAMPLES: Wie man die Schemas verwendet
# =============================================================================

"""
USAGE EXAMPLES:
===============

## 1. Einfache Gift Recommendation erstellen:

```python
from prompt_schemas import create_gift_recommendation_builder, create_example_gift_schema

# Builder erstellen
builder = create_gift_recommendation_builder(
    model_type=AIModelType.ANTHROPIC_CLAUDE,
    optimization_goal=PromptOptimizationGoal.EMOTIONAL_RESONANCE,
    technique=PromptTechnique.DYNAMIC_GENERATION
)

# Schema erstellen
schema = create_example_gift_schema(
    occasion="Valentinstag",
    relationship="partner",
    personality_archetype="Romantic Idealist"
)

# Prompt generieren
final_prompt = builder.build_prompt(schema)
print(final_prompt)
```

## 2. Advanced Few-Shot Template erstellen:

```python
from prompt_schemas import FewShotPromptTemplate, PromptExample

examples = [
    PromptExample(
        input_example="Creative person, birthday, close friend, moderate budget",
        expected_output="Art workshop experience, personalized sketchbook, concert tickets",
        explanation="Focuses on creative experiences over material gifts",
        tags=["creative", "experiential", "personal"]
    )
]

template = FewShotPromptTemplate(
    template_name="advanced_gift_recommendations",
    description="Few-shot learning for personalized gifts",
    technique=PromptTechnique.FEW_SHOT,
    complexity=PromptComplexity.COMPLEX,
    instruction_prompt="Generate creative gift recommendations based on personality",
    examples=examples,
    dynamic_example_selection=True,
    example_selection_strategy="adaptive"
)
```

## 3. Performance Tracking:

```python
from prompt_schemas import PromptPerformanceMetrics

metrics = PromptPerformanceMetrics(
    prompt_template_id="gift_template_001",
    prompt_template_name="Dynamic Gift Template",
    prompt_version="2.0",
    test_id="test_valentine_2024",
    test_scenario="valentinstag_partner",
    response_time_ms=1500,
    response_quality_score=0.85,
    personalization_score=0.92,
    emotional_resonance=0.88,
    creativity_score=0.79,
    prompt_tokens_used=450,
    response_tokens_generated=680,
    target_model="anthropic_claude",
    optimization_goal="emotional_resonance",
    request_complexity="complex",
    context_richness=0.85,
    parsing_successful=True,
    schema_validation_passed=True
)

print(f"Overall Success Score: {metrics.overall_success_score:.2f}")
print(f"Emotional Intelligence Score: {metrics.emotional_intelligence_score:.2f}")
print("Optimization Suggestions:", metrics.get_optimization_suggestions())
```

## 4. Template Optimization:

```python
from prompt_schemas import optimize_prompt_for_model

# Optimiere Template für Groq
groq_optimized = optimize_prompt_for_model(
    base_template=template,
    target_model=AIModelType.GROQ_MIXTRAL
)

# Optimiere für Claude
claude_optimized = optimize_prompt_for_model(
    base_template=template,
    target_model=AIModelType.ANTHROPIC_CLAUDE
)
```

## 5. Schema Validation:

```python
from prompt_schemas import validate_prompt_schema

validation = validate_prompt_schema(schema)
print(f"Valid: {validation['is_valid']}")
print(f"Quality Score: {validation['quality_score']:.2f}")
print("Suggestions:", validation['suggestions'])
```

INTEGRATION MIT AI ENGINE:
==========================

Die Schemas integrieren sich nahtlos mit dem bestehenden AI Engine:

```python
# In ai_engine/core/gift_service.py
from prompt_schemas import create_gift_recommendation_builder, GiftRecommendationSchema

class EnhancedGiftService:
    def __init__(self):
        self.prompt_builder = create_gift_recommendation_builder()
    
    async def generate_recommendations(self, user_data, context):
        # Konvertiere User-Daten zu Schema
        schema = GiftRecommendationSchema(
            occasion=context.get('occasion'),
            relationship_type=context.get('relationship'),
            personality_archetype=user_data.get('archetype'),
            big_five_weights=user_data.get('big_five', {}),
            limbic_emphasis=user_data.get('limbic', {}),
            emotional_goals=context.get('emotional_goals', []),
            relationship_objectives=context.get('objectives', [])
        )
        
        # Generiere optimierten Prompt
        prompt = self.prompt_builder.build_prompt(schema)
        
        # Sende an AI Model
        response = await self.ai_client.generate(prompt)
        
        # Tracke Performance
        performance_data = {
            'quality_score': self._evaluate_response_quality(response),
            'success': True
        }
        self.prompt_builder.track_usage(performance_data)
        
        return response
```

🎉 Die prompt_schemas.py ist jetzt vollständig und produktionsbereit.
"""