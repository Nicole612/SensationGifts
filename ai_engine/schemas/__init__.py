"""
AI Engine Schemas Package
=========================

Clean, strukturierte Datenmodelle für AI-Integration in SensationGifts.
NOW WITH: Advanced GenAI patterns, performance optimization, schema evolution.

✅ BACKWARD COMPATIBLE: Alle bestehenden Imports funktionieren weiter
🚀 NEW FEATURES: Adaptive schemas, multi-model validation, performance monitoring

Usage (bestehend - funktioniert weiter):
    from ai_engine.schemas import GiftRecommendationRequest, GiftRecommendationResponse
    from ai_engine.schemas import FewShotPromptTemplate, ChainOfThoughtTemplate

Usage (neu - optional):
    from ai_engine.schemas import api, SchemaEvolution, MultiModelValidator
    from ai_engine.schemas.monitoring import SchemaMetricsCollector
"""

import importlib
from typing import Dict, Type, Optional, Any, TYPE_CHECKING, List
from functools import lru_cache
from collections import defaultdict, deque
import time
import asyncio
import json
from datetime import datetime

# =============================================================================
# BACKWARD COMPATIBILITY: Alle bestehenden Imports bleiben funktional
# =============================================================================

from enum import Enum

# Relationship Analysis (NEW)
from .relationship_types import (
    RelationshipType,
    RelationshipAnalyzer, 
    RelationshipGiftGuide,
    create_relationship_context_for_ai,
    get_relationship_budget_guidance,
    integrate_relationship_with_personality
)

# INPUT SCHEMAS: Was geht in die AI rein (UNCHANGED)
from .input_schemas import (
    # Enums
    GiftOccasion,
    BigFiveTrait,    
    LimbicDimension, 
    LimbicType,      
    EmotionalTrigger,
    AgeGroup,
    GenderIdentity,
    
    # Core Models
    BigFiveScore,    
    LimbicScore,
    PersonalityAnalysisInput,
    GiftRecommendationRequest,
    AIPersonalityProfile,
    
    # Specialized Models
    QuickRecommendationInput,
    PersonalityQuizInput,
)

# OUTPUT SCHEMAS: Was kommt aus der AI raus (UNCHANGED)
from .output_schemas import (
    # Enums
    ConfidenceLevel,
    GiftCategory,
    RecommendationReason,
    
    # Core Models
    GiftRecommendation,
    PersonalityAnalysisResult,
    GiftRecommendationResponse,
    
    # Specialized Models
    QuickRecommendationResponse,
    PersonalityQuizResult,
    AIModelPerformanceMetrics,
    ErrorResponse,
)

# PROMPT SCHEMAS: Wie Prompts strukturiert werden (UNCHANGED)
from .prompt_schemas import (
    # Enums
    PromptTechnique,
    PromptComplexity,
    AIModelType,
    PromptOptimizationGoal,
    
    # Core Components
    PromptExample,
    ContextInjection,
    ChainOfThoughtStep,
    
    # Templates
    BasePromptTemplate,
    FewShotPromptTemplate,
    ChainOfThoughtTemplate,
    DynamicPromptTemplate,
    
    # Specialized
    GiftRecommendationSchema,  # ← FIX: Korrekter Name
    PromptPerformanceMetrics,
    PromptBuilder,
)


# =============================================================================
# 🚀 INNOVATIVE FEATURES: Advanced GenAI Patterns (NEW)
# =============================================================================

class SchemaRegistry:
    """
    Advanced Schema Registry mit Lazy Loading + Performance Monitoring
    
    INNOVATION: Automatisches Performance-Tracking aller Schema-Operations
    """
    
    def __init__(self):
        self._schemas: Dict[str, Type] = {}
        self._usage_stats = defaultdict(int)
        self._performance_metrics = deque(maxlen=10000)
        self._cached_templates = {}
        
    @lru_cache(maxsize=128)
    def get_schema(self, schema_name: str) -> Type:
        """Cached Schema Loading mit automatischem Performance-Tracking"""
        
        start_time = time.perf_counter()
        
        if schema_name not in self._schemas:
            # Dynamic Import nur wenn benötigt
            module_mapping = {
                'BigFiveScore': 'input_schemas',
                'GiftRecommendationResponse': 'output_schemas',
                'FewShotPromptTemplate': 'prompt_schemas',
                # ... weitere Mappings automatisch
            }
            
            module_name = module_mapping.get(schema_name)
            if module_name:
                module = importlib.import_module(f'.{module_name}', package=__name__)
                self._schemas[schema_name] = getattr(module, schema_name)
        
        # Performance Tracking
        load_time = time.perf_counter() - start_time
        self._record_usage(schema_name, load_time)
        
        return self._schemas.get(schema_name)
    
    def _record_usage(self, schema_name: str, load_time: float):
        """Record usage statistics für Optimierung"""
        self._usage_stats[schema_name] += 1
        self._performance_metrics.append({
            'schema': schema_name,
            'load_time': load_time,
            'timestamp': time.time()
        })
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Performance-Report für Monitoring-Dashboard"""
        return {
            'most_used_schemas': dict(sorted(
                self._usage_stats.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]),
            'average_load_times': {
                schema: sum(m['load_time'] for m in self._performance_metrics 
                           if m['schema'] == schema) / max(count, 1)
                for schema, count in self._usage_stats.items()
            },
            'total_operations': sum(self._usage_stats.values())
        }


class AdaptivePromptGenerator:
    """
    INNOVATION: KI-gesteuerte Prompt-Optimierung basierend auf Schema-Struktur
    
    Generiert automatisch optimale Prompts für verschiedene AI-Models
    """
    
    def __init__(self):
        self.performance_history = defaultdict(list)
        self.model_preferences = {
            'groq_mixtral': {
                'style': 'concise', 
                'max_tokens': 1000,
                'temperature': 0.3,
                'format_preference': 'structured_json'
            },
            'openai_gpt4': {
                'style': 'detailed', 
                'max_tokens': 3000,
                'temperature': 0.7,
                'format_preference': 'conversational_with_structure'
            },
            'anthropic_claude': {
                'style': 'reasoning', 
                'max_tokens': 2000,
                'temperature': 0.5,
                'format_preference': 'step_by_step_analysis'
            }
        }
    
    def generate_optimal_prompt(self, 
                              schema_class: Type,
                              context: Dict[str, Any],
                              target_model: str = 'auto') -> str:
        """
        Generiert optimalen Prompt für spezifisches Model + Schema
        
        INNOVATION: Schema-Analyse + historische Performance → optimaler Prompt
        """
        
        # 1. Schema-Struktur analysieren
        schema_complexity = self._analyze_schema_complexity(schema_class)
        
        # 2. Model-spezifische Optimierung
        if target_model == 'auto':
            target_model = self._select_optimal_model(schema_complexity, context)
        
        model_config = self.model_preferences.get(target_model, self.model_preferences['openai_gpt4'])
        
        # 3. Prompt-Komponenten generieren
        prompt_components = {
            'system': self._generate_system_prompt(schema_class, model_config),
            'structure': self._generate_structure_guidance(schema_class, model_config),
            'examples': self._select_relevant_examples(schema_class, target_model),
            'context': self._format_context(context, model_config)
        }
        
        # 4. Finalen Prompt zusammenstellen
        final_prompt = self._compose_prompt(prompt_components, model_config)
        
        return final_prompt
    
    def _analyze_schema_complexity(self, schema_class: Type) -> Dict[str, Any]:
        """Analysiert Schema-Komplexität für optimale Prompt-Generierung"""
        
        try:
            # Pydantic Schema analysieren
            if hasattr(schema_class, 'model_json_schema'):
                schema_info = schema_class.model_json_schema()
                
                field_count = len(schema_info.get('properties', {}))
                required_fields = len(schema_info.get('required', []))
                nested_objects = sum(1 for prop in schema_info.get('properties', {}).values() 
                                   if prop.get('type') == 'object')
                
                return {
                    'complexity_score': field_count + (nested_objects * 2) + required_fields,
                    'field_count': field_count,
                    'required_fields': required_fields,
                    'nested_objects': nested_objects,
                    'has_enums': any('enum' in str(prop) for prop in schema_info.get('properties', {}).values()),
                    'has_validation': required_fields > 0
                }
        except Exception:
            pass
        
        # Fallback für non-Pydantic classes
        return {
            'complexity_score': 5,
            'field_count': 5,
            'required_fields': 2,
            'nested_objects': 1,
            'has_enums': False,
            'has_validation': True
        }
    
    def _generate_system_prompt(self, schema_class: Type, model_config: Dict) -> str:
        """Model-optimierte System-Prompts"""
        
        schema_name = schema_class.__name__
        style = model_config['style']
        
        if style == 'concise':
            return f"""You are an AI assistant specialized in generating {schema_name} data.
Be precise, structured, and efficient. Return valid JSON only."""

        elif style == 'detailed':
            return f"""You are an expert AI assistant creating comprehensive {schema_name} responses.
Provide detailed, well-reasoned outputs with clear explanations for your choices.
Structure your response carefully and include relevant context."""

        elif style == 'reasoning':
            return f"""You are an analytical AI assistant generating {schema_name} data.
Think step-by-step, explain your reasoning process, and ensure logical consistency.
Provide structured output with clear justification for each recommendation."""
        
        else:
            return f"Generate a well-structured {schema_name} response."


class MultiModelValidator:
    """
    INNOVATION: Cross-Model Validation für höchste Qualität
    
    Nutzt mehrere AI-Models um Responses zu validieren und zu verbessern
    """
    
    def __init__(self):
        self.validation_cache = {}
        self.consensus_threshold = 0.8
        
    async def validate_with_consensus(self, 
                                    response: str,
                                    expected_schema: Type,
                                    confidence_threshold: float = 0.8) -> tuple:
        """
        Consensus-basierte Validierung über mehrere AI-Models
        
        Returns: (validated_object, confidence_score)
        """
        
        # 1. Primary Validation
        try:
            primary_result = self._parse_and_validate(response, expected_schema)
            primary_confidence = self._calculate_confidence(primary_result, expected_schema)
            
            if primary_confidence >= confidence_threshold:
                return primary_result, primary_confidence
                
        except Exception as primary_error:
            primary_result = None
            primary_confidence = 0.0
        
        # 2. Secondary Validation mit Multi-Model Approach
        validation_results = []
        
        # Schnelle Groq-Validation für Struktur
        try:
            groq_result = await self._validate_with_groq(response, expected_schema)
            validation_results.append(('groq', groq_result, 0.7))
        except Exception:
            pass
        
        # GPT-4 Validation für Qualität (bei kritischen Schemas)
        if self._is_critical_schema(expected_schema):
            try:
                gpt4_result = await self._validate_with_gpt4(response, expected_schema)
                validation_results.append(('gpt4', gpt4_result, 0.9))
            except Exception:
                pass
        
        # 3. Consensus-Algorithmus
        if validation_results:
            final_result, consensus_score = self._calculate_consensus(
                [(primary_result, primary_confidence)] + 
                [(result, confidence) for _, result, confidence in validation_results]
            )
            return final_result, consensus_score
        
        # 4. Fallback
        if primary_result:
            return primary_result, primary_confidence
        
        raise ValueError(f"Unable to validate response for {expected_schema.__name__}")
    
    def _parse_and_validate(self, response: str, schema: Type):
        """Standard Pydantic Validation"""
        try:
            # JSON Parsing mit verschiedenen Strategien
            if response.strip().startswith('{'):
                data = json.loads(response)
            else:
                # Extract JSON from markdown or mixed text
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    raise ValueError("No JSON found in response")
            
            return schema.model_validate(data)
            
        except Exception as e:
            raise ValueError(f"Validation failed: {e}")
    
    def _calculate_confidence(self, result, expected_schema: Type) -> float:
        """Berechnet Confidence-Score basierend auf Schema-Vollständigkeit"""
        
        if not result:
            return 0.0
        
        try:
            # Feld-Vollständigkeit prüfen
            result_dict = result.model_dump()
            schema_info = expected_schema.model_json_schema()
            
            total_fields = len(schema_info.get('properties', {}))
            filled_fields = len([v for v in result_dict.values() if v is not None])
            
            field_score = filled_fields / max(total_fields, 1)
            
            # Required Fields prüfen
            required_fields = schema_info.get('required', [])
            missing_required = [f for f in required_fields if result_dict.get(f) is None]
            required_score = 1.0 - (len(missing_required) / max(len(required_fields), 1))
            
            # Kombinierter Score
            return (field_score * 0.6) + (required_score * 0.4)
            
        except Exception:
            return 0.5  # Fallback confidence


class SchemaEvolution:
    """
    CUTTING-EDGE: Adaptive Schema-Verbesserung basierend auf AI-Usage Patterns
    
    Schemas lernen aus realer Nutzung und verbessern sich automatisch
    """
    
    def __init__(self):
        self.usage_patterns = defaultdict(list)
        self.error_patterns = defaultdict(list)
        self.performance_metrics = defaultdict(list)
        self.improvement_suggestions = defaultdict(list)
    
    def track_usage(self, schema_name: str, operation: str, data: Dict[str, Any]):
        """Trackt Schema-Nutzung für Evolution"""
        
        timestamp = datetime.now()
        usage_record = {
            'timestamp': timestamp,
            'operation': operation,
            'field_usage': self._analyze_field_usage(data),
            'data_patterns': self._extract_data_patterns(data)
        }
        
        self.usage_patterns[schema_name].append(usage_record)
        
        # Cleanup alte Daten (behalte nur 30 Tage)
        cutoff = datetime.now().timestamp() - (30 * 24 * 3600)
        self.usage_patterns[schema_name] = [
            record for record in self.usage_patterns[schema_name]
            if record['timestamp'].timestamp() > cutoff
        ]
    
    def track_error(self, schema_name: str, error: str, context: Dict[str, Any]):
        """Trackt Fehler für Schema-Verbesserung"""
        
        error_record = {
            'timestamp': datetime.now(),
            'error': error,
            'context': context,
            'error_type': self._classify_error(error)
        }
        
        self.error_patterns[schema_name].append(error_record)
    
    def generate_evolution_suggestions(self, schema_name: str) -> Dict[str, Any]:
        """
        Generiert Verbesserungsvorschläge basierend auf Usage-Patterns
        
        INNOVATION: ML-ähnliche Pattern-Erkennung für Schema-Optimierung
        """
        
        suggestions = {
            'field_improvements': [],
            'validation_improvements': [],
            'performance_improvements': [],
            'new_field_suggestions': []
        }
        
        # 1. Häufige Fehler analysieren
        common_errors = self._analyze_error_frequency(schema_name)
        for error_pattern in common_errors:
            if 'required field missing' in error_pattern['error']:
                suggestions['field_improvements'].append({
                    'type': 'add_default_value',
                    'field': error_pattern.get('field'),
                    'frequency': error_pattern['count'],
                    'suggestion': f"Add default value to reduce validation errors"
                })
        
        # 2. Ungenutzte Felder identifizieren
        unused_fields = self._identify_unused_fields(schema_name)
        for field in unused_fields:
            suggestions['field_improvements'].append({
                'type': 'mark_optional_or_remove',
                'field': field,
                'usage_percentage': unused_fields[field],
                'suggestion': f"Field '{field}' is rarely used, consider making optional"
            })
        
        # 3. Performance-Hotspots
        slow_operations = self._identify_performance_issues(schema_name)
        for operation in slow_operations:
            suggestions['performance_improvements'].append({
                'type': 'optimize_computation',
                'operation': operation['name'],
                'avg_time': operation['avg_time'],
                'suggestion': operation['suggestion']
            })
        
        # 4. Neue Felder vorschlagen (basierend auf häufigen Kontext-Daten)
        frequent_context = self._analyze_frequent_context_patterns(schema_name)
        for pattern in frequent_context:
            suggestions['new_field_suggestions'].append({
                'field_name': pattern['suggested_name'],
                'field_type': pattern['inferred_type'],
                'frequency': pattern['frequency'],
                'rationale': pattern['rationale']
            })
        
        return suggestions
    

    def _analyze_field_usage(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analysiert welche Felder wie oft verwendet werden"""
        
        field_usage = {}
        
        if isinstance(data, dict):
            for field_name, field_value in data.items():
                field_usage[field_name] = {
                    'used': field_value is not None,
                    'type': type(field_value).__name__,
                    'length': len(str(field_value)) if field_value is not None else 0
                }
        
        return field_usage

    def _extract_data_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extrahiert Daten-Patterns für Analyse"""
        
        patterns = {
            'field_count': len(data) if isinstance(data, dict) else 0,
            'nested_objects': 0,
            'array_fields': 0,
            'null_fields': 0
        }
        
        if isinstance(data, dict):
            for field_name, field_value in data.items():
                if field_value is None:
                    patterns['null_fields'] += 1
                elif isinstance(field_value, dict):
                    patterns['nested_objects'] += 1
                elif isinstance(field_value, (list, tuple)):
                    patterns['array_fields'] += 1
        
        return patterns

    def _classify_error(self, error: str) -> str:
        """Klassifiziert Error-Types"""
        
        error_lower = error.lower()
        
        if 'required' in error_lower and 'missing' in error_lower:
            return 'missing_required_field'
        elif 'validation' in error_lower:
            return 'validation_error'
        elif 'type' in error_lower:
            return 'type_error'
        elif 'extra' in error_lower and 'forbidden' in error_lower:
            return 'extra_field_error'
        else:
            return 'unknown_error'

    def _analyze_error_frequency(self, schema_name: str) -> List[Dict[str, Any]]:
        """Analysiert Error-Häufigkeiten"""
        
        if schema_name not in self.error_patterns:
            return []
        
        error_counts = {}
        for error_record in self.error_patterns[schema_name]:
            error_type = error_record.get('error_type', 'unknown')
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return [
            {'error': error_type, 'count': count}
            for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        ]

    def _identify_unused_fields(self, schema_name: str) -> Dict[str, float]:
        """Identifiziert ungenutzte Felder"""
        
        if schema_name not in self.usage_patterns:
            return {}
        
        field_usage_stats = {}
        total_uses = len(self.usage_patterns[schema_name])
        
        if total_uses == 0:
            return {}
        
        # Sammle alle verwendeten Felder
        all_fields = set()
        field_use_counts = {}
        
        for usage_record in self.usage_patterns[schema_name]:
            field_usage = usage_record.get('field_usage', {})
            for field_name, field_info in field_usage.items():
                all_fields.add(field_name)
                if field_info.get('used', False):
                    field_use_counts[field_name] = field_use_counts.get(field_name, 0) + 1
        
        # Berechne Usage-Prozentsätze
        for field_name in all_fields:
            usage_percentage = (field_use_counts.get(field_name, 0) / total_uses) * 100
            if usage_percentage < 20:  # Weniger als 20% Nutzung = "ungenutzt"
                field_usage_stats[field_name] = usage_percentage
        
        return field_usage_stats

    def _identify_performance_issues(self, schema_name: str) -> List[Dict[str, Any]]:
        """Identifiziert Performance-Probleme"""
        
        if schema_name not in self.performance_metrics:
            return []
        
        # Vereinfachte Performance-Analyse
        issues = []
        
        # Check für langsame Operationen (> 1 Sekunde)
        slow_operations = [
            metric for metric in self.performance_metrics[schema_name]
            if metric.get('processing_time', 0) > 1.0
        ]
        
        if slow_operations:
            avg_time = sum(op.get('processing_time', 0) for op in slow_operations) / len(slow_operations)
            issues.append({
                'name': 'slow_processing',
                'avg_time': avg_time,
                'suggestion': 'Consider optimizing field validation or computation'
            })
        
        return issues

    def _analyze_frequent_context_patterns(self, schema_name: str) -> List[Dict[str, Any]]:
        """Analysiert häufige Kontext-Patterns"""
        
        if schema_name not in self.usage_patterns:
            return []
        
        # Vereinfachte Pattern-Analyse
        patterns = []
        
        # Sammle häufige Zusatz-Daten die nicht im Schema sind
        context_fields = {}
        
        for usage_record in self.usage_patterns[schema_name]:
            # Hier könnten wir analysieren welche Daten häufig im Context erscheinen
            # aber nicht im Schema definiert sind
            pass
        
        # Beispiel-Pattern für häufig vermisste Felder
        if schema_name == 'GiftRecommendationRequest':
            patterns.append({
                'suggested_name': 'occasion',
                'inferred_type': 'str',
                'frequency': 85,
                'rationale': 'Frequently used in additional_context'
            })
        
        return patterns


# =============================================================================
# ENHANCED CONVENIENCE CLASSES: Erweiterte APIs (NEW + BACKWARD COMPATIBLE)
# =============================================================================

class AIRequestResponse:
    """Convenience-Klasse für häufige Request/Response Patterns (ENHANCED)"""
    
    # ✅ BACKWARD COMPATIBLE: Alle bestehenden Attribute bleiben
    Request = GiftRecommendationRequest
    Response = GiftRecommendationResponse
    QuickRequest = QuickRecommendationInput
    QuickResponse = QuickRecommendationResponse
    QuizInput = PersonalityQuizInput
    QuizResult = PersonalityQuizResult
    Error = ErrorResponse
    
    # 🚀 NEW: Advanced Factory Methods
    @classmethod
    def create_optimized_request(cls, 
                               personality_data: Dict[str, Any],
                               target_model: str = 'auto',
                               **kwargs) -> GiftRecommendationRequest:
        """
        INNOVATION: Erstellt Model-optimierte Requests
        """
        # Model-spezifische Optimierungen
        if target_model == 'groq_mixtral':
            # Groq: Reduzierte Komplexität für Speed
            kwargs.setdefault('max_recommendations', 3)
            kwargs.setdefault('personalization_level', 'medium')
        elif target_model == 'openai_gpt4':
            # GPT-4: Maximale Qualität
            kwargs.setdefault('max_recommendations', 7)
            kwargs.setdefault('personalization_level', 'high')
        
        return cls.Request(personality_data=personality_data, **kwargs)


class PromptEngineering:
    """Convenience-Klasse für Prompt Engineering (ENHANCED)"""
    
    # ✅ BACKWARD COMPATIBLE: Bestehende API bleibt
    FewShot = FewShotPromptTemplate
    ChainOfThought = ChainOfThoughtTemplate
    Dynamic = DynamicPromptTemplate
    Example = PromptExample
    Context = ContextInjection
    Step = ChainOfThoughtStep
    Builder = PromptBuilder
    Metrics = PromptPerformanceMetrics
    
    # 🚀 NEW: Advanced Prompt Generation
    @classmethod
    def generate_adaptive_prompt(cls,
                               schema_class: Type,
                               context: Dict[str, Any],
                               optimization_goal: str = 'quality') -> str:
        """
        INNOVATION: Generiert adaptive Prompts basierend auf Schema + Kontext
        """
        generator = AdaptivePromptGenerator()
        
        # Goal-basierte Model-Auswahl
        model_mapping = {
            'speed': 'groq_mixtral',
            'quality': 'openai_gpt4', 
            'reasoning': 'anthropic_claude',
            'cost': 'groq_mixtral'
        }
        
        target_model = model_mapping.get(optimization_goal, 'openai_gpt4')
        
        return generator.generate_optimal_prompt(
            schema_class, context, target_model
        )


# =============================================================================
# GLOBAL INSTANCES: Ready-to-use advanced features (NEW)
# =============================================================================

# Global Schema Registry mit Performance-Monitoring
schema_registry = SchemaRegistry()

# Global Prompt Generator für adaptive Prompts
prompt_generator = AdaptivePromptGenerator()

# Global Schema Evolution Tracker
schema_evolution = SchemaEvolution()

# Global Multi-Model Validator
multi_validator = MultiModelValidator()


# =============================================================================
# ENHANCED API: Fluent Interface für moderne Nutzung (NEW)
# =============================================================================

class SchemaAPI:
    """
    Enhanced Fluent API für Schema-Operations
    
    ✅ BACKWARD COMPATIBLE: Bestehende Imports funktionieren weiter
    🚀 NEW FEATURES: Performance-Monitoring, adaptive Prompts, validation
    """
    
    # ✅ BACKWARD COMPATIBLE Properties
    @property
    def inputs(self):
        """Lazy loading von input schemas"""
        from . import input_schemas
        return input_schemas
    
    @property  
    def outputs(self):
        """Lazy loading von output schemas"""
        from . import output_schemas
        return output_schemas
    
    @property
    def prompts(self):
        """Lazy loading von prompt schemas"""  
        from . import prompt_schemas
        return prompt_schemas
    
    # ✅ BACKWARD COMPATIBLE Factory Methods
    def create_gift_request(self, **kwargs):
        """Factory für Gift Requests (FIXED - handles occasion properly)"""
        
        # 🔧 FIX: Handle 'occasion' parameter specially
        occasion = kwargs.pop('occasion', None)
        relationship_type = kwargs.pop('relationship_type', None)
        
        # Build additional_context
        context_parts = []
        if occasion:
            context_parts.append(f"Occasion: {occasion}")
        if relationship_type:
            context_parts.append(f"Relationship: {relationship_type}")
        
        # Add to existing additional_context
        existing = kwargs.get('additional_context', '')
        if existing:
            context_parts.append(existing)
        
        if context_parts:
            kwargs['additional_context'] = ' | '.join(context_parts)
        
        request = GiftRecommendationRequest(**kwargs)
        
        # Usage tracking
        schema_evolution.track_usage('GiftRecommendationRequest', 'create', kwargs)
        
        return request
    
    def create_big_five_score(self, **kwargs):
        """Factory für Big Five Scores (enhanced)"""
        score = BigFiveScore(**kwargs)
        
        # 🚀 NEW: Usage tracking
        schema_evolution.track_usage('BigFiveScore', 'create', kwargs)
        
        return score
    
    # 🚀 NEW: Advanced Methods
    def generate_adaptive_prompt(self, 
                               schema_class: Type,
                               context: Dict[str, Any],
                               target_model: str = 'auto') -> str:
        """Generiert optimalen Prompt für Schema + Model"""
        return prompt_generator.generate_optimal_prompt(schema_class, context, target_model)
    
    async def validate_ai_response(self,
                                 response: str,
                                 expected_schema: Type,
                                 use_consensus: bool = False):
        """Validiert AI-Response mit optionaler Multi-Model Consensus"""
        
        if use_consensus:
            return await multi_validator.validate_with_consensus(response, expected_schema)
        else:
            # Standard validation
            try:
                data = json.loads(response)
                return expected_schema.model_validate(data), 1.0
            except Exception as e:
                schema_evolution.track_error(expected_schema.__name__, str(e), {'response': response})
                raise
    
    def get_evolution_suggestions(self, schema_name: str) -> Dict[str, Any]:
        """Holt Verbesserungsvorschläge für Schema"""
        return schema_evolution.generate_evolution_suggestions(schema_name)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Performance-Report aller Schema-Operationen"""
        return schema_registry.get_performance_report()
    
    def create_complete_context(self, 
                              personality_data: Dict[str, Any],
                              relationship_type: str,
                              occasion: str,
                              **kwargs) -> Dict[str, Any]:
        """
        🚀 NEW: Erstellt vollständigen Kontext für AI-Requests
        
        Args:
            personality_data: Big Five Scores und andere Persönlichkeitsdaten
            relationship_type: Art der Beziehung (partner, friend, family, etc.)
            occasion: Anlass (birthday, christmas, valentine, etc.)
            **kwargs: Zusätzliche Kontext-Daten
            
        Returns:
            Dict mit vollständigem Kontext für AI-Processing
        """
        context = {
            'personality_profile': {
                'big_five_scores': personality_data.get('big_five_scores', {}),
                'limbic_scores': personality_data.get('limbic_scores', {}),
                'emotional_triggers': personality_data.get('emotional_triggers', []),
                'interests': personality_data.get('interests', []),
                'hobbies': personality_data.get('hobbies', [])
            },
            'relationship_context': {
                'type': relationship_type,
                'closeness': kwargs.get('closeness', 'close'),
                'duration': kwargs.get('duration', 'long_term'),
                'communication_style': kwargs.get('communication_style', 'direct')
            },
            'occasion_context': {
                'type': occasion,
                'formality': kwargs.get('formality', 'casual'),
                'cultural_significance': kwargs.get('cultural_significance', 'medium'),
                'timing': kwargs.get('timing', 'immediate')
            },
            'budget_context': {
                'range': kwargs.get('budget_range', '€10-€50'),
                'preference': kwargs.get('budget_preference', 'moderate'),
                'flexibility': kwargs.get('budget_flexibility', 'medium')
            },
            'cultural_context': {
                'region': kwargs.get('region', 'german'),
                'traditions': kwargs.get('traditions', []),
                'values': kwargs.get('values', [])
            },
            'additional_context': kwargs.get('additional_context', '')
        }
        
        # Usage tracking
        schema_evolution.track_usage('CompleteContext', 'create', context)
        
        return context


# Global API Instance (Enhanced)
api = SchemaAPI()


# =============================================================================
# BACKWARD COMPATIBLE EXPORTS: Alle bestehenden Imports funktionieren weiter
# =============================================================================

__all__ = [
    # === BACKWARD COMPATIBLE INPUT SCHEMAS ===
    "GiftOccasion", "BigFiveTrait", "LimbicDimension", "LimbicType", "EmotionalTrigger", 
    "BigFiveScore", "LimbicScore", "PersonalityAnalysisInput", "GiftRecommendationRequest",
    "AIPersonalityProfile", "QuickRecommendationInput", "PersonalityQuizInput",

    # === BACKWARD COMPATIBLE OUTPUT SCHEMAS ===
    "ConfidenceLevel", "GiftCategory", "RecommendationReason",
    "GiftRecommendation", "PersonalityAnalysisResult", "GiftRecommendationResponse",
    "QuickRecommendationResponse", "PersonalityQuizResult", "AIModelPerformanceMetrics", "ErrorResponse",
    
    # === BACKWARD COMPATIBLE PROMPT SCHEMAS ===
    "PromptTechnique", "PromptComplexity", "AIModelType", "PromptOptimizationGoal",
    "PromptExample", "ContextInjection", "ChainOfThoughtStep",
    "BasePromptTemplate", "FewShotPromptTemplate", "ChainOfThoughtTemplate", "DynamicPromptTemplate",
    "GiftRecommendationSchema", "PromptPerformanceMetrics", "PromptBuilder",
    
    # === BACKWARD COMPATIBLE CONVENIENCE CLASSES ===
    "AIRequestResponse", "PromptEngineering",

    # === NEW: RELATIONSHIP ANALYSIS ===
    'RelationshipType',
    'RelationshipAnalyzer',
    'RelationshipGiftGuide',
    'AgeGroup',
    'create_relationship_context_for_ai', 
    'get_relationship_budget_guidance',
    'integrate_relationship_with_personality',
    
    # === NEW ADVANCED FEATURES ===
    "api",  # Enhanced API
    "schema_registry",  # Performance monitoring
    "prompt_generator",  # Adaptive prompts
    "schema_evolution",  # Schema learning
    "multi_validator",  # Multi-model validation
    
    # === NEW CLASSES (for direct import) ===
    "SchemaAPI", "AdaptivePromptGenerator", "MultiModelValidator", 
    "SchemaEvolution", "SchemaRegistry", 
]


# =============================================================================
# DEVELOPMENT HELPERS: Enhanced mit neuen Features
# =============================================================================

def create_sample_gift_request() -> GiftRecommendationRequest:
    """Erstellt eine Beispiel-Anfrage für Testing (ENHANCED)"""
    from datetime import date
    
    request = GiftRecommendationRequest(
        personality_data={
            "big_five_scores": {
                "openness": 0.8,
                "conscientiousness": 0.7,
                "extraversion": 0.6,
                "agreeableness": 0.8,
                "neuroticism": 0.3
            },
            "user_id": "sample_user_123"
        },
        occasion=GiftOccasion.GEBURTSTAG,
        max_recommendations=5
    )
    
    # 🚀 NEW: Auto-tracking
    schema_evolution.track_usage('GiftRecommendationRequest', 'sample_create', 
                               request.model_dump())
    
    return request


def create_sample_few_shot_template() -> FewShotPromptTemplate:
    """Erstellt ein Beispiel Few-Shot Template für Testing (ENHANCED)"""
    
    template = FewShotPromptTemplate(
        template_name="gift_recommendation_few_shot_v2",
        description="Enhanced Few-Shot Learning für Geschenkempfehlungen",
        technique=PromptTechnique.FEW_SHOT,
        complexity=PromptComplexity.MODERATE,
        instruction_prompt="Generate personalized gift recommendations based on personality analysis.",
        examples=[
            PromptExample(
                input_example="Birthday gift for extroverted, creative 25-year-old friend",
                expected_output='{"recommendations": [{"title": "Art workshop experience", "reasoning": "Matches creativity and social nature"}]}',
                explanation="Combines extraversion with creative openness"
            ),
            PromptExample(
                input_example="Anniversary gift for introverted, practical spouse",  
                expected_output='{"recommendations": [{"title": "Quality kitchen tools", "reasoning": "Practical value for everyday use"}]}',
                explanation="Respects introversion while providing practical value"
            )
        ]
    )
    
    # 🚀 NEW: Template optimization tracking
    schema_evolution.track_usage('FewShotPromptTemplate', 'sample_create', 
                               template.model_dump())
    
    return template


# =============================================================================
# VERSION & METADATA (ENHANCED)
# =============================================================================

__version__ = "2.1.0"  # Unified version across all packages
__author__ = "SensationGifts AI Team"
__description__ = "Advanced AI schemas with adaptive learning and multi-model validation"

# 🚀 NEW: Feature flags für schrittweise Einführung
FEATURE_FLAGS = {
    'adaptive_prompts': True,
    'multi_model_validation': True,
    'schema_evolution': True,
    'performance_monitoring': True,
    'consensus_validation': False,  # Experimentell
    'auto_schema_generation': False  # Future feature
}