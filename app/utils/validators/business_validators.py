"""
Business Validators - Business Logic & System Validation
========================================================

Alle Funktionen für:
- Business Logic Validierung
- User Permissions & Authorization
- Data Integrity Checks
- Budget & Range Validierung
- Template Quality & Performance (from prompt_schemas.py)
"""

from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# === BUDGET & RANGE VALIDIERUNG ===

def validate_budget_range(budget_min: Any, budget_max: Any) -> List[str]:
    """
    🔒 BUDGET-RANGE VALIDIERUNG
    """
    errors = []
    
    try:
        min_val = float(budget_min) if budget_min is not None else None
        max_val = float(budget_max) if budget_max is not None else None
        
        if min_val is not None:
            if min_val < 0:
                errors.append('Mindestbudget muss positiv sein')
            if min_val > 10000:
                errors.append('Mindestbudget zu hoch (max. 10.000€)')
        
        if max_val is not None:
            if max_val < 0:
                errors.append('Maximalbudget muss positiv sein')
            if max_val > 10000:
                errors.append('Maximalbudget zu hoch (max. 10.000€)')
        
        if min_val is not None and max_val is not None:
            if max_val <= min_val:
                errors.append('Maximalbudget muss größer als Mindestbudget sein')
            
            # Realitäts-Check
            if max_val / min_val > 20:  # Mehr als 20x Unterschied ist unrealistisch
                errors.append('Budget-Range zu groß (unrealistisch)')
                
    except (ValueError, TypeError):
        errors.append('Ungültige Budget-Angaben (müssen Zahlen sein)')
    
    return errors


# === BUSINESS LOGIC VALIDATORS ===

def validate_user_permissions(user, resource_user_id: str) -> bool:
    """
    🔒 USER-BERECHTIGUNG VALIDIERUNG
    """
    if not user or not user.is_authenticated:
        return False
    
    return str(user.id) == str(resource_user_id)


def validate_profile_ownership(user, profile) -> bool:
    """
    🔒 PROFIL-BESITZER VALIDIERUNG
    """
    if not user or not user.is_authenticated or not profile:
        return False
    
    return str(profile.buyer_user_id) == str(user.id)


def validate_session_timeout(last_activity: datetime, timeout_minutes: int = 60) -> bool:
    """
    🔒 SESSION-TIMEOUT VALIDIERUNG
    """
    if not last_activity:
        return False
    
    time_diff = datetime.utcnow() - last_activity
    return time_diff.total_seconds() < (timeout_minutes * 60)


# === DATA INTEGRITY VALIDATORS ===

def validate_database_constraints(model_data: dict, model_type: str) -> Tuple[bool, List[str]]:
    """
    🔒 DATENBANK-CONSTRAINTS VALIDIERUNG
    """
    errors = []
    
    # Model-spezifische Constraints
    if model_type == 'PersonalityProfile':
        # Budget-Konsistenz
        if model_data.get('budget_min') and model_data.get('budget_max'):
            if model_data['budget_min'] >= model_data['budget_max']:
                errors.append('Budget-Minimum muss kleiner als Maximum sein')
    
    elif model_type == 'User':
        # Email-Eindeutigkeit (würde in Service geprüft)
        email = model_data.get('email')
        if email and len(email.strip()) == 0:
            errors.append('Email darf nicht leer sein')
    
    return len(errors) == 0, errors


def validate_json_structure(data: Any, expected_structure: dict) -> Tuple[bool, List[str]]:
    """
    🔒 JSON-STRUKTUR VALIDIERUNG
    
    Args:
        data: Zu prüfende Daten
        expected_structure: Erwartete Struktur {'field': type}
    """
    errors = []
    
    if not isinstance(data, dict):
        errors.append('Daten müssen ein JSON-Objekt sein')
        return False, errors
    
    for field, expected_type in expected_structure.items():
        if field in data:
            if not isinstance(data[field], expected_type):
                errors.append(f'{field} muss vom Typ {expected_type.__name__} sein')
        # Optionale Felder werden nicht als Fehler gewertet
    
    return len(errors) == 0, errors


# === TEMPLATE QUALITY & PERFORMANCE VALIDATION (moved from prompt_schemas.py) ===

def validate_template_quality(template: Any) -> Dict[str, Any]:
    """
    🔒 TEMPLATE-QUALITÄT VALIDIERUNG
    
    Validiert die Qualität eines Prompt-Templates
    Moved from prompt_schemas.py
    """
    
    quality_result = {
        "overall_quality": 0.0,
        "completeness": 0.0,
        "effectiveness": 0.0,
        "optimization_potential": 0.0,
        "recommendations": [],
        "warnings": []
    }
    
    try:
        # 1. Completeness Check
        required_fields = ['template_name', 'instruction_prompt', 'technique', 'complexity']
        filled_required = sum(1 for field in required_fields if hasattr(template, field) and getattr(template, field))
        quality_result["completeness"] = filled_required / len(required_fields)
        
        # 2. Effectiveness Check (based on available metrics)
        effectiveness_factors = []
        
        if hasattr(template, 'success_rate') and template.success_rate is not None:
            effectiveness_factors.append(template.success_rate)
        
        if hasattr(template, 'average_response_quality') and template.average_response_quality is not None:
            effectiveness_factors.append(template.average_response_quality)
        
        if hasattr(template, 'user_satisfaction_score') and template.user_satisfaction_score is not None:
            effectiveness_factors.append(template.user_satisfaction_score)
        
        if effectiveness_factors:
            quality_result["effectiveness"] = sum(effectiveness_factors) / len(effectiveness_factors)
        else:
            quality_result["effectiveness"] = 0.5  # Default for new templates
        
        # 3. Optimization Potential
        optimization_factors = []
        
        # Usage-based optimization
        if hasattr(template, 'usage_count'):
            if template.usage_count > 100:
                optimization_factors.append(0.8)  # Well-tested
            elif template.usage_count > 10:
                optimization_factors.append(0.6)  # Some data
            else:
                optimization_factors.append(0.3)  # Limited data
        
        # Performance history
        if hasattr(template, 'performance_history') and template.performance_history:
            recent_performance = template.performance_history[-5:]
            if len(recent_performance) >= 3:
                optimization_factors.append(0.7)
            else:
                optimization_factors.append(0.4)
        
        if optimization_factors:
            quality_result["optimization_potential"] = sum(optimization_factors) / len(optimization_factors)
        else:
            quality_result["optimization_potential"] = 0.2
        
        # 4. Overall Quality Score
        quality_result["overall_quality"] = (
            quality_result["completeness"] * 0.3 +
            quality_result["effectiveness"] * 0.5 +
            quality_result["optimization_potential"] * 0.2
        )
        
        # 5. Generate Recommendations
        if quality_result["completeness"] < 0.8:
            quality_result["recommendations"].append("Complete missing template fields")
        
        if quality_result["effectiveness"] < 0.6:
            quality_result["recommendations"].append("Improve template effectiveness through testing")
        
        if hasattr(template, 'usage_count') and template.usage_count < 10:
            quality_result["recommendations"].append("Increase template usage for better optimization")
        
        # 6. Generate Warnings
        if hasattr(template, 'success_rate') and template.success_rate is not None and template.success_rate < 0.5:
            quality_result["warnings"].append("Low success rate detected - template needs review")
        
        if hasattr(template, 'instruction_prompt') and len(template.instruction_prompt) < 50:
            quality_result["warnings"].append("Instruction prompt is very short")
        
    except Exception as e:
        logger.error(f"Template quality validation failed: {e}")
        quality_result["warnings"].append(f"Validation error: {str(e)}")
    
    return quality_result


def get_optimization_suggestions(template: Any) -> List[str]:
    """
    🔒 OPTIMIERUNGS-VORSCHLÄGE GENERIERUNG
    
    Generiert spezifische Optimierungsvorschläge für Templates
    Moved from prompt_schemas.py
    """
    
    suggestions = []
    
    try:
        # Performance-basierte Vorschläge
        if hasattr(template, 'success_rate') and template.success_rate is not None:
            if template.success_rate < 0.7:
                suggestions.append("Consider simplifying prompt structure for better success rate")
        
        if hasattr(template, 'average_response_quality') and template.average_response_quality is not None:
            if template.average_response_quality < 0.6:
                suggestions.append("Add more specific examples or constraints to improve quality")
        
        if hasattr(template, 'user_satisfaction_score') and template.user_satisfaction_score is not None:
            if template.user_satisfaction_score < 0.7:
                suggestions.append("Focus on user experience and clarity improvements")
        
        # Usage-basierte Vorschläge
        if hasattr(template, 'usage_count'):
            if template.usage_count > 100 and not hasattr(template, 'ab_test_results'):
                suggestions.append("Consider running A/B tests for optimization")
            elif template.usage_count < 5:
                suggestions.append("Increase template usage to gather performance data")
        
        # Cost efficiency
        if hasattr(template, 'cost_efficiency') and template.cost_efficiency is not None:
            if template.cost_efficiency < 1.0:
                suggestions.append("Optimize for cost efficiency")
        
        # Model-spezifische Vorschläge
        if hasattr(template, 'target_model') and hasattr(template, 'instruction_prompt'):
            if template.target_model == 'groq_mixtral' and len(template.instruction_prompt) > 500:
                suggestions.append("Shorten prompt for Groq optimization")
            elif template.target_model == 'openai_gpt4' and len(template.instruction_prompt) < 200:
                suggestions.append("Expand prompt to leverage GPT-4's capabilities")
        
        # Technique-spezifische Vorschläge
        if hasattr(template, 'technique'):
            if template.technique == 'few_shot' and hasattr(template, 'examples'):
                if len(template.examples) < 3:
                    suggestions.append("Add more examples for better few-shot learning")
            elif template.technique == 'chain_of_thought' and hasattr(template, 'reasoning_steps'):
                if len(template.reasoning_steps) < 3:
                    suggestions.append("Add more reasoning steps for better chain-of-thought")
        
        # Personalization suggestions
        if hasattr(template, 'personalization_score'):
            if template.personalization_score < 0.6:
                suggestions.append("Enhance personalization elements")
        
        # Fallback if no specific suggestions
        if not suggestions:
            suggestions.append("Template appears well-optimized - monitor performance metrics")
    
    except Exception as e:
        logger.error(f"Failed to generate optimization suggestions: {e}")
        suggestions.append("Unable to generate specific suggestions - manual review recommended")
    
    return suggestions


def validate_performance_metrics(metrics: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    🔒 PERFORMANCE-METRIKEN VALIDIERUNG
    
    Validiert Performance-Metriken für Templates und AI-Responses
    """
    errors = []
    warnings = []
    
    # Required metrics validation
    required_metrics = ['response_time_ms', 'response_quality_score', 'parsing_successful']
    for metric in required_metrics:
        if metric not in metrics:
            errors.append(f"Missing required metric: {metric}")
    
    # Value range validation
    percentage_metrics = [
        'response_quality_score', 'personalization_score', 'emotional_resonance',
        'creativity_score', 'user_satisfaction', 'success_rate'
    ]
    
    for metric in percentage_metrics:
        if metric in metrics and metrics[metric] is not None:
            value = metrics[metric]
            if not isinstance(value, (int, float)) or value < 0.0 or value > 1.0:
                errors.append(f"{metric} must be between 0.0 and 1.0")
    
    # Response time validation
    if 'response_time_ms' in metrics:
        response_time = metrics['response_time_ms']
        if not isinstance(response_time, (int, float)) or response_time < 0:
            errors.append("response_time_ms must be a positive number")
        elif response_time > 30000:  # 30 seconds
            warnings.append("Response time is very high (>30s)")
    
    # Token usage validation
    for token_field in ['prompt_tokens_used', 'response_tokens_generated']:
        if token_field in metrics:
            tokens = metrics[token_field]
            if not isinstance(tokens, int) or tokens < 0:
                errors.append(f"{token_field} must be a positive integer")
    
    # Business impact validation
    if 'emotional_connection_created' in metrics:
        value = metrics['emotional_connection_created']
        if value is not None and (not isinstance(value, (int, float)) or value < 0.0 or value > 1.0):
            errors.append("emotional_connection_created must be between 0.0 and 1.0")
    
    # Performance threshold warnings
    if metrics.get('response_quality_score', 1.0) < 0.6:
        warnings.append("Low response quality detected")
    
    if metrics.get('user_satisfaction', 1.0) < 0.7:
        warnings.append("Low user satisfaction detected")
    
    if metrics.get('emotional_resonance', 1.0) < 0.6:
        warnings.append("Low emotional resonance detected")
    
    # Log warnings
    for warning in warnings:
        logger.warning(f"Performance metric warning: {warning}")
    
    return len(errors) == 0, errors


def validate_cost_efficiency(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    🔒 KOSTEN-EFFIZIENZ VALIDIERUNG
    
    Bewertet Cost-Efficiency von AI-Operations
    """
    
    efficiency_result = {
        "cost_per_quality_point": 0.0,
        "cost_per_user_satisfaction": 0.0,
        "overall_efficiency": 0.0,
        "efficiency_grade": "Unknown",
        "recommendations": []
    }
    
    try:
        cost_estimate = metrics.get('cost_estimate', 0.0)
        quality_score = metrics.get('response_quality_score', 0.0)
        user_satisfaction = metrics.get('user_satisfaction')
        response_time = metrics.get('response_time_ms', 0)
        
        if cost_estimate > 0:
            # Cost per quality point
            if quality_score > 0:
                efficiency_result["cost_per_quality_point"] = float(cost_estimate) / quality_score
            
            # Cost per user satisfaction
            if user_satisfaction and user_satisfaction > 0:
                efficiency_result["cost_per_user_satisfaction"] = float(cost_estimate) / user_satisfaction
            
            # Overall efficiency (Quality * Speed) / Cost
            if quality_score > 0 and response_time > 0:
                # Normalize response time (faster = better efficiency)
                time_factor = max(0.1, min(1.0, 5000 / response_time))  # 5s baseline
                efficiency_result["overall_efficiency"] = (quality_score * time_factor) / float(cost_estimate)
            
            # Efficiency grading
            overall_eff = efficiency_result["overall_efficiency"]
            if overall_eff > 10:
                efficiency_result["efficiency_grade"] = "Excellent"
            elif overall_eff > 5:
                efficiency_result["efficiency_grade"] = "Good"
            elif overall_eff > 2:
                efficiency_result["efficiency_grade"] = "Fair"
            elif overall_eff > 1:
                efficiency_result["efficiency_grade"] = "Poor"
            else:
                efficiency_result["efficiency_grade"] = "Very Poor"
            
            # Generate recommendations
            if overall_eff < 2:
                efficiency_result["recommendations"].append("Consider using a more cost-effective model")
                efficiency_result["recommendations"].append("Optimize prompt length to reduce token usage")
            
            if efficiency_result["cost_per_quality_point"] > 0.1:
                efficiency_result["recommendations"].append("Quality-to-cost ratio needs improvement")
            
            if response_time > 10000:  # >10s
                efficiency_result["recommendations"].append("Response time optimization needed")
        
        else:
            efficiency_result["recommendations"].append("Cost tracking not available")
    
    except Exception as e:
        logger.error(f"Cost efficiency validation failed: {e}")
        efficiency_result["recommendations"].append("Unable to calculate cost efficiency")
    
    return efficiency_result


# === SYSTEM HEALTH VALIDATORS ===

def validate_system_health() -> Dict[str, Any]:
    """
    🔒 SYSTEM-GESUNDHEIT VALIDIERUNG
    
    Überprüft allgemeine System-Gesundheit und Performance
    """
    
    health_status = {
        "overall_status": "healthy",
        "components": {},
        "warnings": [],
        "errors": [],
        "recommendations": []
    }
    
    try:
        # Database connectivity check
        try:
            from app.models import User  # Test import
            health_status["components"]["database"] = "healthy"
        except Exception as e:
            health_status["components"]["database"] = "error"
            health_status["errors"].append(f"Database connection issue: {e}")
        
        # AI Engine check
        try:
            from ai_engine.schemas import PersonalityAnalysisInput
            health_status["components"]["ai_engine"] = "healthy"
        except Exception as e:
            health_status["components"]["ai_engine"] = "warning"
            health_status["warnings"].append(f"AI Engine import issue: {e}")
        
        # Validator system check
        try:
            # Test if all validator modules are importable
            from . import form_validators, ai_validators
            health_status["components"]["validators"] = "healthy"
        except Exception as e:
            health_status["components"]["validators"] = "error"
            health_status["errors"].append(f"Validator system issue: {e}")
        
        # Overall status determination
        if health_status["errors"]:
            health_status["overall_status"] = "unhealthy"
        elif health_status["warnings"]:
            health_status["overall_status"] = "warning"
        
        # Generate recommendations
        if health_status["errors"]:
            health_status["recommendations"].append("Address critical errors immediately")
        
        if health_status["warnings"]:
            health_status["recommendations"].append("Review and resolve warnings")
        
        if health_status["overall_status"] == "healthy":
            health_status["recommendations"].append("System running optimally")
    
    except Exception as e:
        health_status["overall_status"] = "unhealthy"
        health_status["errors"].append(f"Health check failed: {e}")
    
    return health_status


# === EXPORT ===

__all__ = [
    # Budget & Range Validierung
    'validate_budget_range',
    
    # Business Logic Validators
    'validate_user_permissions', 'validate_profile_ownership', 
    'validate_session_timeout',
    
    # Data Integrity Validators
    'validate_database_constraints', 'validate_json_structure',
    
    # Template Quality & Performance (moved from prompt_schemas.py)
    'validate_template_quality', 'get_optimization_suggestions',
    'validate_performance_metrics', 'validate_cost_efficiency',
    
    # System Health
    'validate_system_health'
]