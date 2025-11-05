"""
Formatters - Zentrale Formatierungs-Utilities für SensationGifts
===============================================================

Alle Funktionen für:
- API Response Formatierung
- Template Formatierung  
- Daten-Transformation
- Enum-Konvertierung
- Score-Berechnungen
"""

from flask import jsonify, request
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)

# === API RESPONSE FORMATIERUNG ===

def api_response(data=None, message=None, status=200, error=None, meta=None):
    """
    🎯 ZENTRALE API-RESPONSE FORMATIERUNG
    
    Verwendet überall im Projekt für konsistente API-Antworten
    """
    response = {
        'success': error is None,
        'timestamp': datetime.utcnow().isoformat(),
        'status': status
    }
    
    if data is not None:
        response['data'] = data
    if message:
        response['message'] = message
    if error:
        response['error'] = error
    if meta:
        response['meta'] = meta
    
    return jsonify(response), status


def format_error_response(error_message: str, status: int = 500, error_code: str = None) -> tuple:
    """Formatiert Fehler-Responses konsistent"""
    
    error_data = {'message': error_message}
    if error_code:
        error_data['code'] = error_code
    
    return api_response(error=error_data, status=status)


def format_success_response(data: Any, message: str = None, meta: Dict = None) -> tuple:
    """Formatiert Erfolgs-Responses konsistent"""
    
    return api_response(data=data, message=message, meta=meta, status=200)


# === REQUEST FORMATIERUNG ===

def is_api_request() -> bool:
    """
    Prüft ob Request JSON-Response erwartet
    """
    return (
        request.is_json or 
        'application/json' in request.headers.get('Accept', '') or
        request.args.get('format') == 'json'
    )


def get_safe_redirect_url(default_url: str = '/') -> str:
    """
    Sichere Redirect-URL aus 'next' Parameter
    """
    next_page = request.args.get('next')
    if not next_page or urlparse(next_page).netloc != '':
        next_page = default_url
    return next_page


def parse_filter_params(args) -> Dict[str, Any]:
    """
    Parsed Query-Parameter für Gift-Filtering
    """
    filters = {}
    
    # Basic Filters
    if args.get('category'):
        filters['category'] = args.get('category')
    if args.get('search'):
        filters['search'] = args.get('search')
    if args.get('occasion'):
        filters['occasion'] = args.get('occasion')
    if args.get('gift_type'):
        filters['gift_type'] = args.get('gift_type')
    
    # Price Filters
    try:
        if args.get('price_min'):
            filters['price_min'] = float(args.get('price_min'))
        if args.get('price_max'):
            filters['price_max'] = float(args.get('price_max'))
    except ValueError:
        pass
    
    # Tag Filters
    if args.get('tags'):
        filters['tags'] = args.get('tags').split(',')
    
    # Pagination
    filters['page'] = max(1, int(args.get('page', 1)))
    filters['limit'] = min(100, max(1, int(args.get('limit', 20))))
    
    # Sorting
    filters['sort_by'] = args.get('sort_by', 'relevance')
    
    return filters


# === TEMPLATE FORMATIERUNG ===

def format_datetime_for_template(dt: datetime) -> str:
    """Formatiert Datetime für Templates"""
    if dt:
        return dt.strftime('%d.%m.%Y %H:%M')
    return ''


def format_date_for_template(dt: datetime) -> str:
    """Formatiert Date für Templates"""
    if dt:
        return dt.strftime('%d.%m.%Y')
    return ''


def format_currency(amount: float, currency: str = 'EUR') -> str:
    """Formatiert Währungsbeträge"""
    if currency == 'EUR':
        return f"{amount:.2f} €"
    else:
        return f"{amount:.2f} {currency}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Formatiert Prozentsätze"""
    return f"{value * 100:.{decimals}f}%"


# === ENUM KONVERTIERUNG ===

def convert_optimization_goal(goal_string: str):
    """Konvertiert String zu PromptOptimizationGoal Enum"""
    try:
        # Import hier um zirkuläre Imports zu vermeiden
        from ai_engine.schemas import PromptOptimizationGoal
        
        goal_mapping = {
            'speed': PromptOptimizationGoal.SPEED,
            'quality': PromptOptimizationGoal.QUALITY,
            'cost': getattr(PromptOptimizationGoal, 'COST', PromptOptimizationGoal.BALANCE),
            'creativity': getattr(PromptOptimizationGoal, 'CREATIVITY', PromptOptimizationGoal.BALANCE),
            'accuracy': getattr(PromptOptimizationGoal, 'ACCURACY', PromptOptimizationGoal.QUALITY),
            'balance': PromptOptimizationGoal.BALANCE
        }
        
        return goal_mapping.get(goal_string, PromptOptimizationGoal.BALANCE)
        
    except ImportError:
        # Fallback wenn AI-Engine nicht verfügbar
        return goal_string


def convert_feedback_type_to_enum(feedback_string: str):
    """Konvertiert String zu FeedbackType Enum"""
    try:
        from app.models.recommendation import FeedbackType
        return FeedbackType(feedback_string)
    except (ImportError, ValueError):
        return None


# === SCORE BERECHNUNGEN ===

def calculate_completion_score(data: dict, total_fields: int = 15) -> float:
    """Berechnet Vollständigkeits-Score der Antworten"""
    completed_fields = 0
    
    # Zähle ausgefüllte Felder
    for field, value in data.items():
        if value is not None and str(value).strip():
            completed_fields += 1
    
    return min(completed_fields / total_fields, 1.0)


def calculate_answer_quality(data: dict) -> float:
    """Berechnet Qualitäts-Score der Antworten"""
    quality_indicators = []
    
    # Hobbies detailliert?
    hobbies = data.get('hobbies', '')
    if hobbies:
        hobby_count = len([h.strip() for h in hobbies.split(',') if h.strip()])
        quality_indicators.append(min(hobby_count / 3.0, 1.0))  # 3+ Hobbies = gut
    
    # Budget realistisch?
    budget_min = data.get('budget_min', 0)
    budget_max = data.get('budget_max', 0)
    if budget_min and budget_max:
        budget_ratio = budget_max / budget_min if budget_min > 0 else 1
        quality_indicators.append(1.0 if 1.5 <= budget_ratio <= 5.0 else 0.5)
    
    # Persönlichkeits-Scores ausgeglichen?
    big_five_values = []  # ← KLARERE BENENNUNG
    for field in ['extraversion', 'openness', 'conscientiousness', 'agreeableness', 'neuroticism']:
        if field in data and data[field] is not None:
            big_five_values.append(data[field])

    limbic_values = []  # ← KLARERE BENENNUNG
    for field in ['stimulanz', 'dominanz', 'balance']:
        if field in data and data[field] is not None:
            limbic_values.append(data[field])

    all_scores = big_five_values + limbic_values
    if all_scores:
        score_variance = max(all_scores) - min(all_scores)
        quality_indicators.append(1.0 if score_variance >= 0.2 else 0.7)  # Variance zeigt Durchdachtheit
        return sum(quality_indicators) / len(quality_indicators) if quality_indicators else 0.5


def calculate_confidence_scores(recommendations: List[dict]) -> dict:
    """Berechnet Confidence-Scores für Empfehlungen"""
    
    if not recommendations:
        return {'average': 0.0, 'min': 0.0, 'max': 0.0}
    
    scores = [rec.get('match_score', 0.5) for rec in recommendations]
    
    return {
        'average': sum(scores) / len(scores),
        'min': min(scores),
        'max': max(scores),
        'distribution': classify_score_distribution(scores)
    }


def classify_score_distribution(scores: List[float]) -> str:
    """Klassifiziert Score-Verteilung"""
    
    high_scores = sum(1 for score in scores if score > 0.8)
    low_scores = sum(1 for score in scores if score < 0.4)
    
    if high_scores >= len(scores) * 0.7:
        return 'excellent_matches'
    elif low_scores >= len(scores) * 0.5:
        return 'diverse_options'
    else:
        return 'balanced_selection'


def calculate_quality_speed_ratio(processing_time: float, recommendation_count: int) -> float:
    """Berechnet Qualität-zu-Geschwindigkeit Verhältnis"""
    
    # Basis-Score: mehr Empfehlungen = bessere Qualität
    quality_score = min(recommendation_count / 8.0, 1.0)  # 8 Empfehlungen = perfekt
    
    # Speed-Score: unter 2s = perfekt
    speed_score = max(0, 1.0 - (processing_time - 2000) / 3000)  # Linear decay nach 2s
    
    return (quality_score + speed_score) / 2.0


# === TEXT FORMATIERUNG ===

def generate_search_interpretation(query: str, results_count: int, filters: Dict[str, Any]) -> str:
    """Generiert AI-Interpretation der Suche"""
    interpretation_parts = []
    
    if results_count == 0:
        interpretation_parts.append("Keine passenden Geschenke gefunden")
    elif results_count < 5:
        interpretation_parts.append("Wenige, aber sehr spezifische Ergebnisse")
    else:
        interpretation_parts.append(f"{results_count} passende Geschenke gefunden")
    
    # Query-Analyse
    if 'geburtstag' in query.lower():
        interpretation_parts.append("Fokus auf Geburtstags-Geschenke")
    elif 'weihnachten' in query.lower():
        interpretation_parts.append("Weihnachts-spezifische Suche")
    
    # Filter-Interpretation
    if filters.get('price_min') or filters.get('price_max'):
        interpretation_parts.append("Budget-Filter angewendet")
    
    return " • ".join(interpretation_parts)


def get_trigger_description(trigger_value: str) -> str:
    """Beschreibungen für emotionale Trigger"""
    descriptions = {
        'nostalgia': "Liebt Erinnerungen und nostalgische Gegenstände",
        'adventure': "Will neue Erfahrungen und aufregende Aktivitäten",
        'care': "Schätzt Fürsorge und Wellness-orientierte Geschenke",
        'status': "Mag hochwertige und prestigeträchtige Dinge",
        'belonging': "Liebt gemeinsame Zeit und Gruppenaktivitäten",
        'creativity': "Ist kreativ und mag künstlerische Ausdrucksformen",
        'achievement': "Schätzt Herausforderungen und Erfolg"
    }
    return descriptions.get(trigger_value, "Emotionaler Trigger")


def get_lifestyle_description(lifestyle_value: str) -> str:
    """Beschreibungen für Lifestyle-Typen"""
    descriptions = {
        'minimalist': "Bevorzugt wenige, hochwertige Gegenstände",
        'collector': "Sammelt gerne Dinge und schätzt Vollständigkeit",
        'adventurer': "Immer unterwegs und aktiv",
        'homebody': "Verbringt gerne Zeit zuhause",
        'socializer': "Liebt Gesellschaft und soziale Aktivitäten",
        'achiever': "Fokussiert auf Karriere und Ziele"
    }
    return descriptions.get(lifestyle_value, "Lifestyle-Typ")


def get_relationship_insights(relationship: str) -> str:
    """Insights basierend auf Beziehung"""
    insights_map = {
        'partner': 'Romantische und persönliche Geschenke funktionieren gut',
        'familie': 'Traditionelle und bedeutungsvolle Geschenke sind ideal',
        'freund': 'Spaßige und geteilte Interessen stehen im Vordergrund',
        'kollege': 'Professionelle und neutrale Geschenke sind angemessen'
    }
    return insights_map.get(relationship, 'Durchdachte Geschenke zeigen Wertschätzung')


def get_occasion_tips(occasion: str) -> str:
    """Tips basierend auf Anlass"""
    tips_map = {
        'geburtstag': 'Persönliche Geschenke mit individueller Note',
        'weihnachten': 'Gemütliche und traditionelle Geschenke',
        'jahrestag': 'Erinnerungen und gemeinsame Erlebnisse',
        'valentinstag': 'Romantische und aufmerksame Gesten'
    }
    return tips_map.get(occasion, 'Passend zum Anlass gewählte Geschenke')


def analyze_budget_range(budget_min: float, budget_max: float) -> str:
    """Analysiert Budget-Bereich"""
    range_size = budget_max - budget_min
    
    if range_size < 20:
        return 'Enger Budget-Bereich - gezielter Focus'
    elif range_size < 100:
        return 'Mittlerer Budget-Bereich - gute Auswahl'
    else:
        return 'Breiter Budget-Bereich - maximale Flexibilität'


# === PERFORMANCE FORMATIERUNG ===

def format_processing_time(start_time: datetime, end_time: datetime = None) -> dict:
    """Formatiert Processing-Time Metriken"""
    if end_time is None:
        end_time = datetime.utcnow()
    
    duration = (end_time - start_time).total_seconds() * 1000
    
    return {
        'processing_time_ms': round(duration, 2),
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'performance_level': 'excellent' if duration < 500 else 'good' if duration < 1000 else 'acceptable'
    }


def format_pagination_meta(page: int, limit: int, total: int) -> dict:
    """Formatiert Pagination-Metadaten"""
    pages = (total + limit - 1) // limit
    
    return {
        'page': page,
        'limit': limit,
        'total': total,
        'pages': pages,
        'has_next': page * limit < total,
        'has_prev': page > 1,
        'from_item': (page - 1) * limit + 1 if total > 0 else 0,
        'to_item': min(page * limit, total)
    }


# === DATENSCHUTZ FORMATIERUNG ===

def sanitize_user_data(user_data: dict) -> dict:
    """Entfernt sensible Daten aus User-Export"""
    sensitive_fields = ['password_hash', 'email_verification_token', 'reset_token']
    
    return {k: v for k, v in user_data.items() if k not in sensitive_fields}


def format_public_profile(user) -> dict:
    """Formatiert öffentliche User-Daten"""
    return {
        'display_name': user.display_name,
        'member_since': user.created_at.isoformat() if user.created_at else None,
        'verified': user.is_verified
    }


# === VALIDATION HELPERS ===

def get_validation_recommendations(data: dict, completion_score: float, quality_score: float) -> List[str]:
    """Gibt Empfehlungen zur Verbesserung der Antworten"""
    recommendations = []
    
    if completion_score < 0.7:
        recommendations.append("Beantworten Sie mehr Fragen für bessere Empfehlungen")
    
    if quality_score < 0.6:
        recommendations.append("Geben Sie detailliertere Antworten für präzisere Ergebnisse")
    
    if not data.get('hobbies'):
        recommendations.append("Fügen Sie Hobbies hinzu für passendere Geschenkvorschläge")
    
    if not data.get('emotional_triggers'):
        recommendations.append("Wählen Sie emotionale Trigger für persönlichere Empfehlungen")
    
    return recommendations


# === EXPORT ===

__all__ = [
    # API Response
    'api_response', 'format_error_response', 'format_success_response',
    
    # Request Formatierung
    'is_api_request', 'get_safe_redirect_url', 'parse_filter_params',
    
    # Template Formatierung
    'format_datetime_for_template', 'format_date_for_template', 
    'format_currency', 'format_percentage',
    
    # Enum Konvertierung
    'convert_optimization_goal', 'convert_feedback_type_to_enum',
    
    # Score Berechnungen
    'calculate_completion_score', 'calculate_answer_quality',
    'calculate_confidence_scores', 'classify_score_distribution',
    'calculate_quality_speed_ratio',
    
    # Text Formatierung
    'generate_search_interpretation', 'get_trigger_description',
    'get_lifestyle_description', 'get_relationship_insights',
    'get_occasion_tips', 'analyze_budget_range',
    
    # Performance Formatierung
    'format_processing_time', 'format_pagination_meta',
    
    # Datenschutz
    'sanitize_user_data', 'format_public_profile',
    
    # Validation Helpers
    'get_validation_recommendations'
]