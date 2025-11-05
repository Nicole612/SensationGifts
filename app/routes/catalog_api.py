# ================================================================
# 📁 app/routes/catalog_api.py - KATALOG EMPFEHLUNGEN
# 🎯 Fallback für AI-Empfehlungen mit kuratierten Geschenken
# ================================================================

from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
import logging
from datetime import datetime

from app.extensions import csrf, db
from app.utils.formatters import api_response

# Katalog Service Import (falls verfügbar)
try:
    from ai_engine.catalog.catalog_service import CatalogIntegrationService as CatalogService
    from ai_engine.catalog.gift_catalog_generator import GeschenkKatalogService as GiftCatalogGenerator
    CATALOG_SERVICE_AVAILABLE = True
except ImportError:
    CATALOG_SERVICE_AVAILABLE = False

logger = logging.getLogger(__name__)

# Blueprint Setup
catalog_bp = Blueprint('catalog', __name__, url_prefix='/api/catalog')

# ================================================================
# 🤖 AI RECOMMENDATIONS
# ================================================================

def generate_ai_recommendations(limbic_type: str, traits: list, 
                               budget_min: float, budget_max: float, 
                               occasion: str, count: int):
    """🤖 Generate AI-powered recommendations using the enhanced AI service"""
    
    try:
        from app.services.enhanced_ai_recommendation_service import get_recommendation_service
        
        # Get AI service
        ai_service = get_recommendation_service()
        
        # Generate AI recommendations
        import asyncio
        result = asyncio.run(ai_service.generate_recommendations(
            user_id=f'ai_user_{limbic_type}',
            occasion=occasion,
            relationship='friend',  # Default
            budget_min=budget_min,
            budget_max=budget_max,
            number_of_recommendations=count,
            optimization_goal='balance'
        ))
        
        if result.get('success'):
            recommendations = result.get('recommendations', [])
            
            # Convert to standard format
            formatted_recommendations = []
            for rec in recommendations:
                formatted_recommendations.append({
                    'id': f'ai_{hash(rec.get("gift_name", ""))}',
                    'name': rec.get('gift_name', 'AI Recommendation'),
                    'description': rec.get('description', 'AI-powered personalized recommendation'),
                    'price_range': f'€{rec.get("price_estimate", budget_min):.0f}',
                    'category': rec.get('category', 'ai_recommendation'),
                    'match_score': rec.get('confidence_score', 0.8),
                    'reasoning': rec.get('reasoning', f'AI-powered recommendation for {limbic_type}'),
                    'emotional_story': rec.get('emotional_story', 'AI-generated emotional story'),
                    'personality_match': rec.get('personality_match', rec.get('reasoning', f'AI analysis for {limbic_type}')),
                    'source': 'ai_recommendation',
                    'tags': [limbic_type] + traits,
                    'rating': 4.5,
                    'availability': 'in_stock'
                })
            
            return formatted_recommendations
        else:
            raise Exception(f"AI service failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"AI recommendations failed: {e}")
        raise

# ================================================================
# 🎯 HAUPT-KATALOG ENDPOINT
# ================================================================

@csrf.exempt
@catalog_bp.route('/recommendations', methods=['POST', 'OPTIONS'])
@cross_origin(origins="*", methods=['POST', 'OPTIONS'], allow_headers=['Content-Type'])
def get_catalog_recommendations():
    """
    🎁 KATALOG-EMPFEHLUNGEN ENDPOINT
    
    POST /api/catalog/recommendations
    {
        "limbic_type": "harmonizer",
        "personality_traits": ["balanced", "creative"],
        "budget_min": 50,
        "budget_max": 200,
        "occasion": "birthday",
        "count": 3
    }
    
    → Gibt 3 kuratierte Geschenk-Empfehlungen zurück
    """
    
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        request_data = request.get_json()
        if not request_data:
            return api_response(error='No request data provided', status=400)
        
        # Extract parameters
        limbic_type = request_data.get('limbic_type', 'harmonizer')
        personality_traits = request_data.get('personality_traits', ['balanced'])
        budget_min = float(request_data.get('budget_min', 50))
        budget_max = float(request_data.get('budget_max', 200))
        occasion = request_data.get('occasion', 'birthday')
        count = min(int(request_data.get('count', 3)), 5)  # Max 5 recommendations
        
        logger.info(f"🎁 Generating {count} catalog recommendations for {limbic_type}")
        
        # Generate recommendations - Try AI first, then fallback
        try:
            # Try AI recommendations first
            recommendations = generate_ai_recommendations(
                limbic_type, personality_traits, budget_min, budget_max, occasion, count
            )
            source = 'ai_recommendations'
        except Exception as e:
            logger.error(f"❌ AI recommendations failed: {e}")
            raise Exception(f"❌ REAL AI ENGINE FAILED: {e}")
        
        response_data = {
            'recommendations': recommendations,
            'total_count': len(recommendations),
            'limbic_type': limbic_type,
            'budget_range': f'€{budget_min:.0f}-{budget_max:.0f}',
            'source': 'ai_recommendations',
            'generated_at': datetime.utcnow().isoformat()
        }
        
        return api_response(
            data=response_data,
            message=f'Generated {len(recommendations)} AI recommendations',
            status=200
        )
        
    except Exception as e:
        logger.error(f"❌ Catalog recommendations failed: {e}")
        return api_response(
            error='Catalog recommendations failed',
            status=500,
            data={'details': str(e)}
        )


# ================================================================
# 🤖 ADVANCED CATALOG SERVICE (mit AI-Engine)
# ================================================================

def generate_catalog_recommendations_advanced(limbic_type: str, traits: list, 
                                            budget_min: float, budget_max: float, 
                                            occasion: str, count: int):
    """🚀 FORCE REAL AI - Keine Mock-Catalogs mehr"""
    raise Exception("❌ REAL AI ENGINE REQUIRED - No mock catalogs allowed")


def generate_catalog_recommendations_fallback(limbic_type: str, traits: list, 
                                            budget_min: float, budget_max: float, 
                                            occasion: str, count: int):
    """🚫 FORCE REAL AI - Keine Mock-Fallbacks mehr"""
    raise Exception("❌ REAL AI ENGINE REQUIRED - No mock catalog fallbacks allowed")


def analyze_prompt_context(user_prompt: str) -> dict:
    """Analyze prompt context for better recommendations"""
    
    context_indicators = {
        'urgency': ['schnell', 'sofort', 'last minute', 'morgen', 'heute'],
        'emotional_state': ['traurig', 'glücklich', 'aufgeregt', 'nervös', 'entspannt'],
        'relationship_closeness': ['beste', 'eng', 'nah', 'vertraut', 'innig'],
        'occasion_hints': ['geburtstag', 'weihnachten', 'hochzeit', 'abschluss', 'erfolg'],
        'personality_hints': ['kreativ', 'sportlich', 'technisch', 'ruhig', 'aktiv']
    }
    
    prompt_lower = user_prompt.lower()
    analysis = {}
    
    for category, keywords in context_indicators.items():
        matches = [keyword for keyword in keywords if keyword in prompt_lower]
        analysis[category] = matches
    
    return analysis


def extract_categories_from_prompt(user_prompt: str) -> list:
    """Extract gift categories from user prompt"""
    
    category_keywords = {
        'tech': ['computer', 'gadget', 'smartphone', 'tablet', 'gaming', 'app', 'digital', 'elektronik'],
        'creative': ['art', 'kunst', 'craft', 'kreativ', 'design', 'musik', 'malen', 'basteln'],
        'experience': ['erlebnis', 'reise', 'abenteuer', 'trip', 'konzert', 'event', 'aktivität'],
        'wellness': ['wellness', 'gesundheit', 'fitness', 'spa', 'entspannung', 'yoga', 'meditation'],
        'luxury': ['luxus', 'premium', 'teuer', 'hochwertig', 'designer', 'exklusiv'],
        'books': ['buch', 'lesen', 'literatur', 'roman', 'autor', 'geschichte'],
        'food': ['essen', 'kochen', 'rezept', 'küche', 'restaurant', 'kaffee', 'wein'],
        'home': ['zuhause', 'deko', 'möbel', 'garten', 'pflanze', 'wohnen'],
        'personal': ['persönlich', 'individuell', 'name', 'foto', 'erinnerung', 'sentimental'],
        'sport': ['sport', 'fitness', 'training', 'laufen', 'fahrrad', 'outdoor']
    }
    
    prompt_lower = user_prompt.lower()
    detected_categories = []
    
    for category, keywords in category_keywords.items():
        if any(keyword in prompt_lower for keyword in keywords):
            detected_categories.append(category)
    
    # Default categories if none detected
    if not detected_categories:
        detected_categories = ['personal', 'thoughtful', 'meaningful']
    
    return detected_categories[:4]  # Max 4 categories


def calculate_prompt_confidence(user_prompt: str, recommendations: list) -> float:
    """Calculate confidence score for prompt-based recommendations"""
    
    base_confidence = 0.6
    
    # Increase confidence based on prompt detail
    if len(user_prompt) > 100:
        base_confidence += 0.1
    if len(user_prompt) > 200:
        base_confidence += 0.1
    
    # Increase confidence based on specific details
    specific_indicators = ['liebt', 'hasst', 'sammelt', 'träumt', 'wünscht']
    specificity_score = sum(1 for indicator in specific_indicators if indicator in user_prompt.lower())
    base_confidence += min(specificity_score * 0.05, 0.15)
    
    # Increase confidence based on number of recommendations
    if len(recommendations) >= 3:
        base_confidence += 0.1
    
    return min(base_confidence, 0.95)


# ================================================================
# 🔍 ADDITIONAL UTILITY FUNCTIONS
# ================================================================

@csrf.exempt
@catalog_bp.route('/health', methods=['GET'])
@cross_origin()
def catalog_health():
    """❤️ Catalog Service Health Check"""
    
    try:
        health_data = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'catalog_service': {
                'available': CATALOG_SERVICE_AVAILABLE,
                'status': 'healthy' if CATALOG_SERVICE_AVAILABLE else 'fallback_mode'
            },
            'endpoints': {
                '/api/catalog/recommendations': 'active',
                '/api/catalog/health': 'active'
            },
            'fallback_categories': {
                'adventurer': 3,
                'performer': 3,
                'harmonizer': 3,
                'disciplined': 3,
                'traditionalist': 3,
                'hedonist': 3
            }
        }
        
        return api_response(data=health_data, message='Catalog Service is healthy')
        
    except Exception as e:
        logger.error(f"❌ Catalog health check failed: {e}")
        return api_response(error='Catalog health check failed', status=503)


@csrf.exempt
@catalog_bp.route('/limbic-types', methods=['GET'])
@cross_origin()
def get_limbic_types():
    """📋 Get available limbic types and their characteristics"""
    
    limbic_types = {
        'adventurer': {
            'name': 'Adventurer',
            'description': 'Aufgeschlossen, risikobereit, liebt neue Erfahrungen',
            'traits': ['high_openness', 'high_extraversion'],
            'gift_focus': ['outdoor', 'experience', 'tech']
        },
        'performer': {
            'name': 'Performer', 
            'description': 'Selbstbewusst, zielstrebig, liebt Aufmerksamkeit',
            'traits': ['high_extraversion', 'high_conscientiousness'],
            'gift_focus': ['tech', 'style', 'experience']
        },
        'harmonizer': {
            'name': 'Harmonizer',
            'description': 'Ausgewogen, empathisch, sucht Balance',
            'traits': ['high_agreeableness', 'balanced'],
            'gift_focus': ['wellness', 'home', 'personal']
        },
        'disciplined': {
            'name': 'Disciplined',
            'description': 'Organisiert, zuverlässig, strukturiert',
            'traits': ['high_conscientiousness', 'practical'],
            'gift_focus': ['productivity', 'office', 'tech']
        },
        'traditionalist': {
            'name': 'Traditionalist',
            'description': 'Bodenständig, wertorientiert, beständig',
            'traits': ['high_agreeableness', 'conservative'],
            'gift_focus': ['books', 'craft', 'classic']
        },
        'hedonist': {
            'name': 'Hedonist',
            'description': 'Genussorientiert, spontan, sinnlich',
            'traits': ['high_extraversion', 'sensation_seeking'],
            'gift_focus': ['food', 'luxury', 'sensory']
        }
    }
    
    return api_response(
        data=limbic_types,
        message='Available limbic types retrieved successfully'
    )


# ================================================================
# 🚀 EXPORTS
# ================================================================

__all__ = ['catalog_bp']