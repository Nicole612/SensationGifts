# ==========================================
# app/routes/gifts.py - EINFACHE VERSION
# ==========================================
"""
Gift API Routes - Flask Version (Ohne Service Dependencies)
==========================================================

🎁 Gift Database API Routes

Features:
- Gift Search & Filtering
- Category Management
- Gift Details & Recommendations
- NUTZT DEINE ECHTEN MODELS direkt
"""

from flask import Blueprint, request, jsonify, stream_with_context, Response
from flask_cors import cross_origin
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from app.services.recommendation_service import get_recommendations_stream
# Import deine Models direkt
from app.models import (
    Gift, GiftCategory, GiftTag, GiftTagAssociation,
    PriceRange, GiftType, PersonalizationLevel, PersonalityProfile,
    get_gifts_by_personality, get_gifts_by_category, 
    get_gifts_by_tags, search_gifts, get_featured_gifts
)
from app.extensions import db

# Logger setup
logger = logging.getLogger(__name__)

# ✅ KORREKTER BLUEPRINT NAME
gifts_bp = Blueprint('gifts', __name__)

# =============================================================================
# BASIC GIFT ROUTES
# =============================================================================

@gifts_bp.route('/test', methods=['GET'])
@cross_origin()
def gifts_test():
    """🧪 Gifts API Test"""
    return jsonify({
        "success": True,
        "message": "Gifts API is working!",
        "features": [
            "Gift Search & Filtering",
            "Category Management", 
            "Price Analysis",
            "Gift Details & Recommendations"
        ],
        "endpoints": [
            "/test", "/search", "/categories", "/gift/<gift_id>", 
            "/by-category/<category_name>", "/featured", "/analytics"
        ],
        "timestamp": datetime.now().isoformat()
    }), 200


@gifts_bp.route('/')
@cross_origin()
def gifts_root():
    """🎁 ROOT GIFTS ENDPOINT - /api/gifts"""
    try:
        # Get quick stats
        total_gifts = Gift.query.filter_by(is_active=True).count()
        total_categories = GiftCategory.query.filter_by(is_active=True).count()
        featured_count = Gift.query.filter_by(is_active=True, is_featured=True).count()
        
        return jsonify({
            "success": True,
            "message": "Gifts API is working",
            "api_info": {
                "total_gifts": total_gifts,
                "total_categories": total_categories,
                "featured_gifts": featured_count
            },
            "available_endpoints": [
                "/search - Gift search with filters",
                "/categories - All gift categories", 
                "/featured - Featured gifts",
                "/gift/<id> - Gift details",
                "/by-category/<name> - Gifts by category",
                "/analytics - Gift statistics",
                "/recommendations/<user_id> - Personalized recommendations"
            ],
            "quick_access": {
                "search_url": "/api/gifts/search",
                "categories_url": "/api/gifts/categories",
                "featured_url": "/api/gifts/featured"
            },
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Gifts root endpoint error: {e}")
        return jsonify({
            "success": False,
            "error": "Gifts API temporarily unavailable",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


@gifts_bp.route('/search', methods=['GET', 'POST'])
@cross_origin()
def search_gifts_endpoint():
    """
    🔍 GIFT SEARCH API
    
    Advanced Gift Search mit Filtering - NUTZT DEINE ECHTEN MODELS!
    """
    
    try:
        # Get search parameters
        if request.method == 'GET':
            search_term = request.args.get('q') or request.args.get('search_term')
            limit = int(request.args.get('limit', 20))
            budget_min = float(request.args.get('budget_min')) if request.args.get('budget_min') else None
            budget_max = float(request.args.get('budget_max')) if request.args.get('budget_max') else None
            categories = request.args.getlist('categories')
        else:
            # POST with JSON body
            data = request.get_json() or {}
            search_term = data.get('search_term')
            limit = int(data.get('limit', 20))
            budget_min = data.get('budget_min')
            budget_max = data.get('budget_max')
            categories = data.get('categories', [])
        
        # ✅ NUTZE DEINE ECHTEN MODEL-FUNKTIONEN!
        if search_term:
            gifts = search_gifts(search_term, limit=limit)
        else:
            gifts = Gift.query.filter_by(is_active=True).limit(limit).all()
        
        # Filter by budget
        if budget_min is not None or budget_max is not None:
            filtered_gifts = []
            for gift in gifts:
                if budget_min and gift.price < budget_min:
                    continue
                if budget_max and gift.price > budget_max:
                    continue
                filtered_gifts.append(gift)
            gifts = filtered_gifts
        
        # Filter by categories
        if categories:
            category_gifts = []
            for category_name in categories:
                cat_gifts = get_gifts_by_category(category_name, limit=limit)
                category_gifts.extend(cat_gifts)
            
            # Intersection with search results
            if gifts:
                gift_ids = {gift.id for gift in gifts}
                gifts = [gift for gift in category_gifts if gift.id in gift_ids]
            else:
                gifts = category_gifts
        
        # Format Results
        formatted_gifts = []
        for gift in gifts[:limit]:
            formatted_gifts.append({
                "id": gift.id,
                "name": gift.name,
                "short_description": gift.short_description,
                "price": gift.price,
                "display_price": gift.display_price,
                "category": {
                    "id": gift.category.id if gift.category else None,
                    "name": gift.category.name if gift.category else "Unknown"
                },
                "gift_type": gift.gift_type.value if gift.gift_type else None,
                "personalization_level": gift.personalization_level.value if gift.personalization_level else None,
                "quality_score": gift.quality_score,
                "popularity_score": gift.popularity_score,
                "is_featured": gift.is_featured,
                "created_at": gift.created_at.isoformat() if gift.created_at else None
            })
        
        return jsonify({
            "success": True,
            "search_results": {
                "gifts": formatted_gifts,
                "total_found": len(formatted_gifts),
                "search_params": {
                    "search_term": search_term,
                    "budget_min": budget_min,
                    "budget_max": budget_max,
                    "categories": categories,
                    "limit": limit
                },
                "search_timestamp": datetime.now().isoformat()
            }
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Gift search failed: {e}")
        return jsonify({
            "success": False,
            "error": "Gift search failed",
            "details": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@gifts_bp.route('/gift/<gift_id>', methods=['GET'])
@cross_origin()
def get_gift_details(gift_id: str):
    """
    📋 GET GIFT DETAILS
    
    Lädt detaillierte Informationen zu einem spezifischen Gift
    """
    
    try:
        # ✅ NUTZE DEINE ECHTEN MODELS!
        gift = Gift.get_by_id(gift_id)
        
        if not gift:
            return jsonify({
                "success": False,
                "error": "Gift not found",
                "error_code": "GIFT_NOT_FOUND"
            }), 404
        
        # Format Detailed Response
        gift_details = {
            "id": gift.id,
            "name": gift.name,
            "short_description": gift.short_description,
            "long_description": gift.long_description,
            "price": gift.price,
            "display_price": gift.display_price,
            "price_range": gift.price_range.value if gift.price_range else None,
            "currency": gift.currency,
            "gift_type": gift.gift_type.value if gift.gift_type else None,
            "personalization_level": gift.personalization_level.value if gift.personalization_level else None,
            "category": {
                "id": gift.category.id if gift.category else None,
                "name": gift.category.name if gift.category else "Unknown",
                "description": gift.category.description if gift.category else None
            },
            "tags": [
                {
                    "name": assoc.tag.name,
                    "display_name": assoc.tag.display_name,
                    "relevance": assoc.relevance_score
                }
                for assoc in gift.tag_associations if assoc.tag
            ],
            "targeting": {
                "target_age_min": gift.target_age_min,
                "target_age_max": gift.target_age_max,
                "target_gender": gift.target_gender
            },
            "media": {
                "image_url": gift.image_url,
                "video_url": gift.video_url
            },
            "ai_scores": {
                "personality_match_scores": gift.personality_match_dict,
                "occasion_suitability": gift.occasion_suitability_dict,
                "relationship_suitability": gift.relationship_suitability_dict
            },
            "purchase_info": {
                "purchase_links": gift.purchase_links_list,
                "availability_status": gift.availability_status,
                "delivery_time_days": gift.delivery_time_days
            },
            "quality_metrics": {
                "quality_score": gift.quality_score,
                "popularity_score": gift.popularity_score,
                "success_rate": gift.success_rate
            },
            "status": {
                "is_active": gift.is_active,
                "is_featured": gift.is_featured,
                "is_seasonal": gift.is_seasonal
            },
            "metadata": {
                "created_at": gift.created_at.isoformat() if gift.created_at else None,
                "updated_at": gift.updated_at.isoformat() if gift.updated_at else None
            }
        }
        
        return jsonify({
            "success": True,
            "gift": gift_details,
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Failed to load gift {gift_id}: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to load gift details",
            "details": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@gifts_bp.route('/by-category/<category_name>', methods=['GET'])
@cross_origin()
def get_gifts_by_category_endpoint(category_name: str):
    """
    📂 GET GIFTS BY CATEGORY
    
    Lädt Gifts gefiltert nach Category - NUTZT DEINE ECHTEN MODELS!
    """
    
    try:
        # Get optional parameters
        limit = int(request.args.get('limit', 50))
        budget_min = float(request.args.get('budget_min')) if request.args.get('budget_min') else None
        budget_max = float(request.args.get('budget_max')) if request.args.get('budget_max') else None
        
        # ✅ NUTZE DEINE ECHTEN MODEL-FUNKTIONEN!
        gifts = get_gifts_by_category(category_name, limit=limit)
        
        # Filter by budget if specified
        if budget_min is not None or budget_max is not None:
            filtered_gifts = []
            for gift in gifts:
                if budget_min and gift.price < budget_min:
                    continue
                if budget_max and gift.price > budget_max:
                    continue
                filtered_gifts.append(gift)
            gifts = filtered_gifts
        
        # Format Results
        formatted_gifts = []
        for gift in gifts:
            formatted_gifts.append({
                "id": gift.id,
                "name": gift.name,
                "short_description": gift.short_description,
                "price": gift.price,
                "display_price": gift.display_price,
                "quality_score": gift.quality_score,
                "popularity_score": gift.popularity_score,
                "is_featured": gift.is_featured
            })
        
        return jsonify({
            "success": True,
            "category_results": {
                "category": category_name,
                "gifts": formatted_gifts,
                "total_found": len(formatted_gifts),
                "filters_applied": {
                    "budget_min": budget_min,
                    "budget_max": budget_max,
                    "limit": limit
                }
            },
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Failed to load gifts for category {category_name}: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to load gifts by category",
            "details": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@gifts_bp.route('/categories', methods=['GET'])
@cross_origin()
def get_all_categories():
    """
    📂 GET ALL CATEGORIES
    
    Lädt alle verfügbaren Gift Categories - NUTZT DEINE ECHTEN MODELS!
    """
    
    try:
        # ✅ NUTZE DEINE ECHTEN MODELS!
        categories = GiftCategory.query.filter_by(is_active=True).order_by(GiftCategory.sort_order).all()
        
        # Format Results
        formatted_categories = []
        for category in categories:
            formatted_categories.append({
                "id": category.id,
                "name": category.name,
                "slug": category.slug,
                "description": category.description,
                "icon": category.icon,
                "color": category.color,
                "image_url": category.image_url,
                "target_traits": category.target_traits_list,
                "target_occasions": category.target_occasions_list,
                "target_relationships": category.target_relationships_list,
                "gift_count": category.gift_count,
                "average_price": category.average_price,
                "sort_order": category.sort_order,
                "created_at": category.created_at.isoformat() if category.created_at else None
            })
        
        return jsonify({
            "success": True,
            "categories": formatted_categories,
            "total_categories": len(formatted_categories),
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Categories loading failed: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to load categories",
            "details": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@gifts_bp.route('/featured', methods=['GET'])
@cross_origin()
def get_featured_gifts_endpoint():
    """
    🏆 GET FEATURED GIFTS
    
    Lädt Featured/Promoted Gifts - NUTZT DEINE ECHTEN MODELS!
    """
    
    try:
        limit = int(request.args.get('limit', 6))
        
        # ✅ NUTZE DEINE ECHTEN MODEL-FUNKTIONEN!
        featured_gifts = get_featured_gifts(limit=limit)
        
        # Format Results
        formatted_gifts = []
        for gift in featured_gifts:
            formatted_gifts.append({
                "id": gift.id,
                "name": gift.name,
                "short_description": gift.short_description,
                "price": gift.price,
                "display_price": gift.display_price,
                "category": gift.category.name if gift.category else "Unknown",
                "quality_score": gift.quality_score,
                "popularity_score": gift.popularity_score,
                "image_url": gift.image_url,
                "is_featured": gift.is_featured
            })
        
        return jsonify({
            "success": True,
            "featured_gifts": {
                "gifts": formatted_gifts,
                "total_found": len(formatted_gifts)
            },
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Featured gifts loading failed: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to load featured gifts",
            "details": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@gifts_bp.route('/analytics', methods=['GET'])
@cross_origin()
def get_gift_analytics():
    """
    📊 GIFT ANALYTICS
    
    Basic Gift Analytics - NUTZT DEINE ECHTEN MODELS!
    """
    
    try:
        # ✅ NUTZE DEINE ECHTEN MODELS für Analytics!
        total_gifts = Gift.query.filter_by(is_active=True).count()
        total_categories = GiftCategory.query.filter_by(is_active=True).count()
        featured_count = Gift.query.filter_by(is_active=True, is_featured=True).count()
        
        # Price Analytics
        gifts_with_prices = Gift.query.filter(Gift.is_active == True, Gift.price > 0).all()
        
        if gifts_with_prices:
            prices = [gift.price for gift in gifts_with_prices]
            avg_price = sum(prices) / len(prices)
            min_price = min(prices)
            max_price = max(prices)
        else:
            avg_price = min_price = max_price = 0
        
        # Price Range Distribution
        price_ranges = {}
        for gift in gifts_with_prices:
            range_name = gift.price_range.value if gift.price_range else 'unknown'
            price_ranges[range_name] = price_ranges.get(range_name, 0) + 1
        
        # Category Distribution
        category_distribution = {}
        for category in GiftCategory.query.filter_by(is_active=True).all():
            category_distribution[category.name] = category.gift_count
        
        analytics = {
            "overview": {
                "total_gifts": total_gifts,
                "total_categories": total_categories,
                "featured_gifts": featured_count,
                "active_gifts_percentage": 100.0  # Alle abgefragten Gifts sind aktiv
            },
            "pricing": {
                "average_price": round(avg_price, 2),
                "min_price": min_price,
                "max_price": max_price,
                "price_range_distribution": price_ranges
            },
            "categories": {
                "distribution": category_distribution,
                "most_popular": max(category_distribution.items(), key=lambda x: x[1]) if category_distribution else None
            },
            "metadata": {
                "analysis_timestamp": datetime.now().isoformat(),
                "data_freshness": "real-time"
            }
        }
        
        return jsonify({
            "success": True,
            "analytics": analytics
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Gift analytics failed: {e}")
        return jsonify({
            "success": False,
            "error": "Analytics generation failed",
            "details": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@gifts_bp.route('/recommendations/<user_id>', methods=['POST'])
@cross_origin()
def get_personality_based_recommendations(user_id: str):
    """
    🧠 PERSONALITY-BASED GIFT RECOMMENDATIONS
    
    Empfehlungen basierend auf Personality Profile - NUTZT DEINE ECHTEN MODELS!
    """
    
    try:
        # Load Personality Profile
        from app.models import PersonalityProfile
        
        profile = PersonalityProfile.query.filter_by(buyer_user_id=user_id).first()
        
        if not profile:
            return jsonify({
                "success": False,
                "error": "Personality profile not found",
                "error_code": "PROFILE_NOT_FOUND"
            }), 404
        
        # ✅ NUTZE DEINE ECHTEN MODEL-FUNKTIONEN!
        recommended_gifts = get_gifts_by_personality(profile, limit=10, min_score=0.3)
        
        # Format Results with Match Scores
        formatted_gifts = []
        for gift in recommended_gifts:
            match_score = gift.calculate_match_score(profile)
            
            formatted_gifts.append({
                "id": gift.id,
                "name": gift.name,
                "short_description": gift.short_description,
                "price": gift.price,
                "display_price": gift.display_price,
                "category": gift.category.name if gift.category else "Unknown",
                "match_score": round(match_score, 2),
                "quality_score": gift.quality_score,
                "popularity_score": gift.popularity_score,
                "image_url": gift.image_url,
                "personality_reasoning": f"Passt gut zu {profile.personality_summary}"
            })
        
        # Sort by match score
        formatted_gifts.sort(key=lambda x: x['match_score'], reverse=True)
        
        return jsonify({
            "success": True,
            "personality_recommendations": {
                "user_id": user_id,
                "profile_id": profile.id,
                "recipient_name": profile.recipient_name,
                "occasion": profile.occasion,
                "budget_range": f"{profile.budget_min}-{profile.budget_max}",
                "gifts": formatted_gifts,
                "total_found": len(formatted_gifts),
                "personality_summary": profile.personality_summary
            },
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Personality-based recommendations failed for user {user_id}: {e}")
        return jsonify({
            "success": False,
            "error": "Personality recommendations failed",
            "details": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


@gifts_bp.route('/recommendations/stream/<int:user_id>', methods=['POST'])
async def stream_recommendations(user_id):
    profile = PersonalityProfile.query.filter_by(user_id=user_id).first()
    context = request.json.get("context", {})

    async def stream():
        async for chunk in get_recommendations_stream(profile, context):
            yield f"data: {chunk.json()}\n\n"

    return Response(stream_with_context(stream()), mimetype='text/event-stream')


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@gifts_bp.errorhandler(404)
def gifts_not_found(error):
    """Gifts API 404 Handler"""
    return jsonify({
        "success": False,
        "error": "Gifts endpoint not found",
        "available_endpoints": [
            "/test", "/search", "/gift/<gift_id>", "/by-category/<category_name>", 
            "/categories", "/featured", "/analytics", "/recommendations/<user_id>"
        ]
    }), 404

@gifts_bp.errorhandler(405)
def gifts_method_not_allowed(error):
    """Gifts API 405 Handler"""
    return jsonify({
        "success": False,
        "error": "HTTP method not allowed for this gifts endpoint"
    }), 405

@gifts_bp.errorhandler(500)
def gifts_server_error(error):
    """Gifts API 500 Handler"""
    logger.error(f"Gifts API server error: {error}")
    return jsonify({
        "success": False,
        "error": "Internal server error in gifts API"
    }), 500

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ['gifts_bp']