# ================================================================
# 📁 app/routes/catalog_classic.py - KLASSISCHER KATALOG ENDPOINT
# 🎯 Fallback für dein bestehendes gift_catalog_generator System
# ================================================================

from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
import logging
from datetime import datetime

from app.extensions import csrf
from app.utils.formatters import api_response

# Import der bestehenden Katalog-Services
try:
    from ai_engine.catalog.gift_catalog_generator import (
        GeschenkKatalogService, 
        GESCHENK_KATALOG,
        get_products_by_age,
        get_products_by_personality,
        get_products_by_relationship
    )
    CLASSIC_CATALOG_AVAILABLE = True
except ImportError:
    CLASSIC_CATALOG_AVAILABLE = False

logger = logging.getLogger(__name__)

# Blueprint Setup
classic_catalog_bp = Blueprint('classic_catalog', __name__, url_prefix='/api/catalog')

# ================================================================
# 🎯 KLASSISCHER KATALOG ENDPOINT
# ================================================================

@csrf.exempt
@classic_catalog_bp.route('/classic-recommendations', methods=['POST', 'OPTIONS'])
@cross_origin(origins="*", methods=['POST', 'OPTIONS'], allow_headers=['Content-Type'])
def get_classic_catalog_recommendations():
    """
    🎁 KLASSISCHER KATALOG ENDPOINT für dein gift_catalog_generator System
    
    POST /api/catalog/classic-recommendations
    {
        "method": "catalog_classic",
        "limbic_type": "harmonizer",
        "age": 30,
        "budget_range": [50, 200],
        "personality_traits": ["balanced", "creative"],
        "relationship": "friend",
        "occasion": "birthday"
    }
    """
    
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        if not CLASSIC_CATALOG_AVAILABLE:
            return api_response(
                error='Classic catalog service not available',
                status=503,
                data={'fallback_available': True}
            )
        
        request_data = request.get_json()
        if not request_data:
            return api_response(error='No request data provided', status=400)
        
        # Extract parameters
        limbic_type = request_data.get('limbic_type', 'harmonizer')
        age = int(request_data.get('age', 30))
        budget_range = request_data.get('budget_range', [50, 200])
        personality_traits = request_data.get('personality_traits', ['balanced'])
        relationship = request_data.get('relationship', 'friend')
        occasion = request_data.get('occasion', 'birthday')
        
        budget_min, budget_max = budget_range[0], budget_range[1]
        
        logger.info(f"🎁 Classic catalog request: {limbic_type}, age {age}, budget €{budget_min}-{budget_max}")
        
        # Initialize classic catalog service
        catalog_service = GeschenkKatalogService()
        
        # Create mock personality profile for classic system
        class MockClassicProfile:
            def __init__(self):
                self.age = age
                self.budget_min = budget_min
                self.budget_max = budget_max
                self.occasion = occasion
                self.relationship = relationship
                
                # Map limbic type to Big Five (simplified)
                limbic_mapping = {
                    'adventurer': {'openness': 0.8, 'extraversion': 0.8, 'conscientiousness': 0.6, 'agreeableness': 0.7, 'neuroticism': 0.3},
                    'performer': {'extraversion': 0.9, 'conscientiousness': 0.8, 'openness': 0.7, 'agreeableness': 0.6, 'neuroticism': 0.4},
                    'harmonizer': {'agreeableness': 0.9, 'conscientiousness': 0.7, 'openness': 0.6, 'extraversion': 0.5, 'neuroticism': 0.3},
                    'disciplined': {'conscientiousness': 0.9, 'agreeableness': 0.7, 'neuroticism': 0.2, 'openness': 0.5, 'extraversion': 0.5},
                    'traditionalist': {'conscientiousness': 0.8, 'agreeableness': 0.8, 'neuroticism': 0.3, 'openness': 0.4, 'extraversion': 0.4},
                    'hedonist': {'extraversion': 0.8, 'openness': 0.7, 'neuroticism': 0.6, 'conscientiousness': 0.4, 'agreeableness': 0.6}
                }
                
                big_five = limbic_mapping.get(limbic_type, limbic_mapping['harmonizer'])
                
                self.openness = big_five['openness']
                self.conscientiousness = big_five['conscientiousness']
                self.extraversion = big_five['extraversion']
                self.agreeableness = big_five['agreeableness']
                self.neuroticism = big_five['neuroticism']
                
                # Additional traits based on personality_traits
                self.creative_type = 'creative' in personality_traits
                self.tech_savvy = 'tech' in personality_traits
                self.practical_type = 'practical' in personality_traits
                self.health_conscious = 'wellness' in personality_traits
        
        mock_profile = MockClassicProfile()
        
        # Get recommendations from classic catalog
        classic_recommendations = catalog_service.get_ai_recommendations(
            personality_profile=mock_profile,
            budget_range=(budget_min, budget_max),
            relationship=relationship,
            occasion=occasion
        )
        
        # Convert to standardized format
        standardized_recommendations = []
        
        for rec in classic_recommendations[:3]:  # Limit to 3
            product_data = rec.get('product_data', {})
            
            standardized_rec = {
                'id': rec.get('product_id', f'classic_{hash(product_data.get("name", ""))}'),
                'name': product_data.get('name', 'Classic Catalog Item'),
                'description': product_data.get('short_description', 'Premium gift from classic catalog'),
                'price_range': f'€{rec.get("recommended_price", budget_min):.0f}',
                'category': product_data.get('category', 'classic'),
                'match_score': rec.get('total_match_score', 0.75),
                'reasoning': rec.get('ai_reasoning', f'Klassische Empfehlung für {limbic_type}'),
                'source': 'classic_catalog',
                'tags': [limbic_type] + personality_traits,
                'rating': 4.3,
                'availability': 'in_stock',
                
                # Additional classic catalog specific data
                'emotional_story': product_data.get('emotional_story', ''),
                'content_components': product_data.get('content_components', {}),
                'price_variants': product_data.get('price_variants', {}),
                'recommended_variant': rec.get('recommended_variant', 'basic'),
                'ki_empfehlung': product_data.get('ki_empfehlung', ''),
                
                # Purchase info
                'purchase_url': f'/gifts/classic/{rec.get("product_id")}',
                'personalization_url': f'/personalize/{rec.get("product_id")}'
            }
            
            standardized_recommendations.append(standardized_rec)
        
        response_data = {
            'recommendations': standardized_recommendations,
            'total_count': len(standardized_recommendations),
            'catalog_source': 'classic_geschenk_katalog',
            'limbic_type': limbic_type,
            'age_group': f'{age} years',
            'budget_range': f'€{budget_min}-{budget_max}',
            'total_catalog_size': len(GESCHENK_KATALOG),
            'generated_at': datetime.utcnow().isoformat()
        }
        
        return api_response(
            data=response_data,
            message=f'Generated {len(standardized_recommendations)} classic catalog recommendations',
            status=200
        )
        
    except Exception as e:
        logger.error(f"❌ Classic catalog recommendations failed: {e}")
        return api_response(
            error='Classic catalog recommendations failed',
            status=500,
            data={'details': str(e), 'fallback_suggestions': get_emergency_fallback(limbic_type, budget_min, budget_max)}
        )


# ================================================================
# 🔧 HELPER FUNCTIONS
# ================================================================

def get_emergency_fallback(limbic_type: str, budget_min: float, budget_max: float):
    """Emergency fallback when everything else fails"""
    
    emergency_gifts = {
        'adventurer': [
            {'name': 'Adventure Starter Kit', 'price': 49.99, 'category': 'outdoor'},
            {'name': 'Experience Voucher', 'price': 75.00, 'category': 'experience'},
            {'name': 'Travel Gear Set', 'price': 89.99, 'category': 'travel'}
        ],
        'performer': [
            {'name': 'Performance Enhancement Kit', 'price': 69.99, 'category': 'lifestyle'},
            {'name': 'Style Upgrade Set', 'price': 89.99, 'category': 'fashion'},
            {'name': 'Tech Accessories', 'price': 99.99, 'category': 'tech'}
        ],
        'harmonizer': [
            {'name': 'Wellness & Balance Set', 'price': 59.99, 'category': 'wellness'},
            {'name': 'Home Harmony Kit', 'price': 69.99, 'category': 'home'},
            {'name': 'Mindfulness Package', 'price': 39.99, 'category': 'personal'}
        ],
        'disciplined': [
            {'name': 'Organization Master Kit', 'price': 79.99, 'category': 'productivity'},
            {'name': 'Professional Upgrade', 'price': 99.99, 'category': 'office'},
            {'name': 'Efficiency Tools', 'price': 89.99, 'category': 'tools'}
        ],
        'traditionalist': [
            {'name': 'Classic Collection', 'price': 69.99, 'category': 'classic'},
            {'name': 'Heritage Set', 'price': 89.99, 'category': 'traditional'},
            {'name': 'Quality Essentials', 'price': 79.99, 'category': 'premium'}
        ],
        'hedonist': [
            {'name': 'Luxury Experience', 'price': 99.99, 'category': 'luxury'},
            {'name': 'Sensory Delight Package', 'price': 79.99, 'category': 'sensory'},
            {'name': 'Gourmet Collection', 'price': 89.99, 'category': 'food'}
        ]
    }
    
    gifts = emergency_gifts.get(limbic_type, emergency_gifts['harmonizer'])
    
    # Filter by budget
    affordable_gifts = [
        gift for gift in gifts 
        if budget_min <= gift['price'] <= budget_max
    ]
    
    if not affordable_gifts:
        # Adjust prices to fit budget
        affordable_gifts = [
            {
                **gift,
                'price': min(max(gift['price'] * 0.8, budget_min), budget_max)
            }
            for gift in gifts
        ]
    
    return affordable_gifts[:3]


# ================================================================
# 🔍 CATALOG BROWSE ENDPOINTS
# ================================================================

@csrf.exempt
@classic_catalog_bp.route('/browse', methods=['GET'])
@cross_origin()
def browse_classic_catalog():
    """Browse the classic catalog by category, age, etc."""
    
    if not CLASSIC_CATALOG_AVAILABLE:
        return api_response(error='Classic catalog not available', status=503)
    
    try:
        # Query parameters
        age = request.args.get('age', type=int)
        category = request.args.get('category')
        personality_traits = request.args.getlist('traits')
        relationship = request.args.get('relationship')
        
        results = []
        
        # Filter by age if provided
        if age:
            age_filtered = get_products_by_age(age)
            results.extend(age_filtered)
        
        # Filter by personality traits if provided
        if personality_traits:
            personality_filtered = get_products_by_personality(personality_traits)
            results.extend(personality_filtered)
        
        # Filter by relationship if provided
        if relationship:
            relationship_filtered = get_products_by_relationship(relationship)
            results.extend(relationship_filtered)
        
        # If no filters, return all products
        if not results:
            results = [
                {'id': k, **v} for k, v in GESCHENK_KATALOG.items()
            ]
        
        # Remove duplicates and limit
        seen_ids = set()
        unique_results = []
        for item in results:
            if item['id'] not in seen_ids:
                seen_ids.add(item['id'])
                unique_results.append(item)
        
        return api_response(
            data={
                'products': unique_results[:20],  # Limit to 20
                'total_catalog_size': len(GESCHENK_KATALOG),
                'filters_applied': {
                    'age': age,
                    'category': category,
                    'personality_traits': personality_traits,
                    'relationship': relationship
                }
            },
            message=f'Found {len(unique_results)} products'
        )
        
    except Exception as e:
        logger.error(f"❌ Catalog browse failed: {e}")
        return api_response(error='Catalog browse failed', status=500)


@csrf.exempt
@classic_catalog_bp.route('/product/<product_id>', methods=['GET'])
@cross_origin()
def get_classic_product_details(product_id: str):
    """Get detailed information about a specific classic catalog product"""
    
    if not CLASSIC_CATALOG_AVAILABLE:
        return api_response(error='Classic catalog not available', status=503)
    
    try:
        if product_id not in GESCHENK_KATALOG:
            return api_response(error='Product not found', status=404)
        
        product = GESCHENK_KATALOG[product_id]
        
        # Enhance product data
        enhanced_product = {
            'id': product_id,
            **product,
            'personalization_options': {
                'name_engraving': True,
                'custom_message': True,
                'photo_upload': product.get('photo_personalization', False),
                'color_variants': product.get('color_options', [])
            },
            'shipping_info': {
                'standard_shipping': '3-5 Werktage',
                'express_shipping': '1-2 Werktage',
                'personalization_time': '+2 Werktage'
            },
            'related_products': get_related_products(product_id, product)
        }
        
        return api_response(
            data=enhanced_product,
            message='Product details retrieved successfully'
        )
        
    except Exception as e:
        logger.error(f"❌ Product details failed: {e}")
        return api_response(error='Product details retrieval failed', status=500)


def get_related_products(current_product_id: str, current_product: dict, limit: int = 3):
    """Get related products based on category and personality match"""
    
    current_category = current_product.get('category', '')
    current_age_categories = current_product.get('age_categories', [])
    
    related = []
    
    for product_id, product in GESCHENK_KATALOG.items():
        if product_id == current_product_id:
            continue
        
        # Same category match
        if product.get('category') == current_category:
            related.append({
                'id': product_id,
                'name': product['name'],
                'price_range': f"€{min(v['price'] for v in product['price_variants'].values())}-{max(v['price'] for v in product['price_variants'].values())}",
                'match_reason': 'Same category'
            })
        
        # Age category overlap
        elif any(age in product.get('age_categories', []) for age in current_age_categories):
            related.append({
                'id': product_id,
                'name': product['name'],
                'price_range': f"€{min(v['price'] for v in product['price_variants'].values())}-{max(v['price'] for v in product['price_variants'].values())}",
                'match_reason': 'Similar age group'
            })
    
    return related[:limit]


# ================================================================
# 🔍 HEALTH CHECK
# ================================================================

@csrf.exempt
@classic_catalog_bp.route('/classic-health', methods=['GET'])
@cross_origin()
def classic_catalog_health():
    """Health check for classic catalog service"""
    
    try:
        health_data = {
            'status': 'healthy' if CLASSIC_CATALOG_AVAILABLE else 'unavailable',
            'classic_catalog_available': CLASSIC_CATALOG_AVAILABLE,
            'total_products': len(GESCHENK_KATALOG) if CLASSIC_CATALOG_AVAILABLE else 0,
            'endpoints': {
                '/api/catalog/classic-recommendations': 'active' if CLASSIC_CATALOG_AVAILABLE else 'disabled',
                '/api/catalog/browse': 'active' if CLASSIC_CATALOG_AVAILABLE else 'disabled',
                '/api/catalog/product/<id>': 'active' if CLASSIC_CATALOG_AVAILABLE else 'disabled'
            },
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if CLASSIC_CATALOG_AVAILABLE:
            # Additional stats
            categories = set()
            price_range = {'min': float('inf'), 'max': 0}
            
            for product in GESCHENK_KATALOG.values():
                categories.add(product.get('category', 'unknown'))
                for variant in product.get('price_variants', {}).values():
                    price_range['min'] = min(price_range['min'], variant['price'])
                    price_range['max'] = max(price_range['max'], variant['price'])
            
            health_data.update({
                'categories_available': len(categories),
                'price_range': f"€{price_range['min']:.0f}-{price_range['max']:.0f}",
                'categories': list(categories)
            })
        
        status_code = 200 if CLASSIC_CATALOG_AVAILABLE else 503
        
        return api_response(
            data=health_data,
            message='Classic catalog health check completed',
            status=status_code
        )
        
    except Exception as e:
        logger.error(f"❌ Classic catalog health check failed: {e}")
        return api_response(error='Classic catalog health check failed', status=503)


# ================================================================
# 🚀 EXPORTS
# ================================================================

__all__ = ['classic_catalog_bp']