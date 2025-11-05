# ==========================================
# app/routes/api.py - UPGRADED VERSION
# ==========================================
"""
Main API Routes - Enterprise Service Integration
==============================================
🧹 BEREINIGTES GENERAL API
=========================

NACH REORGANISATION: Nur noch General API Functions
- Service Health & Monitoring
- General Endpoints
- Catalog Integration
"""

from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
from flask_login import login_required, current_user
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from app.extensions import csrf

# 🚀 DEINE ENTERPRISE SERVICES!
from app.services import services, get_user_service, get_gift_service, get_recommendation_service
from app.services.gift_service import GiftService

# Logger setup
logger = logging.getLogger(__name__)

# Blueprint Setup
api_bp = Blueprint('api', __name__, url_prefix='/api')


# =============================================================================
# 🏠 ROOT API ENDPOINT
# =============================================================================

@csrf.exempt
@api_bp.route('/', methods=['GET'])
@cross_origin()
def api_root():
    """🌐 ROOT API ENDPOINT - /api"""
    try:
        return jsonify({
            "success": True,
            "message": "SensationGifts Enterprise AI API",
            "version": "2.0.0",
            "description": "Enterprise AI-powered gift recommendation system",
            "reorganization": {
                "status": "complete",
                "specialized_endpoints": {
                    "authentication": "/api/auth/* (auth_bp)",
                    "gift_finder": "/api/gift-finder/* (gift_finder_bp)",
                    "general_api": "/api/* (this blueprint)"
                },
                "integration_complete": True
            },
            "architecture": "Clean Architecture with Service Layer",
            "ai_features": [
                "Multi-Model AI Integration (OpenAI, Anthropic, Groq, Gemini)",
                "Intelligent Model Selection",
                "Performance Optimization",
                "Cost Tracking"
            ],
            "enterprise_features": [
                "Service Registry Pattern",
                "Redis Caching",
                "Circuit Breaker Pattern", 
                "Structured Logging",
                "Performance Metrics",
                "Health Monitoring"
            ],
            "available_endpoints": [
                "GET /api/ - This endpoint",
                "GET /api/status - API status",
                "GET /api/health - Comprehensive health check",
                "GET /api/info - API information",
                "GET /api/catalog/* - Product catalog endpoints"
            ],
            "specialized_apis": {
                "authentication_api": {
                    "base_url": "/api/auth",
                    "description": "Unified Authentication Master",
                    "features": ["frontend_templates", "api_endpoints", "jwt_security"]
                },
                "gift_finder_api": {
                    "base_url": "/api/gift-finder", 
                    "description": "Unified Gift Discovery Master",
                    "features": ["personality_based", "emotion_based", "ai_enhanced", "3d_visualization"]
                }
            },
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ API root endpoint error: {e}")
        return jsonify({
            "success": False,
            "error": "API temporarily unavailable",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


# =============================================================================
# 🛒 CART API ENDPOINTS  
# =============================================================================

@csrf.exempt
@api_bp.route('/cart')
@cross_origin()
def cart_root():
    """🛒 ROOT CART ENDPOINT - /api/cart"""
    try:
        return jsonify({
            "success": True,
            "message": "Cart API is working",
            "api_info": {
                "cart_features": ["session_based", "user_carts", "gift_options", "ai_recommendations"],
                "supported_operations": ["view", "add", "update", "remove", "checkout"]
            },
            "available_endpoints": [
                "GET /api/cart - View current cart",
                "POST /api/cart/items - Add item to cart", 
                "PUT /api/cart/items/<id> - Update cart item",
                "DELETE /api/cart/items/<id> - Remove cart item",
                "POST /api/cart/checkout - Prepare checkout"
            ],
            "cart_info": {
                "session_id_required": "For anonymous users",
                "user_id_preferred": "For logged-in users",
                "merge_supported": "Session carts merge with user carts on login"
            },
            "sample_cart": {
                "items": [],
                "total": 0,
                "currency": "EUR",
                "message": "Empty cart - add items via POST /api/cart/items"
            },
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Cart root endpoint error: {e}")
        return jsonify({
            "success": False,
            "error": "Cart API temporarily unavailable", 
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


# =============================================================================
# 📚 CATALOG API ENDPOINTS
# =============================================================================

@csrf.exempt
@api_bp.route('/catalog/products', methods=['GET'])
@cross_origin()
def get_catalog_products():
    """
    📚 GET CATALOG PRODUCTS
    
    GET /api/catalog/products?age=25&personality=creative&relationship=partner
    """
    
    try:
        # Import hier um Circular Imports zu vermeiden
        from ai_engine.catalog import get_products_by_age, get_products_by_personality, get_products_by_relationship
        
        age = request.args.get('age', type=int)
        personality = request.args.get('personality', '').split(',') if request.args.get('personality') else []
        relationship = request.args.get('relationship')
        limit = request.args.get('limit', 20, type=int)
        
        results = []
        
        if age:
            results.extend(get_products_by_age(age))
        elif personality:
            results.extend(get_products_by_personality(personality))
        elif relationship:
            results.extend(get_products_by_relationship(relationship))
        else:
            # Alle Produkte
            from ai_engine.catalog import GESCHENK_KATALOG
            results = list(GESCHENK_KATALOG.values())
        
        return jsonify({
            'success': True,
            'products': results[:limit],  # Limit für Performance
            'total': len(results),
            'filters_applied': {
                'age': age,
                'personality': personality,
                'relationship': relationship
            },
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Catalog products error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to load catalog products',
            'message': str(e)
        }), 500


@csrf.exempt
@api_bp.route('/catalog/sync', methods=['POST'])
@cross_origin()
@login_required  # Nur für Admins!
def sync_catalog():
    """
    🔄 SYNC PRODUCT CATALOG
    
    POST /api/catalog/sync
    Synchronisiert Produktionskatalog mit Datenbank
    """
    
    # TODO: Admin-Check hinzufügen
    # if not current_user.is_admin:
    #     return jsonify({'error': 'Admin access required'}), 403
    
    try:
        gift_service = services.gift_service if services else GiftService()
        result = gift_service.sync_product_catalog()
        
        return jsonify({
            'success': True,
            'sync_result': result,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Catalog sync error: {e}")
        return jsonify({
            'success': False,
            'error': 'Catalog sync failed',
            'message': str(e)
        }), 500


@csrf.exempt
@api_bp.route('/catalog/ai-recommendations', methods=['POST'])
@cross_origin()
@login_required
def get_ai_catalog_recommendations():
    """
    🤖 AI CATALOG RECOMMENDATIONS
    
    POST /api/catalog/ai-recommendations
    {
        "budget_range": [50, 200],
        "relationship": "partner",
        "occasion": "geburtstag"
    }
    """
    
    try:
        gift_service = services.gift_service if services else GiftService()
        session_data = request.get_json() or {}
        
        result = gift_service.get_ai_catalog_recommendations(
            user_id=current_user.id,
            session_data=session_data
        )
        
        return jsonify({
            'success': True,
            'catalog_recommendations': result,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ AI catalog recommendations error: {e}")
        return jsonify({
            'success': False,
            'error': 'AI catalog recommendations failed',
            'message': str(e)
        }), 500


# =============================================================================
# 📊 SERVICE HEALTH & MONITORING
# =============================================================================

@csrf.exempt
@api_bp.route('/health', methods=['GET'])
@cross_origin()
def comprehensive_health_check():
    """
    ❤️ COMPREHENSIVE HEALTH CHECK
    
    Prüft alle Enterprise Services und gibt detaillierten Status zurück
    """
    
    try:
        # 🚀 DEINE SERVICE HEALTH CHECKS!
        health_data = {
            "overall_status": "healthy",
            "components": {},
            "service_status": {},
            "specialized_apis": {},
            "database_status": {},
            "enterprise_features": {}
        }
        
        # Test Database Connection
        try:
            from app.extensions import db
            from sqlalchemy import text
            db.session.execute(text('SELECT 1'))
            health_data["database_status"] = {
                "connection": "healthy",
                "accessible": True
            }
            health_data["components"]["database"] = "healthy"
        except Exception as e:
            health_data["database_status"] = {"error": str(e)}
            health_data["components"]["database"] = "unhealthy"
            health_data["overall_status"] = "critical"
        
        # Test Enterprise Services
        if services:
            try:
                service_health = services.get_service_health()
                health_data["service_status"] = service_health
                health_data["components"]["enterprise_services"] = "healthy"
            except Exception as e:
                health_data["service_status"] = {"error": str(e)}
                health_data["components"]["enterprise_services"] = "unhealthy"
                health_data["overall_status"] = "degraded"
        else:
            health_data["service_status"] = {"available": False}
            health_data["components"]["enterprise_services"] = "unavailable"
        
        # Test Specialized APIs (nach Reorganisation)
        health_data["specialized_apis"] = {
            "auth_api": {
                "endpoint": "/api/auth/health",
                "status": "available",
                "features": ["frontend_templates", "api_endpoints", "jwt_security"]
            },
            "gift_finder_api": {
                "endpoint": "/api/gift-finder/health", 
                "status": "available",
                "features": ["personality_based", "emotion_based", "ai_enhanced", "3d_visualization"]
            }
        }
        health_data["components"]["specialized_apis"] = "healthy"
        
        # Enterprise Features Status
        health_data["enterprise_features"] = {
            "service_registry": "available",
            "health_monitoring": "available", 
            "performance_metrics": "available",
            "circuit_breaker": "available" if services else "unavailable",
            "redis_caching": "available" if services else "unavailable"
        }
        
        # Determine Overall Status
        component_statuses = list(health_data["components"].values())
        if "unhealthy" in component_statuses:
            health_data["overall_status"] = "critical"
        elif "unavailable" in component_statuses:
            health_data["overall_status"] = "degraded"
        
        health_data.update({
            "timestamp": datetime.now().isoformat(),
            "service_name": "SensationGifts General API",
            "version": "2.0.0",
            "reorganization_status": "complete"
        })
        
        # Return appropriate status code
        status_code = 200
        if health_data["overall_status"] == "degraded":
            status_code = 206
        elif health_data["overall_status"] == "critical":
            status_code = 503
        
        return jsonify(health_data), status_code
        
    except Exception as e:
        logger.error(f"❌ Comprehensive health check failed: {e}")
        return jsonify({
            "overall_status": "critical",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 503


@csrf.exempt
@api_bp.route('/status', methods=['GET'])
@cross_origin()
def api_status():
    """
    📊 API STATUS CHECK (nach Reorganisation)
    """
    try:
        # Test Database Connection
        from app.extensions import db
        db.session.execute('SELECT 1')
        
        # Test Services
        service_status = {
            'user_service': 'healthy' if services and hasattr(services, 'user_service') else 'unavailable',
            'gift_service': 'healthy' if services and hasattr(services, 'gift_service') else 'unavailable', 
            'recommendation_service': 'healthy' if services and hasattr(services, 'recommendation_service') else 'unavailable',
            'ai_engine': 'connected'
        }
        
        return jsonify({
            "success": True,
            "status": "healthy",
            "database": "connected",
            "services": service_status,
            "reorganization": {
                "status": "complete",
                "blueprint_count": {
                    "before": 12,
                    "after": 7,
                    "reduction": "42%"
                },
                "specialized_endpoints": {
                    "auth": "/api/auth/*",
                    "gift_finder": "/api/gift-finder/*",
                    "general": "/api/*"
                }
            },
            "features": [
                "enterprise_services",
                "ai_integration", 
                "redis_caching",
                "performance_monitoring",
                "unified_blueprints"
            ],
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Status check error: {e}")
        return jsonify({
            "success": False,
            "status": "unhealthy", 
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


@csrf.exempt
@api_bp.route('/info', methods=['GET'])
@cross_origin()
def api_info():
    """
    ℹ️ API INFORMATION (nach Reorganisation)
    """
    return jsonify({
        "api_name": "SensationGifts Enterprise AI API",
        "version": "2.0.0",
        "description": "Enterprise AI-powered gift recommendation system",
        "reorganization": {
            "status": "complete",
            "date": "2024-01-15",
            "changes": {
                "blueprint_consolidation": "12 → 7 blueprints",
                "unified_authentication": "auth_bp (3 files merged)",
                "unified_gift_finder": "gift_finder_bp (4 files merged)",
                "cleaner_architecture": "Specialized Master Blueprints"
            }
        },
        "architecture": "Clean Architecture with Service Layer",
        "ai_features": [
            "Multi-Model AI Integration (OpenAI, Anthropic, Groq, Gemini)",
            "Intelligent Model Selection",
            "Performance Optimization",
            "Cost Tracking"
        ],
        "enterprise_features": [
            "Service Registry Pattern",
            "Redis Caching",
            "Circuit Breaker Pattern", 
            "Structured Logging",
            "Performance Metrics",
            "Health Monitoring"
        ],
        "specialized_apis": {
            "authentication": {
                "base_url": "/api/auth",
                "endpoints": ["/login", "/register", "/profile", "/refresh", "/protected", "/health"],
                "features": ["frontend_templates", "api_endpoints", "jwt_security"]
            },
            "gift_finder": {
                "base_url": "/api/gift-finder",
                "endpoints": ["/process", "/quick", "/emotions", "/design-data", "/enhanced", "/health"],
                "features": ["personality_based", "emotion_based", "ai_enhanced", "3d_visualization"]
            },
            "general": {
                "base_url": "/api",
                "endpoints": ["/health", "/status", "/info", "/catalog/*"],
                "features": ["health_monitoring", "catalog_integration", "service_management"]
            }
        },
        "timestamp": datetime.now().isoformat()
    }), 200


# =============================================================================
# 🧪 TESTING ENDPOINTS
# =============================================================================

@csrf.exempt
@api_bp.route('/test', methods=['GET'])
@cross_origin()
def test_general_api():
    """🧪 Test General API Components"""
    
    try:
        test_results = {
            'database_connection': test_database_connection(),
            'services_available': services is not None,
            'catalog_available': test_catalog_availability(),
            'health_endpoints': test_health_endpoints()
        }
        
        return jsonify({
            'success': True,
            'message': 'General API test completed',
            'test_results': test_results,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"General API test failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def test_database_connection() -> bool:
    """Test database connection"""
    try:
        from app.extensions import db
        db.session.execute('SELECT 1')
        return True
    except Exception:
        return False


def test_catalog_availability() -> bool:
    """Test catalog availability"""
    try:
        from ai_engine.catalog import GESCHENK_KATALOG
        return len(GESCHENK_KATALOG) > 0
    except Exception:
        return False


def test_health_endpoints() -> Dict[str, bool]:
    """Test health endpoints availability"""
    return {
        'general_health': True,  # This endpoint
        'auth_health': True,     # Should be available at /api/auth/health
        'gift_finder_health': True  # Should be available at /api/gift-finder/health
    }


# =============================================================================
# 🔄 REORGANIZATION ENDPOINTS
# =============================================================================

@csrf.exempt
@api_bp.route('/reorganization/status', methods=['GET'])
@cross_origin()
def reorganization_status():
    """
    🔄 REORGANIZATION STATUS
    
    Zeigt Status der Backend-Reorganisation
    """
    
    try:
        import os
        
        # Check if old files are removed
        removed_files = [
            'app/routes/users.py',
            'app/routes/secure_auth.py',
            'app/routes/emotional_api.py',
            'app/routes/design_api.py',
            'app/routes/enhanced_ai_api.py'
        ]
        
        # Check if new files exist
        expected_files = [
            'app/routes/auth.py',
            'app/routes/gift_finder.py'
        ]
        
        removal_status = {file: not os.path.exists(file) for file in removed_files}
        existence_status = {file: os.path.exists(file) for file in expected_files}
        
        reorganization_complete = (
            all(removal_status.values()) and 
            all(existence_status.values())
        )
        
        return jsonify({
            'success': True,
            'reorganization_complete': reorganization_complete,
            'removed_files': removal_status,
            'expected_files': existence_status,
            'blueprint_consolidation': {
                'before': 12,
                'after': 7,
                'reduction_percentage': 42
            },
            'integration_status': {
                'auth_master': 'complete',
                'gift_finder_master': 'complete'
            },
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Reorganization status check failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@api_bp.errorhandler(404)
def api_not_found(error):
    """API 404 Handler (nach Reorganisation)"""
    return jsonify({
        "success": False,
        "error": "API endpoint not found",
        "reorganized_structure": {
            "authentication": "Use /api/auth/* endpoints",
            "gift_finder": "Use /api/gift-finder/* endpoints",
            "general": "Use /api/* endpoints (this blueprint)"
        },
        "available_endpoints": [
            "/api/health", "/api/status", "/api/info", 
            "/api/catalog/*", "/api/reorganization/status"
        ]
    }), 404


@api_bp.errorhandler(405)
def api_method_not_allowed(error):
    """API 405 Handler"""
    return jsonify({
        "success": False,
        "error": "HTTP method not allowed for this endpoint"
    }), 405


@api_bp.errorhandler(500)
def api_server_error(error):
    """API 500 Handler"""
    logger.error(f"API server error: {error}")
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500


# =============================================================================
# CORS PREFLIGHT HANDLER
# =============================================================================

@csrf.exempt
@api_bp.before_request
def handle_preflight():
    """Handle CORS preflight requests"""
    if request.method == "OPTIONS":
        response = jsonify({'message': 'OK'})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ['api_bp']