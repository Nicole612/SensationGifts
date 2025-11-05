"""
Routes Package - SensationGifts API System
==========================================

Zentrale Route-Registrierung für SensationGifts
- REST API für Frontend Integration
- Flask-Login Authentication  
- Service Layer Integration
- Fehlerbehandlung & Logging
- CORS Support für Frontend
"""

from flask import Flask
from flask_cors import CORS
import logging

# Import aller Blueprint-Module
from .main import main_bp
from .auth import auth_bp
from .api import api_bp  # Deine API Routes
from .personality import personality_bp
from .gifts import gifts_bp
from .gift_finder import gift_finder_bp
from .cart import cart_bp
from .documentation_api import documentation_bp
from .catalog_api import catalog_bp
from .catalog_classic import classic_catalog_bp

# Logger Setup
logger = logging.getLogger(__name__)


def init_routes(app: Flask):
    """
    Registriert alle Routes und Blueprints bei der Flask App
    
    Args:
        app: Flask Application Instance
    """
    
    try:
        # ✨ NEU: Register Main Blueprint ZUERST (globale Routes)
        app.register_blueprint(main_bp)
        logger.info("✅ Main routes registered (/, /health, /api)")
        
        # CORS Setup für Frontend Integration
        CORS(app, resources={
            r"/api/*": {
                "origins": ["http://localhost:3000", "http://localhost:8080"],
                "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization"]
            }
        })
        
        # Register Authentication Blueprint
        app.register_blueprint(auth_bp)
        logger.info("✅ Authentication routes registered")

        # Register API Blueprint (deine bestehende API)
        app.register_blueprint(api_bp, url_prefix='/api')
        logger.info("✅ Main API routes registered")
        
        # 🎯 UNIFIED GIFT FINDER MASTER
        app.register_blueprint(gift_finder_bp)
        logger.info("✅ Gift Finder routes registered")

        # CATALOG APIs
        app.register_blueprint(catalog_bp, url_prefix='/api/catalog')
        logger.info("✅ Catalog API routes registered")
        
        app.register_blueprint(classic_catalog_bp, url_prefix='/api/catalog')
        logger.info("✅ Classic Catalog API routes registered")


        # Register Feature-spezifische Blueprints
        app.register_blueprint(personality_bp, url_prefix='/api/personality')
        logger.info("✅ Personality routes registered")
        
        # GIFT CATALOG (bereinigt - nur Catalog Functions)
        app.register_blueprint(gifts_bp, url_prefix='/api/gifts')
        logger.info("✅ Gift routes registered")

        # SHOPPING CART
        app.register_blueprint(cart_bp)
        logger.info("✅ Cart routes registered")

        # Register Documentation API Routes
        app.register_blueprint(documentation_bp, url_prefix='/api')
        logger.info("📚 Documentation API routes registered")
        
        # Global Error Handlers
        register_error_handlers(app)
        
        logger.info("🚀 All routes successfully registered")
        
    except Exception as e:
        logger.error(f"❌ Failed to register routes: {e}")
        # Versuche trotzdem die wichtigsten Routes zu registrieren
        try:
            app.register_blueprint(main_bp)
            app.register_blueprint(gift_finder_bp)
            logger.info("✅ Emergency routes registered")
        except Exception as e2:
            logger.error(f"❌ Emergency route registration failed: {e2}")
            raise


def register_error_handlers(app: Flask):
    """
    Registriert globale Error Handler
    """
    
    @app.errorhandler(404)
    def not_found_error(error):
        return {
            'success': False,
            'error': 'Endpoint nicht gefunden',
            'status': 404,
            'available_apis': {
                'classic': '/api/',
                'emotional': '/api/emotional/',
                'auth': '/auth/'
            }
        }, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return {
            'success': False,
            'error': 'Interner Server-Fehler',
            'status': 500
        }, 500
    
    @app.errorhandler(405)
    def method_not_allowed(error):
        return {
            'success': False,
            'error': 'HTTP-Methode nicht erlaubt',
            'status': 405
        }, 405


# === BONUS: DEVELOPMENT HELPERS ===

def get_route_info(app: Flask) -> dict:
    """
    Development Helper: Übersicht aller registrierten Routes
    Besonders nützlich für API-Dokumentation
    """
    routes = {
        'total_routes': 0,
        'api_routes': {
            'v1_classic': [],
            'emotional': [],
            'auth': []
        },
        'main_routes': []
    }
    
    for rule in app.url_map.iter_rules():
        route_info = {
            'endpoint': rule.endpoint,
            'methods': list(rule.methods - {'HEAD', 'OPTIONS'}),
            'url': str(rule)
        }
        
        routes['total_routes'] += 1
        
        if rule.rule.startswith('/api/'):
            routes['api_routes']['v1_classic'].append(route_info)
        elif rule.rule.startswith('/api/emotional/'):
            routes['api_routes']['emotional'].append(route_info)
        elif rule.rule.startswith('/auth/'):
            routes['api_routes']['auth'].append(route_info)
        else:
            routes['main_routes'].append(route_info)
    
    return routes


def print_api_overview(app: Flask):
    """
    Development Helper: Druckt API-Übersicht in die Konsole
    Perfekt für Development und Debugging
    """
    routes = get_route_info(app)
    
    print("\n" + "="*60)
    print("🎁 GIFT SHOP API OVERVIEW")
    print("="*60)
    
    print(f"\n📊 TOTAL ROUTES: {routes['total_routes']}")
    
    print(f"\n🔧 CLASSIC API (v1): {len(routes['api_routes']['v1_classic'])} endpoints")
    for route in routes['api_routes']['v1_classic'][:5]:  # Show first 5
        print(f"  {route['methods']} {route['url']}")
    if len(routes['api_routes']['v1_classic']) > 5:
        print(f"  ... and {len(routes['api_routes']['v1_classic']) - 5} more")
    
    print(f"\n🎭 EMOTIONAL API: {len(routes['api_routes']['emotional'])} endpoints")
    for route in routes['api_routes']['emotional']:
        print(f"  {route['methods']} {route['url']}")
    
    print(f"\n🔐 AUTH API: {len(routes['api_routes']['auth'])} endpoints")
    for route in routes['api_routes']['auth'][:3]:  # Show first 3
        print(f"  {route['methods']} {route['url']}")
    
    print("\n" + "="*60 + "\n")


# Export für einfachen Import
__all__ = [
    'init_routes',
    'register_error_handlers', 
    'get_route_info',
    'print_api_overview',
    'auth_bp',
    'api_bp',
    'personality_bp',
    'gifts_bp',
    'main_bp'
]
