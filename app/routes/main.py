"""
Main Routes Blueprint - CLEANED & COMPLETE
==========================================

Alle Frontend-Routes für SensationGifts ohne Duplikate
"""

from flask import Blueprint, jsonify, render_template, request, redirect, url_for, flash
from app.services import services
import logging
from flask_login import current_user, login_required

logger = logging.getLogger(__name__)

# Main Blueprint
main_bp = Blueprint('main', __name__)

# =============================================================================
# 🏠 GLOBAL ROUTES
# =============================================================================

@main_bp.route('/')
def index():
    """🏠 Homepage - Haupteinstieg für SensationGifts"""
    try:
        return render_template('index.html', 
                             title='SensationGifts - AI-Powered Gift Recommendations',
                             version='1.0.0')
    except Exception as e:
        logger.warning(f"Template not found, returning JSON: {e}")
        return jsonify({
            'message': 'Welcome to SensationGifts AI',
            'description': 'AI-powered gift recommendation system',
            'version': '1.0.0',
            'api_endpoints': {
                'health': '/health',
                'api': '/api',
                'authentication': '/auth',
                'personality': '/api/personality',
                'gifts': '/api/gifts'
            },
            'frontend_info': 'Frontend template will be loaded here'
        })


    """🎁 GIFT FINDER HUB - English Alias"""
    # Redirect to German version or render English template
    try:
        template_data = {
            'title': 'Find the Perfect Gift - SensationGifts',
            'user_id': current_user.id if current_user.is_authenticated else None,
            'is_authenticated': current_user.is_authenticated,
            'page_type': 'gift_finder',
            'language': 'en'
        }
        
        # Try English template first, fallback to German
        try:
            return render_template('gift_finder_hub_en.html', **template_data)
        except:
            # Fallback to German template
            template_data['title'] = 'Geschenke finden - SensationGifts'
            return render_template('gift_finder_hub.html', **template_data)
        
    except Exception as e:
        logger.error(f"Gift Finder Hub English error: {e}")
        # JSON fallback
        return jsonify({
            'message': 'Gift Finder Hub',
            'description': 'Find the perfect personalized gift',
            'redirect_to': '/geschenke-finden',
            'api_endpoints': {
                'gifts': '/api/gifts',
                'personality': '/api/personality',
                'recommendations': '/api/recommendations'
            }
        })


@main_bp.route('/health')
def global_health_check():
    """📊 Globaler Health Check - AKTIVE SERVICE INITIALISIERUNG"""
    try:
        logger.info("🔄 Initializing services for health check...")
        
        # Services aktiv laden (löst Lazy Loading aus)
        user_svc = services.user_service
        gift_svc = services.gift_service
        rec_svc = services.recommendation_service
        model_sel = services.model_selector
        model_fact = services.model_factory
        
        logger.info("✅ All services initialized successfully")
        
        # Health data abrufen
        health_data = services.get_service_health()
        
        # Status Code bestimmen
        status_code = 200
        if health_data.get('overall_status') == 'degraded':
            status_code = 206
        elif health_data.get('overall_status') in ['critical', 'error']:
            status_code = 503
        
        # App-Info hinzufügen
        health_data.update({
            'app_name': 'SensationGifts',
            'app_version': '1.0.0',
            'initialization_status': 'completed',
            'services_loaded': len(health_data.get('services', {})),
            'endpoints': {
                'api_health': '/api/health',
                'personality': '/api/personality/health',
                'gifts': '/api/gifts/health'
            }
        })
        
        logger.info(f"✅ Health check completed - Status: {health_data.get('overall_status')}")
        
        return jsonify(health_data), status_code
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return jsonify({
            'overall_status': 'error',
            'error': str(e),
            'app_name': 'SensationGifts',
            'message': 'Service Registry nicht verfügbar'
        }), 500


@main_bp.route('/api')
@main_bp.route('/api/')
def api_info():
    """📋 API Information Endpoint"""
    return jsonify({
        'message': 'SensationGifts API',
        'version': '1.0.0',
        'description': 'AI-powered gift recommendation system',
        'endpoints': {
            'health': {
                'url': '/health',
                'description': 'Global health check with service initialization',
                'methods': ['GET']
            },
            'auth': {
                'base_url': '/auth',
                'endpoints': ['login', 'register', 'logout'],
                'description': 'User authentication'
            },
            'api_v1': {
                'base_url': '/api',
                'health': '/api/health',
                'description': 'Main API endpoints'
            },
            'personality': {
                'base_url': '/api/personality',
                'analyze': '/api/personality/analyze',
                'health': '/api/personality/health',
                'description': 'Big Five + Limbic personality analysis'
            },
            'gifts': {
                'base_url': '/api/gifts',
                'search': '/api/gifts/search',
                'categories': '/api/gifts/categories',
                'recommendations': '/api/gifts/recommendations',
                'health': '/api/gifts/health',
                'description': 'Gift catalog and recommendations'
            }
        },
        'ai_features': {
            'personality_analysis': 'Big Five + Limbic System',
            'ai_models': ['OpenAI GPT-4', 'Groq Mixtral', 'Anthropic Claude', 'Google Gemini'],
            'recommendation_engine': 'Multi-AI Powered Recommendations'
        }
    })


@main_bp.route('/status')
def quick_status():
    """⚡ Quick Status Check"""
    try:
        return jsonify({
            'status': 'ok',
            'app': 'SensationGifts',
            'services_available': bool(services),
            'timestamp': services.get_service_health().get('timestamp', 'unknown')
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@main_bp.route('/warm-up')
def warm_up_services():
    """🔥 Service Warm-up Endpoint"""
    try:
        logger.info("🔥 Warming up all services...")
        
        services_to_warm = [
            ('user_service', services.user_service),
            ('gift_service', services.gift_service),
            ('recommendation_service', services.recommendation_service),
            ('model_selector', services.model_selector),
            ('model_factory', services.model_factory)
        ]
        
        initialized = []
        errors = []
        
        for service_name, service_instance in services_to_warm:
            try:
                _ = service_instance  # Lazy Loading auslösen
                initialized.append(service_name)
                logger.info(f"✅ {service_name} warmed up")
            except Exception as e:
                errors.append(f"{service_name}: {str(e)}")
                logger.error(f"❌ {service_name} warm-up failed: {e}")
        
        return jsonify({
            'status': 'completed',
            'services_initialized': initialized,
            'services_count': len(initialized),
            'errors': errors,
            'message': f"Warmed up {len(initialized)} services"
        })
        
    except Exception as e:
        logger.error(f"❌ Service warm-up failed: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


# =============================================================================
# 🎁 GIFT FINDER FRONTEND ROUTES
# =============================================================================

@main_bp.route('/geschenke-finden')
def gift_finder_hub():
    """🎁 GIFT FINDER HUB - Hauptseite für Geschenksuche"""
    try:
        template_data = {
            'title': 'Geschenke finden - SensationGifts',
            'user_id': current_user.id if current_user.is_authenticated else None,
            'is_authenticated': current_user.is_authenticated,
            'page_type': 'gift_finder'
        }
        
        return render_template('gift_finder_hub.html', **template_data)
        
    except Exception as e:
        logger.error(f"Gift Finder Hub Fehler: {e}")
        flash("Fehler beim Laden der Geschenke-Suche", 'error')
        return redirect(url_for('main.index'))


@main_bp.route('/geschenke-finden/direkt')
def direct_recommendation():
    """🤖 DIREKTE AI-EMPFEHLUNG"""
    try:
        template_data = {
            'title': 'AI-Empfehlung - SensationGifts',
            'mount_island': 'direct-recommendation-island',
            'user_id': current_user.id if current_user.is_authenticated else None,
            'page_type': 'direct_recommendation'
        }
        
        return render_template('direct_recommendation.html', **template_data)
        
    except Exception as e:
        logger.error(f"Direct Recommendation Fehler: {e}")
        flash("Fehler beim Laden der AI-Empfehlung", 'error')
        return redirect(url_for('main.gift_finder_hub'))


@main_bp.route('/geschenke-finden/quiz')
def personality_quiz_page():
    """🧠 PERSÖNLICHKEITS-QUIZ"""
    try:
        template_data = {
            'title': 'Persönlichkeits-Quiz - SensationGifts',
            'mount_island': 'personality-quiz-island',
            'user_id': current_user.id if current_user.is_authenticated else None,
            'return_url': request.args.get('return_url', '/empfehlungen'),
            'page_type': 'personality_quiz'
        }
        
        return render_template('personality_quiz.html', **template_data)
        
    except Exception as e:
        logger.error(f"Personality Quiz Fehler: {e}")
        flash("Fehler beim Laden des Quiz", 'error')
        return redirect(url_for('main.gift_finder_hub'))


# =============================================================================
# 📋 EMPFEHLUNGS-ERGEBNIS ROUTES  
# =============================================================================

@main_bp.route('/empfehlungen')
def recommendations_page():
    """📋 EMPFEHLUNGS-ERGEBNISSE"""
    try:
        # Parameter aus URL holen
        recommendation_id = request.args.get('rec_id')
        quiz_completed = request.args.get('quiz_completed', False)
        personality_type = request.args.get('personality_type', '')
        demo_mode = request.args.get('demo', False)
        
        logger.info(f"Recommendations page accessed - Quiz: {quiz_completed}, Type: {personality_type}")
        
        template_data = {
            'title': 'Deine Geschenk-Empfehlungen - SensationGifts',
            'mount_island': 'recommendations-island',
            'recommendation_id': recommendation_id,
            'user_id': current_user.id if current_user.is_authenticated else None,
            'quiz_completed': quiz_completed,
            'personality_type': personality_type,
            'demo_mode': demo_mode,
            'page_type': 'recommendations'
        }
        
        return render_template('recommendations.html', **template_data)
        
    except Exception as e:
        logger.error(f"Recommendations Fehler: {e}")
        flash("Fehler beim Laden der Empfehlungen", 'error')
        return redirect(url_for('main.gift_finder_hub'))


# =============================================================================
# 🎁 GESCHENK-DETAIL ROUTES
# =============================================================================

@main_bp.route('/geschenk/<gift_id>')
def gift_detail_page(gift_id: str):
    """🎁 GESCHENK-DETAIL SEITE"""
    try:
        template_data = {
            'title': f'Geschenk Details - SensationGifts',
            'mount_island': 'gift-detail-island',
            'gift_id': gift_id,
            'user_id': current_user.id if current_user.is_authenticated else None,
            'page_type': 'gift_detail'
        }
        
        return render_template('gift_detail.html', **template_data)
        
    except Exception as e:
        logger.error(f"Gift Detail Fehler für {gift_id}: {e}")
        flash("Fehler beim Laden des Geschenks", 'error')
        return redirect(url_for('main.gift_finder_hub'))


@main_bp.route('/geschenk/<gift_id>/personalisieren')
def gift_personalization_page(gift_id: str):
    """✨ GESCHENK PERSONALISIERUNG"""
    try:
        template_data = {
            'title': 'Geschenk personalisieren - SensationGifts',
            'mount_island': 'personalization-island',
            'gift_id': gift_id,
            'user_id': current_user.id if current_user.is_authenticated else None,
            'page_type': 'personalization'
        }
        
        return render_template('gift_personalization.html', **template_data)
        
    except Exception as e:
        logger.error(f"Personalization Fehler für {gift_id}: {e}")
        flash("Fehler beim Laden der Personalisierung", 'error')
        return redirect(url_for('main.gift_detail_page', gift_id=gift_id))


# =============================================================================
# 🛒 WARENKORB ROUTES
# =============================================================================

@main_bp.route('/warenkorb')
def cart_page():
    """🛒 WARENKORB"""
    try:
        template_data = {
            'title': 'Warenkorb - SensationGifts',
            'mount_island': 'cart-island',
            'user_id': current_user.id if current_user.is_authenticated else None,
            'page_type': 'cart'
        }
        
        return render_template('cart.html', **template_data)
        
    except Exception as e:
        logger.error(f"Cart Fehler: {e}")
        flash("Fehler beim Laden des Warenkorbs", 'error')
        return redirect(url_for('main.index'))


# =============================================================================
# 📱 USER DASHBOARD
# =============================================================================

@main_bp.route('/dashboard')
@login_required
def user_dashboard():
    """📊 USER DASHBOARD"""
    try:
        template_data = {
            'title': 'Mein Dashboard - SensationGifts',
            'mount_island': 'user-dashboard-island',
            'user_id': current_user.id,
            'user_name': current_user.display_name,
            'page_type': 'dashboard'
        }
        
        return render_template('user_dashboard.html', **template_data)
        
    except Exception as e:
        logger.error(f"Dashboard Fehler: {e}")
        flash("Fehler beim Laden des Dashboards", 'error')
        return redirect(url_for('main.index'))


# =============================================================================
# ⚙️ FRONTEND CONFIG API
# =============================================================================

@main_bp.route('/api/frontend/config')
def frontend_config():
    """⚙️ FRONTEND CONFIGURATION für React Islands"""
    try:
        config = {
            'api_base_url': '/api',
            'features': {
                'personality_quiz': True,
                'ai_recommendations': True,
                'cart_system': True,
                'user_accounts': True
            },
            'user': {
                'is_authenticated': current_user.is_authenticated if current_user else False,
                'user_id': current_user.id if current_user and current_user.is_authenticated else None,
                'display_name': current_user.display_name if current_user and current_user.is_authenticated else None
            },
            'endpoints': {
                'personality_quiz': '/api/personality/quiz-questions',
                'create_profile': '/api/personality/create-profile',
                'gift_search': '/api/gifts/search',
                'recommendations': '/api/gifts/recommendations',
                'cart_summary': '/api/cart/summary',
                'add_to_cart': '/api/cart/items'
            }
        }
        
        return jsonify(config)
        
    except Exception as e:
        logger.error(f"Frontend config error: {e}")
        return jsonify({'error': 'Configuration not available'}), 500