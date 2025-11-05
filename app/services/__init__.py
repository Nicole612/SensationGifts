"""
Service Layer - Clean Architecture Implementation
==============================================

Service Registry und Dependency Injection für saubere Architektur.
Zentrale Koordination aller Business Logic Services.

🏗️ Architektur-Prinzipien:
- Dependency Injection für Testbarkeit
- Service Isolation für Maintainability  
- Async Support für Performance
- Error Handling für Robustheit
- Integration mit AI-Engine für Intelligence
"""

from typing import Optional, Dict, Any
import logging
from datetime import datetime

# Import aller Services
from .user_service import UserService
from .gift_service import GiftService  
from .recommendation_service import OptimizedRecommendationService
from .enhanced_ai_recommendation_service import EnhancedRecommendationService, get_recommendation_service
from .security_service import SecureUserService, setup_security_middleware
from .api_documentation_service import setup_api_documentation, create_documented_api, create_api_models

# AI-Engine Integration
from ai_engine.processors.model_selector import ModelSelector
from ai_engine.models.model_factory import AIModelFactory


class ServiceRegistry:
    """
    🏗️ Service Registry für Dependency Injection
    
    Zentraler Service-Manager der alle Business Logic Services koordiniert.
    Implementiert Singleton Pattern für konsistente Service-Instanzen.
    
    Features:
    - Lazy Loading von Services
    - Dependency Injection 
    - Health Monitoring
    - Performance Tracking
    """
    
    _instance: Optional['ServiceRegistry'] = None
    _initialized: bool = False
    
    def __new__(cls) -> 'ServiceRegistry':
        """Singleton Pattern - Eine Service Registry Instanz pro App"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialisiert Service Registry (nur einmal)"""
        if self._initialized:
            return
            
        # Core Services
        self._user_service: Optional[UserService] = None
        self._gift_service: Optional[GiftService] = None  
        self._recommendation_service: Optional[OptimizedRecommendationService] = None
        
        # AI-Engine Components (deine existierende Engine!)
        self._model_selector: Optional[ModelSelector] = None
        self._model_factory: Optional[AIModelFactory] = None
        
        # Service Health Tracking
        self._service_health: Dict[str, Dict[str, Any]] = {}
        self._service_metrics: Dict[str, Dict[str, Any]] = {}
        
        self._initialized = True
        
        
    # ==========================================================================
    # SERVICE GETTERS (Lazy Loading)
    # ==========================================================================
    
    @property
    def user_service(self) -> UserService:
        """
        🧑‍💼 User Service für Authentication & Profile Management
        
        Lazy Loading: Service wird nur erstellt wenn benötigt
        """
        if self._user_service is None:
            self._user_service = UserService()
            self._register_service_health("user_service")
            logging.info("👤 UserService initialized")
        
        return self._user_service
    
    @property  
    def gift_service(self) -> GiftService:
        """
        🎁 Gift Service für Database Operations & Gift Management
        
        Handles: Gift CRUD, Categories, Filtering, Search
        """
        if self._gift_service is None:
            self._gift_service = GiftService()
            self._register_service_health("gift_service")
            logging.info("🎁 GiftService initialized")
        
        return self._gift_service
    
    @property
    def recommendation_service(self) -> OptimizedRecommendationService:
        """
        🤖 Recommendation Service - DEINE AI-ENGINE INTEGRATION!
        
        Das ist der KERN deines Systems:
        - Nutzt deinen ModelSelector für intelligente AI-Model Auswahl
        - Orchestriert alle AI-Clients über ModelFactory
        - Implementiert Advanced AI-Features
        """
        if self._recommendation_service is None:
            # Dependency Injection: AI-Engine Components
            self._recommendation_service = OptimizedRecommendationService(
                model_selector=self.model_selector,
                model_factory=self.model_factory,
                gift_service=self.gift_service,
                user_service=self.user_service
            )
            self._register_service_health("recommendation_service")
            logging.info("🤖 OptimizedRecommendationService initialized with AI-Engine")
        
        return self._recommendation_service
    
    @property
    def enhanced_recommendation_service(self) -> EnhancedRecommendationService:
        """
        🚀 Enhanced Recommendation Service - NEUE AI-ENGINE INTEGRATION!
        
        Das ist die ERWEITERTE Version mit:
        - Verbesserte Error Handling & Fallbacks
        - Performance Optimierung
        - Async Integration
        - Mock Services für Testing
        """
        if not hasattr(self, '_enhanced_recommendation_service') or self._enhanced_recommendation_service is None:
            self._enhanced_recommendation_service = get_recommendation_service()
            self._register_service_health("enhanced_recommendation_service")
            logging.info("🚀 EnhancedRecommendationService initialized")
        
        return self._enhanced_recommendation_service

    @property
    def secure_user_service(self) -> SecureUserService:
        """
        🔐 Secure User Service - Production-Ready Authentication
        
        Das ist die SICHERE Version mit:
        - JWT Token Management
        - Rate Limiting
        - Input Sanitization
        - Brute Force Protection
        - Password Policies
        """
        if not hasattr(self, '_secure_user_service') or self._secure_user_service is None:
            self._secure_user_service = SecureUserService()
            self._register_service_health("secure_user_service")
            logging.info("🔐 SecureUserService initialized")
        return self._secure_user_service
    
    @property
    def api_documentation_service(self):
        """
        📚 API Documentation Service - OpenAPI/Swagger Integration
        
        Das ist die DOKUMENTATION Version mit:
        - OpenAPI/Swagger Integration
        - Interactive API Explorer
        - Auto-generated Documentation
        - Request/Response Examples
        - Testing Framework
        """
        if not hasattr(self, '_api_documentation_service') or self._api_documentation_service is None:
            self._api_documentation_service = setup_api_documentation
            self._register_service_health("api_documentation_service")
            logging.info("📚 API Documentation Service initialized")
        return self._api_documentation_service
    
    @property
    def model_selector(self) -> ModelSelector:
        """
        🧠 Dein AI Model Selector - Intelligente Model-Auswahl
        
        Das ist DEINE brillante Implementation aus:
        ai_engine/processors/model_selector.py
        """
        if self._model_selector is None:
            self._model_selector = ModelSelector()
            self._register_service_health("model_selector")
            logging.info("🧠 ModelSelector initialized")
        
        return self._model_selector
    
    @property
    def model_factory(self) -> AIModelFactory:
        """
        🏭 Model Factory für AI-Client Management
        
        Deine AI-Engine Factory für alle AI-Provider:
        - OpenAI GPT-4
        - Groq Mixtral  
        - Anthropic Claude
        - Google Gemini
        """
        if self._model_factory is None:
            self._model_factory = AIModelFactory()
            self._register_service_health("model_factory")
            logging.info("🏭 ModelFactory initialized")
        
        return self._model_factory
    
    # ==========================================================================
    # SERVICE HEALTH & MONITORING
    # ==========================================================================
    
    def _register_service_health(self, service_name: str):
        """Registriert Service für Health Monitoring"""
        self._service_health[service_name] = {
            "status": "healthy",
            "initialized_at": datetime.now().isoformat(),
            "last_health_check": datetime.now().isoformat(),
            "error_count": 0,
            "total_requests": 0
        }
        
        self._service_metrics[service_name] = {
            "total_calls": 0,
            "avg_response_time_ms": 0.0,
            "success_rate": 1.0,
            "last_error": None
        }
    
    def get_service_health(self) -> Dict[str, Any]:
        """
        📊 Service Health Dashboard
        
        Returns umfassende Health-Informationen aller Services
        """
        return {
            "overall_status": self._calculate_overall_health(),
            "services": self._service_health.copy(),
            "metrics": self._service_metrics.copy(),
            "ai_engine_status": self._get_ai_engine_health(),
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_overall_health(self) -> str:
        """Berechnet Overall Health Status"""
        if not self._service_health:
            return "initializing"
        
        unhealthy_services = [
            name for name, health in self._service_health.items()
            if health["status"] != "healthy"
        ]
        
        if not unhealthy_services:
            return "healthy"
        elif len(unhealthy_services) < len(self._service_health) / 2:
            return "degraded"
        else:
            return "critical"
    
    def _get_ai_engine_health(self) -> Dict[str, Any]:
        """Gibt AI-Engine Health Status zurück"""
        if self._model_selector and self._model_factory:
            try:
                # Nutze deine ModelSelector Health-Funktion
                model_health = self._model_selector.get_model_health_status()
                
                return {
                    "ai_engine_available": True,
                    "model_selector_ready": True,
                    "model_factory_ready": True,
                    "model_health": model_health,
                }
            except Exception as e:
                return {
                    "ai_engine_available": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        return {
            "ai_engine_available": False,
            "reason": "not_initialized"
        }
    
    # ==========================================================================
    # SERVICE LIFECYCLE MANAGEMENT  
    # ==========================================================================
    
    async def initialize_all_services(self):
        """
        🚀 Initialisiert alle Services für bessere Startup Performance
        
        Useful für App-Startup um alle Services warm zu machen
        """
        services_to_init = [
            "user_service",
            "gift_service", 
            "recommendation_service",
            "model_selector",
            "model_factory"
        ]
        
        for service_name in services_to_init:
            try:
                # Trigger Lazy Loading
                getattr(self, service_name)
                logging.info(f"✅ {service_name} initialized successfully")
                
            except Exception as e:
                logging.error(f"❌ Failed to initialize {service_name}: {e}")
                self._service_health[service_name] = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
    
    def shutdown_services(self):
        """Graceful Shutdown aller Services"""
        logging.info("🔄 Shutting down services...")
        
        # AI-Engine Cleanup
        if self._recommendation_service:
            # Cleanup AI connections, release resources
            self._recommendation_service.cleanup()
        
        if self._model_selector:
            # Save performance metrics, cleanup monitoring
            # self._model_selector.save_metrics() # wenn implementiert
            pass
        
        logging.info("✅ Services shutdown complete")


# =============================================================================
# GLOBAL SERVICE REGISTRY INSTANCE
# =============================================================================

# Globale Service Registry Instanz (Singleton)
services = ServiceRegistry()

# Convenience Functions für einfachen Service-Zugriff
def get_user_service() -> UserService:
    """🧑‍💼 Quick Access zu User Service"""
    return services.user_service

def get_gift_service() -> GiftService:
    """🎁 Quick Access zu Gift Service"""  
    return services.gift_service

def get_recommendation_service() -> OptimizedRecommendationService:
    """🤖 Quick Access zu Recommendation Service (AI-Engine!)"""
    return services.recommendation_service

def get_service_health() -> Dict[str, Any]:
    """📊 Quick Access zu Service Health"""
    return services.get_service_health()

# =============================================================================
# get_service_registry Function 
# =============================================================================

def get_service_registry() -> ServiceRegistry:
    """
    🔄 Returns the global service registry instance
    
    Funktioniert mit deinem bestehenden Singleton Pattern.
    Löst den Import-Error: 'cannot import name get_service_registry'
    """
    return services


# =============================================================================  
# Service Patches für Runtime-Probleme
# =============================================================================

def _patch_redis_cache_logger():
    """
    🔧 Patches Redis Cache Logger Issue
    
    Löst: 'RedisCache' object has no attribute 'logger'
    """
    try:
        # Check recommendation service für Redis Cache
        if hasattr(services, '_recommendation_service') and services._recommendation_service:
            rec_service = services._recommendation_service
            
            # Suche nach cache attribute
            if hasattr(rec_service, 'cache') and rec_service.cache:
                if not hasattr(rec_service.cache, 'logger'):
                    rec_service.cache.logger = logging.getLogger('redis_cache')
                    logging.info("🔧 Redis cache logger patched")
            
            # Suche nach _cache attribute        
            if hasattr(rec_service, '_cache') and rec_service._cache:
                if not hasattr(rec_service._cache, 'logger'):
                    rec_service._cache.logger = logging.getLogger('redis_cache')
                    logging.info("🔧 Redis _cache logger patched")
                    
    except Exception as e:
        logging.error(f"Redis cache patch failed: {e}")


def _patch_async_error_handler():
    """
    🔧 Patches Missing Async Error Handler
    
    Löst: 'OptimizedRecommendationService' object has no attribute '_handle_recommendation_error_async'
    """
    try:
        if hasattr(services, '_recommendation_service') and services._recommendation_service:
            rec_service = services._recommendation_service
            
            # Prüfe ob die Methode fehlt
            if not hasattr(rec_service, '_handle_recommendation_error_async'):
                import types
                
                async def _handle_recommendation_error_async(self, error, user_id, request_id):
                    """Async error handler for recommendations"""
                    error_response = {
                        "success": False,
                        "error": str(error),
                        "recommendations": [],
                        "fallback": True,
                        "user_id": user_id,
                        "request_id": request_id,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Log error mit bestehender logger structure
                    if hasattr(self, 'logger'):
                        self.logger.error(f"Async recommendation error for user {user_id}: {error}")
                    else:
                        logging.error(f"Async recommendation error for user {user_id}: {error}")
                    
                    return error_response
                
                # Dynamisch die Methode hinzufügen (preserviert bestehende Architektur)
                rec_service._handle_recommendation_error_async = types.MethodType(
                    _handle_recommendation_error_async, rec_service
                )
                
                logging.info("🔧 Async error handler patched")
                
    except Exception as e:
        logging.error(f"Async error handler patch failed: {e}")


def apply_service_patches():
    """
    🔧 Applies all necessary service patches on startup
    
    Wird automatisch beim Import ausgeführt um Runtime-Probleme zu beheben.
    """
    _patch_redis_cache_logger()
    _patch_async_error_handler()


# =============================================================================
# Enhanced Service Registry für bessere Error Resilience
# =============================================================================

def get_service_safely(service_name: str):
    """
    🛡️ Safe Service Getter mit Error Handling
    
    Fallback wenn ein Service nicht verfügbar ist.
    """
    try:
        return getattr(services, service_name)
    except Exception as e:
        logging.error(f"Failed to get service {service_name}: {e}")
        return None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main Registry
    'ServiceRegistry',
    'services',
    
    # Service Access Functions  
    'get_user_service',
    'get_gift_service', 
    'get_recommendation_service',
    'get_service_health',
    
    # Individual Services
    'UserService',
    'GiftService',
    'OptimizedRecommendationService',

    # NEW
    'get_service_registry', 
    'apply_service_patches',
    'get_service_safely'
]

# Führe Patches beim Import aus (non-breaking)
try:
    apply_service_patches()
    logging.info("🔧 Service patches applied successfully")
except Exception as e:
    logging.error(f"Service patches failed: {e}")
    