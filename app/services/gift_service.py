"""
Gift Service - Gift Database Operations & Management
==================================================

Business Logic für Gift-Management, Database Operations und Gift-Filtering.
Clean Architecture Service für alle Gift-bezogenen Operations.

Features:
- Gift CRUD Operations
- Advanced Filtering & Search
- Category Management
- Price Analysis
- Popular Gifts Analytics
- Gift Recommendation Support
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

# Database Models (deine bestehenden Models)
from app.models.gift import Gift, GiftCategory
from app.models.user import User

from ai_engine.catalog import CatalogIntegrationService
# Database Connection (wird später von Flask App injected)
from app import db


class GiftService:
    """
    🎁 Gift Service für Database Operations & Business Logic
    
    Handles:
    - Gift CRUD Operations
    - Advanced Search & Filtering
    - Category Management
    - Price Analytics
    - Popular Gift Analysis
    - Gift Recommendation Database Support
    
    Clean Architecture: Trennt Database Logic von AI Logic
    """
    
    def __init__(self, database_session: Optional[Session] = None):
        """
        Initialize Gift Service
        
        Args:
            database_session: Optional SQLAlchemy session (für Testing)
        """
        self.db = database_session or db.session
        
        # Performance Tracking
        self.query_metrics = {
            "total_queries": 0,
            "avg_query_time_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Simple Caching für häufige Queries
        self._category_cache = {}
        self._popular_gifts_cache = {}
        
        # 📦 NEU: AI-Katalog-Integration aktivieren
        self.catalog_service = CatalogIntegrationService()

        logging.info("🎁 GiftService initialized")
    
    # ==========================================================================
    # GIFT CRUD OPERATIONS
    # ==========================================================================
    
    async def get_gift_by_id(self, gift_id: int) -> Optional[Gift]:
        """
        Lädt einzelnes Gift by ID
        
        Args:
            gift_id: Gift ID
            
        Returns:
            Gift oder None
        """
        try:
            gift = self.db.query(Gift).filter(Gift.id == gift_id).first()
            return gift
            
        except Exception as e:
            logging.error(f"Error loading gift {gift_id}: {e}")
            return None
    
    async def get_gifts_by_category(
        self, 
        category_name: str, 
        limit: int = 50,
        budget_min: Optional[float] = None,
        budget_max: Optional[float] = None
    ) -> List[Gift]:
        """
        Lädt Gifts by Category mit optional Budget Filter
        
        Args:
            category_name: Category Name
            limit: Max anzahl Results
            budget_min: Minimum Budget
            budget_max: Maximum Budget
            
        Returns:
            List von Gifts
        """
        try:
            query = self.db.query(Gift).join(GiftCategory).filter(
                GiftCategory.name.ilike(f"%{category_name}%")
            )
            
            # Budget Filtering
            if budget_min is not None:
                query = query.filter(Gift.price >= budget_min)
            if budget_max is not None:
                query = query.filter(Gift.price <= budget_max)
            
            # Order by popularity/rating
            query = query.order_by(Gift.popularity_score.desc())
            
            gifts = query.limit(limit).all()
            return gifts
            
        except Exception as e:
            logging.error(f"Error loading gifts by category {category_name}: {e}")
            return []
    
    async def search_gifts(
        self,
        search_term: str,
        limit: int = 20,
        budget_min: Optional[float] = None,
        budget_max: Optional[float] = None,
        categories: Optional[List[str]] = None,
        exclude_allergens: Optional[List[str]] = None
    ) -> List[Gift]:
        """
        🔍 Advanced Gift Search mit Filtering
        
        Args:
            search_term: Search Query
            limit: Max Results
            budget_min/max: Budget Range
            categories: Category Filter
            exclude_allergens: Allergen Exclusions
            
        Returns:
            Filtered & Ranked Gift List
        """
        try:
            # Base Query
            query = self.db.query(Gift)
            
            # Text Search (Name + Description)
            if search_term:
                search_filter = or_(
                    Gift.name.ilike(f"%{search_term}%"),
                    Gift.description.ilike(f"%{search_term}%"),
                    Gift.tags.ilike(f"%{search_term}%")
                )
                query = query.filter(search_filter)
            
            # Budget Filtering
            if budget_min is not None:
                query = query.filter(Gift.price >= budget_min)
            if budget_max is not None:
                query = query.filter(Gift.price <= budget_max)
            
            # Category Filtering
            if categories:
                query = query.join(GiftCategory).filter(
                    GiftCategory.name.in_(categories)
                )
            
            # Allergen Exclusions
            if exclude_allergens:
                for allergen in exclude_allergens:
                    query = query.filter(
                        ~Gift.allergen_info.ilike(f"%{allergen}%")
                    )
            
            # Ranking: Popularity + Relevance
            query = query.order_by(
                Gift.popularity_score.desc(),
                Gift.rating.desc(),
                Gift.created_at.desc()
            )
            
            gifts = query.limit(limit).all()
            return gifts
            
        except Exception as e:
            logging.error(f"Error searching gifts with term '{search_term}': {e}")
            return []
    
    # ==========================================================================
    # POPULAR GIFTS & ANALYTICS
    # ==========================================================================
    
    async def get_popular_gifts_by_occasion(
        self, 
        occasion: str, 
        limit: int = 10
    ) -> List[Gift]:
        """
        Lädt beliebte Gifts für specific Occasion
        
        Args:
            occasion: Occasion (Birthday, Christmas, etc.)
            limit: Max Results
            
        Returns:
            List beliebter Gifts für Occasion
        """
        cache_key = f"popular_gifts_{occasion}_{limit}"
        
        # Cache Check
        if cache_key in self._popular_gifts_cache:
            cache_data = self._popular_gifts_cache[cache_key]
            if (datetime.now() - cache_data["cached_at"]).seconds < 1800:  # 30min Cache
                return cache_data["data"]
        
        try:
            # Query Popular Gifts for Occasion
            gifts = self.db.query(Gift).filter(
                or_(
                    Gift.occasions.ilike(f"%{occasion}%"),
                    Gift.tags.ilike(f"%{occasion}%")
                )
            ).order_by(
                Gift.popularity_score.desc(),
                Gift.rating.desc()
            ).limit(limit).all()
            
            # Cache Results
            self._popular_gifts_cache[cache_key] = {
                "data": gifts,
                "cached_at": datetime.now()
            }
            
            return gifts
            
        except Exception as e:
            logging.error(f"Error loading popular gifts for occasion '{occasion}': {e}")
            return []
    
    async def filter_gifts_by_personality(
        self,
        personality_preferences: Dict[str, Any],
        budget_min: Optional[float] = None,
        budget_max: Optional[float] = None,
        limit: int = 50
    ) -> List[Gift]:
        """
        Filtert Gifts basierend auf Personality Preferences
        
        Args:
            personality_preferences: Dict mit Preferences
            budget_min/max: Budget Constraints
            limit: Max Results
            
        Returns:
            Filtered Gift List für AI-Engine
        """
        try:
            # Base Query
            query = self.db.query(Gift)
            
            # Budget Filtering
            if budget_min is not None:
                query = query.filter(Gift.price >= budget_min)
            if budget_max is not None:
                query = query.filter(Gift.price <= budget_max)
            
            # Allergen Exclusions
            allergies = personality_preferences.get('allergies', [])
            for allergy in allergies:
                query = query.filter(
                    ~Gift.allergen_info.ilike(f"%{allergy}%")
                )
            
            # Dislikes Exclusions
            dislikes = personality_preferences.get('dislikes', [])
            for dislike in dislikes:
                query = query.filter(
                    and_(
                        ~Gift.name.ilike(f"%{dislike}%"),
                        ~Gift.description.ilike(f"%{dislike}%"),
                        ~Gift.tags.ilike(f"%{dislike}%")
                    )
                )
            
            # Order by match potential
            query = query.order_by(
                Gift.popularity_score.desc(),
                Gift.rating.desc()
            )
            
            gifts = query.limit(limit).all()
            return gifts
            
        except Exception as e:
            logging.error(f"Error filtering gifts by personality: {e}")
            return []
      
    
    def get_ai_catalog_recommendations(self, user_id: str, session_data: Dict = None) -> Dict:
        """
        Neue Methode: Holt AI-Empfehlungen aus dem Produktionskatalog
        """
        return self.catalog_service.get_ai_recommendations_for_user(user_id, session_data)
    
    def sync_product_catalog(self) -> Dict:
        """
        Sync-Funktion für Admin: Katalog zu Datenbank synchronisieren
        """
        return self.catalog_service.sync_catalog_to_database()



# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ['GiftService']