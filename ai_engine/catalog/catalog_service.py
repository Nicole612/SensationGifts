"""
ai_engine/catalog/catalog_service.py - ERWEITERTE VERSION

🔄 DIE BESTEHENDE DATEI + HELDENREISE-KATALOG
===============================================

Diese Datei erweitert die bestehende catalog_service.py um den 
neuen Heldenreise-Katalog, ohne die alte Funktionalität zu verlieren.

"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

# Imports aus dem bestehenden System
from app.models.personality import PersonalityProfile
from app.models.gift import Gift, GiftCategory
from app.models.recommendation import RecommendationSession, Recommendation, AIModelType
from app.extensions import db

# Imports: Alter Katalog (BEHALTEN für Rückwärtskompatibilität)
from .gift_catalog_generator import GeschenkKatalogService, GESCHENK_KATALOG

# Imports: Neuer Heldenreise-Katalog (NEU!)
from .heroic_journey_catalog import (
    HEROIC_GIFT_CATALOG, 
    HeroicJourneyIntegration,
    LifeStage,
    HeroicGiftBox
)

logger = logging.getLogger(__name__)


class CatalogIntegrationService:
    """
    ERWEITERTE Integration zwischen Katalogen und dem Flask-System
    
    Unterstützt jetzt:
    - Alter GESCHENK_KATALOG (bestehende Funktionalität)
    - Neuer HEROIC_GIFT_CATALOG (Heldenreise-Konzept)
    - Schrittweise Migration möglich
    """
    
    def __init__(self, use_heroic_catalog: bool = True):
        """
        Args:
            use_heroic_catalog: True = Neuer Heldenreise-Katalog, False = Alter Katalog
        """
        self.use_heroic_catalog = use_heroic_catalog
        self.logger = logger
        
        # Services für beide Kataloge
        self.old_catalog_service = GeschenkKatalogService()  # Alter Katalog
        self.heroic_integration = HeroicJourneyIntegration()  # Neuer Katalog
        
        # 🚀 ENHANCED INTEGRATION FEATURES
        self.integration_enabled = True
        self.performance_tracking_enabled = True
        self.adaptive_catalog_selection = True
        
        # Performance metrics
        self.performance_metrics = {
            "total_recommendations": 0,
            "successful_recommendations": 0,
            "catalog_usage": {"old": 0, "heroic": 0},
            "average_response_time": 0.0,
            "user_satisfaction": 0.0
        }
        
        # Integration state
        self.integration_state = {
            "catalog_sync_status": "pending",
            "last_sync_time": None,
            "sync_errors": [],
            "optimization_enabled": True
        }
        
        self.logger.info(f"🚀 Enhanced CatalogIntegrationService initialisiert mit {'Heldenreise' if use_heroic_catalog else 'klassischem'} Katalog")
    
    def sync_catalog_to_database(self, catalog_type: str = "auto") -> Dict[str, Any]:
        """
        Synchronisiert Katalog(e) mit der SQL-Datenbank
        
        Args:
            catalog_type: "old", "heroic", oder "auto" (basierend auf use_heroic_catalog)
            
        Returns:
            Dict mit Sync-Ergebnissen
        """
        
        if catalog_type == "auto":
            catalog_type = "heroic" if self.use_heroic_catalog else "old"
        
        try:
            # Enhanced sync with optimization
            if self.integration_enabled:
                self._update_sync_status("in_progress")
            
            if catalog_type == "heroic":
                result = self._sync_heroic_catalog()
            elif catalog_type == "old":
                result = self._sync_old_catalog()
            elif catalog_type == "both":
                # Beide Kataloge synchronisieren mit Enhanced Features
                old_result = self._sync_old_catalog()
                heroic_result = self._sync_heroic_catalog()
                
                result = {
                    'success': old_result['success'] and heroic_result['success'],
                    'old_catalog': old_result,
                    'heroic_catalog': heroic_result,
                    'total_synced': old_result.get('synced_products', 0) + heroic_result.get('synced_products', 0)
                }
            else:
                result = {'success': False, 'error': f'Unbekannter catalog_type: {catalog_type}'}
            
            # Update integration state
            if self.integration_enabled:
                self._update_sync_status("completed" if result['success'] else "failed")
                if result['success']:
                    self.integration_state["last_sync_time"] = datetime.now()
                else:
                    self.integration_state["sync_errors"].append(result.get('error', 'Unknown error'))
            
            return result
        
        except Exception as e:
            self.logger.error(f"❌ Enhanced Katalog-Sync fehlgeschlagen: {str(e)}")
            if self.integration_enabled:
                self._update_sync_status("failed")
                self.integration_state["sync_errors"].append(str(e))
            return {'success': False, 'error': str(e)}
    
    def _sync_old_catalog(self) -> Dict[str, Any]:
        """Synchronisiert den alten GESCHENK_KATALOG (die bestehende Logik)"""
        
        try:
            synced_count = 0
            
            for product_id, product_data in GESCHENK_KATALOG.items():
                
                # Prüfe ob Produkt bereits existiert
                existing_gift = Gift.query.filter_by(
                    name=product_data['name']
                ).first()
                
                if existing_gift:
                    self.logger.info(f"Überspringe existierendes Produkt: {product_data['name']}")
                    continue
                
                # Erstelle/finde Kategorie
                category = self._get_or_create_category(product_data['category'])
                
                # Erstelle Gift für Basic-Variante
                basic_variant = product_data['price_variants']['basic']
                
                new_gift = Gift(
                    name=product_data['name'],
                    short_description=product_data['short_description'],
                    long_description=product_data['long_description'],
                    price=basic_variant['price'],
                    category_id=category.id,
                    
                    # Felder aus altem Katalog
                    age_categories=','.join(product_data['age_categories']),
                    target_age_min=product_data['target_age_min'],
                    target_age_max=product_data['target_age_max'],
                    emotional_story=product_data['emotional_story'],
                    
                    # Template Info
                    is_generated=True,
                    template_name=product_id,
                    catalog_source="old_catalog",  # NEU: Kennzeichne Quelle
                    
                    # JSON Felder
                    personality_match_scores=str(product_data.get('personality_match_scores', {})),
                    relationship_suitability=str(product_data.get('relationship_suitability', {})),
                    occasion_suitability=str(product_data.get('occasion_suitability', {})),
                    
                    # Metadaten
                    is_active=True,
                    is_featured=True if 'premium' in product_data['price_variants'] else False
                )
                
                db.session.add(new_gift)
                synced_count += 1
                
                self.logger.info(f"Erstelle altes Produkt: {product_data['name']}")
            
            db.session.commit()
            self.logger.info(f"✅ Alter Katalog-Sync abgeschlossen: {synced_count} neue Produkte")
            
            return {
                'success': True,
                'synced_products': synced_count,
                'total_catalog_products': len(GESCHENK_KATALOG),
                'catalog_type': 'old'
            }
            
        except Exception as e:
            db.session.rollback()
            self.logger.error(f"❌ Alter Katalog-Sync fehlgeschlagen: {str(e)}")
            return {'success': False, 'error': str(e), 'catalog_type': 'old'}
    
    def _sync_heroic_catalog(self) -> Dict[str, Any]:
        """Synchronisiert den neuen HEROIC_GIFT_CATALOG"""
        
        try:
            synced_count = 0
            
            # Hole Daten vom Heldenreise-Katalog
            heroic_data = self.heroic_integration.export_for_database_sync()
            
            for product_data in heroic_data:
                
                # Prüfe ob Produkt bereits existiert
                existing_gift = Gift.query.filter_by(
                    name=product_data['name']
                ).first()
                
                if existing_gift:
                    self.logger.info(f"Überspringe existierendes Heldenreise-Produkt: {product_data['name']}")
                    continue
                
                # Erstelle/finde Kategorie
                category = self._get_or_create_category(product_data['category'])
                
                # Erstelle Gift mit allen Heldenreise-Daten
                new_gift = Gift(
                    name=product_data['name'],
                    short_description=product_data['short_description'],
                    long_description=product_data['long_description'],
                    price=product_data['price'],
                    category_id=category.id,
                    
                    # Heldenreise-spezifische Felder
                    age_categories=product_data['age_categories'],
                    target_age_min=product_data['target_age_min'],
                    target_age_max=product_data['target_age_max'],
                    emotional_story=product_data['emotional_story'],
                    
                    # Template Info
                    is_generated=True,
                    template_name=product_data['template_name'],
                    catalog_source="heroic_catalog",  # NEU: Kennzeichne Quelle
                    
                    # JSON Felder (erweitert für Heldenreise)
                    personality_match_scores=product_data['personality_match_scores'],
                    relationship_suitability=product_data['relationship_suitability'],
                    ai_prompt_keywords=product_data['ai_prompt_keywords'],
                    emotional_tags=product_data['emotional_tags'],
                    heldenreise_data=product_data.get('heldenreise_data'),  
                    
                    # Metadaten
                    is_active=product_data['is_active'],
                    is_featured=product_data['is_featured']
                )
                
                db.session.add(new_gift)
                synced_count += 1
                
                self.logger.info(f"Erstelle Heldenreise-Produkt: {product_data['name']}")
            
            db.session.commit()
            self.logger.info(f"✅ Heldenreise Katalog-Sync abgeschlossen: {synced_count} neue Produkte")
            
            return {
                'success': True,
                'synced_products': synced_count,
                'total_catalog_products': len(HEROIC_GIFT_CATALOG) * 3,  # 3 Varianten
                'catalog_type': 'heroic'
            }
            
        except Exception as e:
            db.session.rollback()
            self.logger.error(f"❌ Heldenreise Katalog-Sync fehlgeschlagen: {str(e)}")
            return {'success': False, 'error': str(e), 'catalog_type': 'heroic'}

    def _update_sync_status(self, status: str):
        """Update sync status for integration tracking"""
        try:
            self.integration_state["catalog_sync_status"] = status
            self.logger.info(f"📊 Sync status updated: {status}")
        except Exception as e:
            self.logger.warning(f"Sync status update failed: {e}")

    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status and metrics"""
        try:
            return {
                "integration_enabled": self.integration_enabled,
                "performance_tracking_enabled": self.performance_tracking_enabled,
                "adaptive_catalog_selection": self.adaptive_catalog_selection,
                "integration_state": self.integration_state,
                "performance_metrics": self.performance_metrics
            }
        except Exception as e:
            self.logger.warning(f"Integration status retrieval failed: {e}")
            return {}

    def optimize_catalog_performance(self) -> Dict[str, Any]:
        """Optimize catalog performance based on usage patterns"""
        try:
            if not self.performance_tracking_enabled:
                return {"success": False, "error": "Performance tracking disabled"}
            
            # Analyze usage patterns
            usage_analysis = self._analyze_catalog_usage()
            
            # Apply optimizations
            optimizations = self._apply_catalog_optimizations(usage_analysis)
            
            return {
                "success": True,
                "usage_analysis": usage_analysis,
                "optimizations_applied": optimizations
            }
            
        except Exception as e:
            self.logger.error(f"Catalog performance optimization failed: {e}")
            return {"success": False, "error": str(e)}

    def _analyze_catalog_usage(self) -> Dict[str, Any]:
        """Analyze catalog usage patterns for optimization"""
        try:
            total_recommendations = self.performance_metrics["total_recommendations"]
            successful_recommendations = self.performance_metrics["successful_recommendations"]
            
            analysis = {
                "success_rate": successful_recommendations / max(1, total_recommendations),
                "catalog_preference": self.performance_metrics["catalog_usage"],
                "average_response_time": self.performance_metrics["average_response_time"],
                "user_satisfaction": self.performance_metrics["user_satisfaction"]
            }
            
            # Determine optimal catalog strategy
            if analysis["success_rate"] > 0.8:
                analysis["recommended_strategy"] = "current"
            elif self.performance_metrics["catalog_usage"]["heroic"] > self.performance_metrics["catalog_usage"]["old"]:
                analysis["recommended_strategy"] = "heroic"
            else:
                analysis["recommended_strategy"] = "old"
            
            return analysis
            
        except Exception as e:
            self.logger.warning(f"Catalog usage analysis failed: {e}")
            return {}

    def _apply_catalog_optimizations(self, usage_analysis: Dict[str, Any]) -> List[str]:
        """Apply catalog optimizations based on usage analysis"""
        try:
            optimizations = []
            
            # Switch catalog strategy if recommended
            if usage_analysis.get("recommended_strategy") != "current":
                new_strategy = usage_analysis.get("recommended_strategy")
                if new_strategy == "heroic":
                    self.use_heroic_catalog = True
                    optimizations.append("switched_to_heroic_catalog")
                elif new_strategy == "old":
                    self.use_heroic_catalog = False
                    optimizations.append("switched_to_old_catalog")
            
            # Enable adaptive selection if performance is good
            if usage_analysis.get("success_rate", 0) > 0.7:
                self.adaptive_catalog_selection = True
                optimizations.append("enabled_adaptive_catalog_selection")
            
            return optimizations
            
        except Exception as e:
            self.logger.warning(f"Catalog optimization application failed: {e}")
            return []

    def track_recommendation_performance(self, user_id: str, recommendations: List[Dict], success: bool, response_time: float):
        """Track recommendation performance for optimization"""
        try:
            if not self.performance_tracking_enabled:
                return
            
            # Update basic metrics
            self.performance_metrics["total_recommendations"] += 1
            if success:
                self.performance_metrics["successful_recommendations"] += 1
            
            # Track catalog usage
            catalog_used = "heroic" if self.use_heroic_catalog else "old"
            self.performance_metrics["catalog_usage"][catalog_used] += 1
            
            # Update response time
            current_avg = self.performance_metrics["average_response_time"]
            total_recommendations = self.performance_metrics["total_recommendations"]
            self.performance_metrics["average_response_time"] = (
                (current_avg * (total_recommendations - 1) + response_time) / total_recommendations
            )
            
            # Track user satisfaction (placeholder - would need actual feedback)
            if success:
                self.performance_metrics["user_satisfaction"] = min(1.0, 
                    self.performance_metrics["user_satisfaction"] + 0.1)
            else:
                self.performance_metrics["user_satisfaction"] = max(0.0, 
                    self.performance_metrics["user_satisfaction"] - 0.1)
            
        except Exception as e:
            self.logger.warning(f"Performance tracking failed: {e}")

    def get_enhanced_ai_recommendations_for_user(self, user_id: str, session_data: Dict = None) -> Dict[str, Any]:
        """Enhanced AI recommendations with performance tracking and optimization"""
        try:
            start_time = datetime.now()
            
            # Get recommendations using existing method
            result = self.get_ai_recommendations_for_user(user_id, session_data)
            
            # Track performance
            response_time = (datetime.now() - start_time).total_seconds()
            success = result.get('success', False)
            
            if self.performance_tracking_enabled:
                self.track_recommendation_performance(user_id, result.get('recommendations', []), success, response_time)
            
            # Apply adaptive optimizations
            if self.adaptive_catalog_selection and self.integration_enabled:
                self._apply_adaptive_optimizations(result, user_id)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Enhanced AI recommendations failed: {e}")
            return {"success": False, "error": str(e), "recommendations": []}

    def _apply_adaptive_optimizations(self, result: Dict[str, Any], user_id: str):
        """Apply adaptive optimizations based on user patterns"""
        try:
            # Analyze user-specific patterns
            user_patterns = self._analyze_user_patterns(user_id)
            
            # Apply user-specific optimizations
            if user_patterns.get("prefers_heroic_catalog"):
                self.use_heroic_catalog = True
            elif user_patterns.get("prefers_old_catalog"):
                self.use_heroic_catalog = False
            
            # Apply performance-based optimizations
            if result.get('success') and len(result.get('recommendations', [])) > 0:
                # Success - maintain current strategy
                pass
            else:
                # Failure - try alternative strategy
                self.use_heroic_catalog = not self.use_heroic_catalog
                
        except Exception as e:
            self.logger.warning(f"Adaptive optimization failed: {e}")

    def _analyze_user_patterns(self, user_id: str) -> Dict[str, Any]:
        """Analyze user-specific patterns for optimization"""
        try:
            # This would analyze historical user data
            # For now, return default patterns
            return {
                "prefers_heroic_catalog": False,
                "prefers_old_catalog": False,
                "success_rate": 0.8
            }
        except Exception as e:
            self.logger.warning(f"User pattern analysis failed: {e}")
            return {}
    
    def get_ai_recommendations_for_user(self, user_id: str, session_data: Dict = None) -> Dict[str, Any]:
        """
        ERWEITERTE AI-Empfehlungen für einen User
        
        Nutzt jetzt:
        - Heldenreise-Katalog für altersgerechte Empfehlungen (Standard)
        - Alter Katalog als Fallback
        
        Args:
            user_id: User-ID aus deinem System
            session_data: Optional - Zusätzliche Session-Daten
            
        Returns:
            Dict mit AI-optimierten Geschenk-Empfehlungen
        """
        
        try:
            # Hole neuestes PersonalityProfile
            personality_profile = PersonalityProfile.query.filter_by(
                buyer_user_id=user_id
            ).order_by(PersonalityProfile.created_at.desc()).first()
            
            if not personality_profile:
                return {
                    'success': False,
                    'error': 'Kein PersonalityProfile gefunden. Bitte erst Persönlichkeitstest machen.',
                    'recommendations': []
                }
            
            # Session-Daten oder Defaults aus Profile
            budget_range = session_data.get('budget_range', (
                personality_profile.budget_min or 10,
                personality_profile.budget_max or 500
            )) if session_data else (
                personality_profile.budget_min or 10,
                personality_profile.budget_max or 500
            )
            
            relationship = session_data.get('relationship') or getattr(personality_profile, 'relationship', 'friend')
            occasion = session_data.get('occasion') or getattr(personality_profile, 'occasion', 'birthday')
            recipient_age = session_data.get('recipient_age') or getattr(personality_profile, 'recipient_age', 25)
            
            recommendations = []
            
            if self.use_heroic_catalog:
                # Nutze NEUEN Heldenreise-Katalog
                recommendations = self._get_heroic_recommendations(
                    personality_profile, budget_range, relationship, occasion, recipient_age
                )
                
                # Fallback: Wenn keine Heldenreise-Empfehlungen, nutze alten Katalog
                if not recommendations:
                    self.logger.info("Kein Heldenreise-Match gefunden, fallback zu altem Katalog")
                    recommendations = self._get_old_catalog_recommendations(
                        personality_profile, budget_range, relationship, occasion
                    )
            else:
                # Nutze ALTEN Katalog
                recommendations = self._get_old_catalog_recommendations(
                    personality_profile, budget_range, relationship, occasion
                )
            
            # Erweitere mit deinen System-Daten
            enhanced_recommendations = []
            
            for rec in recommendations:
                
                # Suche entsprechendes Gift in der DB
                db_gift = Gift.query.filter_by(
                    template_name=rec.get('product_id') or rec.get('template_name')
                ).first()
                
                enhanced_rec = {
                    **rec,
                    'database_gift_id': db_gift.id if db_gift else None,
                    'purchase_url': f"/gifts/{db_gift.id}" if db_gift else None,
                    'personalization_url': f"/personalize/{rec.get('product_id') or rec.get('template_name')}",
                    'generated_at': datetime.now().isoformat(),
                    'catalog_source': 'heroic' if self.use_heroic_catalog else 'old'
                }
                
                enhanced_recommendations.append(enhanced_rec)
            
            return {
                'success': True,
                'user_id': user_id,
                'personality_summary': getattr(personality_profile, 'personality_summary', 'No summary available'),
                'recommendations': enhanced_recommendations,
                'total_found': len(enhanced_recommendations),
                'budget_range': budget_range,
                'catalog_used': 'heroic' if self.use_heroic_catalog else 'old',
                'search_criteria': {
                    'relationship': relationship,
                    'occasion': occasion,
                    'recipient_age': recipient_age
                }
            }
            
        except Exception as e:
            self.logger.error(f"AI-Empfehlungen fehlgeschlagen: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'recommendations': []
            }
    
    def _get_heroic_recommendations(self, personality_profile, budget_range, relationship, occasion, recipient_age) -> List[Dict]:
        """Hole Empfehlungen aus dem Heldenreise-Katalog"""
        
        try:
            # Nutze die Heldenreise-Integration
            matching_gifts = self.heroic_integration.get_gifts_for_age_group(recipient_age)
            
            recommendations = []
            
            for gift in matching_gifts:
                # Prüfe Budget
                affordable_variants = []
                if budget_range[0] <= gift.price_basic <= budget_range[1]:
                    affordable_variants.append(('basic', gift.price_basic))
                if budget_range[0] <= gift.price_medium <= budget_range[1]:
                    affordable_variants.append(('medium', gift.price_medium))
                if budget_range[0] <= gift.price_premium <= budget_range[1]:
                    affordable_variants.append(('premium', gift.price_premium))
                
                if not affordable_variants:
                    continue
                
                # Berechne Match-Score
                personality_score = self._calculate_heroic_personality_match(personality_profile, gift)
                relationship_score = gift.relationship_contexts.get(relationship, 0.5)
                
                # Occasion Matching (vereinfacht)
                occasion_score = 0.8 if occasion in ['birthday', 'christmas', 'graduation'] else 0.6
                
                total_score = (personality_score * 0.5 + relationship_score * 0.3 + occasion_score * 0.2)
                
                if total_score > 0.4:  # Minimum threshold
                    best_variant = min(affordable_variants, key=lambda x: x[1])
                    
                    recommendations.append({
                        'product_id': gift.id,
                        'template_name': gift.id,
                        'product_name': gift.name,
                        'recommended_variant': best_variant[0],
                        'recommended_price': best_variant[1],
                        'total_match_score': total_score,
                        'emotional_story': gift.emotional_story,
                        'transformation_goal': gift.heroic_goal,
                        'ai_reasoning': f"Heldenreise-Match für {recipient_age}-Jährige: {gift.transformation}",
                        'match_reasons': {
                            'personality_match': personality_score,
                            'relationship_fit': relationship_score,
                            'occasion_suitability': occasion_score,
                        }
                    })
            
            # Sortiere nach Score
            recommendations.sort(key=lambda x: x['total_match_score'], reverse=True)
            return recommendations[:5]  # Top 5
            
        except Exception as e:
            self.logger.error(f"Heldenreise-Empfehlungen fehlgeschlagen: {str(e)}")
            return []
    
    def _get_old_catalog_recommendations(self, personality_profile, budget_range, relationship, occasion) -> List[Dict]:
        """Hole Empfehlungen aus dem alten Katalog (deine bestehende Logic)"""
        
        try:
            recommendations = self.old_catalog_service.get_ai_recommendations(
                personality_profile=personality_profile,
                budget_range=budget_range,
                relationship=relationship,
                occasion=occasion
            )
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Alte Katalog-Empfehlungen fehlgeschlagen: {str(e)}")
            return []
    
    def _calculate_heroic_personality_match(self, profile, gift: HeroicGiftBox) -> float:
        """Berechne Persönlichkeits-Match für Heldenreise-Geschenke"""
        
        total_match = 0.0
        matched_traits = 0
        
        # Big Five Matching
        big_five_traits = {
            'high_openness': getattr(profile, 'openness', 0.5),
            'high_conscientiousness': getattr(profile, 'conscientiousness', 0.5),
            'high_extraversion': getattr(profile, 'extraversion', 0.5),
            'high_agreeableness': getattr(profile, 'agreeableness', 0.5),
            'high_neuroticism': getattr(profile, 'neuroticism', 0.5)
        }
        
        for trait, gift_score in gift.personality_match.items():
            if trait in big_five_traits:
                user_score = big_five_traits[trait]
                if user_score > 0.6:  # Nur wenn stark ausgeprägt
                    total_match += gift_score * user_score
                    matched_traits += 1
            elif hasattr(profile, trait.replace('high_', '')):
                # Andere Traits
                if getattr(profile, trait.replace('high_', ''), False):
                    total_match += gift_score
                    matched_traits += 1
        
        return total_match / matched_traits if matched_traits > 0 else 0.5
    
    def create_recommendation_session(self, user_id: str, ai_recommendations: List[Dict], 
                                    ai_model: str = "catalog_ai") -> Optional[RecommendationSession]:
        """
        Erstelle RecommendationSession (erweitert für beide Kataloge)
        """
        
        try:
            personality_profile = PersonalityProfile.query.filter_by(
                buyer_user_id=user_id
            ).order_by(PersonalityProfile.created_at.desc()).first()
            
            if not personality_profile:
                return None
            
            # Bestimme AI-Model basierend auf verwendetem Katalog
            model_used = AIModelType.OPENAI_GPT4  # Default
            prompt_version = f"{'heroic_' if self.use_heroic_catalog else 'classic_'}catalog_v1.0"
            
            # Erstelle Session
            session = RecommendationSession(
                personality_profile_id=personality_profile.id,
                ai_model_used=model_used,
                prompt_version=prompt_version,
                recommendations_count=len(ai_recommendations),
                overall_confidence=sum(r.get('total_match_score', 0.5) for r in ai_recommendations) / len(ai_recommendations) if ai_recommendations else 0.5,
                ai_reasoning=f"{'Heldenreise-' if self.use_heroic_catalog else 'Klassischer '}Katalog mit KI-Persönlichkeitsmatching"
            )
            
            db.session.add(session)
            db.session.flush()
            
            # Erstelle einzelne Recommendations
            for i, rec in enumerate(ai_recommendations, 1):
                
                # Finde Gift in DB
                db_gift = Gift.query.filter_by(
                    template_name=rec.get('product_id') or rec.get('template_name')
                ).first()
                
                if db_gift:
                    recommendation = Recommendation(
                        session_id=session.id,
                        personality_profile_id=personality_profile.id,
                        gift_id=db_gift.id,
                        rank_position=i,
                        match_score=rec.get('total_match_score', 0.5),
                        confidence_score=rec.get('total_match_score', 0.5),
                        reasoning=rec.get('ai_reasoning', ''),
                        
                        # Zusätzliche Daten
                        personality_match_factors=str(rec.get('match_reasons', {})),
                        occasion_relevance=rec.get('match_reasons', {}).get('occasion_suitability', 0.7),
                        budget_fit=1.0 if rec.get('match_reasons', {}).get('budget_friendly') else 0.5,
                    )
                    
                    db.session.add(recommendation)
            
            db.session.commit()
            self.logger.info(f"✅ RecommendationSession erstellt: {session.id} ({'Heldenreise' if self.use_heroic_catalog else 'Klassisch'})")
            
            return session
            
        except Exception as e:
            db.session.rollback()
            self.logger.error(f"❌ RecommendationSession Erstellung fehlgeschlagen: {str(e)}")
            return None
    
    def _get_or_create_category(self, category_name: str) -> GiftCategory:
        """Erstelle oder finde Kategorie (bestehende Logik)"""
        
        category = GiftCategory.query.filter_by(name=category_name).first()
        
        if not category:
            category = GiftCategory(
                name=category_name,
                slug=category_name.lower().replace(' ', '_').replace('&', 'und'),
                description=f"Kategorie für {category_name}",
                is_active=True
            )
            db.session.add(category)
            db.session.flush()
        
        return category
    
    def get_catalog_statistics(self) -> Dict[str, Any]:
        """
        NEU: Zeigt Statistiken über beide Kataloge
        """
        
        old_stats = {
            'products': len(GESCHENK_KATALOG),
            'categories': len(set(p['category'] for p in GESCHENK_KATALOG.values())),
            'price_range': {
                'min': min(min(p['price_variants'][v]['price'] for v in p['price_variants']) for p in GESCHENK_KATALOG.values()),
                'max': max(max(p['price_variants'][v]['price'] for v in p['price_variants']) for p in GESCHENK_KATALOG.values())
            }
        }
        
        heroic_stats = {
            'products': len(HEROIC_GIFT_CATALOG),
            'total_variants': len(HEROIC_GIFT_CATALOG) * 3,
            'age_groups': len(set(g.life_stage for g in HEROIC_GIFT_CATALOG.values())),
            'price_range': {
                'min': min(g.price_basic for g in HEROIC_GIFT_CATALOG.values()),
                'max': max(g.price_premium for g in HEROIC_GIFT_CATALOG.values())
            }
        }
        
        return {
            'active_catalog': 'heroic' if self.use_heroic_catalog else 'old',
            'old_catalog': old_stats,
            'heroic_catalog': heroic_stats,
            'recommendation': 'Nutze Heldenreise-Katalog für bessere altersgerechte Empfehlungen'
        }


# =============================================================================
# BACKWARDS COMPATIBILITY & MIGRATION
# =============================================================================

def migrate_to_heroic_catalog(test_mode: bool = True) -> Dict[str, Any]:
    """
    Hilfsfunktion für Migration vom alten zum neuen Katalog
    
    Args:
        test_mode: Wenn True, nur Simulation
        
    Returns:
        Migration-Bericht
    """
    
    migration_report = {
        'old_products_found': 0,
        'heroic_products_available': len(HEROIC_GIFT_CATALOG),
        'migration_recommended': False,
        'benefits': [],
        'next_steps': []
    }
    
    if SYSTEM_AVAILABLE:
        old_products = Gift.query.filter_by(catalog_source='old_catalog').count()
        heroic_products = Gift.query.filter_by(catalog_source='heroic_catalog').count()
        
        migration_report.update({
            'old_products_found': old_products,
            'heroic_products_in_db': heroic_products,
            'migration_recommended': heroic_products == 0,
        })
    
    migration_report['benefits'] = [
        "🎯 Altersgerechte Heldenreise-Geschenke für alle Lebensphasen",
        "🤖 Verbesserte AI-Prompts für präzisere Empfehlungen",
        "📈 Höherer emotionaler Impact durch Story-basierte Geschenke",
        "🛍️ 3 Preisstufen pro Geschenk für bessere Monetarisierung",
        "🔄 Bessere Personalisierung durch symbolische Items"
    ]
    
    migration_report['next_steps'] = [
        "1. Heldenreise-Katalog in DB synchronisieren",
        "2. A/B Testing zwischen alten und neuen Katalog",
        "3. Schrittweise Umstellung der AI-Empfehlungen",
        "4. Marketing-Update für Heldenreise-Konzept"
    ]
    
    return migration_report


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_catalog_service(heroic: bool = True) -> CatalogIntegrationService:
    """
    Convenience-Funktion für schnelle Service-Erstellung
    
    Args:
        heroic: True für Heldenreise-Katalog, False für alten Katalog
        
    Returns:
        Konfigurierter CatalogIntegrationService
    """
    return CatalogIntegrationService(use_heroic_catalog=heroic)


def quick_sync_heroic_catalog() -> Dict[str, Any]:
    """
    Quick-Sync für Heldenreise-Katalog (für Testing)
    """
    service = get_catalog_service(heroic=True)
    return service.sync_catalog_to_database(catalog_type="heroic")


def quick_test_recommendations(user_id: str, recipient_age: int = 25) -> Dict[str, Any]:
    """
    Quick-Test für Empfehlungen mit Heldenreise-Katalog
    """
    service = get_catalog_service(heroic=True)
    return service.get_ai_recommendations_for_user(
        user_id=user_id,
        session_data={'recipient_age': recipient_age}
    )


# =============================================================================
# SYSTEM COMPATIBILITY CHECK
# =============================================================================

try:
    from app.models.personality import PersonalityProfile
    from app.models.gift import Gift
    SYSTEM_AVAILABLE = True
    logger.info("✅ System-Integration verfügbar")
except ImportError:
    SYSTEM_AVAILABLE = False
    logger.warning("⚠️ System-Integration nicht verfügbar - nur Standalone-Modus")


if __name__ == "__main__":
    print("🚀 ERWEITERTE CATALOG INTEGRATION SERVICE")
    print("=" * 60)
    
    # Test beide Kataloge
    old_service = get_catalog_service(heroic=False)
    heroic_service = get_catalog_service(heroic=True)
    
    print("📊 Katalog-Vergleich:")
    old_stats = old_service.get_catalog_statistics()
    heroic_stats = heroic_service.get_catalog_statistics()
    
    print(f"   Alter Katalog: {old_stats['old_catalog']['products']} Produkte")
    print(f"   Heldenreise-Katalog: {heroic_stats['heroic_catalog']['products']} Templates ({heroic_stats['heroic_catalog']['total_variants']} Varianten)")
    
    # Migration-Bericht
    migration = migrate_to_heroic_catalog()
    print(f"\n🔄 Migration-Empfehlung: {'✅ Empfohlen' if migration['migration_recommended'] else '⚠️ Bereits migriert'}")
    
    print(f"\n💡 Nächste Schritte:")
    for step in migration['next_steps'][:3]:
        print(f"   {step}")
    
    print(f"\n🎉 Service bereit für beide Kataloge!")