import unittest
from app import create_app
from app.extensions import db
from app.models.user import User
from app.models.personality import PersonalityProfile
from ai_engine.catalog import CatalogIntegrationService, GESCHENK_KATALOG


class TestCatalogIntegration(unittest.TestCase):
    
    def setUp(self):
        """Test-Setup"""
        self.app = create_app({'TESTING': True, 'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:'})
        self.app_context = self.app.app_context()
        self.app_context.push()
        self.client = self.app.test_client()
        
        db.create_all()
        
        # Test-User erstellen
        self.test_user = User(
            display_name="testuser",
            email="test@example.com"
        )
        db.session.add(self.test_user)
        db.session.commit()
        
        # Test PersonalityProfile erstellen
        self.test_profile = PersonalityProfile(
            buyer_user_id=self.test_user.id,
            recipient_name="Test Person",
            occasion="geburtstag",
            relationship="partner",
            budget_min=50,
            budget_max=200,
            
            # Big Five Scores
            openness=0.8,
            creative_type=True,
            tech_savvy=False
        )
        db.session.add(self.test_profile)
        db.session.commit()
        
        self.catalog_service = CatalogIntegrationService()
    
    def tearDown(self):
        """Test-Cleanup"""
        db.session.remove()
        db.drop_all()
        self.app_context.pop()
    
    def test_catalog_exists(self):
        """Test: Katalog ist vorhanden"""
        self.assertGreater(len(GESCHENK_KATALOG), 0)
        self.assertIn('mutige_helden_box', GESCHENK_KATALOG)
    
    def test_catalog_sync(self):
        """Test: Katalog-Sync funktioniert"""
        result = self.catalog_service.sync_catalog_to_database()
        
        self.assertTrue(result['success'])
        self.assertGreater(result['synced_products'], 0)
    
    def test_ai_recommendations(self):
        """Test: AI-Empfehlungen funktionieren"""
        recommendations = self.catalog_service.get_ai_recommendations_for_user(
            user_id=self.test_user.id
        )
        
        self.assertTrue(recommendations['success'])
        self.assertGreater(len(recommendations['recommendations']), 0)
        
        # Erste Empfehlung prüfen
        first_rec = recommendations['recommendations'][0]
        self.assertIn('product_id', first_rec)
        self.assertIn('total_match_score', first_rec)
        self.assertIn('ai_reasoning', first_rec)
    
    def test_recommendation_session_creation(self):
        """Test: RecommendationSession wird korrekt erstellt"""
        
        # Erst Katalog syncen
        self.catalog_service.sync_catalog_to_database()
        
        # AI-Empfehlungen holen
        recommendations = self.catalog_service.get_ai_recommendations_for_user(
            user_id=self.test_user.id
        )
        
        # Session erstellen
        session = self.catalog_service.create_recommendation_session(
            user_id=self.test_user.id,
            ai_recommendations=recommendations['recommendations']
        )
        
        self.assertIsNotNone(session)
        self.assertGreater(session.recommendations_count, 0)