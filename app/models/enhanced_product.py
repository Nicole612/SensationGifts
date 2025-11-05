# 📁 app/models/enhanced_product.py
# ================================================================
# ENHANCED PRODUCT MODEL - Für emotionalen Online Shop
# ================================================================

from sqlalchemy import Column, String, Integer, Float, Boolean, Text, JSON, DateTime, ForeignKey, Table
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from typing import Dict, List, Optional
import uuid

from app.models.base import BaseModel

# Many-to-many association tables
product_categories = Table(
    'product_categories',
    BaseModel.metadata,
    Column('product_id', String, ForeignKey('products.id'), primary_key=True),
    Column('category_id', String, ForeignKey('categories.id'), primary_key=True)
)

product_emotions = Table(
    'product_emotions',
    BaseModel.metadata,
    Column('product_id', String, ForeignKey('products.id'), primary_key=True),
    Column('emotion_id', String, ForeignKey('emotions.id'), primary_key=True)
)

class EnhancedProduct(BaseModel):
    """
    Enhanced Product Model für emotionalen Online Shop
    Kombiniert klassische E-Commerce Features mit AI-Persönlichkeitsmatching
    """
    __tablename__ = 'products'

    # Basic Product Info
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(200), nullable=False, index=True)
    slug = Column(String(250), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=False)
    short_description = Column(String(500))
    
    # Pricing
    price_basic = Column(Float, nullable=False)
    price_premium = Column(Float)
    currency = Column(String(3), default='EUR')
    cost_price = Column(Float)  # For profit calculation
    
    # Inventory
    stock_quantity = Column(Integer, default=0)
    low_stock_threshold = Column(Integer, default=5)
    track_inventory = Column(Boolean, default=True)
    allow_backorders = Column(Boolean, default=False)
    
    # Physical Properties
    weight = Column(Float)  # in grams
    dimensions = Column(JSON)  # {"length": 10, "width": 5, "height": 3}
    requires_shipping = Column(Boolean, default=True)
    
    # Status & Visibility
    status = Column(String(20), default='active')  # active, inactive, draft
    is_featured = Column(Boolean, default=False)
    is_digital = Column(Boolean, default=False)
    
    # SEO & Marketing
    meta_title = Column(String(60))
    meta_description = Column(String(160))
    seo_keywords = Column(JSON)  # ["geschenk", "personalisiert", "romantisch"]
    
    # AI & Personalization Features
    personality_match = Column(JSON)  # Big Five + Limbic matching scores
    emotional_tags = Column(JSON)    # Primary emotions this gift evokes
    target_demographics = Column(JSON)  # Age, gender, relationship context
    occasion_tags = Column(JSON)     # Birthday, wedding, anniversary, etc.
    
    # Emotional Storytelling
    emotional_story = Column(Text)   # AI-generated emotional story
    user_stories = Column(JSON)      # Real customer stories
    emotional_impact_score = Column(Float, default=0.0)  # 0-10 rating
    
    # Customization Options
    customization_options = Column(JSON)  # Text, images, colors, etc.
    personalization_time = Column(Integer, default=0)  # Days for customization
    
    # Gift Features
    gift_wrap_available = Column(Boolean, default=True)
    gift_message_available = Column(Boolean, default=True)
    gift_packaging_options = Column(JSON)
    
    # Image & Media
    primary_image = Column(String(500))
    image_gallery = Column(JSON)     # Array of image URLs
    lifestyle_images = Column(JSON)  # Lifestyle/context images
    video_url = Column(String(500))
    
    # Reviews & Ratings
    average_rating = Column(Float, default=0.0)
    total_reviews = Column(Integer, default=0)
    emotional_satisfaction_score = Column(Float, default=0.0)
    
    # Analytics
    view_count = Column(Integer, default=0)
    purchase_count = Column(Integer, default=0)
    wishlist_count = Column(Integer, default=0)
    conversion_rate = Column(Float, default=0.0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    categories = relationship("Category", secondary=product_categories, back_populates="products")
    emotions = relationship("Emotion", secondary=product_emotions, back_populates="products")
    reviews = relationship("ProductReview", back_populates="product")
    cart_items = relationship("CartItem", back_populates="product")
    order_items = relationship("OrderItem", back_populates="product")

    def to_dict(self, include_sensitive: bool = False) -> Dict:
        """Convert to dictionary for API responses"""
        data = {
            'id': self.id,
            'name': self.name,
            'slug': self.slug,
            'description': self.description,
            'short_description': self.short_description,
            'price': {
                'basic': self.price_basic,
                'premium': self.price_premium,
                'currency': self.currency
            },
            'stock': {
                'quantity': self.stock_quantity if self.track_inventory else None,
                'in_stock': self.is_in_stock(),
                'low_stock': self.is_low_stock()
            },
            'images': {
                'primary': self.primary_image,
                'gallery': self.image_gallery or [],
                'lifestyle': self.lifestyle_images or []
            },
            'personality_match': self.personality_match or {},
            'emotional_tags': self.emotional_tags or [],
            'emotional_story': self.emotional_story,
            'rating': {
                'average': self.average_rating,
                'count': self.total_reviews,
                'emotional_satisfaction': self.emotional_satisfaction_score
            },
            'customization': {
                'available': bool(self.customization_options),
                'options': self.customization_options or {},
                'production_time': self.personalization_time
            },
            'gift_options': {
                'wrap_available': self.gift_wrap_available,
                'message_available': self.gift_message_available,
                'packaging_options': self.gift_packaging_options or []
            },
            'categories': [cat.to_dict() for cat in self.categories],
            'status': self.status,
            'is_featured': self.is_featured
        }
        
        if include_sensitive:
            data.update({
                'cost_price': self.cost_price,
                'analytics': {
                    'views': self.view_count,
                    'purchases': self.purchase_count,
                    'conversion_rate': self.conversion_rate
                }
            })
        
        return data

    def is_in_stock(self) -> bool:
        """Check if product is in stock"""
        if not self.track_inventory:
            return True
        return self.stock_quantity > 0 or self.allow_backorders

    def is_low_stock(self) -> bool:
        """Check if product is low in stock"""
        if not self.track_inventory:
            return False
        return self.stock_quantity <= self.low_stock_threshold

    def calculate_personality_match(self, user_personality: Dict) -> float:
        """Calculate personality match score for user"""
        if not self.personality_match or not user_personality:
            return 0.0
        
        total_score = 0.0
        factors = 0
        
        # Big Five matching
        big_five_traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        for trait in big_five_traits:
            if trait in self.personality_match and trait in user_personality:
                product_score = self.personality_match[trait]
                user_score = user_personality[trait]
                
                # Calculate compatibility (higher for matching extremes)
                difference = abs(product_score - user_score)
                compatibility = max(0, 100 - (difference * 2))
                
                total_score += compatibility
                factors += 1
        
        # Limbic type bonus
        if 'limbic_type' in self.personality_match and 'limbic_type' in user_personality:
            if self.personality_match['limbic_type'] == user_personality['limbic_type']:
                total_score += 20  # Bonus for exact limbic match
                factors += 1
        
        return total_score / max(factors, 1) if factors > 0 else 0.0

    def get_estimated_delivery(self) -> Dict:
        """Get estimated delivery time"""
        base_days = 2 if self.is_in_stock() else 7
        personalization_days = self.personalization_time or 0
        
        total_days = base_days + personalization_days
        
        return {
            'min_days': total_days,
            'max_days': total_days + 2,
            'business_days_only': True,
            'includes_customization': personalization_days > 0
        }

class Category(BaseModel):
    """Product Categories"""
    __tablename__ = 'categories'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    slug = Column(String(120), unique=True, nullable=False)
    description = Column(Text)
    parent_id = Column(String, ForeignKey('categories.id'))
    
    # SEO
    meta_title = Column(String(60))
    meta_description = Column(String(160))
    
    # Display
    image = Column(String(500))
    icon = Column(String(50))  # Icon class or emoji
    color = Column(String(7))  # Hex color
    sort_order = Column(Integer, default=0)
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Relationships
    products = relationship("EnhancedProduct", secondary=product_categories, back_populates="categories")
    parent = relationship("Category", remote_side=[id], backref="children")

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'slug': self.slug,
            'description': self.description,
            'image': self.image,
            'icon': self.icon,
            'color': self.color,
            'product_count': len(self.products)
        }

class Emotion(BaseModel):
    """Emotional Tags for Products"""
    __tablename__ = 'emotions'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(50), nullable=False, unique=True)
    description = Column(String(200))
    emoji = Column(String(10))
    color = Column(String(7))  # Hex color for UI
    intensity_scale = Column(JSON)  # 1-5 intensity definitions
    
    # Relationships
    products = relationship("EnhancedProduct", secondary=product_emotions, back_populates="emotions")

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'emoji': self.emoji,
            'color': self.color,
            'product_count': len(self.products)
        }

# ================================================================
# FLASK ROUTES - Enhanced Product API
# ================================================================

from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from app.models import db
from app.services.recommendation_service import RecommendationService
from app.services.image_service import ImageService

products_bp = Blueprint('products', __name__)

@products_bp.route('/api/products', methods=['GET'])
def get_products():
    """Get products with advanced filtering and personality matching"""
    
    # Parse query parameters
    category = request.args.get('category')
    emotion = request.args.get('emotion')
    min_price = request.args.get('min_price', type=float)
    max_price = request.args.get('max_price', type=float)
    in_stock_only = request.args.get('in_stock', type=bool, default=False)
    featured_only = request.args.get('featured', type=bool, default=False)
    sort_by = request.args.get('sort', default='relevance')
    page = request.args.get('page', type=int, default=1)
    per_page = min(request.args.get('per_page', type=int, default=12), 50)
    
    # User personality for matching
    user_personality = None
    if current_user.is_authenticated:
        user_personality = current_user.get_personality_profile()
    
    # Build query
    query = EnhancedProduct.query.filter(EnhancedProduct.status == 'active')
    
    # Apply filters
    if category:
        query = query.join(EnhancedProduct.categories).filter(Category.slug == category)
    
    if emotion:
        query = query.join(EnhancedProduct.emotions).filter(Emotion.name == emotion)
    
    if min_price is not None:
        query = query.filter(EnhancedProduct.price_basic >= min_price)
    
    if max_price is not None:
        query = query.filter(EnhancedProduct.price_basic <= max_price)
    
    if in_stock_only:
        query = query.filter(
            (EnhancedProduct.track_inventory == False) |
            (EnhancedProduct.stock_quantity > 0) |
            (EnhancedProduct.allow_backorders == True)
        )
    
    if featured_only:
        query = query.filter(EnhancedProduct.is_featured == True)
    
    # Apply sorting
    if sort_by == 'price_low':
        query = query.order_by(EnhancedProduct.price_basic.asc())
    elif sort_by == 'price_high':
        query = query.order_by(EnhancedProduct.price_basic.desc())
    elif sort_by == 'rating':
        query = query.order_by(EnhancedProduct.average_rating.desc())
    elif sort_by == 'newest':
        query = query.order_by(EnhancedProduct.created_at.desc())
    elif sort_by == 'popular':
        query = query.order_by(EnhancedProduct.purchase_count.desc())
    else:  # relevance (default)
        if user_personality:
            # Complex personality-based sorting would be done via service
            pass
        else:
            query = query.order_by(EnhancedProduct.is_featured.desc(), EnhancedProduct.average_rating.desc())
    
    # Paginate
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    products = pagination.items
    
    # Calculate personality matches if user is logged in
    product_data = []
    for product in products:
        product_dict = product.to_dict()
        
        if user_personality:
            match_score = product.calculate_personality_match(user_personality)
            product_dict['personality_match_score'] = match_score
        
        product_data.append(product_dict)
    
    # Sort by personality match if we have user data
    if user_personality and sort_by == 'relevance':
        product_data.sort(key=lambda x: x.get('personality_match_score', 0), reverse=True)
    
    return jsonify({
        'products': product_data,
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total': pagination.total,
            'pages': pagination.pages,
            'has_next': pagination.has_next,
            'has_prev': pagination.has_prev
        },
        'filters_applied': {
            'category': category,
            'emotion': emotion,
            'price_range': [min_price, max_price],
            'in_stock_only': in_stock_only,
            'featured_only': featured_only,
            'sort_by': sort_by
        },
        'user_personality_available': user_personality is not None
    })

@products_bp.route('/api/products/<product_id>', methods=['GET'])
def get_product_detail(product_id):
    """Get detailed product information"""
    
    product = EnhancedProduct.query.get_or_404(product_id)
    
    # Increment view count
    product.view_count += 1
    db.session.commit()
    
    # Get related products
    recommendation_service = RecommendationService()
    related_products = recommendation_service.get_related_products(product_id, limit=6)
    
    # Get reviews
    reviews = product.reviews[:10]  # Latest 10 reviews
    
    # Calculate personality match for current user
    personality_match_score = None
    if current_user.is_authenticated:
        user_personality = current_user.get_personality_profile()
        if user_personality:
            personality_match_score = product.calculate_personality_match(user_personality)
    
    product_data = product.to_dict()
    product_data.update({
        'personality_match_score': personality_match_score,
        'related_products': [p.to_dict() for p in related_products],
        'recent_reviews': [review.to_dict() for review in reviews],
        'delivery_estimate': product.get_estimated_delivery(),
        'seo': {
            'title': product.meta_title or product.name,
            'description': product.meta_description or product.short_description,
            'keywords': product.seo_keywords or []
        }
    })
    
    return jsonify(product_data)

@products_bp.route('/api/products/<product_id>/personalization', methods=['GET'])
def get_personalization_options(product_id):
    """Get available personalization options for product"""
    
    product = EnhancedProduct.query.get_or_404(product_id)
    
    if not product.customization_options:
        return jsonify({'available': False})
    
    options = product.customization_options
    
    # Add pricing for customization options
    for option_type, option_data in options.items():
        if 'price' not in option_data:
            option_data['price'] = 0  # Default no extra cost
    
    return jsonify({
        'available': True,
        'options': options,
        'production_time': product.personalization_time,
        'max_production_time': product.personalization_time + 2,
        'guidelines': {
            'text': 'Bitte verwende nur jugendfreie und respektvolle Texte.',
            'images': 'Bilder sollten hochauflösend (min. 300 DPI) und im JPG/PNG Format sein.',
            'general': 'Personalisierte Produkte können nicht zurückgegeben werden.'
        }
    })

@products_bp.route('/api/categories', methods=['GET'])
def get_categories():
    """Get all active categories"""
    
    categories = Category.query.filter(Category.is_active == True).order_by(Category.sort_order, Category.name).all()
    
    return jsonify({
        'categories': [cat.to_dict() for cat in categories]
    })

@products_bp.route('/api/emotions', methods=['GET'])
def get_emotions():
    """Get all available emotional tags"""
    
    emotions = Emotion.query.all()
    
    return jsonify({
        'emotions': [emotion.to_dict() for emotion in emotions]
    })

# ================================================================
# SEED DATA - Beispiel Produkte für den Shop
# ================================================================

SEED_PRODUCTS = [
    {
        'name': 'Personalisierte Sterne-Karte',
        'slug': 'personalisierte-sterne-karte',
        'description': 'Eine einzigartige Karte des Sternenhimmels von einem besonderen Moment in deinem Leben. Perfekt für Jahrestage, Geburtstage oder als romantisches Geschenk.',
        'short_description': 'Deine persönliche Sternenkarte von einem besonderen Moment',
        'price_basic': 45.00,
        'price_premium': 85.00,
        'stock_quantity': 100,
        'personality_match': {
            'openness': 85,
            'conscientiousness': 60,
            'extraversion': 40,
            'agreeableness': 80,
            'neuroticism': 30,
            'limbic_type': 'Balance'
        },
        'emotional_tags': ['love', 'nostalgia', 'wonder'],
        'emotional_story': 'Diese Sterne-Karte zeigt den Himmel genau so, wie er an eurem ersten Kuss aussah. Ein Geschenk, das Tränen der Freude auslöst und für immer an diesen magischen Moment erinnert.',
        'customization_options': {
            'date': {'type': 'date', 'required': True, 'label': 'Besonderes Datum'},
            'location': {'type': 'text', 'required': True, 'label': 'Ort'},
            'message': {'type': 'text', 'max_length': 100, 'label': 'Persönliche Nachricht'},
            'frame_color': {'type': 'choice', 'options': ['Schwarz', 'Weiß', 'Gold'], 'default': 'Schwarz'}
        },
        'personalization_time': 3,
        'categories': ['romantic', 'personalized'],
        'primary_image': '/images/products/star-map-romantic.webp'
    },
    {
        'name': 'Kreatives Mal-Set für Erwachsene',
        'slug': 'kreatives-mal-set-erwachsene',
        'description': 'Hochwertiges Mal-Set mit Acrylfarben, Pinseln und Leinwänden. Perfekt für kreative Auszeiten und künstlerische Entfaltung.',
        'short_description': 'Alles für deine künstlerische Reise',
        'price_basic': 35.00,
        'price_premium': 65.00,
        'stock_quantity': 50,
        'personality_match': {
            'openness': 95,
            'conscientiousness': 70,
            'extraversion': 60,
            'agreeableness': 75,
            'neuroticism': 45,
            'limbic_type': 'Dominance'
        },
        'emotional_tags': ['creativity', 'relaxation', 'pride'],
        'emotional_story': 'Tauche ein in eine Welt der Farben und lass deiner Kreativität freien Lauf. Dieses Set hilft dir dabei, Stress abzubauen und stolz auf deine eigenen Kunstwerke zu sein.',
        'categories': ['creative', 'wellness'],
        'primary_image': '/images/products/art-set-creative.webp'
    }
]

def seed_products():
    """Seed initial products to database"""
    for product_data in SEED_PRODUCTS:
        # Check if product already exists
        existing = EnhancedProduct.query.filter_by(slug=product_data['slug']).first()
        if existing:
            continue
        
        # Create product
        product = EnhancedProduct(**{k: v for k, v in product_data.items() if k != 'categories'})
        db.session.add(product)
        db.session.flush()  # To get the ID
        
        # Add categories
        for cat_name in product_data.get('categories', []):
            category = Category.query.filter_by(slug=cat_name).first()
            if category:
                product.categories.append(category)
    
    db.session.commit()
    print(f"Seeded {len(SEED_PRODUCTS)} products")