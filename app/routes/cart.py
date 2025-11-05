"""
Cart Routes - Flask API für SensationGifts Warenkorb-System
===========================================================

🛒 LOCATION: app/routes/cart.py

API ENDPOINTS:
- GET /cart - Aktuellen Cart anzeigen
- POST /cart/items - Item hinzufügen
- PUT /cart/items/{id} - Item aktualisieren
- DELETE /cart/items/{id} - Item entfernen
- POST /cart/discount - Discount-Code anwenden
- POST /cart/gift-options - Gift-Optionen setzen
- POST /cart/checkout - Checkout starten
- POST /cart/ai-recommendation - AI-Empfehlung hinzufügen

FEATURES:
- Session-basierte anonyme Carts
- User-Cart Merge bei Login
- Comprehensive Error Handling
- Input Validation mit Pydantic
- Cart Analytics Tracking

CORRECTED LOGIC:
- PersonalityProfile beschreibt EMPFÄNGER (nicht User/Käufer)
- Cart gehört dem USER (Käufer)
- CartItems sind FÜR Empfänger basierend auf deren Persönlichkeit
"""

from flask import Blueprint, request, jsonify, session, g
from flask_cors import cross_origin
from flask_jwt_extended import jwt_required, get_jwt_identity, verify_jwt_in_request
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from decimal import Decimal
from pydantic import BaseModel, Field, ValidationError, validator
from enum import Enum

from app.extensions import db
from app.models.user import User
from app.services.cart_service import CartService, AICartService
from app.models.cart import CartItemType, GiftWrappingType, DeliveryMethod

# Logger
logger = logging.getLogger(__name__)

# Blueprint
cart_bp = Blueprint('cart', __name__)

# Services
cart_service = CartService()
ai_cart_service = AICartService()


# =============================================================================
# PYDANTIC SCHEMAS (REQUEST/RESPONSE VALIDATION)
# =============================================================================

class AddItemRequest(BaseModel):
    """Request Schema für Item hinzufügen"""
    item_name: str = Field(..., min_length=1, max_length=200)
    unit_price: float = Field(..., gt=0)
    quantity: int = Field(default=1, ge=1, le=99)
    item_type: str = Field(default="physical_gift")
    item_description: Optional[str] = Field(None, max_length=1000)
    item_image_url: Optional[str] = Field(None, max_length=500)
    item_url: Optional[str] = Field(None, max_length=500)
    gift_id: Optional[str] = Field(None, max_length=36)
    
    # EMPFÄNGER-INFORMATIONEN (KORRIGIERT!)
    recipient_name: Optional[str] = Field(None, max_length=100)
    recipient_email: Optional[str] = Field(None, max_length=100)
    personal_message: Optional[str] = Field(None, max_length=500)
    
    # AI & PERSONALITY CONTEXT (KORRIGIERT!)
    source_recommendation_id: Optional[str] = Field(None, max_length=36)
    source_personality_profile_id: Optional[str] = Field(None, max_length=36)  # EMPFÄNGER's Profile!
    personality_match_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    recommendation_reasoning: Optional[str] = Field(None, max_length=1000)
    
    # Gift Options
    gift_wrapping: Optional[str] = Field(None)
    
    @validator('item_type')
    def validate_item_type(cls, v):
        valid_types = [t.value for t in CartItemType]
        if v not in valid_types:
            raise ValueError(f'item_type must be one of: {valid_types}')
        return v
    
    @validator('gift_wrapping')
    def validate_gift_wrapping(cls, v):
        if v is not None:
            valid_wrapping = [t.value for t in GiftWrappingType]
            if v not in valid_wrapping:
                raise ValueError(f'gift_wrapping must be one of: {valid_wrapping}')
        return v


class UpdateItemRequest(BaseModel):
    """Request Schema für Item aktualisieren"""
    quantity: Optional[int] = Field(None, ge=0, le=99)
    personal_message: Optional[str] = Field(None, max_length=500)
    gift_wrapping: Optional[str] = Field(None)
    preferred_delivery_date: Optional[str] = Field(None)  # ISO format
    
    @validator('gift_wrapping')
    def validate_gift_wrapping(cls, v):
        if v is not None:
            valid_wrapping = [t.value for t in GiftWrappingType]
            if v not in valid_wrapping:
                raise ValueError(f'gift_wrapping must be one of: {valid_wrapping}')
        return v


class UpdateGiftOptionsRequest(BaseModel):
    """Request Schema für Gift-Optionen"""
    recipient_name: Optional[str] = Field(None, max_length=100)
    recipient_email: Optional[str] = Field(None, max_length=100)
    personal_message: Optional[str] = Field(None, max_length=500)
    gift_wrapping: Optional[str] = Field(None)
    delivery_date: Optional[str] = Field(None)  # ISO format


class ApplyDiscountRequest(BaseModel):
    """Request Schema für Discount-Code"""
    discount_code: str = Field(..., min_length=1, max_length=20)


class SaveWishlistRequest(BaseModel):
    """Request Schema für Wishlist speichern"""
    wishlist_name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    is_public: bool = Field(default=False)


class AddAIRecommendationRequest(BaseModel):
    """Request Schema für AI-Empfehlung hinzufügen"""
    recommendation_data: Dict[str, Any] = Field(...)
    recipient_personality_profile_id: str = Field(..., min_length=1, max_length=36)


class CartItemResponse(BaseModel):
    """Response Schema für Cart Item"""
    id: str
    item_name: str
    quantity: int
    unit_price: float
    line_total: float
    recipient_name: Optional[str]
    personal_message: Optional[str]
    gift_wrapping: Optional[str]
    is_personalized: bool
    personality_match_score: Optional[float]


class CartResponse(BaseModel):
    """Response Schema für kompletten Cart"""
    id: str
    status: str
    items_count: int
    subtotal: float
    tax_amount: float
    shipping_cost: float
    discount_amount: float
    total_amount: float
    currency: str
    is_gift: bool
    last_activity: str
    items: List[CartItemResponse]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_current_user_id() -> Optional[str]:
    """Holt aktuelle User-ID falls eingeloggt"""
    try:
        verify_jwt_in_request(optional=True)
        return get_jwt_identity()
    except:
        return None


def get_session_id() -> str:
    """Holt oder erstellt Session-ID"""
    if 'cart_session_id' not in session:
        import uuid
        session['cart_session_id'] = str(uuid.uuid4())
    return session['cart_session_id']


def get_cart_context() -> Dict[str, Optional[str]]:
    """Holt Cart-Kontext (User + Session)"""
    return {
        'user_id': get_current_user_id(),
        'session_id': get_session_id()
    }


def handle_cart_error(error_msg: str, status_code: int = 400) -> tuple:
    """Standardisierte Fehlerbehandlung"""
    logger.error(f"Cart error: {error_msg}")
    return jsonify({
        'success': False,
        'error': error_msg,
        'timestamp': datetime.utcnow().isoformat()
    }), status_code


def validate_pydantic_request(schema_class, request_data: dict):
    """Validiert Request mit Pydantic Schema"""
    try:
        return schema_class(**request_data), None
    except ValidationError as e:
        error_details = []
        for error in e.errors():
            field = ' -> '.join(str(loc) for loc in error['loc'])
            error_details.append(f"{field}: {error['msg']}")
        return None, f"Validierungsfehler: {'; '.join(error_details)}"


# =============================================================================
# CART MANAGEMENT ROUTES
# =============================================================================

@cart_bp.route('/cart', methods=['GET'])
@cross_origin()
def get_cart():
    """
    📋 GET CURRENT CART
    
    Holt aktuellen Cart für User oder Session
    """
    try:
        context = get_cart_context()
        cart = cart_service.get_cart(
            user_id=context['user_id'],
            session_id=context['session_id']
        )
        
        if not cart:
            return jsonify({
                'success': True,
                'cart': None,
                'message': 'Kein aktiver Warenkorb'
            }), 200
        
        # Get cart summary
        cart_summary = cart_service.get_cart_summary(cart.id)
        
        return jsonify({
            'success': True,
            'cart': cart_summary,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        return handle_cart_error(f"Fehler beim Laden des Warenkorbs: {str(e)}")


@cart_bp.route('/cart/summary', methods=['GET'])
@cross_origin()
def get_cart_summary():
    """
    📊 GET CART SUMMARY
    
    Lightweight Cart-Übersicht
    """
    try:
        context = get_cart_context()
        cart = cart_service.get_cart(
            user_id=context['user_id'],
            session_id=context['session_id']
        )
        
        if not cart:
            return jsonify({
                'success': True,
                'summary': {
                    'items_count': 0,
                    'total_amount': 0.0,
                    'currency': 'EUR'
                }
            }), 200
        
        summary = {
            'items_count': cart.items_count,
            'total_amount': float(cart.total_amount),
            'currency': cart.currency_code,
            'last_activity': cart.last_activity.isoformat()
        }
        
        return jsonify({
            'success': True,
            'summary': summary
        }), 200
        
    except Exception as e:
        return handle_cart_error(f"Fehler beim Laden der Cart-Übersicht: {str(e)}")


@cart_bp.route('/cart/merge', methods=['POST'])
@jwt_required()
def merge_carts():
    """
    🔗 MERGE CARTS
    
    Mergt Session-Cart in User-Cart bei Login
    """
    try:
        user_id = get_jwt_identity()
        session_id = get_session_id()
        
        merged_cart = cart_service.merge_user_session_carts(user_id, session_id)
        cart_summary = cart_service.get_cart_summary(merged_cart.id)
        
        return jsonify({
            'success': True,
            'cart': cart_summary,
            'message': 'Warenkörbe erfolgreich zusammengeführt'
        }), 200
        
    except Exception as e:
        return handle_cart_error(f"Fehler beim Zusammenführen der Warenkörbe: {str(e)}")


# =============================================================================
# ITEM MANAGEMENT ROUTES
# =============================================================================

@cart_bp.route('/cart/items', methods=['POST'])
@cross_origin()
def add_item_to_cart():
    """
    ➕ ADD ITEM TO CART
    
    Fügt Item zum Warenkorb hinzu
    """
    try:
        request_data = request.get_json()
        if not request_data:
            return handle_cart_error("Keine Daten empfangen")
        
        # Validiere Request
        add_request, validation_error = validate_pydantic_request(AddItemRequest, request_data)
        if validation_error:
            return handle_cart_error(validation_error)
        
        # Hole oder erstelle Cart
        context = get_cart_context()
        cart = cart_service.get_cart(
            user_id=context['user_id'],
            session_id=context['session_id']
        )
        
        if not cart:
            cart = cart_service.create_new_cart(
                user_id=context['user_id'],
                session_id=context['session_id']
            )
        
        # Konvertiere Item-Type Enum
        item_type = CartItemType(add_request.item_type)
        
        # Bereite Item-Details vor
        item_details = add_request.model_dump(exclude={'item_name', 'unit_price', 'quantity', 'item_type'})
        
        # Gift Wrapping konvertieren
        if item_details.get('gift_wrapping'):
            item_details['gift_wrapping'] = GiftWrappingType(item_details['gift_wrapping'])
        
        # Füge Item hinzu
        success, message, cart_item = cart_service.add_item_to_cart(
            cart_id=cart.id,
            item_name=add_request.item_name,
            unit_price=Decimal(str(add_request.unit_price)),
            quantity=add_request.quantity,
            item_type=item_type,
            **item_details
        )
        
        if success:
            cart_summary = cart_service.get_cart_summary(cart.id)
            return jsonify({
                'success': True,
                'message': message,
                'cart': cart_summary,
                'item_id': cart_item.id if cart_item else None
            }), 201
        else:
            return handle_cart_error(message)
        
    except Exception as e:
        return handle_cart_error(f"Fehler beim Hinzufügen des Items: {str(e)}")


@cart_bp.route('/cart/items/<item_id>', methods=['PUT'])
@cross_origin()
def update_cart_item(item_id: str):
    """
    ✏️ UPDATE CART ITEM
    
    Aktualisiert Cart-Item (Quantity, Gift-Optionen, etc.)
    """
    try:
        request_data = request.get_json()
        if not request_data:
            return handle_cart_error("Keine Daten empfangen")
        
        # Validiere Request
        update_request, validation_error = validate_pydantic_request(UpdateItemRequest, request_data)
        if validation_error:
            return handle_cart_error(validation_error)
        
        # Hole Cart
        context = get_cart_context()
        cart = cart_service.get_cart(
            user_id=context['user_id'],
            session_id=context['session_id']
        )
        
        if not cart:
            return handle_cart_error("Kein aktiver Warenkorb gefunden", 404)
        
        # Update Quantity falls angegeben
        if update_request.quantity is not None:
            success, message = cart_service.update_item_quantity(
                cart_id=cart.id,
                item_id=item_id,
                new_quantity=update_request.quantity
            )
            if not success:
                return handle_cart_error(message)
        
        # Update Gift Options falls angegeben
        gift_options = {}
        if update_request.personal_message is not None:
            gift_options['personal_message'] = update_request.personal_message
        
        if update_request.gift_wrapping is not None:
            gift_options['gift_wrapping'] = GiftWrappingType(update_request.gift_wrapping)
        
        if update_request.preferred_delivery_date is not None:
            try:
                gift_options['delivery_date'] = datetime.fromisoformat(update_request.preferred_delivery_date.replace('Z', '+00:00'))
            except ValueError:
                return handle_cart_error("Ungültiges Datumsformat")
        
        if gift_options:
            success, message = cart_service.update_item_gift_options(
                cart_id=cart.id,
                item_id=item_id,
                **gift_options
            )
            if not success:
                return handle_cart_error(message)
        
        cart_summary = cart_service.get_cart_summary(cart.id)
        return jsonify({
            'success': True,
            'message': 'Item erfolgreich aktualisiert',
            'cart': cart_summary
        }), 200
        
    except Exception as e:
        return handle_cart_error(f"Fehler beim Aktualisieren des Items: {str(e)}")


@cart_bp.route('/cart/items/<item_id>', methods=['DELETE'])
@cross_origin()
def remove_cart_item(item_id: str):
    """
    🗑️ REMOVE CART ITEM
    
    Entfernt Item aus dem Warenkorb
    """
    try:
        # Hole Cart
        context = get_cart_context()
        cart = cart_service.get_cart(
            user_id=context['user_id'],
            session_id=context['session_id']
        )
        
        if not cart:
            return handle_cart_error("Kein aktiver Warenkorb gefunden", 404)
        
        # Entferne Item
        success, message = cart_service.remove_item_from_cart(cart.id, item_id)
        
        if success:
            cart_summary = cart_service.get_cart_summary(cart.id)
            return jsonify({
                'success': True,
                'message': message,
                'cart': cart_summary
            }), 200
        else:
            return handle_cart_error(message)
        
    except Exception as e:
        return handle_cart_error(f"Fehler beim Entfernen des Items: {str(e)}")


@cart_bp.route('/cart/clear', methods=['POST'])
@cross_origin()
def clear_cart():
    """
    🧹 CLEAR CART
    
    Leert kompletten Warenkorb
    """
    try:
        # Hole Cart
        context = get_cart_context()
        cart = cart_service.get_cart(
            user_id=context['user_id'],
            session_id=context['session_id']
        )
        
        if not cart:
            return handle_cart_error("Kein aktiver Warenkorb gefunden", 404)
        
        # Leere Cart
        success, message = cart_service.clear_cart(cart.id)
        
        if success:
            return jsonify({
                'success': True,
                'message': message
            }), 200
        else:
            return handle_cart_error(message)
        
    except Exception as e:
        return handle_cart_error(f"Fehler beim Leeren des Warenkorbs: {str(e)}")


# =============================================================================
# GIFT FEATURES ROUTES
# =============================================================================

@cart_bp.route('/cart/gift-options', methods=['POST'])
@cross_origin()
def update_gift_options():
    """
    🎁 UPDATE GIFT OPTIONS
    
    Aktualisiert Gift-Optionen für Items
    """
    try:
        request_data = request.get_json()
        if not request_data:
            return handle_cart_error("Keine Daten empfangen")
        
        item_id = request_data.get('item_id')
        if not item_id:
            return handle_cart_error("item_id ist erforderlich")
        
        # Validiere Request
        gift_request, validation_error = validate_pydantic_request(UpdateGiftOptionsRequest, request_data)
        if validation_error:
            return handle_cart_error(validation_error)
        
        # Hole Cart
        context = get_cart_context()
        cart = cart_service.get_cart(
            user_id=context['user_id'],
            session_id=context['session_id']
        )
        
        if not cart:
            return handle_cart_error("Kein aktiver Warenkorb gefunden", 404)
        
        # Bereite Gift-Optionen vor
        gift_options = {}
        if gift_request.recipient_name is not None:
            gift_options['recipient_name'] = gift_request.recipient_name
        
        if gift_request.recipient_email is not None:
            gift_options['recipient_email'] = gift_request.recipient_email
        
        if gift_request.personal_message is not None:
            gift_options['personal_message'] = gift_request.personal_message
        
        if gift_request.gift_wrapping is not None:
            gift_options['gift_wrapping'] = GiftWrappingType(gift_request.gift_wrapping)
        
        if gift_request.delivery_date is not None:
            try:
                gift_options['delivery_date'] = datetime.fromisoformat(gift_request.delivery_date.replace('Z', '+00:00'))
            except ValueError:
                return handle_cart_error("Ungültiges Datumsformat")
        
        # Update Gift Options
        success, message = cart_service.update_item_gift_options(
            cart_id=cart.id,
            item_id=item_id,
            **gift_options
        )
        
        if success:
            cart_summary = cart_service.get_cart_summary(cart.id)
            return jsonify({
                'success': True,
                'message': message,
                'cart': cart_summary
            }), 200
        else:
            return handle_cart_error(message)
        
    except Exception as e:
        return handle_cart_error(f"Fehler beim Aktualisieren der Gift-Optionen: {str(e)}")


@cart_bp.route('/cart/global-message', methods=['POST'])
@cross_origin()
def set_global_gift_message():
    """
    💌 SET GLOBAL GIFT MESSAGE
    
    Setzt globale Geschenk-Nachricht für den gesamten Cart
    """
    try:
        request_data = request.get_json()
        if not request_data:
            return handle_cart_error("Keine Daten empfangen")
        
        message = request_data.get('message', '')
        
        # Hole Cart
        context = get_cart_context()
        cart = cart_service.get_cart(
            user_id=context['user_id'],
            session_id=context['session_id']
        )
        
        if not cart:
            return handle_cart_error("Kein aktiver Warenkorb gefunden", 404)
        
        # Setze globale Nachricht
        success, response_message = cart_service.set_global_gift_message(cart.id, message)
        
        if success:
            return jsonify({
                'success': True,
                'message': response_message
            }), 200
        else:
            return handle_cart_error(response_message)
        
    except Exception as e:
        return handle_cart_error(f"Fehler beim Setzen der globalen Nachricht: {str(e)}")


# =============================================================================
# DISCOUNT & PRICING ROUTES
# =============================================================================

@cart_bp.route('/cart/discount', methods=['POST'])
@cross_origin()
def apply_discount_code():
    """
    💰 APPLY DISCOUNT CODE
    
    Wendet Discount-Code auf Cart an
    """
    try:
        request_data = request.get_json()
        if not request_data:
            return handle_cart_error("Keine Daten empfangen")
        
        # Validiere Request
        discount_request, validation_error = validate_pydantic_request(ApplyDiscountRequest, request_data)
        if validation_error:
            return handle_cart_error(validation_error)
        
        # Hole Cart
        context = get_cart_context()
        cart = cart_service.get_cart(
            user_id=context['user_id'],
            session_id=context['session_id']
        )
        
        if not cart:
            return handle_cart_error("Kein aktiver Warenkorb gefunden", 404)
        
        # Wende Discount an
        success, message, discount_amount = cart_service.apply_discount_code(
            cart.id, 
            discount_request.discount_code
        )
        
        if success:
            cart_summary = cart_service.get_cart_summary(cart.id)
            return jsonify({
                'success': True,
                'message': message,
                'discount_amount': float(discount_amount) if discount_amount else 0.0,
                'cart': cart_summary
            }), 200
        else:
            return handle_cart_error(message)
        
    except Exception as e:
        return handle_cart_error(f"Fehler beim Anwenden des Discount-Codes: {str(e)}")


# =============================================================================
# AI INTEGRATION ROUTES (KORRIGIERT!)
# =============================================================================

@cart_bp.route('/cart/ai-recommendation', methods=['POST'])
@cross_origin()
def add_ai_recommendation_to_cart():
    """
    🤖 ADD AI RECOMMENDATION TO CART
    
    Fügt AI-Empfehlung basierend auf EMPFÄNGER-Persönlichkeit zum Cart hinzu
    
    KORRIGIERT: PersonalityProfile beschreibt EMPFÄNGER, nicht Käufer!
    """
    try:
        request_data = request.get_json()
        if not request_data:
            return handle_cart_error("Keine Daten empfangen")
        
        # Validiere Request
        ai_request, validation_error = validate_pydantic_request(AddAIRecommendationRequest, request_data)
        if validation_error:
            return handle_cart_error(validation_error)
        
        # Hole Cart (gehört dem KÄUFER)
        context = get_cart_context()
        cart = cart_service.get_cart(
            user_id=context['user_id'],
            session_id=context['session_id']
        )
        
        if not cart:
            cart = cart_service.create_new_cart(
                user_id=context['user_id'],
                session_id=context['session_id']
            )
        
        # Füge AI-Empfehlung hinzu (basierend auf EMPFÄNGER-Persönlichkeit)
        success, message, cart_item = ai_cart_service.add_ai_recommendation_to_cart(
            cart_id=cart.id,  # KÄUFER's Cart
            recommendation_data=ai_request.recommendation_data,
            recipient_personality_profile_id=ai_request.recipient_personality_profile_id  # EMPFÄNGER's Profile!
        )
        
        if success:
            cart_summary = cart_service.get_cart_summary(cart.id)
            return jsonify({
                'success': True,
                'message': message,
                'cart': cart_summary,
                'item_id': cart_item.id if cart_item else None
            }), 201
        else:
            return handle_cart_error(message)
        
    except Exception as e:
        return handle_cart_error(f"Fehler beim Hinzufügen der AI-Empfehlung: {str(e)}")


@cart_bp.route('/cart/suggestions', methods=['GET'])
@cross_origin()
def get_cart_suggestions():
    """
    💡 GET CART SUGGESTIONS
    
    Holt intelligente Verbesserungsvorschläge für Cart
    """
    try:
        # Hole Cart
        context = get_cart_context()
        cart = cart_service.get_cart(
            user_id=context['user_id'],
            session_id=context['session_id']
        )
        
        if not cart:
            return jsonify({
                'success': True,
                'suggestions': []
            }), 200
        
        # Hole Suggestions
        suggestions = ai_cart_service.suggest_cart_optimizations(cart.id)
        
        return jsonify({
            'success': True,
            'suggestions': suggestions
        }), 200
        
    except Exception as e:
        return handle_cart_error(f"Fehler beim Laden der Vorschläge: {str(e)}")


# =============================================================================
# WISHLIST & SAVED CARTS ROUTES
# =============================================================================

@cart_bp.route('/cart/save-wishlist', methods=['POST'])
@jwt_required()
def save_cart_as_wishlist():
    """
    💾 SAVE CART AS WISHLIST
    
    Speichert aktuellen Cart als Wishlist
    """
    try:
        user_id = get_jwt_identity()
        request_data = request.get_json()
        
        if not request_data:
            return handle_cart_error("Keine Daten empfangen")
        
        # Validiere Request
        wishlist_request, validation_error = validate_pydantic_request(SaveWishlistRequest, request_data)
        if validation_error:
            return handle_cart_error(validation_error)
        
        # Hole Cart
        cart = cart_service.get_cart(user_id=user_id)
        
        if not cart:
            return handle_cart_error("Kein aktiver Warenkorb gefunden", 404)
        
        # Speichere als Wishlist
        success, message, saved_cart = cart_service.save_cart_as_wishlist(
            cart_id=cart.id,
            wishlist_name=wishlist_request.wishlist_name,
            description=wishlist_request.description,
            is_public=wishlist_request.is_public
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': message,
                'wishlist_id': saved_cart.id if saved_cart else None,
                'share_code': saved_cart.share_code if saved_cart and saved_cart.is_public else None
            }), 201
        else:
            return handle_cart_error(message)
        
    except Exception as e:
        return handle_cart_error(f"Fehler beim Speichern der Wishlist: {str(e)}")


# =============================================================================
# CHECKOUT ROUTES
# =============================================================================

@cart_bp.route('/cart/checkout', methods=['POST'])
@cross_origin()
def prepare_checkout():
    """
    🛒 PREPARE CHECKOUT
    
    Bereitet Cart für Checkout vor
    """
    try:
        # Hole Cart
        context = get_cart_context()
        cart = cart_service.get_cart(
            user_id=context['user_id'],
            session_id=context['session_id']
        )
        
        if not cart:
            return handle_cart_error("Kein aktiver Warenkorb gefunden", 404)
        
        # Bereite Checkout vor
        success, message, checkout_data = cart_service.prepare_cart_for_checkout(cart.id)
        
        if success:
            return jsonify({
                'success': True,
                'message': message,
                'checkout_data': checkout_data
            }), 200
        else:
            return handle_cart_error(message)
        
    except Exception as e:
        return handle_cart_error(f"Fehler bei Checkout-Vorbereitung: {str(e)}")


# =============================================================================
# ANALYTICS ROUTES
# =============================================================================

@cart_bp.route('/cart/analytics', methods=['GET'])
@jwt_required()
def get_cart_analytics():
    """
    📊 GET CART ANALYTICS
    
    Holt Analytics-Daten für Cart (nur für eingeloggte User)
    """
    try:
        user_id = get_jwt_identity()
        cart = cart_service.get_cart(user_id=user_id)
        
        if not cart:
            return handle_cart_error("Kein aktiver Warenkorb gefunden", 404)
        
        analytics = cart_service.get_cart_analytics(cart.id)
        
        return jsonify({
            'success': True,
            'analytics': analytics
        }), 200
        
    except Exception as e:
        return handle_cart_error(f"Fehler beim Laden der Analytics: {str(e)}")


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@cart_bp.errorhandler(404)
def cart_not_found(error):
    """Cart API 404 Handler"""
    return jsonify({
        'success': False,
        'error': 'Cart endpoint not found',
        'available_endpoints': [
            'GET /cart', 'POST /cart/items', 'PUT /cart/items/<id>', 
            'DELETE /cart/items/<id>', 'POST /cart/discount',
            'POST /cart/ai-recommendation', 'POST /cart/checkout'
        ]
    }), 404


@cart_bp.errorhandler(405)
def cart_method_not_allowed(error):
    """Cart API 405 Handler"""
    return jsonify({
        'success': False,
        'error': 'HTTP method not allowed for this cart endpoint'
    }), 405


@cart_bp.errorhandler(500)
def cart_server_error(error):
    """Cart API 500 Handler"""
    logger.error(f"Cart API server error: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal server error in cart API'
    }), 500


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ['cart_bp']