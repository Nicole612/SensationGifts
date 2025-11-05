"""
Cart Service - Business Logic für SensationGifts Warenkorb-System
=================================================================

🛒 LOCATION: app/services/cart_service.py

FEATURES:
- Complete Cart Management (Add, Update, Remove Items)
- Gift-spezifische Funktionen (Wrapping, Messages, Recipients)
- Pricing & Discount Management
- Cart Abandonment Recovery
- Analytics & Conversion Tracking
- Integration mit AI-Recommendations

CORRECTED LOGIC:
- PersonalityProfile beschreibt EMPFÄNGER (nicht User/Käufer)
- Cart gehört dem USER (Käufer)
- CartItems sind FÜR Empfänger basierend auf deren Persönlichkeit
"""

from typing import Optional, List, Dict, Any, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
import logging
import json

from app.extensions import db
from app.models.user import User
from app.models.cart import (
    Cart, CartItem, CartActivity, SavedCart,
    CartStatus, CartItemType, DeliveryMethod, GiftWrappingType,
    get_or_create_cart, merge_carts
)


logger = logging.getLogger(__name__)


class CartService:
    """
    🛒 Core Cart Service für alle Warenkorb-Operationen
    
    Handles:
    - Item Management (Add/Update/Remove)
    - Price Calculations
    - Gift Features
    - User Experience Optimization
    """
    
    def __init__(self):
        self.default_currency = 'EUR'
        self.default_tax_rate = Decimal('0.19')  # 19% MwSt Deutschland
        self.free_shipping_threshold = Decimal('50.00')
    
    # =============================================================================
    # CART MANAGEMENT
    # =============================================================================
    
    def get_cart(self, user_id: str = None, session_id: str = None) -> Optional[Cart]:
        """
        Holt aktuellen Cart für User oder Session
        
        Args:
            user_id: ID des eingeloggten Users (Käufer)
            session_id: Session ID für anonyme Users
            
        Returns:
            Cart oder None wenn nicht gefunden
        """
        try:
            return get_or_create_cart(user_id=user_id, session_id=session_id)
        except Exception as e:
            logger.error(f"Error getting cart: {e}")
            return None
    
    def create_new_cart(self, user_id: str = None, session_id: str = None, cart_name: str = None) -> Cart:
        """
        Erstellt explizit neuen Cart
        
        Args:
            user_id: User ID (Käufer)
            session_id: Session ID 
            cart_name: Name für den Cart
            
        Returns:
            Neuer Cart
        """
        cart = Cart(
            user_id=user_id,
            session_id=session_id,
            cart_name=cart_name or "Neuer Warenkorb",
            currency_code=self.default_currency,
            tax_rate=self.default_tax_rate,
            cart_status=CartStatus.ACTIVE
        )
        
        db.session.add(cart)
        db.session.commit()
        
        self._log_cart_activity(
            cart_id=cart.id,
            activity_type='cart_created',
            activity_description=f"New cart created: {cart.cart_name}",
            user_id=user_id,
            session_id=session_id
        )
        
        return cart
    
    def merge_user_session_carts(self, user_id: str, session_id: str) -> Cart:
        """
        Mergt Session-Cart in User-Cart bei Login
        
        Args:
            user_id: User der sich einloggt (Käufer)
            session_id: Session ID des anonymen Carts
            
        Returns:
            Gemergter User-Cart
        """
        try:
            # Hole beide Carts
            user_cart = Cart.query.filter_by(
                user_id=user_id,
                cart_status=CartStatus.ACTIVE
            ).first()
            
            session_cart = Cart.query.filter_by(
                session_id=session_id,
                cart_status=CartStatus.ACTIVE
            ).first()
            
            if not session_cart:
                # Kein Session-Cart vorhanden
                return user_cart or self.create_new_cart(user_id=user_id)
            
            if not user_cart:
                # Kein User-Cart, übernehme Session-Cart
                session_cart.user_id = user_id
                session_cart.session_id = None
                db.session.commit()
                return session_cart
            
            # Beide Carts vorhanden - merge sie
            merged_cart = merge_carts(user_cart, session_cart)
            db.session.commit()
            
            self._log_cart_activity(
                cart_id=merged_cart.id,
                activity_type='carts_merged',
                activity_description=f"Session cart merged into user cart",
                user_id=user_id
            )
            
            return merged_cart
            
        except Exception as e:
            logger.error(f"Error merging carts: {e}")
            db.session.rollback()
            return self.get_cart(user_id=user_id)
    
    # =============================================================================
    # ITEM MANAGEMENT
    # =============================================================================
    
    def add_item_to_cart(
        self,
        cart_id: str,
        item_name: str,
        unit_price: Decimal,
        quantity: int = 1,
        item_type: CartItemType = CartItemType.PHYSICAL_GIFT,
        **item_details
    ) -> Tuple[bool, str, Optional[CartItem]]:
        """
        Fügt Item zum Cart hinzu
        
        Args:
            cart_id: Cart ID (Käufer's Cart)
            item_name: Name des Items
            unit_price: Einzelpreis
            quantity: Anzahl
            item_type: Typ des Items
            **item_details: Zusätzliche Item-Details (inkl. recipient_name, source_personality_profile_id)
            
        Returns:
            (success, message, cart_item)
        """
        try:
            cart = Cart.query.get(cart_id)
            if not cart:
                return False, "Cart nicht gefunden", None
            
            if cart.cart_status != CartStatus.ACTIVE:
                return False, "Cart ist nicht aktiv", None
            
            # Prüfe ob Item bereits existiert (für Mengen-Update)
            existing_item = None
            gift_id = item_details.get('gift_id')
            recipient_name = item_details.get('recipient_name')
            
            if gift_id and recipient_name:
                # Suche nach gleichem Geschenk für gleichen Empfänger
                existing_item = CartItem.query.filter_by(
                    cart_id=cart_id,
                    gift_id=gift_id,
                    recipient_name=recipient_name
                ).first()
            
            if existing_item:
                # Update existing item quantity
                old_quantity = existing_item.quantity
                existing_item.quantity += quantity
                existing_item.calculate_line_total()
                
                self._log_cart_activity(
                    cart_id=cart_id,
                    activity_type='item_quantity_updated',
                    activity_description=f"Quantity updated for {item_name} (recipient: {recipient_name})",
                    item_id=existing_item.id,
                    old_value=json.dumps({'quantity': old_quantity}),
                    new_value=json.dumps({'quantity': existing_item.quantity}),
                    user_id=cart.user_id
                )
                
                cart_item = existing_item
            else:
                # Create new item
                cart_item = CartItem(
                    cart_id=cart_id,
                    item_name=item_name,
                    unit_price=unit_price,
                    quantity=quantity,
                    item_type=item_type,
                    gift_id=item_details.get('gift_id'),
                    item_description=item_details.get('item_description'),
                    item_image_url=item_details.get('item_image_url'),
                    item_url=item_details.get('item_url'),
                    
                    # EMPFÄNGER-INFORMATIONEN (KORRIGIERT!)
                    recipient_name=item_details.get('recipient_name'),
                    recipient_email=item_details.get('recipient_email'),
                    personal_message=item_details.get('personal_message'),
                    
                    # AI & PERSONALITY CONTEXT (KORRIGIERT!)
                    source_recommendation_id=item_details.get('source_recommendation_id'),
                    source_personality_profile_id=item_details.get('source_personality_profile_id'),  # EMPFÄNGER's Profile!
                    personality_match_score=item_details.get('personality_match_score'),
                    recommendation_reasoning=item_details.get('recommendation_reasoning'),
                    
                    # Gift Wrapping
                    gift_wrapping=item_details.get('gift_wrapping', GiftWrappingType.BASIC)
                )
                
                cart_item.calculate_line_total()
                if cart_item.gift_wrapping:
                    cart_item.set_gift_wrapping(cart_item.gift_wrapping)
                
                db.session.add(cart_item)
                
                self._log_cart_activity(
                    cart_id=cart_id,
                    activity_type='item_added',
                    activity_description=f"Added {item_name} for {recipient_name or 'unknown recipient'}",
                    item_id=cart_item.id,
                    new_value=json.dumps({
                        'item_name': item_name,
                        'quantity': quantity,
                        'unit_price': float(unit_price),
                        'recipient_name': recipient_name,
                        'source_personality_profile_id': item_details.get('source_personality_profile_id')
                    }),
                    user_id=cart.user_id
                )
            
            # Update cart totals
            cart.calculate_totals()
            cart.update_activity()
            
            db.session.commit()
            
            return True, f"{item_name} wurde zum Warenkorb hinzugefügt", cart_item
            
        except Exception as e:
            logger.error(f"Error adding item to cart: {e}")
            db.session.rollback()
            return False, f"Fehler beim Hinzufügen: {str(e)}", None
    
    def update_item_quantity(
        self,
        cart_id: str,
        item_id: str,
        new_quantity: int
    ) -> Tuple[bool, str]:
        """
        Aktualisiert Quantity eines Cart-Items
        
        Args:
            cart_id: Cart ID
            item_id: Item ID
            new_quantity: Neue Anzahl
            
        Returns:
            (success, message)
        """
        try:
            cart_item = CartItem.query.filter_by(
                id=item_id,
                cart_id=cart_id
            ).first()
            
            if not cart_item:
                return False, "Item nicht gefunden"
            
            if new_quantity <= 0:
                return self.remove_item_from_cart(cart_id, item_id)
            
            old_quantity = cart_item.quantity
            cart_item.quantity = new_quantity
            cart_item.calculate_line_total()
            
            # Update cart
            cart = cart_item.cart
            cart.calculate_totals()
            cart.update_activity()
            
            self._log_cart_activity(
                cart_id=cart_id,
                activity_type='item_quantity_updated',
                activity_description=f"Quantity updated for {cart_item.item_name}",
                item_id=item_id,
                old_value=json.dumps({'quantity': old_quantity}),
                new_value=json.dumps({'quantity': new_quantity}),
                user_id=cart.user_id
            )
            
            db.session.commit()
            
            return True, "Anzahl aktualisiert"
            
        except Exception as e:
            logger.error(f"Error updating item quantity: {e}")
            db.session.rollback()
            return False, f"Fehler beim Aktualisieren: {str(e)}"
    
    def remove_item_from_cart(self, cart_id: str, item_id: str) -> Tuple[bool, str]:
        """
        Entfernt Item aus Cart
        
        Args:
            cart_id: Cart ID
            item_id: Item ID
            
        Returns:
            (success, message)
        """
        try:
            cart_item = CartItem.query.filter_by(
                id=item_id,
                cart_id=cart_id
            ).first()
            
            if not cart_item:
                return False, "Item nicht gefunden"
            
            item_name = cart_item.item_name
            recipient_name = cart_item.recipient_name
            
            self._log_cart_activity(
                cart_id=cart_id,
                activity_type='item_removed',
                activity_description=f"Removed {item_name} for {recipient_name}",
                item_id=item_id,
                old_value=json.dumps({
                    'item_name': item_name,
                    'quantity': cart_item.quantity,
                    'unit_price': float(cart_item.unit_price),
                    'recipient_name': recipient_name
                }),
                user_id=cart_item.cart.user_id
            )
            
            db.session.delete(cart_item)
            
            # Update cart
            cart = Cart.query.get(cart_id)
            cart.calculate_totals()
            cart.update_activity()
            
            db.session.commit()
            
            return True, f"{item_name} wurde entfernt"
            
        except Exception as e:
            logger.error(f"Error removing item from cart: {e}")
            db.session.rollback()
            return False, f"Fehler beim Entfernen: {str(e)}"
    
    def clear_cart(self, cart_id: str) -> Tuple[bool, str]:
        """
        Leert kompletten Cart
        
        Args:
            cart_id: Cart ID
            
        Returns:
            (success, message)
        """
        try:
            cart = Cart.query.get(cart_id)
            if not cart:
                return False, "Cart nicht gefunden"
            
            items_count = len(cart.cart_items)
            
            # Delete all items
            CartItem.query.filter_by(cart_id=cart_id).delete()
            
            # Reset cart totals
            cart.subtotal = Decimal('0.00')
            cart.tax_amount = Decimal('0.00')
            cart.shipping_cost = Decimal('0.00')
            cart.total_amount = Decimal('0.00')
            cart.items_count = 0
            cart.update_activity()
            
            self._log_cart_activity(
                cart_id=cart_id,
                activity_type='cart_cleared',
                activity_description=f"Cart cleared - {items_count} items removed",
                user_id=cart.user_id
            )
            
            db.session.commit()
            
            return True, f"Warenkorb geleert ({items_count} Items entfernt)"
            
        except Exception as e:
            logger.error(f"Error clearing cart: {e}")
            db.session.rollback()
            return False, f"Fehler beim Leeren: {str(e)}"
    
    # =============================================================================
    # GIFT FEATURES
    # =============================================================================
    
    def update_item_gift_options(
        self,
        cart_id: str,
        item_id: str,
        recipient_name: str = None,
        recipient_email: str = None,
        personal_message: str = None,
        gift_wrapping: GiftWrappingType = None,
        delivery_date: datetime = None
    ) -> Tuple[bool, str]:
        """
        Aktualisiert Gift-Optionen für Cart-Item
        
        Args:
            cart_id: Cart ID
            item_id: Item ID
            recipient_name: Name des Empfängers
            recipient_email: E-Mail des Empfängers
            personal_message: Persönliche Nachricht
            gift_wrapping: Geschenkverpackung
            delivery_date: Lieferdatum
            
        Returns:
            (success, message)
        """
        try:
            cart_item = CartItem.query.filter_by(
                id=item_id,
                cart_id=cart_id
            ).first()
            
            if not cart_item:
                return False, "Item nicht gefunden"
            
            # Update gift options
            if recipient_name is not None:
                cart_item.recipient_name = recipient_name
            
            if recipient_email is not None:
                cart_item.recipient_email = recipient_email
            
            if personal_message is not None:
                cart_item.personal_message = personal_message
            
            if gift_wrapping is not None:
                cart_item.set_gift_wrapping(gift_wrapping)
            
            if delivery_date is not None:
                cart_item.preferred_delivery_date = delivery_date
            
            # Update cart totals (wegen gift wrapping costs)
            cart = cart_item.cart
            cart.calculate_totals()
            cart.update_activity()
            
            self._log_cart_activity(
                cart_id=cart_id,
                activity_type='gift_options_updated',
                activity_description=f"Gift options updated for {cart_item.item_name}",
                item_id=item_id,
                user_id=cart.user_id
            )
            
            db.session.commit()
            
            return True, "Geschenk-Optionen aktualisiert"
            
        except Exception as e:
            logger.error(f"Error updating gift options: {e}")
            db.session.rollback()
            return False, f"Fehler beim Aktualisieren: {str(e)}"
    
    def set_global_gift_message(self, cart_id: str, message: str) -> Tuple[bool, str]:
        """
        Setzt globale Geschenk-Nachricht für gesamten Cart
        
        Args:
            cart_id: Cart ID
            message: Globale Nachricht
            
        Returns:
            (success, message)
        """
        try:
            cart = Cart.query.get(cart_id)
            if not cart:
                return False, "Cart nicht gefunden"
            
            cart.global_gift_message = message
            cart.update_activity()
            
            self._log_cart_activity(
                cart_id=cart_id,
                activity_type='global_gift_message_set',
                activity_description="Global gift message updated",
                user_id=cart.user_id
            )
            
            db.session.commit()
            
            return True, "Globale Nachricht gesetzt"
            
        except Exception as e:
            logger.error(f"Error setting global gift message: {e}")
            db.session.rollback()
            return False, f"Fehler beim Setzen: {str(e)}"
    
    # =============================================================================
    # DISCOUNTS & PRICING
    # =============================================================================
    
    def apply_discount_code(
        self,
        cart_id: str,
        discount_code: str
    ) -> Tuple[bool, str, Optional[Decimal]]:
        """
        Wendet Discount-Code an
        
        Args:
            cart_id: Cart ID
            discount_code: Discount Code
            
        Returns:
            (success, message, discount_amount)
        """
        try:
            cart = Cart.query.get(cart_id)
            if not cart:
                return False, "Cart nicht gefunden", None
            
            # Validiere Discount Code (vereinfacht - könnte komplexer sein)
            discount_info = self._validate_discount_code(discount_code, cart)
            
            if not discount_info['valid']:
                return False, discount_info['message'], None
            
            discount_amount = discount_info['amount']
            
            # Prüfe ob bereits angewendet
            if discount_code in cart.discount_codes_list:
                return False, "Discount-Code bereits angewendet", None
            
            # Wende Discount an
            cart.apply_discount_code(discount_code, discount_amount, discount_info)
            
            self._log_cart_activity(
                cart_id=cart_id,
                activity_type='discount_applied',
                activity_description=f"Discount code applied: {discount_code}",
                new_value=json.dumps({
                    'code': discount_code,
                    'amount': float(discount_amount)
                }),
                user_id=cart.user_id
            )
            
            db.session.commit()
            
            return True, f"Discount angewendet: -{discount_amount}€", discount_amount
            
        except Exception as e:
            logger.error(f"Error applying discount code: {e}")
            db.session.rollback()
            return False, f"Fehler beim Anwenden: {str(e)}", None
    
    def _validate_discount_code(self, code: str, cart: Cart) -> Dict[str, Any]:
        """
        Validiert Discount Code (vereinfacht)
        
        In production würde das eine richtige Discount-Engine sein
        """
        
        # Beispiel-Discounts (hardcoded für Demo)
        discount_codes = {
            'WELCOME10': {
                'type': 'percentage',
                'value': 10,
                'min_order': 20,
                'valid': True,
                'description': '10% Willkommensrabatt'
            },
            'FREESHIP': {
                'type': 'free_shipping',
                'value': 0,
                'min_order': 0,
                'valid': True,
                'description': 'Kostenloser Versand'
            },
            'GIFT20': {
                'type': 'fixed',
                'value': 20,
                'min_order': 100,
                'valid': True,
                'description': '20€ Geschenkrabatt'
            }
        }
        
        if code not in discount_codes:
            return {'valid': False, 'message': 'Ungültiger Discount-Code'}
        
        discount = discount_codes[code]
        
        if cart.subtotal < discount['min_order']:
            return {
                'valid': False,
                'message': f'Mindestbestellwert: {discount["min_order"]}€'
            }
        
        # Berechne Discount-Betrag
        if discount['type'] == 'percentage':
            amount = cart.subtotal * (Decimal(discount['value']) / 100)
        elif discount['type'] == 'fixed':
            amount = Decimal(discount['value'])
        elif discount['type'] == 'free_shipping':
            amount = cart.shipping_cost
        else:
            amount = Decimal('0.00')
        
        return {
            'valid': True,
            'message': discount['description'],
            'amount': amount,
            'type': discount['type'],
            'code': code
        }
    
    # =============================================================================
    # CART ANALYTICS & ABANDONMENT
    # =============================================================================
    
    def get_abandonment_carts(self, hours_threshold: int = 24) -> List[Cart]:
        """
        Holt verlassene Carts für Recovery-Kampagnen
        
        Args:
            hours_threshold: Stunden nach denen Cart als verlassen gilt
            
        Returns:
            Liste von verlassenen Carts
        """
        threshold_time = datetime.utcnow() - timedelta(hours=hours_threshold)
        
        abandoned_carts = Cart.query.filter(
            Cart.cart_status == CartStatus.ACTIVE,
            Cart.items_count > 0,
            Cart.last_activity < threshold_time,
            Cart.abandonment_email_sent == False
        ).all()
        
        return abandoned_carts
    
    def mark_abandonment_email_sent(self, cart_id: str):
        """Markiert dass Abandonment-E-Mail versendet wurde"""
        cart = Cart.query.get(cart_id)
        if cart:
            cart.abandonment_email_sent = True
            cart.cart_status = CartStatus.ABANDONED
            db.session.commit()
    
    def get_cart_analytics(self, cart_id: str) -> Dict[str, Any]:
        """
        Holt Analytics-Daten für Cart
        
        Args:
            cart_id: Cart ID
            
        Returns:
            Analytics Dictionary
        """
        cart = Cart.query.get(cart_id)
        if not cart:
            return {}
        
        activities = CartActivity.query.filter_by(cart_id=cart_id).all()
        
        return {
            'cart_id': cart_id,
            'created_at': cart.created_at.isoformat(),
            'last_activity': cart.last_activity.isoformat(),
            'items_count': cart.items_count,
            'total_amount': float(cart.total_amount),
            'abandonment_risk': cart.abandonment_risk,
            'activities_count': len(activities),
            'activity_timeline': [
                {
                    'type': activity.activity_type,
                    'description': activity.activity_description,
                    'timestamp': activity.activity_timestamp.isoformat()
                }
                for activity in activities[-10:]  # Last 10 activities
            ]
        }
    
    # =============================================================================
    # SAVED CARTS & WISHLIST
    # =============================================================================
    
    def save_cart_as_wishlist(
        self,
        cart_id: str,
        wishlist_name: str,
        description: str = None,
        is_public: bool = False
    ) -> Tuple[bool, str, Optional[SavedCart]]:
        """
        Speichert Cart als Wishlist
        
        Args:
            cart_id: Cart ID
            wishlist_name: Name der Wishlist
            description: Beschreibung
            is_public: Öffentlich sichtbar?
            
        Returns:
            (success, message, saved_cart)
        """
        try:
            cart = Cart.query.get(cart_id)
            if not cart:
                return False, "Cart nicht gefunden", None
            
            if cart.is_empty:
                return False, "Leerer Cart kann nicht gespeichert werden", None
            
            # Erstelle Snapshot der Cart-Daten
            cart_data = {
                'cart_name': cart.cart_name,
                'currency_code': cart.currency_code,
                'is_gift': cart.is_gift,
                'global_gift_message': cart.global_gift_message,
                'total_amount': float(cart.total_amount)
            }
            
            items_data = []
            for item in cart.cart_items:
                items_data.append({
                    'item_name': item.item_name,
                    'item_description': item.item_description,
                    'unit_price': float(item.unit_price),
                    'quantity': item.quantity,
                    'item_image_url': item.item_image_url,
                    'recipient_name': item.recipient_name,
                    'personal_message': item.personal_message,
                    'source_personality_profile_id': item.source_personality_profile_id
                })
            
            saved_cart = SavedCart(
                user_id=cart.user_id,
                original_cart_id=cart_id,
                saved_cart_name=wishlist_name,
                saved_cart_description=description,
                cart_data=json.dumps(cart_data),
                items_snapshot=json.dumps(items_data),
                is_public=is_public
            )
            
            if is_public:
                saved_cart.generate_share_code()
            
            db.session.add(saved_cart)
            db.session.commit()
            
            return True, "Wishlist gespeichert", saved_cart
            
        except Exception as e:
            logger.error(f"Error saving cart as wishlist: {e}")
            db.session.rollback()
            return False, f"Fehler beim Speichern: {str(e)}", None
    
    # =============================================================================
    # CHECKOUT PREPARATION
    # =============================================================================
    
    def prepare_cart_for_checkout(self, cart_id: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Bereitet Cart für Checkout vor
        
        Args:
            cart_id: Cart ID
            
        Returns:
            (success, message, checkout_data)
        """
        try:
            cart = Cart.query.get(cart_id)
            if not cart:
                return False, "Cart nicht gefunden", None
            
            if cart.is_empty:
                return False, "Cart ist leer", None
            
            # Validiere Cart-Items
            validation_issues = []
            
            for item in cart.cart_items:
                # Prüfe ob Gift-Informationen vollständig sind
                if cart.is_gift and not item.recipient_name:
                    validation_issues.append(f"Empfänger fehlt für: {item.item_name}")
                
                # Prüfe Delivery-Anforderungen
                if item.requires_delivery_date and not item.preferred_delivery_date:
                    validation_issues.append(f"Lieferdatum fehlt für: {item.item_name}")
            
            if validation_issues:
                return False, "Validierungsfehler: " + "; ".join(validation_issues), None
            
            # Starte Checkout
            cart.start_checkout()
            
            # Erstelle Checkout-Daten
            checkout_data = {
                'cart_id': cart_id,
                'buyer_user_id': cart.user_id,  # KÄUFER
                'items_count': cart.items_count,
                'subtotal': float(cart.subtotal),
                'tax_amount': float(cart.tax_amount),
                'shipping_cost': float(cart.shipping_cost),
                'discount_amount': float(cart.discount_amount),
                'total_amount': float(cart.total_amount),
                'currency': cart.currency_code,
                'items': [
                    {
                        'id': item.id,
                        'name': item.item_name,
                        'quantity': item.quantity,
                        'unit_price': float(item.unit_price),
                        'line_total': float(item.line_total),
                        'recipient_name': item.recipient_name,  # EMPFÄNGER
                        'personal_message': item.personal_message,
                        'gift_wrapping': item.gift_wrapping.value if item.gift_wrapping else None,
                        'delivery_date': item.preferred_delivery_date.isoformat() if item.preferred_delivery_date else None,
                        'source_personality_profile_id': item.source_personality_profile_id  # EMPFÄNGER's Personality
                    }
                    for item in cart.cart_items
                ],
                'delivery_info': {
                    'method': cart.delivery_method.value if cart.delivery_method else None,
                    'preferred_date': cart.preferred_delivery_date.isoformat() if cart.preferred_delivery_date else None,
                    'notes': cart.delivery_notes
                },
                'gift_info': {
                    'is_gift': cart.is_gift,
                    'global_message': cart.global_gift_message,
                    'wrapping_preference': cart.gift_wrapping_preference.value if cart.gift_wrapping_preference else None
                }
            }
            
            db.session.commit()
            
            return True, "Cart für Checkout vorbereitet", checkout_data
            
        except Exception as e:
            logger.error(f"Error preparing cart for checkout: {e}")
            db.session.rollback()
            return False, f"Fehler bei Checkout-Vorbereitung: {str(e)}", None
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    def _log_cart_activity(
        self,
        cart_id: str,
        activity_type: str,
        activity_description: str,
        user_id: str = None,
        session_id: str = None,
        item_id: str = None,
        old_value: str = None,
        new_value: str = None
    ):
        """Loggt Cart-Aktivität für Analytics"""
        
        try:
            activity = CartActivity(
                cart_id=cart_id,
                user_id=user_id,
                activity_type=activity_type,
                activity_description=activity_description,
                item_id=item_id,
                old_value=old_value,
                new_value=new_value,
                session_id=session_id
            )
            
            db.session.add(activity)
            # Nicht committen - das macht der aufrufende Code
            
        except Exception as e:
            logger.error(f"Error logging cart activity: {e}")
    
    def get_cart_summary(self, cart_id: str) -> Optional[Dict[str, Any]]:
        """
        Holt Cart-Summary für Frontend
        
        Args:
            cart_id: Cart ID
            
        Returns:
            Cart Summary Dictionary oder None
        """
        cart = Cart.query.get(cart_id)
        if not cart:
            return None
        
        return {
            'id': cart.id,
            'status': cart.cart_status.value,
            'items_count': cart.items_count,
            'subtotal': float(cart.subtotal),
            'tax_amount': float(cart.tax_amount),
            'shipping_cost': float(cart.shipping_cost),
            'discount_amount': float(cart.discount_amount),
            'total_amount': float(cart.total_amount),
            'currency': cart.currency_code,
            'is_gift': cart.is_gift,
            'last_activity': cart.last_activity.isoformat(),
            'items': [
                {
                    'id': item.id,
                    'name': item.item_name,
                    'quantity': item.quantity,
                    'unit_price': float(item.unit_price),
                    'line_total': float(item.line_total),
                    'image_url': item.item_image_url,
                    'recipient_name': item.recipient_name,  # EMPFÄNGER
                    'is_personalized': item.is_personalized,
                    'personality_match_score': item.personality_match_score
                }
                for item in cart.cart_items
            ]
        }


# =============================================================================
# AI INTEGRATION FEATURES (KORRIGIERT!)
# =============================================================================

class AICartService(CartService):
    """
    Erweiterte Cart-Service mit AI-Integration
    
    🤖 Features:
    - AI-Recommendation-to-Cart (basierend auf EMPFÄNGER-Persönlichkeit)
    - Personality-basierte Cart-Optimierung
    - Smart Gift Suggestions
    
    KORRIGIERT: PersonalityProfile beschreibt EMPFÄNGER, nicht Käufer!
    """
    
    def add_ai_recommendation_to_cart(
        self,
        cart_id: str,  # KÄUFER's Cart
        recommendation_data: Dict[str, Any],
        recipient_personality_profile_id: str  # EMPFÄNGER's PersonalityProfile
    ) -> Tuple[bool, str, Optional[CartItem]]:
        """
        Fügt AI-Recommendation direkt zum Cart hinzu
        
        Args:
            cart_id: Cart ID (gehört dem KÄUFER)
            recommendation_data: AI-Recommendation Daten
            recipient_personality_profile_id: PersonalityProfile des EMPFÄNGERS
            
        Returns:
            (success, message, cart_item)
        """
        try:
            # Importiere PersonalityProfile hier um circular imports zu vermeiden
            from app.models.personality import PersonalityProfile
            
            # Lade EMPFÄNGER's PersonalityProfile
            recipient_personality = PersonalityProfile.query.get(recipient_personality_profile_id)
            
            if not recipient_personality:
                return False, "Empfänger-Persönlichkeit nicht gefunden", None
            
            # Extrahiere Recommendation-Details
            item_name = recommendation_data.get('title', recommendation_data.get('gift_name', 'AI Empfehlung'))
            price_str = recommendation_data.get('price', recommendation_data.get('price_estimate', '€0'))
            
            # Parse Price
            price_clean = price_str.replace('€', '').replace(',', '.').strip()
            try:
                unit_price = Decimal(price_clean)
            except (ValueError, TypeError):
                unit_price = Decimal('0.00')
            
            # Baue Item-Details mit EMPFÄNGER-Info
            item_details = {
                'item_description': recommendation_data.get('description', recommendation_data.get('reasoning', '')),
                'source_recommendation_id': recommendation_data.get('id'),
                'source_personality_profile_id': recipient_personality_profile_id,  # EMPFÄNGER's Profile!
                'recommendation_reasoning': recommendation_data.get('reasoning', recommendation_data.get('why_perfect', '')),
                'personality_match_score': recommendation_data.get('confidence_score', recommendation_data.get('confidence', 0.8)),
                
                # EMPFÄNGER-Informationen
                'recipient_name': recipient_personality.recipient_name,
                'recipient_email': getattr(recipient_personality, 'recipient_email', None),
                
                # Auto-generate personal message basierend auf EMPFÄNGER-Personality
                'personal_message': self._generate_personality_message(
                    item_name, 
                    recipient_personality.limbic_type_auto,
                    recommendation_data
                )
            }
            
            return self.add_item_to_cart(
                cart_id=cart_id,  # KÄUFER's Cart
                item_name=item_name,
                unit_price=unit_price,
                quantity=1,
                item_type=CartItemType.PHYSICAL_GIFT,
                **item_details
            )
            
        except Exception as e:
            logger.error(f"Error adding AI recommendation to cart: {e}")
            return False, f"Fehler beim Hinzufügen der AI-Empfehlung: {str(e)}", None
    
    def _generate_personality_message(
        self,
        item_name: str,
        limbic_type,  # LimbicType Enum
        recommendation_data: Dict[str, Any]
    ) -> str:
        """Generiert personalisierten Message basierend auf EMPFÄNGER's Limbic Type"""
        
        if not limbic_type:
            return f"Speziell für dich ausgewählt: {item_name}! 💝"
        
        limbic_type_str = limbic_type.value if hasattr(limbic_type, 'value') else str(limbic_type)
        
        messages = {
            'adventurer': f"Für neue Abenteuer mit {item_name} - perfekt für deinen Entdeckergeist! 🌟",
            'performer': f"Mit {item_name} wirst du brillieren - zeig allen was in dir steckt! 💪",
            'harmonizer': f"{item_name} für entspannte Momente - gönn dir diese Auszeit! 🕯️",
            'hedonist': f"Purer Genuss mit {item_name} - das Leben ist zu kurz für weniger! 🎉",
            'disciplined': f"Durchdachte Qualität: {item_name} - eine Investition in dich selbst! ✨",
            'traditionalist': f"Bewährte Werte, neue Freude: {item_name} für dich! 🏠",
            'pioneer': f"Innovation trifft auf {item_name} - sei Vorreiter! 🚀"
        }
        
        return messages.get(limbic_type_str.lower(), f"Speziell für dich ausgewählt: {item_name}! 💝")
    
    def suggest_cart_optimizations(self, cart_id: str) -> List[Dict[str, Any]]:
        """
        Schlägt Cart-Optimierungen vor basierend auf Inhalt
        
        Args:
            cart_id: Cart ID
            
        Returns:
            Liste von Optimierungs-Vorschlägen
        """
        cart = Cart.query.get(cart_id)
        if not cart:
            return []
        
        suggestions = []
        
        # Free Shipping Suggestion
        if cart.subtotal < self.free_shipping_threshold:
            missing_amount = self.free_shipping_threshold - cart.subtotal
            suggestions.append({
                'type': 'free_shipping',
                'title': 'Kostenloser Versand',
                'description': f'Noch {missing_amount}€ für kostenlosen Versand',
                'action': 'add_items',
                'value': float(missing_amount)
            })
        
        # Gift Wrapping Suggestion
        unwrapped_items = [item for item in cart.cart_items if item.gift_wrapping == GiftWrappingType.NONE]
        if unwrapped_items and cart.is_gift:
            suggestions.append({
                'type': 'gift_wrapping',
                'title': 'Geschenkverpackung',
                'description': f'{len(unwrapped_items)} Items ohne Geschenkverpackung',
                'action': 'add_wrapping',
                'items': [item.id for item in unwrapped_items]
            })
        
        # Missing Recipients Suggestion
        items_without_recipients = [item for item in cart.cart_items if not item.recipient_name]
        if items_without_recipients and cart.is_gift:
            suggestions.append({
                'type': 'missing_recipients',
                'title': 'Empfänger hinzufügen',
                'description': f'{len(items_without_recipients)} Items ohne Empfänger',
                'action': 'add_recipients',
                'items': [item.id for item in items_without_recipients]
            })
        
        # Bundle Suggestion (vereinfacht)
        if len(cart.cart_items) == 1:
            suggestions.append({
                'type': 'bundle',
                'title': 'Komplettiere dein Geschenk',
                'description': 'Passende Zusatzprodukte verfügbar',
                'action': 'show_bundles'
            })
        
        return suggestions


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'CartService',
    'AICartService'
]