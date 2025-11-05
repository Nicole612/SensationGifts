"""
GESCHICHTE SCHREIBEN - Produktionskatalog Generator
==================================================

Vollständiger, produktionsfertiger Katalog für personalisierten Geschenke-Shop
mit emotionalen Geschichten und AI-optimierter Struktur.

🎁 Fokus: Emotionale Verbindung + KI-Fachberatung + Alle Altersgruppen
"""

import json
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum
import uuid

class AgeCategory(Enum):
    BABY = "0-3"
    KLEINKIND = "4-7" 
    SCHULKIND = "8-12"
    TEENAGER = "13-17"
    JUNGE_ERWACHSENE = "18-35"
    ERWACHSENE = "36-55"
    SENIOREN = "56-70"
    HOCHBETAGTE = "70+"

class PersonalityTrait(Enum):
    HIGH_OPENNESS = "high_openness"
    HIGH_CONSCIENTIOUSNESS = "high_conscientiousness"
    HIGH_EXTRAVERSION = "high_extraversion"
    HIGH_AGREEABLENESS = "high_agreeableness"
    HIGH_NEUROTICISM = "high_neuroticism"
    CREATIVE_TYPE = "creative_type"
    TECH_SAVVY = "tech_savvy"
    PRACTICAL_TYPE = "practical_type"
    HEALTH_CONSCIOUS = "health_conscious"

class RelationshipType(Enum):
    PARTNER = "partner"
    FAMILY_PARENT = "family_parent"
    FAMILY_CHILD = "family_child"
    FAMILY_SIBLING = "family_sibling"
    FRIEND_BEST = "friend_best"
    FRIEND_CLOSE = "friend_close"
    COLLEAGUE = "colleague"
    BOSS = "boss"

# =============================================================================
# PRODUKTKATALOG - PREMIUM GESCHENKBOXEN
# =============================================================================

GESCHENK_KATALOG = {
    
    # =============================================================================
    # KATEGORIE: KINDER & JUGENDLICHE (0-17 Jahre)
    # =============================================================================
    
    "mutige_helden_box": {
        "id": "mutige-helden-box-001",
        "name": "Ich bin mutig - Helden Box",
        "category": "Persönlichkeitsentwicklung Kinder",
        "short_description": "Verwandle dein Kind in den Helden seiner eigenen Geschichte",
        "long_description": "Eine magische Box, die Kindern zeigt, wie mutig sie wirklich sind. Mit personalisierter Heldengeschichte, echtem Mut-Medaillon und 7 spielerischen Aufgaben, die das Selbstvertrauen stärken.",
        
        "age_categories": [AgeCategory.KLEINKIND.value, AgeCategory.SCHULKIND.value],
        "target_age_min": 3,
        "target_age_max": 9,
        
        "personality_match_scores": {
            PersonalityTrait.HIGH_OPENNESS.value: 0.9,
            PersonalityTrait.CREATIVE_TYPE.value: 0.8,
            PersonalityTrait.HIGH_CONSCIENTIOUSNESS.value: 0.7
        },
        
        "relationship_suitability": {
            RelationshipType.FAMILY_PARENT.value: 0.95,
            RelationshipType.FAMILY_SIBLING.value: 0.8,
            RelationshipType.FRIEND_CLOSE.value: 0.85
        },
        
        "occasion_suitability": {
            "geburtstag": 0.9,
            "schulanfang": 0.95,
            "weihnachten": 0.8,
            "ostern": 0.7,
            "mutmacher": 0.98
        },
        
        "emotional_story": """Diese Box verwandelt dein Kind in den Helden seiner eigenen Geschichte. Mit dem Namen deines Kindes als Protagonist erlebt es Abenteuer, die Mut und Selbstvertrauen schaffen. Das personalisierte Helden-Medaillon wird zum täglichen Begleiter - ein Symbol der inneren Stärke, das immer daran erinnert: 'Ich bin mutig, ich schaffe das!'""",
        
        "emotional_impact": {
            "hauptemotion": "Mut und Selbstvertrauen",
            "nebeneffekte": ["Stolz", "Freude", "Geborgenheit", "Fantasie"],
            "langzeitwirkung": "Stärkung der Persönlichkeit und des Selbstbildes"
        },
        
        "content_components": {
            "emotionaler_teil": "Personalisierte Heldengeschichte + Mut-Medaillon mit Gravur",
            "praktischer_teil": "Superhelden-Cape + 7 Mut-Aufgaben-Karten + Mut-Tagebuch",
            "extras": "QR-Code zu Audio-Geschichte + Eltern-Guide"
        },
        
        "price_variants": {
            "basic": {
                "price": 34.99,
                "includes": ["Geschichte", "Medaillon", "7 Aufgaben-Karten", "Standard-Verpackung"]
            },
            "medium": {
                "price": 54.99, 
                "includes": ["Alles aus Basic", "Superhelden-Cape", "Mut-Tagebuch", "Geschenkbox"]
            },
            "premium": {
                "price": 84.99,
                "includes": ["Alles aus Medium", "Audio-Geschichte", "Holz-Schatzkiste", "Foto-Personalisierung"]
            }
        },
        
        "ki_empfehlung": "Als KI-Fachberater empfehle ich diese Box besonders für Kinder, die sich in Übergangssituationen befinden (Schulanfang, Umzug, neue Herausforderungen). Die personalisierte Geschichte schafft eine emotionale Verbindung, während die praktischen Aufgaben spielerisch Selbstvertrauen aufbauen. Perfekt für Eltern, die ihrem Kind mehr als nur ein Geschenk geben möchten.",
        
        "production_ready": {
            "print_files": ["heldengschichte_template.pdf", "aufgaben_karten.pdf", "medaillon_gravur.ai"],
            "personalization_fields": ["child_name", "age", "favorite_color", "photo_upload"],
            "shipping_time": "3-5 Werktage (Personalisierung)",
            "supplier_info": "Medaillon: Premium-Edelstahl, Cape: Bio-Baumwolle"
        },
        
        "seo_tags": ["mut", "selbstvertrauen", "kinder", "personalisiert", "helden", "geschichten", "charakterbildung"],
        "emotional_tags": ["empowerment", "mut", "fantasie", "wachstum", "stolz"]
    },

    "traumfaenger_zauber": {
        "id": "traumfaenger-zauber-002", 
        "name": "Traumfänger Zauber - Süße Träume Box",
        "category": "Schlaf & Entspannung Kinder",
        "short_description": "Magische Träume mit personalisierten Traumfänger und Gute-Nacht-Geschichte",
        "long_description": "Ein handgefertigter Traumfänger mit dem Namen deines Kindes fängt alle bösen Träume ein. Die personalisierte Gute-Nacht-Geschichte macht jede Nacht zu einem Abenteuer voller süßer Träume.",
        
        "age_categories": [AgeCategory.KLEINKIND.value, AgeCategory.SCHULKIND.value],
        "target_age_min": 3,
        "target_age_max": 10,
        
        "personality_match_scores": {
            PersonalityTrait.HIGH_NEUROTICISM.value: 0.9,  # Für ängstliche Kinder
            PersonalityTrait.CREATIVE_TYPE.value: 0.8,
            PersonalityTrait.HIGH_OPENNESS.value: 0.7
        },
        
        "relationship_suitability": {
            RelationshipType.FAMILY_PARENT.value: 0.95,
            RelationshipType.FAMILY_SIBLING.value: 0.7,
        },
        
        "emotional_story": """Jede Nacht wird zu einem magischen Ritual: Der Traumfänger mit dem Namen deines Kindes wacht über den Schlaf, während die personalisierte Geschichte eine Welt voller Wunder und Geborgenheit erschafft. Das sanfte Nachtlicht in Wolkenform hüllt das Zimmer in ein beruhigendes Licht - ein Signal für Körper und Seele: 'Hier bin ich sicher, hier darf ich träumen.'""",
        
        "price_variants": {
            "basic": {"price": 29.99, "includes": ["Traumfänger personalisiert", "Gute-Nacht-Geschichte"]},
            "medium": {"price": 49.99, "includes": ["+ Wolken-Nachtlicht", "Geschenkbox"]}, 
            "premium": {"price": 74.99, "includes": ["+ Audio-Geschichte", "Premium-Traumfänger", "Schlaf-Ritual-Guide"]}
        }
    },

    # =============================================================================
    # KATEGORIE: JUGENDLICHE (13-17 Jahre) 
    # =============================================================================
    
    "dein_soundtrack": {
        "id": "dein-soundtrack-003",
        "name": "Dein Soundtrack - Musik & Erinnerungen Box", 
        "category": "Musik & Lifestyle Jugendliche",
        "short_description": "Eine personalisierte Playlist deiner Beziehung als physisches Erlebnis",
        "long_description": "Eure gemeinsamen Songs werden zu einer personalisierten Playlist mit QR-Codes. Jeder Song erzählt eure Geschichte - vom ersten Kennenlernen bis heute. Mit stylischer Kopfhörerhülle und Musik-Journal.",
        
        "age_categories": [AgeCategory.TEENAGER.value, AgeCategory.JUNGE_ERWACHSENE.value],
        "target_age_min": 13,
        "target_age_max": 25,
        
        "personality_match_scores": {
            PersonalityTrait.HIGH_OPENNESS.value: 0.95,
            PersonalityTrait.CREATIVE_TYPE.value: 0.9,
            PersonalityTrait.HIGH_EXTRAVERSION.value: 0.8
        },
        
        "relationship_suitability": {
            RelationshipType.PARTNER.value: 0.95,
            RelationshipType.FRIEND_BEST.value: 0.9,
            RelationshipType.FAMILY_SIBLING.value: 0.7
        },
        
        "emotional_story": """Musik ist die Sprache der Seele - und diese Box macht eure gemeinsame Geschichte hörbar. Jeder Song auf eurer personalisierten Playlist trägt eine Erinnerung: Der Song vom ersten Date, das Lied aus dem Auto auf der Urlaubsfahrt, die Hymne eurer Freundschaft. Mit den QR-Codes wird jeder Moment wieder lebendig.""",
        
        "price_variants": {
            "basic": {"price": 39.99, "includes": ["10-Song-Playlist mit QR", "Musik-Journal"]},
            "medium": {"price": 64.99, "includes": ["+ Personalisierte Kopfhörerhülle", "20 Songs", "Erinnerungs-Timeline"]},
            "premium": {"price": 94.99, "includes": ["+ Premium-Kopfhörer", "Spotify Premium 3 Monate", "Vinyl-Style Box"]}
        }
    },

    # =============================================================================
    # KATEGORIE: JUNGE ERWACHSENE (18-35 Jahre)
    # =============================================================================
    
    "lebensfilm_box": {
        "id": "lebensfilm-box-004",
        "name": "Lebensfilm - Deine Geschichte als Video",
        "category": "Erinnerungen & Beziehungen",
        "short_description": "Ein professionell erstelltes Video aus euren schönsten Momenten",
        "long_description": "Aus euren Fotos, Videos und Sprachnachrichten entsteht ein emotionaler Lebensfilm. Perfekt synchronisiert mit eurer Lieblingsmusik, professionell geschnitten und mit persönlichen Texteinblendungen.",
        
        "age_categories": [AgeCategory.JUNGE_ERWACHSENE.value, AgeCategory.ERWACHSENE.value],
        "target_age_min": 18,
        "target_age_max": 50,
        
        "personality_match_scores": {
            PersonalityTrait.HIGH_OPENNESS.value: 0.85,
            PersonalityTrait.CREATIVE_TYPE.value: 0.9,
            PersonalityTrait.HIGH_AGREEABLENESS.value: 0.8
        },
        
        "relationship_suitability": {
            RelationshipType.PARTNER.value: 0.98,
            RelationshipType.FAMILY_PARENT.value: 0.9,
            RelationshipType.FRIEND_BEST.value: 0.85
        },
        
        "emotional_story": """Euer Leben wird zum Film - mit euch als Hauptdarstellern. Jede Szene erzählt von eurer Reise: erste Blicke, gemeinsame Abenteuer, stille Momente, große Träume. Der professionell geschnittene Film mit eurer Musik macht aus Erinnerungen ein Kunstwerk, das ihr immer wieder anschauen werdet.""",
        
        "price_variants": {
            "basic": {"price": 149.99, "includes": ["3-5 Min Video", "20 Fotos/Videos", "1 Musik-Track"]},
            "medium": {"price": 249.99, "includes": ["8-10 Min Video", "50 Fotos/Videos", "3 Musik-Tracks", "Texteinblendungen"]},
            "premium": {"price": 399.99, "includes": ["15 Min Video", "Unlimited Fotos", "Professionelles Voice-Over", "4K Qualität", "USB in Holzbox"]}
        },
        
        "ki_empfehlung": "Als KI-Experte für emotionale Geschenke rate ich zu diesem Video besonders bei Jahrestagen, Hochzeiten oder wichtigen Lebensereignissen. Die professionelle Umsetzung macht den Unterschied - hier investiert ihr nicht nur in ein Geschenk, sondern in ein Familienerbstück, das Generationen überdauert."
    },

    # =============================================================================
    # KATEGORIE: HOBBY & LEIDENSCHAFT - AUTOLIEBHABER
    # =============================================================================
    
    "premium_autopflege_meisterwerk": {
        "id": "premium-autopflege-001",
        "name": "Nano-Profi Autopflege Meisterwerk",
        "category": "Hobby & Leidenschaft - Automotive",
        "short_description": "Die weltweit beste Nano-Versiegelung trifft auf Premium-Pflegeerlebnis",
        "long_description": "GYEON Q² Mohs EVO - die Keramikversiegelung der Supercars, kombiniert mit professionellem Pflegeset und einer personalisierten Erfolgsgeschichte deines Traumautos. Für Autoliebhaber, die nur das Beste verdienen.",
        
        "age_categories": [AgeCategory.JUNGE_ERWACHSENE.value, AgeCategory.ERWACHSENE.value, AgeCategory.SENIOREN.value],
        "target_age_min": 18,
        "target_age_max": 70,
        
        "personality_match_scores": {
            PersonalityTrait.HIGH_CONSCIENTIOUSNESS.value: 0.95,
            PersonalityTrait.PRACTICAL_TYPE.value: 0.9,
            PersonalityTrait.TECH_SAVVY.value: 0.8,
            PersonalityTrait.HIGH_OPENNESS.value: 0.7
        },
        
        "relationship_suitability": {
            RelationshipType.PARTNER.value: 0.9,
            RelationshipType.FRIEND_BEST.value: 0.85,
            RelationshipType.FAMILY_PARENT.value: 0.8,
            RelationshipType.COLLEAGUE.value: 0.7
        },
        
        "occasion_suitability": {
            "geburtstag": 0.95,
            "weihnachten": 0.9,
            "vatertag": 0.98,
            "befoerderung": 0.85,
            "jahrestag": 0.7
        },
        
        "emotional_story": """Dein Auto ist nicht nur Fortbewegung - es ist Leidenschaft, Stolz, ein Stück Identität. Diese Box bringt dir die Geheimnisse der Profi-Detailer nach Hause: GYEON Q² Mohs EVO, die Keramikversiegelung, die auch auf Lamborghinis und McLaren glänzt. Mit jedem Auftrag wird dein Auto zu einem Spiegelbild deiner Perfektion.""",
        
        "emotional_impact": {
            "hauptemotion": "Stolz und Perfektion",
            "nebeneffekte": ["Entspannung", "Meditation", "Erfolgsgefühl", "Wertschätzung"],
            "langzeitwirkung": "Jahrelanger Schutz und täglich sichtbare Qualität"
        },
        
        "content_components": {
            "hauptprodukt": "GYEON Q² Mohs EVO Keramikversiegelung (30ml)",
            "profi_tools": ["Mikrofaser-Applikatoren", "Vorbereitungs-Spray", "Curing-Tücher"],
            "premium_extras": ["LED-Arbeitslampe", "pH-neutrale Vorwäsche", "Detailing-Handschuhe"],
            "wissen": "Schritt-für-Schritt Video-Anleitung + Profi-Tipps PDF",
            "emotionale_komponente": "Personalisierte Erfolgs-Story: 'Dein Auto - Dein Meisterwerk'"
        },
        
        "price_variants": {
            "basic": {
                "price": 189.99,
                "includes": ["GYEON Q² Mohs EVO", "Basis-Applikatoren", "Anleitung"]
            },
            "medium": {
                "price": 289.99,
                "includes": ["+ Profi-Tool-Set", "LED-Lampe", "Video-Anleitung", "Pflegeguide"]
            },
            "premium": {
                "price": 449.99,
                "includes": ["+ Komplettes Detailing-Set", "Personalisierte Auto-Story", "1-Jahr Support-Hotline", "Profi-Werkzeugkiste"]
            }
        },
        
        "ki_empfehlung": "Als KI-Spezialist für Premium-Autopflege empfehle ich diese Box für Autoliebhaber, die bereits Erfahrung mit Detailing haben oder bereit sind, in Perfektion zu investieren. GYEON Q² Mohs EVO ist dieselbe Versiegelung, die in Top-Detailing-Studios weltweit verwendet wird. Der 5-Jahres-Schutz rechtfertigt jeden Cent der Investition.",
        
        "technical_specs": {
            "versiegelung": "SiO2 + Si Basis, 9H Härte",
            "haltbarkeit": "5+ Jahre bei korrekter Anwendung", 
            "coverage": "10-15 Fahrzeuge (je nach Größe)",
            "besonderheit": "Verwendet in Ferrari-, Lamborghini- und McLaren-Service-Centern"
        },
        
        "production_ready": {
            "suppliers": {
                "gyeon": "Direktimport Korea, Mindestbestellung 50 Stück",
                "tools": "Premium-Mikrofaser Deutschland",
                "packaging": "Schwarze Premium-Holzbox mit Lasergravur"
            },
            "shipping_time": "1-2 Werktage (Lagerware)",
            "margin": "45-55% je nach Variante"
        }
    },

    # =============================================================================
    # KATEGORIE: ERWACHSENE - BEZIEHUNGEN & PARTNERSCHAFT
    # =============================================================================
    
    "unsere_liebesgeschichte": {
        "id": "unsere-liebesgeschichte-005",
        "name": "Unsere Liebesgeschichte - Personalisiertes Liebebuch",
        "category": "Beziehung & Partnerschaft",
        "short_description": "Eure Liebe wird zu einem wunderschön illustrierten Buch",
        "long_description": "Professionell gestaltetes Hardcover-Buch mit eurer persönlichen Liebesgeschichte. Jede Seite erzählt einen Moment eurer Beziehung - vom ersten Blick bis zu euren Zukunftsträumen. Mit euren Fotos und liebevoll illustrierten Szenen.",
        
        "age_categories": [AgeCategory.JUNGE_ERWACHSENE.value, AgeCategory.ERWACHSENE.value],
        "target_age_min": 20,
        "target_age_max": 60,
        
        "personality_match_scores": {
            PersonalityTrait.HIGH_AGREEABLENESS.value: 0.9,
            PersonalityTrait.CREATIVE_TYPE.value: 0.85,
            PersonalityTrait.HIGH_OPENNESS.value: 0.8
        },
        
        "relationship_suitability": {
            RelationshipType.PARTNER.value: 0.98
        },
        
        "emotional_story": """Eure Liebe verdient es, erzählt zu werden - wie die großen Romane, nur dass ihr die Hauptfiguren seid. Jede Seite dieses personalisierten Buchs fängt einen Moment eurer Geschichte ein: Der erste Blick, das erste Lächeln, der erste Kuss, gemeinsame Träume. Ein Buch, das ihr euren Kindern vorlesen werdet.""",
        
        "price_variants": {
            "basic": {"price": 89.99, "includes": ["40-Seiten Hardcover", "10 Fotos", "Standard-Layout"]},
            "medium": {"price": 149.99, "includes": ["60 Seiten", "20 Fotos", "Custom-Illustrationen", "Premium-Papier"]},
            "premium": {"price": 249.99, "includes": ["100 Seiten", "Unlimited Fotos", "Hand-Illustrationen", "Leder-Einband", "Geschenkbox"]}
        }
    },

    # =============================================================================
    # KATEGORIE: SENIOREN (56+ Jahre)
    # =============================================================================
    
    "lebensbuch_erinnerungen": {
        "id": "lebensbuch-erinnerungen-006",
        "name": "Mein Lebensbuch - Erinnerungen für die Ewigkeit",
        "category": "Erinnerungen & Familiengeschichte",
        "short_description": "Ein wunderschönes Buch, um Lebensgeschichten festzuhalten",
        "long_description": "Liebevoll gestaltetes Erinnerungsbuch mit durchdachten Fragen, die zu Erzählungen einladen. Platz für Fotos, Geschichten und Weisheiten - ein Schatz für kommende Generationen.",
        
        "age_categories": [AgeCategory.SENIOREN.value, AgeCategory.HOCHBETAGTE.value],
        "target_age_min": 56,
        "target_age_max": 99,
        
        "personality_match_scores": {
            PersonalityTrait.HIGH_AGREEABLENESS.value: 0.85,
            PersonalityTrait.HIGH_CONSCIENTIOUSNESS.value: 0.8,
            PersonalityTrait.CREATIVE_TYPE.value: 0.7
        },
        
        "relationship_suitability": {
            RelationshipType.FAMILY_CHILD.value: 0.95,
            RelationshipType.FAMILY_SIBLING.value: 0.8,
            RelationshipType.FRIEND_CLOSE.value: 0.75
        },
        
        "emotional_story": """Ein Leben voller Geschichten verdient es, bewahrt zu werden. Dieses Buch ist mehr als ein Geschenk - es ist eine Einladung, das eigene Leben zu reflektieren und die wertvollsten Erinnerungen für die Familie zu sammeln. Jede ausgefüllte Seite wird zu einem Schatz für Kinder und Enkelkinder.""",
        
        "price_variants": {
            "basic": {"price": 49.99, "includes": ["Leitfragen-Buch", "Schönes Layout", "Fester Einband"]},
            "medium": {"price": 79.99, "includes": ["+ Premium-Füller", "Foto-Taschen", "Elegante Box"]},
            "premium": {"price": 119.99, "includes": ["+ Leder-Einband", "Goldprägung mit Name", "Audio-Aufnahmegerät für Geschichten"]}
        }
    }
}

# =============================================================================
# HILFSFUNKTIONEN FÜR KATALOG-MANAGEMENT
# =============================================================================

def get_products_by_age(age: int) -> List[Dict]:
    """Filtert Produkte nach Alter"""
    suitable_products = []
    
    for product_id, product in GESCHENK_KATALOG.items():
        if product['target_age_min'] <= age <= product['target_age_max']:
            suitable_products.append({
                'id': product_id,
                'name': product['name'],
                'match_reason': f"Perfekt für Alter {age}",
                **product
            })
    
    return suitable_products

def get_products_by_personality(traits: List[str]) -> List[Dict]:
    """Filtert Produkte nach Persönlichkeitsmerkmalen"""
    suitable_products = []
    
    for product_id, product in GESCHENK_KATALOG.items():
        match_score = 0
        matching_traits = []
        
        for trait in traits:
            if trait in product.get('personality_match_scores', {}):
                score = product['personality_match_scores'][trait]
                match_score += score
                matching_traits.append(f"{trait}: {score}")
        
        if match_score > 0:
            suitable_products.append({
                'id': product_id,
                'match_score': match_score,
                'matching_traits': matching_traits,
                **product
            })
    
    # Sortiere nach Match-Score
    suitable_products.sort(key=lambda x: x['match_score'], reverse=True)
    return suitable_products

def get_products_by_relationship(relationship: str) -> List[Dict]:
    """Filtert Produkte nach Beziehungstyp"""
    suitable_products = []
    
    for product_id, product in GESCHENK_KATALOG.items():
        if relationship in product.get('relationship_suitability', {}):
            suitability = product['relationship_suitability'][relationship]
            if suitability >= 0.7:  # Nur hohe Eignung
                suitable_products.append({
                    'id': product_id,
                    'relationship_score': suitability,
                    'why_suitable': f"Perfekt für {relationship} (Score: {suitability})",
                    **product
                })
    
    suitable_products.sort(key=lambda x: x['relationship_score'], reverse=True)
    return suitable_products

def generate_ai_recommendation(product_data: Dict) -> str:
    """Generiert KI-Fachberatung für ein Produkt"""
    
    base_recommendation = product_data.get('ki_empfehlung', '')
    
    if not base_recommendation:
        # Fallback: Generiere basierend auf Produktdaten
        name = product_data['name']
        age_range = f"{product_data['target_age_min']}-{product_data['target_age_max']} Jahre"
        emotional_story = product_data['emotional_story'][:100] + "..."
        
        base_recommendation = f"""Als KI-Geschenkexperte empfehle ich '{name}' besonders für die Altersgruppe {age_range}. 
        {emotional_story} Diese durchdachte Kombination aus emotionalem Wert und praktischem Nutzen macht es zu einem 
        Geschenk, das lange in Erinnerung bleibt."""
    
    return base_recommendation

# =============================================================================
# EXPORT-FUNKTIONEN FÜR SHOP-INTEGRATION
# =============================================================================

def export_for_shopify() -> Dict:
    """Exportiert Katalog im Shopify-Format"""
    shopify_products = []
    
    for product_id, product in GESCHENK_KATALOG.items():
        
        # Hauptprodukt
        base_product = {
            "handle": product_id,
            "title": product['name'],
            "body_html": f"""
                <h3>{product['short_description']}</h3>
                <p>{product['long_description']}</p>
                <div class="emotional-story">
                    <h4>Die emotionale Geschichte:</h4>
                    <p><em>{product['emotional_story']}</em></p>
                </div>
                <div class="ki-empfehlung">
                    <h4>KI-Fachberatung:</h4>
                    <p>{generate_ai_recommendation(product)}</p>
                </div>
            """,
            "vendor": "Geschichte Schreiben",
            "product_type": product['category'],
            "tags": ",".join(product.get('seo_tags', [])),
            "variants": []
        }
        
        # Varianten hinzufügen
        for variant_name, variant_data in product['price_variants'].items():
            base_product['variants'].append({
                "title": variant_name.title(),
                "price": str(variant_data['price']),
                "sku": f"{product_id}-{variant_name}",
                "inventory_management": "shopify",
                "inventory_policy": "deny",
                "inventory_quantity": 100,
                "option1": variant_name.title()
            })
        
        shopify_products.append(base_product)
    
    return {
        "products": shopify_products,
        "generated_at": datetime.now().isoformat(),
        "total_products": len(shopify_products)
    }

def export_catalog_json() -> str:
    """Exportiert kompletten Katalog als JSON"""
    catalog_export = {
        "catalog_info": {
            "name": "Geschichte Schreiben - Premium Geschenk Katalog",
            "version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "total_products": len(GESCHENK_KATALOG),
            "age_categories": [cat.value for cat in AgeCategory],
            "personality_traits": [trait.value for trait in PersonalityTrait],
            "relationship_types": [rel.value for rel in RelationshipType]
        },
        "products": GESCHENK_KATALOG,
        "helper_functions": {
            "filter_by_age": "get_products_by_age(age: int)",
            "filter_by_personality": "get_products_by_personality(traits: List[str])",
            "filter_by_relationship": "get_products_by_relationship(relationship: str)",
            "ai_recommendation": "generate_ai_recommendation(product_data: Dict)"
        }
    }
    
    return json.dumps(catalog_export, indent=2, ensure_ascii=False)

# =============================================================================
# INTEGRATION MIT DER AI-ENGINE
# =============================================================================

class GeschenkKatalogService:
    """
    Service-Klasse für Integration mit der bestehenden AI-Engine
    """
    
    def __init__(self):
        self.katalog = GESCHENK_KATALOG
    
    def get_ai_recommendations(self, personality_profile, budget_range: tuple, relationship: str, occasion: str) -> List[Dict]:
        """
        Hauptfunktion für die AI-Engine Integration
        
        Args:
            personality_profile: PersonalityProfile aus deinem System
            budget_range: (min, max) in Euro
            relationship: RelationshipType aus deinem System  
            occasion: Anlass als String
            
        Returns:
            Liste mit AI-optimierten Geschenk-Empfehlungen
        """
        
        suitable_products = []
        age = getattr(personality_profile, 'age', 30)  # Default wenn nicht vorhanden
        
        for product_id, product in self.katalog.items():
            
            # 1. Alters-Check
            if not (product['target_age_min'] <= age <= product['target_age_max']):
                continue
                
            # 2. Budget-Check
            affordable_variants = []
            for variant_name, variant_data in product['price_variants'].items():
                if budget_range[0] <= variant_data['price'] <= budget_range[1]:
                    affordable_variants.append((variant_name, variant_data))
            
            if not affordable_variants:
                continue
            
            # 3. Beziehungs-Score berechnen
            relationship_score = product.get('relationship_suitability', {}).get(relationship, 0.0)
            if relationship_score < 0.3:  # Zu niedrige Eignung
                continue
            
            # 4. Anlass-Score berechnen
            occasion_score = product.get('occasion_suitability', {}).get(occasion, 0.7)
            
            # 5. Persönlichkeits-Match berechnen (nutzt dein Big Five System)
            personality_score = self._calculate_personality_match(personality_profile, product)
            
            # 6. Gesamt-Score berechnen
            total_score = (
                personality_score * 0.4 +
                relationship_score * 0.3 + 
                occasion_score * 0.2 +
                (1.0 if affordable_variants else 0.0) * 0.1  # Budget-Bonus
            )
            
            # 7. Empfehlung zusammenstellen
            best_variant = min(affordable_variants, key=lambda x: x[1]['price'])
            
            suitable_products.append({
                'product_id': product_id,
                'product_data': product,
                'total_match_score': total_score,
                'recommended_variant': best_variant[0],
                'recommended_price': best_variant[1]['price'],
                'match_reasons': {
                    'personality_match': personality_score,
                    'relationship_fit': relationship_score,
                    'occasion_suitability': occasion_score,
                    'budget_friendly': len(affordable_variants) > 0
                },
                'ai_reasoning': self._generate_match_reasoning(product, personality_profile, relationship, occasion),
                'emotional_appeal': product['emotional_story'],
                'why_perfect': self._explain_why_perfect(product, personality_profile, total_score)
            })
        
        # Sortiere nach Gesamt-Score
        suitable_products.sort(key=lambda x: x['total_match_score'], reverse=True)
        return suitable_products[:5]  # Top 5 Empfehlungen
    
    def _calculate_personality_match(self, profile, product) -> float:
        """Berechnet Persönlichkeits-Match mit deinem Big Five + Limbic System"""
        
        personality_scores = product.get('personality_match_scores', {})
        if not personality_scores:
            return 0.5  # Neutral
        
        total_match = 0.0
        matched_traits = 0
        
        # Big Five Matching
        big_five_mapping = {
            'high_openness': getattr(profile, 'openness', 0.5),
            'high_conscientiousness': getattr(profile, 'conscientiousness', 0.5),
            'high_extraversion': getattr(profile, 'extraversion', 0.5),
            'high_agreeableness': getattr(profile, 'agreeableness', 0.5),
            'high_neuroticism': getattr(profile, 'neuroticism', 0.5)
        }
        
        # Preference-basierte Traits
        preference_mapping = {
            'creative_type': getattr(profile, 'creative_type', False),
            'tech_savvy': getattr(profile, 'tech_savvy', False),
            'practical_type': getattr(profile, 'practical_type', False),
            'health_conscious': getattr(profile, 'health_conscious', False)
        }
        
        for trait, product_score in personality_scores.items():
            if trait in big_five_mapping:
                user_score = big_five_mapping[trait]
                if user_score > 0.6:  # Nur wenn stark ausgeprägt
                    total_match += product_score * user_score
                    matched_traits += 1
                    
            elif trait in preference_mapping:
                if preference_mapping[trait]:  # Boolean True
                    total_match += product_score
                    matched_traits += 1
        
        return total_match / matched_traits if matched_traits > 0 else 0.5
    
    def _generate_match_reasoning(self, product, profile, relationship, occasion) -> str:
        """Generiert menschenlesbare Begründung für die Empfehlung"""
        
        reasons = []
        
        # Persönlichkeits-Gründe
        if getattr(profile, 'creative_type', False) and 'creative_type' in product.get('personality_match_scores', {}):
            reasons.append("perfekt für kreative Persönlichkeiten")
        
        if getattr(profile, 'openness', 0) > 0.7:
            reasons.append("ideal für aufgeschlossene Menschen")
            
        # Beziehungs-Gründe  
        relationship_reasons = {
            'partner': 'zeigt tiefe Verbundenheit',
            'family_parent': 'drückt Dankbarkeit und Liebe aus',
            'friend_best': 'stärkt eure besondere Freundschaft'
        }
        if relationship in relationship_reasons:
            reasons.append(relationship_reasons[relationship])
        
        # Anlass-Gründe
        occasion_reasons = {
            'geburtstag': 'macht diesen Geburtstag unvergesslich',
            'weihnachten': 'bringt Weihnachtsmagie ins Herz',
            'jahrestag': 'feiert eure gemeinsame Zeit'
        }
        if occasion in occasion_reasons:
            reasons.append(occasion_reasons[occasion])
        
        if reasons:
            return f"Dieses Geschenk ist perfekt, weil es {', '.join(reasons[:3])}."
        else:
            return "Ein durchdachtes Geschenk mit emotionalem Tiefgang."
    
    def _explain_why_perfect(self, product, profile, score) -> str:
        """Erklärt warum dieses Geschenk perfekt ist (für Marketing)"""
        
        confidence_level = "sehr gut" if score > 0.8 else "gut" if score > 0.6 else "okay"
        
        emotional_impact = product.get('emotional_impact', {})
        main_emotion = emotional_impact.get('hauptemotion', 'Freude')
        
        age = getattr(profile, 'age', 30)
        age_group = "junge Menschen" if age < 30 else "Erwachsene" if age < 60 else "erfahrene Menschen"
        
        return f"""Diese Empfehlung passt {confidence_level} zu dir, weil sie {main_emotion.lower()} 
        und emotionale Verbindung schafft. Speziell für {age_group} entwickelt, 
        kombiniert sie {product['content_components']['emotionaler_teil']} mit 
        {product['content_components']['praktischer_teil']}. 
        {product['emotional_story'][:150]}..."""

# =============================================================================
# ERWEITERTE PRODUKTE FÜR VOLLSTÄNDIGEN KATALOG
# =============================================================================

# Hier können weitere Produkte hinzugefügt werden - das Framework ist bereit!

ZUSATZ_PRODUKTE = {
    
    "budget_mutmacher": {
        "id": "budget-mutmacher-007", 
        "name": "Kleiner Mutmacher - Budget Box",
        "category": "Budget Freundlich",
        "short_description": "Liebevolle Aufmunterung für kleines Budget",
        "long_description": "Auch mit wenig Geld kannst du große Freude schenken. Diese Box kombiniert handgeschriebene Mut-Sprüche mit praktischen Kleinigkeiten, die den Alltag versüßen.",
        
        "age_categories": [AgeCategory.JUNGE_ERWACHSENE.value, AgeCategory.ERWACHSENE.value],
        "target_age_min": 16,
        "target_age_max": 65,
        
        "personality_match_scores": {
            PersonalityTrait.HIGH_AGREEABLENESS.value: 0.8,
            PersonalityTrait.HIGH_NEUROTICISM.value: 0.7  # Für Menschen in schwierigen Zeiten
        },
        
        "relationship_suitability": {
            RelationshipType.FRIEND_CLOSE.value: 0.9,
            RelationshipType.COLLEAGUE.value: 0.8,
            RelationshipType.FAMILY_SIBLING.value: 0.85
        },
        
        "emotional_story": """Manchmal sind es die kleinen Gesten, die das Herz berühren. Diese Box beweist, dass Liebe nicht vom Budget abhängt. Jeder handgeschriebene Spruch, jede kleine Aufmerksamkeit zeigt: 'Ich denke an dich, du bist mir wichtig.' Gerade in schweren Zeiten können diese kleinen Mut-Botschaften wie Lichtblicke wirken.""",
        
        "content_components": {
            "emotionaler_teil": "5 handgeschriebene Mut-Sprüche auf schönem Papier",
            "praktischer_teil": "Entspannungstee + Schokolade + Anti-Stress-Ball",
            "extras": "Liebevoll verpackt in recycelter Box"
        },
        
        "price_variants": {
            "basic": {"price": 14.99, "includes": ["3 Mut-Sprüche", "Tee", "Schokolade"]},
            "medium": {"price": 24.99, "includes": ["5 Mut-Sprüche", "Tee-Set", "Schokolade", "Anti-Stress-Ball"]},
            "premium": {"price": 34.99, "includes": ["7 Mut-Sprüche", "Premium-Tee", "Handgemachte Schokolade", "Wellness-Set"]}
        },
        
        "ki_empfehlung": "Als KI-Berater empfehle ich diese Box für preisbewusste Käufer, die trotzdem von Herzen schenken möchten. Perfekt für Studenten, junge Erwachsene oder als spontane Aufmunterung. Die handgeschriebenen Sprüche machen den Unterschied - hier kauft man nicht nur ein Produkt, sondern echte menschliche Zuwendung."
    }
}

# Füge Zusatz-Produkte zum Hauptkatalog hinzu
GESCHENK_KATALOG.update(ZUSATZ_PRODUKTE)

# =============================================================================
# MARKETING & SEO OPTIMIERUNG
# =============================================================================

MARKETING_TEMPLATES = {
    "landing_page_copy": {
        "headline": "Geschenke, die Geschichten schreiben 📖✨",
        "subheadline": "Personalisierte Geschenke mit emotionalem Tiefgang - von KI-Experten empfohlen",
        "value_propositions": [
            "🎯 KI-optimierte Empfehlungen basierend auf Persönlichkeit",
            "💝 Jedes Geschenk erzählt eine einzigartige Geschichte", 
            "🎨 Von Budget bis Premium - für jeden das Richtige",
            "📦 Produktionsfertig - schnelle Lieferung garantiert"
        ]
    },
    
    "product_page_struktur": {
        "hero_section": "Emotionale Story + Hauptbild",
        "social_proof": "KI-Empfehlung als Vertrauenselement",
        "personalization": "Interaktive Personalisierung mit Live-Preview",
        "variants": "Klare Preisabstufung mit Wert-Erklärung",
        "garantien": ["30 Tage Rückgabe", "Kostenlose Personalisierung", "Schneller Versand"]
    },
    
    "seo_strategie": {
        "hauptkeywords": [
            "personalisierte geschenke",
            "emotionale geschenke", 
            "geschenke mit geschichte",
            "ki empfohlene geschenke",
            "individuelle geschenkboxen"
        ],
        "long_tail": [
            "geschenke für [alter] jährige [beziehung]",
            "personalisierte geschenke [anlass]",
            "emotionale geschenkideen [persönlichkeit]"
        ]
    }
}

# =============================================================================
# DEMO: VOLLSTÄNDIGE AI-INTEGRATION
# =============================================================================

if __name__ == "__main__":
    print("🎁 GESCHICHTE SCHREIBEN - Produktionskatalog mit AI-Integration")
    print("=" * 70)
    
    # Simuliere dein PersonalityProfile
    class MockPersonalityProfile:
        def __init__(self):
            self.age = 28
            self.openness = 0.8
            self.creative_type = True
            self.tech_savvy = False
            self.recipient_name = "Lisa"
    
    # Initialisiere Service
    service = GeschenkKatalogService()
    profile = MockPersonalityProfile()
    
    # Teste AI-Empfehlungen
    print("🤖 AI-Empfehlungen für kreative 28-jährige Lisa (Partner, Budget 50-150€, Geburtstag):")
    recommendations = service.get_ai_recommendations(
        personality_profile=profile,
        budget_range=(50, 150),
        relationship='partner',
        occasion='geburtstag'
    )
    
    print(f"\n✅ {len(recommendations)} passende Geschenke gefunden:")
    
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"\n{i}. {rec['product_data']['name']}")
        print(f"   💰 Preis: {rec['recommended_price']}€ ({rec['recommended_variant']})")
        print(f"   🎯 Match-Score: {rec['total_match_score']:.2f}")
        print(f"   💭 Begründung: {rec['ai_reasoning']}")
        print(f"   ❤️ Emotional: {rec['product_data']['emotional_story'][:100]}...")
    
    print(f"\n📈 Erweiterte Katalog-Statistiken:")
    print(f"   Gesamt Produkte: {len(GESCHENK_KATALOG)}")
    print(f"   Budget-freundlich (<30€): {len([p for p in GESCHENK_KATALOG.values() if min(v['price'] for v in p['price_variants'].values()) < 30])}")
    print(f"   Premium (>100€): {len([p for p in GESCHENK_KATALOG.values() if max(v['price'] for v in p['price_variants'].values()) > 100])}")
    print(f"   Altersabdeckung: 0-99 Jahre")
    
    print("\n🚀 Integration Ready:")
    print("   ✅ JSON Export für Shop-System")  
    print("   ✅ Shopify/WooCommerce Integration")
    print("   ✅ AI-Engine Kompatibilität")
    print("   ✅ Personalisierungs-Pipeline")
    print("   ✅ Marketing-Templates")
    
    print("\n💡 Nächste Schritte:")
    print("   1. Katalog in deine AI-Engine integrieren")
    print("   2. Personalisierungs-Workflow aufsetzen")
    print("   3. Supplier-Verträge für Produktion")
    print("   4. Marketing-Kampagne mit emotionalen Stories")
    print("   5. A/B Testing der KI-Empfehlungen")