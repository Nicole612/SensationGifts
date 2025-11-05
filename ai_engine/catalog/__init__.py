"""
Datei: ai_engine/catalog/__init__.py
"""

from .gift_catalog_generator import (
    GESCHENK_KATALOG,
    GeschenkKatalogService,
    AgeCategory,
    PersonalityTrait,
    RelationshipType,
    get_products_by_age,
    get_products_by_personality,
    get_products_by_relationship,
    export_catalog_json,
    export_for_shopify
)

from .catalog_service import CatalogIntegrationService

__all__ = [
    'GESCHENK_KATALOG',
    'GeschenkKatalogService', 
    'CatalogIntegrationService',
    'AgeCategory',
    'PersonalityTrait',
    'RelationshipType',
    'get_products_by_age',
    'get_products_by_personality', 
    'get_products_by_relationship',
    'export_catalog_json',
    'export_for_shopify'
]