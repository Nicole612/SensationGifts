# 📁 app/services/seo_service.py
# ================================================================
# SEO & PERFORMANCE OPTIMIZATION SERVICE
# Komplett System für Search Engine Optimization
# ================================================================

import json
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from urllib.parse import urlparse, urljoin
from flask import current_app, request, url_for
from bs4 import BeautifulSoup
import requests
from dataclasses import dataclass
import gzip
from xml.etree.ElementTree import Element, SubElement, tostring

@dataclass
class SEOAnalysis:
    url: str
    title: str
    description: str
    keywords: List[str]
    h1_tags: List[str]
    h2_tags: List[str]
    image_alts: List[str]
    internal_links: int
    external_links: int
    word_count: int
    loading_time: float
    mobile_friendly: bool
    has_schema: bool
    meta_robots: str
    canonical_url: str
    social_tags: Dict
    performance_score: int
    seo_score: int
    recommendations: List[str]

class SEOService:
    """
    Comprehensive SEO Service für SensationGifts Online Shop
    - Automatische SEO-Optimierung
    - Meta Tags Generation
    - Schema.org Structured Data
    - Sitemap Generation
    - Performance Monitoring
    """
    
    def __init__(self):
        self.base_url = current_app.config.get('BASE_URL', 'https://sensationgifts.com')
        self.site_name = 'SensationGifts'
        self.default_description = 'Entdecke emotionale Geschenke mit AI-powered Persönlichkeitsmatching. Perfekte Geschenke für jeden Anlass.'
        
    # ================================================================
    # META TAGS & SEO OPTIMIZATION
    # ================================================================
    
    def optimize_product_seo(self, product) -> Dict:
        """Generate optimized SEO data for product pages"""
        
        # Generate optimized title
        title = self._generate_product_title(product)
        
        # Generate meta description
        description = self._generate_product_description(product)
        
        # Generate keywords
        keywords = self._generate_product_keywords(product)
        
        # Generate schema.org data
        schema_data = self._generate_product_schema(product)
        
        # Generate Open Graph tags
        og_tags = self._generate_product_og_tags(product, title, description)
        
        # Generate Twitter Card tags
        twitter_tags = self._generate_product_twitter_tags(product, title, description)
        
        return {
            'title': title,
            'meta_description': description,
            'keywords': keywords,
            'canonical_url': f"{self.base_url}/produkt/{product.slug}",
            'schema_data': schema_data,
            'og_tags': og_tags,
            'twitter_tags': twitter_tags,
            'robots': 'index, follow',
            'structured_data': self._generate_structured_data(product)
        }
    
    def _generate_product_title(self, product) -> str:
        """Generate SEO-optimized product title"""
        
        # Base title with product name
        base_title = product.name
        
        # Add emotional context
        if product.emotional_tags:
            primary_emotion = product.emotional_tags[0] if isinstance(product.emotional_tags, list) else product.emotional_tags.get('primary')
            emotion_map = {
                'love': 'Romantisch',
                'joy': 'Freude',
                'surprise': 'Überraschung',
                'excitement': 'Aufregend',
                'comfort': 'Gemütlich',
                'gratitude': 'Dankbarkeit'
            }
            emotion_text = emotion_map.get(primary_emotion, '')
            if emotion_text:
                base_title = f"{emotion_text} {base_title}"
        
        # Add price range for commercial appeal
        if product.price_basic:
            if product.price_basic < 50:
                price_text = "Günstig"
            elif product.price_basic < 100:
                price_text = "Premium"
            else:
                price_text = "Luxus"
            base_title = f"{base_title} - {price_text}"
        
        # Add site name
        title = f"{base_title} | {self.site_name}"
        
        # Ensure title is within Google's limit (60 characters)
        if len(title) > 60:
            title = f"{base_title[:50]}... | {self.site_name}"
        
        return title
    
    def _generate_product_description(self, product) -> str:
        """Generate SEO-optimized meta description"""
        
        description_parts = []
        
        # Start with emotional hook
        if product.emotional_story:
            # Extract first sentence from emotional story
            first_sentence = product.emotional_story.split('.')[0] + '.'
            if len(first_sentence) < 100:
                description_parts.append(first_sentence)
        
        # Add product benefits
        if product.customization_options:
            description_parts.append("Personalisierbar")
        
        if product.gift_wrap_available:
            description_parts.append("Geschenkverpackung verfügbar")
        
        # Add price
        if product.price_basic:
            description_parts.append(f"Ab €{product.price_basic}")
        
        # Add delivery info
        if hasattr(product, 'delivery_time'):
            description_parts.append(f"Lieferung in {product.delivery_time}")
        
        # Combine and ensure length
        description = ' • '.join(description_parts)
        
        if len(description) > 160:
            description = description[:157] + '...'
        elif len(description) < 120:
            # Add more context if too short
            description += f" Jetzt bei {self.site_name} bestellen!"
        
        return description
    
    def _generate_product_keywords(self, product) -> List[str]:
        """Generate relevant keywords for product"""
        
        keywords = []
        
        # Product name variations
        keywords.append(product.name.lower())
        keywords.extend(product.name.lower().split())
        
        # Category keywords
        if product.categories:
            for category in product.categories:
                keywords.append(category.name.lower())
                keywords.append(f"{category.name.lower()} geschenk")
        
        # Emotional keywords
        if product.emotional_tags:
            emotion_keywords = {
                'love': ['romantisch', 'liebe', 'partner', 'beziehung'],
                'joy': ['freude', 'glück', 'fröhlich', 'positiv'],
                'surprise': ['überraschung', 'überraschen', 'spontan'],
                'excitement': ['aufregend', 'spannend', 'abenteuer'],
                'comfort': ['gemütlich', 'entspannen', 'ruhe', 'wellness'],
                'gratitude': ['dankbarkeit', 'danke', 'wertschätzung']
            }
            
            for emotion in product.emotional_tags if isinstance(product.emotional_tags, list) else [product.emotional_tags.get('primary')]:
                if emotion in emotion_keywords:
                    keywords.extend(emotion_keywords[emotion])
        
        # Price-based keywords
        if product.price_basic:
            if product.price_basic < 30:
                keywords.extend(['günstig', 'preiswert', 'budget'])
            elif product.price_basic > 100:
                keywords.extend(['premium', 'luxus', 'hochwertig'])
        
        # Occasion keywords
        keywords.extend([
            'geschenk', 'geschenkidee', 'schenken',
            'geburtstag', 'weihnachten', 'valentinstag',
            'hochzeit', 'anniversary', 'jubiläum'
        ])
        
        # Personalization keywords
        if product.customization_options:
            keywords.extend(['personalisiert', 'individuell', 'custom', 'gravur'])
        
        # Remove duplicates and return top 20
        unique_keywords = list(set(keywords))
        return unique_keywords[:20]
    
    def _generate_product_schema(self, product) -> Dict:
        """Generate Schema.org structured data for product"""
        
        schema = {
            "@context": "https://schema.org/",
            "@type": "Product",
            "name": product.name,
            "description": product.description,
            "brand": {
                "@type": "Brand",
                "name": self.site_name
            },
            "offers": {
                "@type": "Offer",
                "url": f"{self.base_url}/produkt/{product.slug}",
                "priceCurrency": product.currency or "EUR",
                "price": str(product.price_basic),
                "availability": "https://schema.org/InStock" if product.is_in_stock() else "https://schema.org/OutOfStock",
                "seller": {
                    "@type": "Organization",
                    "name": self.site_name
                }
            },
            "category": product.categories[0].name if product.categories else "Geschenke"
        }
        
        # Add images
        if product.primary_image:
            schema["image"] = [
                f"{self.base_url}{product.primary_image}",
            ]
            if product.image_gallery:
                schema["image"].extend([f"{self.base_url}{img}" for img in product.image_gallery])
        
        # Add ratings if available
        if product.average_rating and product.total_reviews:
            schema["aggregateRating"] = {
                "@type": "AggregateRating",
                "ratingValue": str(product.average_rating),
                "reviewCount": str(product.total_reviews),
                "bestRating": "5",
                "worstRating": "1"
            }
        
        # Add reviews
        if hasattr(product, 'reviews') and product.reviews:
            schema["review"] = []
            for review in product.reviews[:3]:  # Top 3 reviews
                schema["review"].append({
                    "@type": "Review",
                    "reviewRating": {
                        "@type": "Rating",
                        "ratingValue": str(review.rating),
                        "bestRating": "5",
                        "worstRating": "1"
                    },
                    "author": {
                        "@type": "Person",
                        "name": review.author_name
                    },
                    "reviewBody": review.content[:200] + "..." if len(review.content) > 200 else review.content
                })
        
        return schema
    
    def _generate_product_og_tags(self, product, title: str, description: str) -> Dict:
        """Generate Open Graph tags for social media"""
        
        return {
            'og:title': title,
            'og:description': description,
            'og:type': 'product',
            'og:url': f"{self.base_url}/produkt/{product.slug}",
            'og:image': f"{self.base_url}{product.primary_image}" if product.primary_image else '',
            'og:image:width': '800',
            'og:image:height': '600',
            'og:site_name': self.site_name,
            'product:price:amount': str(product.price_basic),
            'product:price:currency': product.currency or 'EUR',
            'product:availability': 'in stock' if product.is_in_stock() else 'out of stock'
        }
    
    def _generate_product_twitter_tags(self, product, title: str, description: str) -> Dict:
        """Generate Twitter Card tags"""
        
        return {
            'twitter:card': 'summary_large_image',
            'twitter:site': '@SensationGifts',  # Your Twitter handle
            'twitter:title': title,
            'twitter:description': description,
            'twitter:image': f"{self.base_url}{product.primary_image}" if product.primary_image else '',
            'twitter:creator': '@SensationGifts'
        }
    
    # ================================================================
    # SITEMAP GENERATION
    # ================================================================
    
    def generate_sitemap(self) -> str:
        """Generate XML sitemap for the entire site"""
        
        from app.models.enhanced_product import EnhancedProduct, Category
        
        # Root element
        urlset = Element('urlset')
        urlset.set('xmlns', 'http://www.sitemaps.org/schemas/sitemap/0.9')
        urlset.set('xmlns:image', 'http://www.google.com/schemas/sitemap-image/1.1')
        
        # Homepage
        url = SubElement(urlset, 'url')
        SubElement(url, 'loc').text = self.base_url
        SubElement(url, 'lastmod').text = datetime.now().strftime('%Y-%m-%d')
        SubElement(url, 'changefreq').text = 'daily'
        SubElement(url, 'priority').text = '1.0'
        
        # Static pages
        static_pages = [
            ('/', 1.0, 'daily'),
            ('/geschenke-finden', 0.9, 'weekly'),
            ('/geschenke-finden/quiz', 0.8, 'weekly'),  
            ('/geschenke-finden/direkt', 0.8, 'weekly'),
            ('/ueber-uns', 0.5, 'monthly'),
            ('/kontakt', 0.5, 'monthly'),
            ('/datenschutz', 0.3, 'yearly'),
            ('/impressum', 0.3, 'yearly')
        ]
        
        for page_url, priority, changefreq in static_pages:
            if page_url != '/':  # Skip homepage as already added
                url = SubElement(urlset, 'url')
                SubElement(url, 'loc').text = f"{self.base_url}{page_url}"
                SubElement(url, 'lastmod').text = datetime.now().strftime('%Y-%m-%d')
                SubElement(url, 'changefreq').text = changefreq
                SubElement(url, 'priority').text = str(priority)
        
        # Products
        products = EnhancedProduct.query.filter_by(status='active').all()
        for product in products:
            url = SubElement(urlset, 'url')
            SubElement(url, 'loc').text = f"{self.base_url}/produkt/{product.slug}"
            SubElement(url, 'lastmod').text = product.updated_at.strftime('%Y-%m-%d')
            SubElement(url, 'changefreq').text = 'weekly'
            SubElement(url, 'priority').text = '0.8'
            
            # Add product images
            if product.primary_image:
                image = SubElement(url, 'image:image')
                SubElement(image, 'image:loc').text = f"{self.base_url}{product.primary_image}"
                SubElement(image, 'image:title').text = product.name
                SubElement(image, 'image:caption').text = product.short_description or product.name
            
            # Additional gallery images
            if product.image_gallery:
                for img_url in product.image_gallery[:3]:  # Limit to 3 additional images
                    image = SubElement(url, 'image:image')
                    SubElement(image, 'image:loc').text = f"{self.base_url}{img_url}"
                    SubElement(image, 'image:title').text = f"{product.name} - Gallery"
        
        # Categories
        categories = Category.query.filter_by(is_active=True).all()
        for category in categories:
            url = SubElement(urlset, 'url')
            SubElement(url, 'loc').text = f"{self.base_url}/kategorie/{category.slug}"
            SubElement(url, 'lastmod').text = datetime.now().strftime('%Y-%m-%d')
            SubElement(url, 'changefreq').text = 'weekly'
            SubElement(url, 'priority').text = '0.7'
        
        return tostring(urlset, encoding='utf-8', method='xml').decode('utf-8')
    
    def generate_robots_txt(self) -> str:
        """Generate robots.txt file"""
        
        robots_content = f"""User-agent: *
Allow: /

# Sitemap
Sitemap: {self.base_url}/sitemap.xml

# Disallow admin and sensitive areas
Disallow: /admin/
Disallow: /api/
Disallow: /auth/login
Disallow: /auth/register
Disallow: /warenkorb
Disallow: /checkout/
Disallow: /*?*

# Allow important pages
Allow: /geschenke-finden
Allow: /produkt/
Allow: /kategorie/

# Crawl delay for politeness
Crawl-delay: 1

# Popular bots specific rules
User-agent: Googlebot
Crawl-delay: 0

User-agent: Bingbot
Crawl-delay: 1
"""
        return robots_content
    
    # ================================================================
    # PERFORMANCE OPTIMIZATION
    # ================================================================
    
    def analyze_page_performance(self, url: str) -> Dict:
        """Analyze page performance and SEO metrics"""
        
        try:
            # Fetch page
            start_time = datetime.now()
            response = requests.get(url, timeout=10)
            load_time = (datetime.now() - start_time).total_seconds()
            
            if response.status_code != 200:
                return {'error': f'HTTP {response.status_code}'}
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Basic SEO elements
            title = soup.find('title').text if soup.find('title') else ''
            description = ''
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                description = meta_desc.get('content', '')
            
            # Keywords
            keywords = []
            meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
            if meta_keywords:
                keywords = [k.strip() for k in meta_keywords.get('content', '').split(',')]
            
            # Headings
            h1_tags = [h1.text.strip() for h1 in soup.find_all('h1')]
            h2_tags = [h2.text.strip() for h2 in soup.find_all('h2')]
            
            # Images
            images = soup.find_all('img')
            image_alts = [img.get('alt', '') for img in images]
            missing_alts = sum(1 for alt in image_alts if not alt)
            
            # Links
            links = soup.find_all('a', href=True)
            internal_links = sum(1 for link in links if self._is_internal_link(link['href'], url))
            external_links = len(links) - internal_links
            
            # Word count
            text_content = soup.get_text()
            word_count = len(text_content.split())
            
            # Mobile-friendly check
            viewport_meta = soup.find('meta', attrs={'name': 'viewport'})
            mobile_friendly = viewport_meta is not None
            
            # Schema.org check
            has_schema = soup.find('script', attrs={'type': 'application/ld+json'}) is not None
            
            # Canonical URL
            canonical = soup.find('link', attrs={'rel': 'canonical'})
            canonical_url = canonical.get('href', '') if canonical else ''
            
            # Social tags
            social_tags = self._extract_social_tags(soup)
            
            # Calculate scores
            seo_score = self._calculate_seo_score(
                title, description, h1_tags, image_alts, missing_alts,
                internal_links, has_schema, mobile_friendly
            )
            
            performance_score = self._calculate_performance_score(
                load_time, len(response.content), len(images)
            )
            
            # Generate recommendations
            recommendations = self._generate_seo_recommendations(
                title, description, h1_tags, missing_alts, has_schema,
                mobile_friendly, load_time
            )
            
            return SEOAnalysis(
                url=url,
                title=title,
                description=description,
                keywords=keywords,
                h1_tags=h1_tags,
                h2_tags=h2_tags,
                image_alts=image_alts,
                internal_links=internal_links,
                external_links=external_links,
                word_count=word_count,
                loading_time=load_time,
                mobile_friendly=mobile_friendly,
                has_schema=has_schema,
                meta_robots='',
                canonical_url=canonical_url,
                social_tags=social_tags,
                performance_score=performance_score,
                seo_score=seo_score,
                recommendations=recommendations
            ).__dict__
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_seo_score(self, title, description, h1_tags, image_alts, missing_alts, internal_links, has_schema, mobile_friendly) -> int:
        """Calculate SEO score (0-100)"""
        
        score = 0
        
        # Title (20 points)
        if title:
            if 30 <= len(title) <= 60:
                score += 20
            elif len(title) <= 70:
                score += 15
            else:
                score += 10
        
        # Description (20 points)  
        if description:
            if 140 <= len(description) <= 160:
                score += 20
            elif 120 <= len(description) <= 170:
                score += 15
            else:
                score += 10
        
        # H1 tags (15 points)
        if len(h1_tags) == 1:
            score += 15
        elif len(h1_tags) > 1:
            score += 10
        
        # Images with alt text (15 points)
        if image_alts:
            alt_ratio = (len(image_alts) - missing_alts) / len(image_alts)
            score += int(15 * alt_ratio)
        
        # Internal links (10 points)
        if internal_links >= 3:
            score += 10
        elif internal_links >= 1:
            score += 5
        
        # Schema.org (10 points)
        if has_schema:
            score += 10
        
        # Mobile friendly (10 points)
        if mobile_friendly:
            score += 10
        
        return min(score, 100)
    
    def _calculate_performance_score(self, load_time, page_size, image_count) -> int:
        """Calculate performance score (0-100)"""
        
        score = 100
        
        # Loading time penalties
        if load_time > 3.0:
            score -= 30
        elif load_time > 2.0:
            score -= 20
        elif load_time > 1.0:
            score -= 10
        
        # Page size penalties
        if page_size > 2 * 1024 * 1024:  # 2MB
            score -= 20
        elif page_size > 1 * 1024 * 1024:  # 1MB
            score -= 10
        
        # Image count penalties
        if image_count > 20:
            score -= 15
        elif image_count > 10:
            score -= 5
        
        return max(score, 0)
    
    def _generate_seo_recommendations(self, title, description, h1_tags, missing_alts, has_schema, mobile_friendly, load_time) -> List[str]:
        """Generate SEO improvement recommendations"""
        
        recommendations = []
        
        if not title:
            recommendations.append("Füge einen aussagekräftigen Title-Tag hinzu")
        elif len(title) > 60:
            recommendations.append("Kürze den Title-Tag auf unter 60 Zeichen")
        elif len(title) < 30:
            recommendations.append("Erweitere den Title-Tag auf mindestens 30 Zeichen")
        
        if not description:
            recommendations.append("Füge eine Meta-Description hinzu")
        elif len(description) > 160:
            recommendations.append("Kürze die Meta-Description auf unter 160 Zeichen")
        elif len(description) < 140:
            recommendations.append("Erweitere die Meta-Description auf mindestens 140 Zeichen")
        
        if len(h1_tags) == 0:
            recommendations.append("Füge einen H1-Tag hinzu")
        elif len(h1_tags) > 1:
            recommendations.append("Verwende nur einen H1-Tag pro Seite")
        
        if missing_alts > 0:
            recommendations.append(f"Füge Alt-Text zu {missing_alts} Bildern hinzu")
        
        if not has_schema:
            recommendations.append("Implementiere Schema.org Structured Data")
        
        if not mobile_friendly:
            recommendations.append("Füge Viewport Meta-Tag für Mobile-Optimierung hinzu")
        
        if load_time > 2.0:
            recommendations.append("Optimiere die Ladezeit (derzeit {:.1f}s)".format(load_time))
        
        return recommendations
    
    def _extract_social_tags(self, soup) -> Dict:
        """Extract Open Graph and Twitter Card tags"""
        
        social_tags = {}
        
        # Open Graph tags
        og_tags = soup.find_all('meta', attrs={'property': lambda x: x and x.startswith('og:')})
        for tag in og_tags:
            social_tags[tag['property']] = tag.get('content', '')
        
        # Twitter Card tags
        twitter_tags = soup.find_all('meta', attrs={'name': lambda x: x and x.startswith('twitter:')})
        for tag in twitter_tags:
            social_tags[tag['name']] = tag.get('content', '')
        
        return social_tags
    
    def _is_internal_link(self, href: str, base_url: str) -> bool:
        """Check if link is internal"""
        
        if href.startswith('/'):
            return True
        
        parsed_href = urlparse(href)
        parsed_base = urlparse(base_url)
        
        return parsed_href.netloc == parsed_base.netloc
    
    def _generate_structured_data(self, product) -> Dict:
        """Generate additional structured data for product"""
        
        # FAQ Schema for common questions
        faq_schema = {
            "@context": "https://schema.org",
            "@type": "FAQPage",
            "mainEntity": [
                {
                    "@type": "Question",
                    "name": "Ist dieses Geschenk personalisierbar?",
                    "acceptedAnswer": {
                        "@type": "Answer",
                        "text": "Ja, dieses Geschenk kann personalisiert werden." if product.customization_options else "Nein, dieses Geschenk ist nicht personalisierbar."
                    }
                },
                {
                    "@type": "Question", 
                    "name": "Wie lange dauert die Lieferung?",
                    "acceptedAnswer": {
                        "@type": "Answer",
                        "text": f"Die Lieferung dauert {product.get_estimated_delivery()['min_days']}-{product.get_estimated_delivery()['max_days']} Werktage."
                    }
                }
            ]
        }
        
        return faq_schema

# ================================================================
# FLASK ROUTES - SEO API
# ================================================================

from flask import Blueprint, Response, jsonify, render_template_string

seo_bp = Blueprint('seo', __name__)

@seo_bp.route('/sitemap.xml')
def sitemap():
    """Serve XML sitemap"""
    
    seo_service = SEOService()
    sitemap_xml = seo_service.generate_sitemap()
    
    response = Response(sitemap_xml, mimetype='application/xml')
    response.headers['Content-Encoding'] = 'gzip'
    
    # Compress for better performance
    compressed_xml = gzip.compress(sitemap_xml.encode('utf-8'))
    response.data = compressed_xml
    
    return response

@seo_bp.route('/robots.txt')
def robots():
    """Serve robots.txt"""
    
    seo_service = SEOService()
    robots_content = seo_service.generate_robots_txt()
    
    return Response(robots_content, mimetype='text/plain')

@seo_bp.route('/api/seo/analyze')
def analyze_page():
    """Analyze page SEO performance"""
    
    url = request.args.get('url')
    if not url:
        return jsonify({'error': 'URL parameter required'}), 400
    
    seo_service = SEOService()
    analysis = seo_service.analyze_page_performance(url)
    
    return jsonify(analysis)

@seo_bp.route('/api/seo/optimize-product/<product_id>')
def optimize_product(product_id):
    """Get SEO optimization data for product"""
    
    from app.models.enhanced_product import EnhancedProduct
    product = EnhancedProduct.query.get_or_404(product_id)
    
    seo_service = SEOService()
    seo_data = seo_service.optimize_product_seo(product)
    
    return jsonify(seo_data)

# ================================================================
# JINJA2 TEMPLATE HELPERS - For rendering meta tags
# ================================================================

def register_seo_helpers(app):
    """Register SEO template helpers"""
    
    @app.template_global()
    def render_meta_tags(seo_data):
        """Render meta tags in template"""
        
        meta_html = f"""
        <title>{seo_data.get('title', 'SensationGifts')}</title>
        <meta name="description" content="{seo_data.get('meta_description', '')}">
        <meta name="keywords" content="{', '.join(seo_data.get('keywords', []))}">
        <meta name="robots" content="{seo_data.get('robots', 'index, follow')}">
        <link rel="canonical" href="{seo_data.get('canonical_url', '')}">
        """
        
        # Open Graph tags
        for key, value in seo_data.get('og_tags', {}).items():
            if value:
                meta_html += f'\n        <meta property="{key}" content="{value}">'
        
        # Twitter tags
        for key, value in seo_data.get('twitter_tags', {}).items():
            if value:
                meta_html += f'\n        <meta name="{key}" content="{value}">'
        
        return meta_html
    
    @app.template_global()
    def render_structured_data(seo_data):
        """Render structured data JSON-LD"""
        
        schema_data = seo_data.get('schema_data', {})
        if not schema_data:
            return ''
        
        return f'<script type="application/ld+json">{json.dumps(schema_data, ensure_ascii=False)}</script>'
    
    @app.template_global()
    def generate_breadcrumbs(product=None, category=None):
        """Generate breadcrumb structured data"""
        
        breadcrumbs = {
            "@context": "https://schema.org",
            "@type": "BreadcrumbList",
            "itemListElement": [
                {
                    "@type": "ListItem",
                    "position": 1,
                    "name": "Home",
                    "item": "https://sensationgifts.com"
                }
            ]
        }
        
        if category:
            breadcrumbs["itemListElement"].append({
                "@type": "ListItem",
                "position": 2,
                "name": category.name,
                "item": f"https://sensationgifts.com/kategorie/{category.slug}"
            })
        
        if product:
            position = 3 if category else 2
            breadcrumbs["itemListElement"].append({
                "@type": "ListItem",
                "position": position,
                "name": product.name,
                "item": f"https://sensationgifts.com/produkt/{product.slug}"
            })
        
        return f'<script type="application/ld+json">{json.dumps(breadcrumbs, ensure_ascii=False)}</script>'