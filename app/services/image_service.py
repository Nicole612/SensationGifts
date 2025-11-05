# 📁 app/services/image_service.py
# ================================================================
# IMAGE MANAGEMENT SERVICE - Emotionale Bilder für Online Shop
# ================================================================

import os
import uuid
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageEnhance, ImageFilter
import boto3
from flask import current_app, request
import requests
from io import BytesIO
import hashlib
from datetime import datetime, timedelta

class ImageService:
    """
    Comprehensive Image Management Service
    - Image Upload & Processing
    - Automatic Optimization (WebP, different sizes)
    - Emotional Context Detection
    - CDN Integration
    - AI-powered Image Enhancement
    """
    
    def __init__(self):
        self.s3_client = None
        self.bucket_name = current_app.config.get('S3_BUCKET_NAME')
        self.cdn_base_url = current_app.config.get('CDN_BASE_URL', '')
        
        # Initialize S3 if configured
        if current_app.config.get('AWS_ACCESS_KEY_ID'):
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=current_app.config.get('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=current_app.config.get('AWS_SECRET_ACCESS_KEY'),
                region_name=current_app.config.get('AWS_REGION', 'eu-central-1')
            )
    
    # ================================================================
    # IMAGE UPLOAD & PROCESSING
    # ================================================================
    
    def upload_product_image(
        self, 
        image_file, 
        product_id: str,
        image_type: str = 'primary',  # primary, gallery, lifestyle, detail
        emotional_context: Optional[str] = None
    ) -> Dict:
        """
        Upload and process product image with multiple optimizations
        """
        try:
            # Generate unique filename
            file_extension = image_file.filename.split('.')[-1].lower()
            unique_id = str(uuid.uuid4())
            base_filename = f"products/{product_id}/{image_type}_{unique_id}"
            
            # Open and validate image
            image = Image.open(image_file)
            
            # Validate image
            validation_result = self._validate_image(image, image_type)
            if not validation_result['valid']:
                return {'success': False, 'error': validation_result['error']}
            
            # Enhance image if needed
            enhanced_image = self._enhance_image(image, emotional_context)
            
            # Generate multiple sizes and formats
            image_variants = self._generate_image_variants(enhanced_image, base_filename)
            
            # Upload all variants
            uploaded_urls = {}
            for variant_name, (variant_image, variant_filename) in image_variants.items():
                url = self._upload_to_storage(variant_image, variant_filename)
                uploaded_urls[variant_name] = url
            
            # Generate metadata
            metadata = self._extract_image_metadata(enhanced_image, emotional_context)
            
            # Save to database
            image_record = self._save_image_record(
                product_id=product_id,
                image_type=image_type,
                urls=uploaded_urls,
                metadata=metadata,
                emotional_context=emotional_context
            )
            
            return {
                'success': True,
                'image_id': image_record['id'],
                'urls': uploaded_urls,
                'metadata': metadata,
                'optimizations_applied': len(image_variants)
            }
            
        except Exception as e:
            current_app.logger.error(f"Image upload failed: {str(e)}")
            return {'success': False, 'error': 'Image processing failed'}
    
    def _validate_image(self, image: Image.Image, image_type: str) -> Dict:
        """Validate image requirements"""
        
        # Size requirements by type
        size_requirements = {
            'primary': {'min_width': 800, 'min_height': 600, 'max_size': 10*1024*1024},
            'gallery': {'min_width': 600, 'min_height': 400, 'max_size': 8*1024*1024},
            'lifestyle': {'min_width': 1000, 'min_height': 700, 'max_size': 12*1024*1024},
            'detail': {'min_width': 1200, 'min_height': 800, 'max_size': 15*1024*1024}
        }
        
        requirements = size_requirements.get(image_type, size_requirements['primary'])
        width, height = image.size
        
        # Check dimensions
        if width < requirements['min_width'] or height < requirements['min_height']:
            return {
                'valid': False,
                'error': f'Image too small. Minimum size: {requirements["min_width"]}x{requirements["min_height"]}px'
            }
        
        # Check format
        if image.format not in ['JPEG', 'PNG', 'WEBP']:
            return {
                'valid': False,
                'error': 'Unsupported format. Please use JPEG, PNG, or WebP'
            }
        
        # Check for inappropriate content (basic)
        if self._contains_inappropriate_content(image):
            return {
                'valid': False,
                'error': 'Image contains inappropriate content'
            }
        
        return {'valid': True}
    
    def _enhance_image(self, image: Image.Image, emotional_context: str = None) -> Image.Image:
        """Apply AI-powered image enhancements based on emotional context"""
        
        enhanced = image.copy()
        
        # Apply emotional enhancements
        if emotional_context:
            if emotional_context == 'romantic':
                # Warm tone, soft contrast
                enhancer = ImageEnhance.Color(enhanced)
                enhanced = enhancer.enhance(1.1)  # Slightly more colorful
                
                enhancer = ImageEnhance.Brightness(enhanced)
                enhanced = enhancer.enhance(1.05)  # Slightly brighter
                
                # Add subtle blur for dreamy effect
                enhanced = enhanced.filter(ImageFilter.GaussianBlur(radius=0.5))
                
            elif emotional_context == 'excitement':
                # High contrast, vibrant colors
                enhancer = ImageEnhance.Contrast(enhanced)
                enhanced = enhancer.enhance(1.2)
                
                enhancer = ImageEnhance.Color(enhanced)
                enhanced = enhancer.enhance(1.3)
                
            elif emotional_context == 'comfort':
                # Warm, soft tones
                enhancer = ImageEnhance.Brightness(enhanced)
                enhanced = enhancer.enhance(1.1)
                
                enhancer = ImageEnhance.Contrast(enhanced)
                enhanced = enhancer.enhance(0.9)  # Softer contrast
        
        # Standard enhancements for all images
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(1.1)  # Slight sharpening
        
        return enhanced
    
    def _generate_image_variants(self, image: Image.Image, base_filename: str) -> Dict:
        """Generate multiple sizes and formats for responsive design"""
        
        variants = {}
        
        # Size variants
        sizes = {
            'thumbnail': (150, 150),
            'card': (400, 300),
            'medium': (600, 450),
            'large': (800, 600),
            'xlarge': (1200, 900),
            'hero': (1920, 1080)
        }
        
        for size_name, (max_width, max_height) in sizes.items():
            # Calculate dimensions maintaining aspect ratio
            img_width, img_height = image.size
            ratio = min(max_width / img_width, max_height / img_height)
            
            if ratio < 1:  # Only resize if image is larger than target
                new_width = int(img_width * ratio)
                new_height = int(img_height * ratio)
                resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            else:
                resized = image.copy()
            
            # WebP format (primary)
            webp_filename = f"{base_filename}_{size_name}.webp"
            variants[f"{size_name}_webp"] = (resized, webp_filename)
            
            # JPEG fallback
            if resized.mode in ('RGBA', 'LA', 'P'):
                rgb_image = Image.new('RGB', resized.size, (255, 255, 255))
                rgb_image.paste(resized, mask=resized.split()[-1] if resized.mode in ('RGBA', 'LA') else None)
                resized = rgb_image
            
            jpg_filename = f"{base_filename}_{size_name}.jpg"
            variants[f"{size_name}_jpg"] = (resized, jpg_filename)
        
        return variants
    
    def _upload_to_storage(self, image: Image.Image, filename: str) -> str:
        """Upload image to storage (S3 or local)"""
        
        if self.s3_client:
            # Upload to S3
            buffer = BytesIO()
            
            if filename.endswith('.webp'):
                image.save(buffer, format='WebP', quality=85, optimize=True)
            else:
                image.save(buffer, format='JPEG', quality=85, optimize=True)
            
            buffer.seek(0)
            
            try:
                self.s3_client.upload_fileobj(
                    buffer,
                    self.bucket_name,
                    filename,
                    ExtraArgs={
                        'ContentType': 'image/webp' if filename.endswith('.webp') else 'image/jpeg',
                        'CacheControl': 'max-age=31536000',  # 1 year cache
                        'ACL': 'public-read'
                    }
                )
                
                return f"{self.cdn_base_url}/{filename}"
                
            except Exception as e:
                current_app.logger.error(f"S3 upload failed: {str(e)}")
                # Fallback to local storage
                return self._upload_to_local(image, filename)
        else:
            # Local storage
            return self._upload_to_local(image, filename)
    
    def _upload_to_local(self, image: Image.Image, filename: str) -> str:
        """Upload image to local storage"""
        
        upload_dir = current_app.config.get('UPLOAD_FOLDER', 'app/static/uploads')
        os.makedirs(os.path.dirname(os.path.join(upload_dir, filename)), exist_ok=True)
        
        full_path = os.path.join(upload_dir, filename)
        
        if filename.endswith('.webp'):
            image.save(full_path, format='WebP', quality=85, optimize=True)
        else:
            image.save(full_path, format='JPEG', quality=85, optimize=True)
        
        return f"/static/uploads/{filename}"
    
    # ================================================================
    # EMOTIONAL IMAGE ANALYSIS
    # ================================================================
    
    def analyze_emotional_content(self, image_url: str) -> Dict:
        """
        Analyze image for emotional content using AI
        (This would integrate with services like Azure Cognitive Services or AWS Rekognition)
        """
        
        try:
            # For demo purposes, we'll simulate emotional analysis
            # In production, this would call external AI services
            
            emotions_detected = {
                'primary_emotion': 'joy',
                'confidence': 0.85,
                'all_emotions': {
                    'joy': 0.85,
                    'surprise': 0.12,
                    'love': 0.78,
                    'comfort': 0.45,
                    'excitement': 0.23
                },
                'demographics': {
                    'age_groups': ['young_adults', 'adults'],
                    'gender_appeal': ['universal'],
                    'cultural_context': ['western', 'family_oriented']
                },
                'visual_elements': {
                    'color_mood': 'warm',
                    'brightness': 'high',
                    'contrast': 'medium',
                    'composition': 'balanced'
                }
            }
            
            return {
                'success': True,
                'analysis': emotions_detected
            }
            
        except Exception as e:
            current_app.logger.error(f"Emotional analysis failed: {str(e)}")
            return {
                'success': False,
                'error': 'Analysis failed'
            }
    
    def suggest_image_improvements(self, image_analysis: Dict) -> List[str]:
        """Suggest improvements based on emotional analysis"""
        
        suggestions = []
        
        analysis = image_analysis.get('analysis', {})
        
        # Brightness suggestions
        if analysis.get('visual_elements', {}).get('brightness') == 'low':
            suggestions.append("Bild könnte heller sein für mehr positive Ausstrahlung")
        
        # Color suggestions
        if analysis.get('visual_elements', {}).get('color_mood') == 'cold':
            suggestions.append("Wärmere Farbtöne könnten emotionale Verbindung stärken")
        
        # Emotional suggestions
        primary_emotion = analysis.get('primary_emotion')
        if primary_emotion not in ['joy', 'love', 'excitement']:
            suggestions.append("Bild könnte positivere Emotionen vermitteln")
        
        # Composition suggestions
        if analysis.get('visual_elements', {}).get('composition') == 'unbalanced':
            suggestions.append("Bildkomposition könnte ausgewogener sein")
        
        return suggestions
    
    # ================================================================
    # IMAGE OPTIMIZATION & DELIVERY
    # ================================================================
    
    def get_optimized_image_url(
        self,
        base_url: str,
        size: str = 'medium',
        format: str = 'webp',
        quality: int = 85
    ) -> str:
        """Get optimized image URL with fallback"""
        
        # Check if browser supports WebP
        user_agent = request.headers.get('User-Agent', '').lower()
        supports_webp = 'chrome' in user_agent or 'firefox' in user_agent or 'edge' in user_agent
        
        if not supports_webp:
            format = 'jpg'
        
        # Construct optimized URL
        base_name = base_url.rsplit('.', 1)[0]
        optimized_url = f"{base_name}_{size}.{format}"
        
        return optimized_url
    
    def generate_responsive_image_srcset(self, base_url: str) -> Dict:
        """Generate srcset for responsive images"""
        
        sizes = ['thumbnail', 'card', 'medium', 'large', 'xlarge']
        widths = [150, 400, 600, 800, 1200]
        
        webp_srcset = []
        jpg_srcset = []
        
        base_name = base_url.rsplit('.', 1)[0]
        
        for size, width in zip(sizes, widths):
            webp_srcset.append(f"{base_name}_{size}.webp {width}w")
            jpg_srcset.append(f"{base_name}_{size}.jpg {width}w")
        
        return {
            'webp_srcset': ', '.join(webp_srcset),
            'jpg_srcset': ', '.join(jpg_srcset),
            'sizes': '(max-width: 400px) 400px, (max-width: 800px) 600px, 800px'
        }
    
    # ================================================================
    # IMAGE METADATA & DATABASE OPERATIONS
    # ================================================================
    
    def _extract_image_metadata(self, image: Image.Image, emotional_context: str = None) -> Dict:
        """Extract comprehensive metadata from image"""
        
        return {
            'dimensions': {
                'width': image.size[0],
                'height': image.size[1]
            },
            'format': image.format,
            'mode': image.mode,
            'has_transparency': image.mode in ('RGBA', 'LA', 'P'),
            'emotional_context': emotional_context,
            'file_size_kb': 0,  # Will be calculated during upload
            'color_profile': self._analyze_color_profile(image),
            'upload_timestamp': datetime.utcnow().isoformat()
        }
    
    def _analyze_color_profile(self, image: Image.Image) -> Dict:
        """Analyze color characteristics for emotional context"""
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Sample colors from image
        colors = list(image.getdata())
        
        # Calculate averages
        total_r = sum(color[0] for color in colors)
        total_g = sum(color[1] for color in colors)
        total_b = sum(color[2] for color in colors)
        num_pixels = len(colors)
        
        avg_r = total_r / num_pixels
        avg_g = total_g / num_pixels
        avg_b = total_b / num_pixels
        
        # Determine color mood
        if avg_r > avg_g and avg_r > avg_b:
            color_mood = 'warm'
        elif avg_b > avg_r and avg_b > avg_g:
            color_mood = 'cool'
        else:
            color_mood = 'neutral'
        
        # Calculate brightness
        brightness = (avg_r + avg_g + avg_b) / 3
        brightness_level = 'high' if brightness > 170 else 'medium' if brightness > 85 else 'low'
        
        return {
            'dominant_rgb': [int(avg_r), int(avg_g), int(avg_b)],
            'color_mood': color_mood,
            'brightness_level': brightness_level,
            'brightness_value': brightness
        }
    
    def _save_image_record(
        self,
        product_id: str,
        image_type: str,
        urls: Dict,
        metadata: Dict,
        emotional_context: str = None
    ) -> Dict:
        """Save image record to database"""
        
        from app.models.product_image import ProductImage
        from app.models import db
        
        image_record = ProductImage(
            id=str(uuid.uuid4()),
            product_id=product_id,
            image_type=image_type,
            primary_url=urls.get('medium_webp', ''),
            url_variants=urls,
            metadata=metadata,
            emotional_context=emotional_context,
            is_active=True
        )
        
        db.session.add(image_record)
        db.session.commit()
        
        return {
            'id': image_record.id,
            'product_id': product_id,
            'urls': urls
        }
    
    def _contains_inappropriate_content(self, image: Image.Image) -> bool:
        """Basic content filtering (would use external AI services in production)"""
        
        # For demo purposes, always return False
        # In production, this would integrate with content moderation APIs
        return False
    
    # ================================================================
    # BULK IMAGE OPERATIONS
    # ================================================================
    
    def bulk_optimize_images(self, product_ids: List[str]) -> Dict:
        """Optimize images for multiple products"""
        
        results = {
            'processed': 0,
            'failed': 0,
            'total_size_saved': 0,
            'errors': []
        }
        
        for product_id in product_ids:
            try:
                # Get all images for product
                from app.models.product_image import ProductImage
                images = ProductImage.query.filter_by(product_id=product_id).all()
                
                for image_record in images:
                    # Re-optimize existing images
                    optimization_result = self._reoptimize_image(image_record)
                    
                    if optimization_result['success']:
                        results['processed'] += 1
                        results['total_size_saved'] += optimization_result.get('size_saved', 0)
                    else:
                        results['failed'] += 1
                        results['errors'].append(f"Product {product_id}: {optimization_result.get('error')}")
                        
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(f"Product {product_id}: {str(e)}")
        
        return results
    
    def _reoptimize_image(self, image_record) -> Dict:
        """Re-optimize existing image with latest algorithms"""
        
        try:
            # Download current image
            primary_url = image_record.primary_url
            response = requests.get(primary_url)
            image = Image.open(BytesIO(response.content))
            
            # Apply latest optimizations
            enhanced_image = self._enhance_image(image, image_record.emotional_context)
            
            # Generate new variants
            base_filename = f"products/{image_record.product_id}/optimized_{image_record.id}"
            image_variants = self._generate_image_variants(enhanced_image, base_filename)
            
            # Upload optimized versions
            optimized_urls = {}
            for variant_name, (variant_image, variant_filename) in image_variants.items():
                url = self._upload_to_storage(variant_image, variant_filename)
                optimized_urls[variant_name] = url
            
            # Update database record
            image_record.url_variants = optimized_urls
            image_record.metadata['last_optimized'] = datetime.utcnow().isoformat()
            
            from app.models import db
            db.session.commit()
            
            return {
                'success': True,
                'size_saved': 0  # Would calculate actual size difference
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# ================================================================
# FLASK ROUTES - Image API
# ================================================================

from flask import Blueprint, request, jsonify, send_file
from flask_login import login_required
from werkzeug.utils import secure_filename

images_bp = Blueprint('images', __name__)

@images_bp.route('/api/images/upload', methods=['POST'])
@login_required
def upload_image():
    """Upload and process product image"""
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    product_id = request.form.get('product_id')
    image_type = request.form.get('image_type', 'primary')
    emotional_context = request.form.get('emotional_context')
    
    if not product_id:
        return jsonify({'error': 'Product ID required'}), 400
    
    image_service = ImageService()
    result = image_service.upload_product_image(
        image_file=image_file,
        product_id=product_id,
        image_type=image_type,
        emotional_context=emotional_context
    )
    
    if result['success']:
        return jsonify(result), 201
    else:
        return jsonify(result), 400

@images_bp.route('/api/images/<image_id>/analyze', methods=['POST'])
@login_required
def analyze_image_emotions(image_id):
    """Analyze emotional content of image"""
    
    from app.models.product_image import ProductImage
    image_record = ProductImage.query.get_or_404(image_id)
    
    image_service = ImageService()
    analysis = image_service.analyze_emotional_content(image_record.primary_url)
    
    if analysis['success']:
        # Save analysis results
        image_record.emotional_analysis = analysis['analysis']
        from app.models import db
        db.session.commit()
        
        # Generate improvement suggestions
        suggestions = image_service.suggest_image_improvements(analysis)
        
        return jsonify({
            'analysis': analysis['analysis'],
            'suggestions': suggestions
        })
    else:
        return jsonify(analysis), 500

@images_bp.route('/api/images/optimized')
def get_optimized_image():
    """Get optimized image with automatic format selection"""
    
    product_id = request.args.get('productId')
    size = request.args.get('size', 'medium')
    format = request.args.get('format', 'webp')
    
    if not product_id:
        return jsonify({'error': 'Product ID required'}), 400
    
    image_service = ImageService()
    
    # Get product's primary image
    from app.models.product_image import ProductImage
    image_record = ProductImage.query.filter_by(
        product_id=product_id,
        image_type='primary',
        is_active=True
    ).first()
    
    if not image_record:
        return jsonify({'error': 'Image not found'}), 404
    
    # Get optimized URL
    optimized_url = image_service.get_optimized_image_url(
        base_url=image_record.primary_url,
        size=size,
        format=format
    )
    
    return jsonify({
        'url': optimized_url,
        'responsive_srcset': image_service.generate_responsive_image_srcset(image_record.primary_url)
    })

@images_bp.route('/api/images/bulk-optimize', methods=['POST'])
@login_required
def bulk_optimize_images():
    """Bulk optimize images for multiple products"""
    
    data = request.get_json()
    product_ids = data.get('product_ids', [])
    
    if not product_ids:
        return jsonify({'error': 'Product IDs required'}), 400
    
    image_service = ImageService()
    results = image_service.bulk_optimize_images(product_ids)
    
    return jsonify(results)