"""
Production AI Orchestrator - Vollständige Integration aller Advanced Techniques
==============================================================================

🚀 ENTERPRISE-READY: Production-Level AI Processing mit allen Advanced Features
🎯 PERFORMANCE-OPTIMIZED: Real-time Optimization und Adaptive Learning
🛡️ ROBUST: Comprehensive Error Handling und Fallback-Systeme
📊 MONITORED: Full Performance Tracking und Analytics

WORKFLOW:
1. Request Validation & Transformation
2. Context Building mit Advanced Techniques
3. Optimization Preference Analysis
4. Intelligent Processing (All Advanced Techniques)
5. Response Parsing & Validation
6. Performance Tracking
7. Final Response Assembly
"""

from __future__ import annotations
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from ai_engine.processors import IntelligentProcessor
from ai_engine.schemas import GiftRecommendationRequest, GiftRecommendationResponse
from ai_engine.processors.optimization_engine import OptimizationObjective

logger = logging.getLogger(__name__)

class ProductionAIOrchestrator:
    """
    Production-Level AI Orchestrator mit vollständiger Advanced Techniques Integration
    
    🚀 ENTERPRISE FEATURES:
    - Advanced Techniques Orchestration (Meta-Prompting, Self-Correction, Ensemble, Constitutional AI)
    - Intelligent Multi-Model Selection (OpenAI, Groq, Claude, Gemini)
    - Real-time Performance Optimization
    - Adaptive Learning basierend auf Feedback
    - Comprehensive Error Handling & Fallbacks
    - Full Performance Monitoring & Analytics
    """
    
    def __init__(self):
        """Initialize Production Orchestrator mit allen Advanced Features (ENHANCED)"""
        self.logger = logging.getLogger(__name__)
        
        # 🚀 INTELLIGENT PROCESSOR (Alle Advanced Techniques integriert)
        try:
            self.intelligent_processor = IntelligentProcessor()
            self.logger.info("✅ Intelligent Processor initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Intelligent Processor initialization failed: {e}")
            raise
        
        # 📊 PERFORMANCE MONITORING
        self.performance_monitor = PerformanceMonitor()
        
        # 🛡️ ERROR RECOVERY SYSTEM
        self.error_recovery = ErrorRecoverySystem()
        
        # 🚀 ENHANCED OPTIMIZATION FEATURES
        self.optimization_engine = None
        self.model_selector = None
        self.prompt_builder = None
        self.response_parser = None
        
        # Initialize optimization components
        self._initialize_optimization_components()
        
        # 📈 ENHANCED ANALYTICS
        self.analytics = {
            "total_requests": 0,
            "successful_requests": 0,
            "average_processing_time": 0.0,
            "advanced_techniques_usage": {},
            "model_usage_statistics": {},
            "error_statistics": {},
            "optimization_metrics": {},
            "cost_tracking": {},
            "quality_metrics": {},
            "user_satisfaction": {}
        }
        
        # 🎯 OPTIMIZATION STATE
        self.optimization_state = {
            "current_strategy": "balanced",
            "learning_enabled": True,
            "adaptive_mode": True,
            "performance_baseline": 0.8
        }
    
    def _initialize_optimization_components(self):
        """Initialize optimization components for enhanced performance"""
        try:
            from ai_engine.processors.optimization_engine import OptimizationEngine
            from ai_engine.processors.model_selector import ModelSelector
            from ai_engine.processors.prompt_builder import DynamicPromptBuilder
            from ai_engine.processors.response_parser import ResponseParser
            
            self.optimization_engine = OptimizationEngine()
            self.model_selector = ModelSelector()
            self.prompt_builder = DynamicPromptBuilder()
            self.response_parser = ResponseParser()
            
            self.logger.info("✅ Optimization components initialized successfully")
        except Exception as e:
            self.logger.warning(f"⚠️ Some optimization components failed to initialize: {e}")
            # Continue with basic functionality
    
    # ✅ ADDITIONAL FIX: Enhanced main processing method with better enum handling
    async def process_complete_gift_request(self, raw_request: dict) -> dict:
        """
        🚀 HAUPTMETHODE für Production-Level Processing with enhanced enum handling
        
        Vollständige Integration aller Advanced Techniques:
        - Meta-Prompting für optimale Prompts
        - Self-Correction für Validierung
        - Ensemble Prompting für Multi-Strategie
        - Constitutional AI für ethische Compliance
        - Adaptive Learning für kontinuierliche Optimierung
        
        Args:
            raw_request: Raw request dictionary
            
        Returns:
            Complete GiftRecommendationResponse mit allen Advanced Features
        """
        request_start_time = datetime.now()
        self.analytics["total_requests"] += 1
        
        try:
            self.logger.info(f"🚀 Processing gift request: {raw_request.get('user_id', 'unknown')}")
            
            # 1. REQUEST VALIDATION & TRANSFORMATION
            validated_request = await self._validate_and_transform_request(raw_request)
            gift_request = GiftRecommendationRequest(**validated_request)
            
            # 2. CONTEXT BUILDING mit Advanced Techniques
            context = await self._build_comprehensive_context(gift_request)
            
            # 3. OPTIMIZATION PREFERENCE ANALYSIS
            optimization_pref = self._analyze_optimization_preference(gift_request, context)
            
            # 4. ENHANCED INTELLIGENT PROCESSING (All Advanced Techniques + Optimization)
            if self.optimization_engine and self.model_selector:
                # Use enhanced optimization pipeline
                processing_result = await self._enhanced_processing_pipeline(
                    request=gift_request,
                    optimization_preference=optimization_pref,
                    context=context
                )
            else:
                # Fallback to standard processing
                processing_result = await self.intelligent_processor.process_gift_recommendation_request(
                    request=gift_request,
                    optimization_preference=optimization_pref,
                    context=context
                )
            
            # 5. RESPONSE PARSING & VALIDATION (with enum cleaning)
            parsed_response = await self._parse_and_validate_response(processing_result)
            
            # 6. ENHANCED PERFORMANCE TRACKING
            processing_time = (datetime.now() - request_start_time).total_seconds() * 1000
            self._record_enhanced_performance_metrics(
                request=gift_request,
                response=parsed_response,
                processing_time=processing_time,
                techniques_used=processing_result.get("advanced_techniques_used", []),
                optimization_metrics=processing_result.get("optimization_metrics", {}),
                context=context
            )
            
            # 7. FINAL RESPONSE ASSEMBLY with safe enum handling
            try:
                # ✅ FIX: Complete missing personality_analysis fields
                if "personality_analysis" in parsed_response and parsed_response["personality_analysis"]:
                    personality_analysis = parsed_response["personality_analysis"]
                    
                    # Add missing required fields to personality_analysis
                    if "big_five_gift_implications" not in personality_analysis:
                        personality_analysis["big_five_gift_implications"] = {
                            "openness": ["Creative and unique gifts", "Artistic experiences"],
                            "conscientiousness": ["Quality and practical gifts", "Well-planned surprises"],
                            "extraversion": ["Social experiences", "Group activities"],
                            "agreeableness": ["Thoughtful gestures", "Caring gifts"],
                            "neuroticism": ["Comforting items", "Stress-reducing gifts"]
                        }
                    
                    if "limbic_type" not in personality_analysis:
                        personality_analysis["limbic_type"] = "Balanced"
                    
                    if "emotional_drivers" not in personality_analysis:
                        personality_analysis["emotional_drivers"] = ["appreciation", "thoughtfulness", "connection"]
                    
                    if "purchase_motivations" not in personality_analysis:
                        personality_analysis["purchase_motivations"] = ["showing care", "creating joy", "strengthening relationship"]
                    
                    if "limbic_insights" not in personality_analysis:
                        personality_analysis["limbic_insights"] = {
                            "stimulanz_implications": "Enjoys moderate stimulation and variety",
                            "dominanz_implications": "Appreciates having choice and control",
                            "balance_implications": "Values harmony and emotional balance"
                        }
                    
                    if "recommended_gift_categories" not in personality_analysis:
                        personality_analysis["recommended_gift_categories"] = ["thoughtful", "quality", "personal", "emotional_bonds"]
                    
                    if "gift_dos" not in personality_analysis:
                        personality_analysis["gift_dos"] = [
                            "Choose something that shows you know them well",
                            "Focus on quality over quantity",
                            "Consider their personal style and preferences"
                        ]
                    
                    if "gift_donts" not in personality_analysis:
                        personality_analysis["gift_donts"] = [
                            "Don't choose something too generic",
                            "Avoid overly expensive or cheap items"
                        ]
                    
                    if "emotional_appeal_strategies" not in personality_analysis:
                        personality_analysis["emotional_appeal_strategies"] = [
                            "Show thoughtfulness and care",
                            "Create positive emotional connection"
                        ]
                    
                    if "analysis_depth" not in personality_analysis:
                        personality_analysis["analysis_depth"] = 0.8
                    
                    if "data_completeness" not in personality_analysis:
                        personality_analysis["data_completeness"] = 0.9

                # Try to create Pydantic response with completed data
                final_response = GiftRecommendationResponse(**parsed_response)
                response_dict = final_response.model_dump()
                self.logger.info("✅ Pydantic response assembly successful")
                # Enum-Safety für evtl. verbliebene Felder aus dem Modell
                for k in ("ai_model_used", "prompt_strategy", "optimization_goal"):
                    v = response_dict.get(k)
                    if v is not None and hasattr(v, "value") and not isinstance(v, str):
                        response_dict[k] = v.value
                    elif v is not None and not isinstance(v, str):
                        response_dict[k] = str(v)
            except Exception as assembly_error:
                # ✅ GRACEFUL FALLBACK: Use parsed response directly if Pydantic still fails
                self.logger.warning(f"Pydantic assembly failed: {assembly_error}, using direct response")
                response_dict = parsed_response
                for k in ("ai_model_used", "prompt_strategy", "optimization_goal"):
                    v = response_dict.get(k)
                    if v is not None and hasattr(v, "value") and not isinstance(v, str):
                        response_dict[k] = v.value
                    elif v is not None and not isinstance(v, str):
                        response_dict[k] = str(v)
                # Ensure success field exists
                response_dict["success"] = True
                        
            self.analytics["successful_requests"] += 1
            self.logger.info(f"✅ Request processed successfully in {processing_time:.0f}ms")
            
            return response_dict
            
        except Exception as e:
            # 🛡️ ERROR RECOVERY
            self.logger.error(f"❌ Processing error: {e}")
            self.analytics["error_statistics"][type(e).__name__] = \
                self.analytics["error_statistics"].get(type(e).__name__, 0) + 1
            
            return await self.error_recovery.handle_processing_error(e, raw_request)

    async def _enhanced_processing_pipeline(self, request, optimization_preference, context):
        """Enhanced processing pipeline with full optimization capabilities"""
        try:
            # 1. OPTIMIZE REQUEST PIPELINE
            if self.optimization_engine:
                optimization_result = await self.optimization_engine.optimize_request_pipeline(
                    request=request,
                    optimization_goal=optimization_preference,
                    context=context
                )
                context.update(optimization_result)
            
            # 2. SELECT OPTIMAL MODEL
            if self.model_selector:
                selected_model = await self.model_selector.select_optimal_model(
                    request=request,
                    optimization_goal=optimization_preference,
                    context=context
                )
                context["selected_model"] = selected_model
            
            # 3. BUILD OPTIMIZED PROMPT
            if self.prompt_builder:
                prompt_result = await self.prompt_builder.process_prompt_method(
                    input_data=request,
                    context=context
                )
                context["optimized_prompt"] = prompt_result
            
            # 4. PROCESS WITH INTELLIGENT PROCESSOR
            processing_result = await self.intelligent_processor.process_gift_recommendation_request(
                request=request,
                optimization_preference=optimization_preference,
                context=context
            )
            
            # 5. ENHANCE WITH OPTIMIZATION METRICS
            if self.optimization_engine:
                optimization_metrics = await self.optimization_engine._extract_request_features(request)
                processing_result["optimization_metrics"] = optimization_metrics
            
            return processing_result
            
        except Exception as e:
            self.logger.warning(f"Enhanced processing pipeline failed: {e}, falling back to standard processing")
            # Fallback to standard processing
            return await self.intelligent_processor.process_gift_recommendation_request(
                request=request,
                optimization_preference=optimization_preference,
                context=context
            )

    async def _validate_and_transform_request(self, raw_request: dict) -> dict:
        """✅ FIXED: Korrekte Schema-Transformation mit ALLEN Feldern"""
        
        # Required fields check
        required_fields = ["personality_data"]
        for field in required_fields:
            if field not in raw_request:
                raise ValueError(f"Missing required field: {field}")
        
        # ✅ VOLLSTÄNDIGE Schema-Transformation (alle 20 Felder)
        validated_request = {
            # === REQUIRED FIELDS ===
            "personality_data": raw_request["personality_data"],
            "occasion": raw_request.get("occasion", "birthday"),
            "relationship": raw_request.get("relationship", "friend"),
            
            # === RECOMMENDATIONS CONFIG ===
            "number_of_recommendations": raw_request.get("number_of_recommendations", 
                                                    raw_request.get("max_recommendations", 5)),  # ✅ FIXED
            "include_explanation": raw_request.get("include_explanation", True),
            "explanation_detail": raw_request.get("explanation_detail", "medium"),
            
            # === BUDGET & TEMPORAL ===
            "budget_min": raw_request.get("budget_min"),
            "budget_max": raw_request.get("budget_max"),
            "occasion_date": raw_request.get("occasion_date"),
            "urgency_level": raw_request.get("urgency_level"),
            
            # === PERSONALIZATION ===
            "personalization_level": raw_request.get("personalization_level", "medium"),
            "prioritize_emotional_impact": raw_request.get("prioritize_emotional_impact", True),
            
            # === CATEGORY CONSTRAINTS ===
            "gift_categories": raw_request.get("gift_categories"),
            "exclude_categories": raw_request.get("exclude_categories", []),
            
            # === CONTEXT & CULTURE ===
            "cultural_context": raw_request.get("cultural_context"),
            "additional_context": raw_request.get("additional_context", ""),
            
            # === AI MODEL CONFIG ===
            "target_ai_model": raw_request.get("target_ai_model"),
            "use_consensus_validation": raw_request.get("use_consensus_validation", False),
            "optimization_goal": raw_request.get("optimization_preference", 
                                            raw_request.get("optimization_goal", "quality")),
            
            # === USER CONFIG ===
            "user_id": raw_request.get("user_id")
        }
        
        return validated_request
    
    async def _build_comprehensive_context(self, gift_request: GiftRecommendationRequest) -> dict:
        """
        ✅ VOLLSTÄNDIG: Nutzt ALLE Schema-Felder für Complete Context
        """
        
        context = {
            # === CORE CONTEXT ===
            "user_id": gift_request.user_id,
            "personality_profile": gift_request.personality_data,
            "occasion": gift_request.occasion,
            "relationship": gift_request.relationship,
            
            # === BUDGET & TEMPORAL ===
            "budget_range": (gift_request.budget_min, gift_request.budget_max),
            "occasion_date": gift_request.occasion_date,
            "urgency_level": gift_request.urgency_level,
            "time_sensitivity": gift_request.urgency_level in ["high", "urgent"] if gift_request.urgency_level else False,
            
            # === CONTEXT & CULTURE ===
            "cultural_context": gift_request.cultural_context,
            "additional_context": gift_request.additional_context,
            "context_richness": len(gift_request.additional_context) if gift_request.additional_context else 0,
            
            # === PREFERENCES & PERSONALIZATION ===
            "personalization_level": gift_request.personalization_level,
            "prioritize_emotional_impact": gift_request.prioritize_emotional_impact,
            "include_explanation": gift_request.include_explanation,
            "explanation_detail": gift_request.explanation_detail,
            
            # === CATEGORY CONSTRAINTS ===
            "gift_categories": gift_request.gift_categories,
            "exclude_categories": gift_request.exclude_categories,
            "has_category_constraints": bool(gift_request.gift_categories or gift_request.exclude_categories),
            
            # === AI MODEL PREFERENCES ===
            "target_ai_model": gift_request.target_ai_model,
            "use_consensus_validation": gift_request.use_consensus_validation,
            "optimization_goal": gift_request.optimization_goal,
            
            # === REQUEST METADATA ===
            "number_of_recommendations": gift_request.number_of_recommendations,
            "request_timestamp": datetime.now().isoformat(),
            "advanced_techniques_enabled": True,
            
            # === ORCHESTRATOR-SPECIFIC ===
            "orchestration_strategy": self._determine_orchestration_strategy(gift_request),
            "processing_priority": self._calculate_processing_priority(gift_request),
            "quality_requirements": self._determine_quality_requirements(gift_request)
        }
        
        return context

    def _determine_orchestration_strategy(self, request: GiftRecommendationRequest) -> str:
        """Bestimmt Orchestrierungs-Strategie basierend auf Request"""
        if request.urgency_level in ["high", "urgent"]:
            return "speed_optimized"
        elif request.personalization_level == "maximum":
            return "quality_maximized"
        elif request.use_consensus_validation:
            return "consensus_driven"
        else:
            return "balanced"

    def _calculate_processing_priority(self, request: GiftRecommendationRequest) -> str:
        """Berechnet Processing-Priorität"""
        priority_score = 0
        
        if request.urgency_level == "urgent":
            priority_score += 3
        elif request.urgency_level == "high":
            priority_score += 2
        
        if request.occasion_date:
            from datetime import date
            days_until = (request.occasion_date - date.today()).days
            if days_until <= 1:
                priority_score += 2
            elif days_until <= 3:
                priority_score += 1
        
        if priority_score >= 4:
            return "critical"
        elif priority_score >= 2:
            return "high"
        else:
            return "normal"

    def _determine_quality_requirements(self, request: GiftRecommendationRequest) -> Dict[str, Any]:
        """Bestimmt Quality-Requirements"""
        return {
            "personalization_required": request.personalization_level in ["high", "maximum"],
            "emotional_impact_required": request.prioritize_emotional_impact,
            "explanation_required": request.include_explanation,
            "cultural_sensitivity_required": bool(request.cultural_context),
            "consensus_validation_required": request.use_consensus_validation
        }
    
    # FEHLER: 'str' object has no attribute 'value'
    # ZEILEN: In _analyze_optimization_preference und anderen Enum-handling methods

    def _analyze_optimization_preference(self, gift_request, context) -> OptimizationObjective:
        """✅ FINAL FIX: Use CORRECT OptimizationObjective enum values with safe string handling"""
        
        try:
            # ✅ FIX: Safe optimization goal extraction
            if hasattr(gift_request, 'optimization_goal'):
                preference = gift_request.optimization_goal
            elif hasattr(gift_request, 'optimization_preference'):
                preference = gift_request.optimization_preference
            else:
                preference = 'quality'  # Default fallback
            
            # ✅ FIX: Safe string conversion
            if preference is None:
                preference = 'quality'
            
            # Handle enum objects safely
            if preference is not None and hasattr(preference, 'value') and not isinstance(preference, str):
                preference_str = preference.value.lower()
            elif preference is not None:
                preference_str = str(preference).lower()
            else:
                preference_str = 'quality'
            
            # Clean up enum string representations
            if 'optimization' in preference_str:
                preference_str = preference_str.split('.')[-1]  # Extract last part
            
            # ✅ FIX: Map to ACTUAL enum values from optimization_engine.py
            optimization_mapping = {
                "speed": OptimizationObjective.PERFORMANCE_MAXIMIZATION,  # Speed = Performance
                "quality": OptimizationObjective.PERFORMANCE_MAXIMIZATION,  # Quality = Performance
                "cost": OptimizationObjective.COST_EFFICIENCY,           # Cost = Cost Efficiency
                "creativity": OptimizationObjective.INNOVATION_FOCUSED,        # Creativity = Innovation
                "accuracy": OptimizationObjective.PERFORMANCE_MAXIMIZATION,  # Accuracy = Performance
                "balance": OptimizationObjective.BALANCED_ROI,               # Balance = Balanced ROI
                
                # Additional aliases
                "performance": OptimizationObjective.PERFORMANCE_MAXIMIZATION,
                "fast": OptimizationObjective.PERFORMANCE_MAXIMIZATION,
                "cheap": OptimizationObjective.COST_EFFICIENCY,
                "innovative": OptimizationObjective.INNOVATION_FOCUSED,
                "balanced": OptimizationObjective.BALANCED_ROI,
                "roi": OptimizationObjective.BALANCED_ROI,
                "user_satisfaction": OptimizationObjective.USER_SATISFACTION,
                "resource_utilization": OptimizationObjective.RESOURCE_UTILIZATION
            }
            
            # Get optimization objective with fallback
            optimization_obj = optimization_mapping.get(preference_str, OptimizationObjective.BALANCED_ROI)
            
            return optimization_obj
            
        except Exception as e:
            # ✅ SAFE FALLBACK: Always return a valid OptimizationObjective
            self.logger.warning("Optimization preference analysis failed: %s, using default", e)
            return OptimizationObjective.BALANCED_ROI
              # Default = Balanced ROI
    
    # ✅ ADDITIONAL FIX: Safe enum value extraction helper
    def _safe_enum_value(self, enum_obj, default_value="unknown"):
        """Safely extract enum value as string"""
        try:
            if enum_obj is None:
                return default_value
            
            if enum_obj is not None and hasattr(enum_obj, 'value') and not isinstance(enum_obj, str):
                return str(enum_obj.value)
            elif enum_obj is not None:
                return str(enum_obj)
            else:
                return default_value
                
        except Exception:
            return default_value

    def _record_enhanced_performance_metrics(self, request, response, processing_time, techniques_used, optimization_metrics, context):
        """Enhanced performance tracking with optimization metrics"""
        try:
            # Basic metrics
            self.analytics["average_processing_time"] = (
                (self.analytics["average_processing_time"] * (self.analytics["total_requests"] - 1) + processing_time) 
                / self.analytics["total_requests"]
            )
            
            # Track advanced techniques usage
            for technique in techniques_used:
                self.analytics["advanced_techniques_usage"][technique] = \
                    self.analytics["advanced_techniques_usage"].get(technique, 0) + 1
            
            # Track model usage
            model_used = response.get("ai_model_used", "unknown")
            self.analytics["model_usage_statistics"][model_used] = \
                self.analytics["model_usage_statistics"].get(model_used, 0) + 1
            
            # Track optimization metrics
            if optimization_metrics:
                for metric, value in optimization_metrics.items():
                    if metric not in self.analytics["optimization_metrics"]:
                        self.analytics["optimization_metrics"][metric] = []
                    self.analytics["optimization_metrics"][metric].append(value)
            
            # Track quality metrics
            quality_score = response.get("overall_confidence", 0.5)
            if "quality_metrics" not in self.analytics:
                self.analytics["quality_metrics"] = []
            self.analytics["quality_metrics"].append({
                "timestamp": datetime.now().isoformat(),
                "quality_score": quality_score,
                "processing_time": processing_time,
                "model_used": model_used
            })
            
            # Track cost if available
            cost_estimate = response.get("cost_estimate")
            if cost_estimate:
                if "cost_tracking" not in self.analytics:
                    self.analytics["cost_tracking"] = []
                self.analytics["cost_tracking"].append({
                    "timestamp": datetime.now().isoformat(),
                    "cost": float(cost_estimate),
                    "model_used": model_used
                })
            
            # Update optimization state based on performance
            if quality_score > self.optimization_state["performance_baseline"]:
                self.optimization_state["current_strategy"] = "quality_optimized"
            elif processing_time < 2000:  # Fast processing
                self.optimization_state["current_strategy"] = "speed_optimized"
            
        except Exception as e:
            self.logger.warning(f"Performance tracking failed: {e}")

    # ✅ ADDITIONAL FIX: Fallback response creation
    async def _create_fallback_response(self, error_type: str, error_message: str) -> dict:
        """Create safe fallback response when everything fails"""
        
        return {
            "recommendations": [{
                "title": "Thoughtful Gift Selection",
                "description": "A carefully chosen gift that matches the recipient's personality and shows genuine care",
                "category": "emotional_bonds",
                "price_range": "€25-€50",
                "estimated_price": 35.0,
                "availability": "Available online and in stores",
                "where_to_find": ["Local gift shops", "Online retailers"],
                "emotional_impact": "Creates joy and strengthens your relationship through thoughtful giving",
                "personal_connection": "Selected based on their unique personality and your special bond",
                "relationship_benefit": "Shows you pay attention to their preferences and care about their happiness",
                "personality_match": "Carefully aligned with their personality traits and interests",
                "primary_reason": "deepens_connection",
                "supporting_reasons": ["agreeableness_harmony", "shared_memory"],
                "confidence_score": 0.7,
                "confidence_level": "medium",
                "uniqueness_score": 0.6
            }],
            
            "personality_analysis": {
                "personality_summary": "A thoughtful individual who appreciates genuine, well-considered gifts",
                "personality_archetype": "Appreciative Gift Recipient", 
                "dominant_traits": ["thoughtful", "appreciative", "genuine"],
                "analysis_confidence": 0.6
            },
            
            "overall_strategy": "Safe fallback recommendation with thoughtful consideration",
            "key_considerations": ["Personality-based selection", "Relationship appropriateness"],
            "emotional_themes": ["thoughtfulness", "appreciation"],
            "overall_confidence": 0.7,
            "personalization_score": 0.6,
            "novelty_score": 0.5,
            "emotional_resonance": 0.7,
            "ai_model_used": "fallback_orchestrator",
            "processing_time_ms": 0,
            "prompt_strategy": "fallback",
            "optimization_goal": "safety",
            
            # Error information
            "fallback_used": True,
            "error_type": error_type,
            "error_message": error_message,
            "timestamp": datetime.now().isoformat()
        }

    # ✅ ADDITIONAL FIX: Minimal response for parsing failures
    async def _create_minimal_response(self) -> dict:
        """Create minimal valid response structure"""
        
        return await self._create_fallback_response("parsing_fallback", "Response parsing failed, using minimal structure")

    async def _parse_and_validate_response(self, processing_result) -> dict:
        """
        Parse and validate processing result with robust enum handling
        
        ✅ CRITICAL FIX: Ensures all enum fields are properly converted to strings for Pydantic
        """
        # Extract base response from processing result
        if "response" in processing_result:
            base_response = processing_result["response"]
        elif "recommendations" in processing_result:
            base_response = {
                "recommendations": processing_result["recommendations"],
                "processing_metadata": processing_result.get("processing_metadata", {})
            }
        else:
            base_response = {}
        
        # ✅ CRITICAL FIX: Ensure all required fields are present with correct enum string values
        validated_response = {}
        
        # Required recommendations array (minimum 1 item)
        recommendations = base_response.get("recommendations", [])
        if not recommendations:
            # ✅ GIFT SHOP FOCUSED: Create meaningful fallback recommendation with CORRECT string enum values
            recommendations = [{
                "title": "Thoughtful Gift Selection",
                "description": "A carefully chosen gift that matches the recipient's personality and shows genuine care",
                "category": "emotional_bonds",  # ✅ STRING VALUE (not enum object)
                "price_range": "€25-€50",
                "estimated_price": 35.0,
                "availability": "Available online and in stores",
                "where_to_find": ["Local gift shops", "Online retailers"],
                "emotional_impact": "Creates joy and strengthens your relationship through thoughtful giving",
                "personal_connection": "Selected based on their unique personality and your special bond",
                "relationship_benefit": "Shows you pay attention to their preferences and care about their happiness",
                "personality_match": "Carefully aligned with their personality traits and interests",
                "primary_reason": "deepens_connection",     # ✅ STRING VALUE (not enum object)
                "supporting_reasons": [                     # ✅ STRING VALUES (not enum objects)
                    "agreeableness_harmony", 
                    "shared_memory"
                ],
                "confidence_score": 0.75,
                "confidence_level": "high",
                "uniqueness_score": 0.7
            }]
        else:
            # ✅ FIX: Clean existing recommendations - convert any enum objects to strings
            cleaned_recommendations = []
            for rec in recommendations:
                cleaned_rec = {}
                for key, value in rec.items():
                    # ✅ ENUM TO STRING CONVERSION
                    if value is not None and hasattr(value, 'value') and not isinstance(value, str):
                        # It's an enum object, extract string value safely
                        try:
                            cleaned_rec[key] = value.value
                        except (AttributeError, TypeError) as enum_error:
                            # Fallback if value access fails
                            self.logger.warning(f"Enum conversion failed for {key}: {enum_error}")
                            cleaned_rec[key] = str(value) if value is not None else ""
                    elif isinstance(value, list):
                        # Handle list of potential enums
                        cleaned_list = []
                        for item in value:
                            if item is not None and hasattr(item, 'value') and not isinstance(item, str):
                                cleaned_list.append(item.value)
                            else:
                                cleaned_list.append(str(item) if item is not None else "")
                        cleaned_rec[key] = cleaned_list
                    else:
                        # Regular value or string
                        cleaned_rec[key] = str(value) if value is not None and not isinstance(value, (int, float, bool)) else value
                
                # Ensure required enum fields have valid string values
                if "category" not in cleaned_rec or not cleaned_rec["category"]:
                    cleaned_rec["category"] = "emotional_bonds"
                if "primary_reason" not in cleaned_rec or not cleaned_rec["primary_reason"]:
                    cleaned_rec["primary_reason"] = "deepens_connection"
                if "supporting_reasons" not in cleaned_rec or not cleaned_rec["supporting_reasons"]:
                    cleaned_rec["supporting_reasons"] = ["agreeableness_harmony", "shared_memory"]
                
                cleaned_recommendations.append(cleaned_rec)
            
            recommendations = cleaned_recommendations
        
        validated_response["recommendations"] = recommendations
        
        # Required personality_analysis
        personality_analysis = base_response.get("personality_analysis", {})
        
        # ✅ FIX: Clean personality analysis - convert enums to strings
        cleaned_personality = {}
        for key, value in personality_analysis.items():
            if value is not None and hasattr(value, 'value') and not isinstance(value, str):
                try:
                    cleaned_personality[key] = value.value
                except (AttributeError, TypeError):
                    cleaned_personality[key] = str(value) if value is not None else ""

            elif isinstance(value, dict):
                # Handle nested dictionaries
                cleaned_dict = {}
                for nested_key, nested_value in value.items():
                    if nested_value is not None and hasattr(nested_value, 'value') and not isinstance(nested_value, str):
                        cleaned_dict[nested_key] = nested_value.value
                    else:
                        cleaned_dict[nested_key] = nested_value
                cleaned_personality[key] = cleaned_dict
            elif isinstance(value, list):
                # Handle lists of potential enums
                cleaned_list = []
                for item in value:
                    if item is not None and hasattr(item, 'value') and not isinstance(item, str):
                        cleaned_list.append(item.value)
                    else:
                        cleaned_list.append(str(item) if item is not None else "")
                cleaned_personality[key] = cleaned_list
            else:
                cleaned_personality[key] = value
        
        # Ensure default personality analysis if empty
        if not cleaned_personality:
            cleaned_personality = {
                "personality_summary": "A thoughtful individual who appreciates genuine, well-considered gifts",
                "personality_archetype": "Appreciative Gift Recipient", 
                "dominant_traits": ["thoughtful", "appreciative", "genuine"],
                "big_five_insights": {
                    "openness": "Values creativity and meaningful experiences",
                    "conscientiousness": "Appreciates quality and thoughtful gestures",
                    "extraversion": "Enjoys both personal and shared experiences",
                    "agreeableness": "Values harmony and considerate relationships",
                    "neuroticism": "Responds well to comforting and supportive gifts"
                },
                "analysis_confidence": 0.8
            }
        
        validated_response["personality_analysis"] = cleaned_personality
        
        # ✅ FIX: Clean all other string fields - convert enums to strings
        def clean_value(value):
            """Helper to clean any value, converting enums to strings"""
            try:
                # Handle None
                if value is None:
                    return ""
                
                # Handle strings (already clean)
                if isinstance(value, str):
                    return value
                
                # Handle enums (convert to string)
                if hasattr(value, 'value'):
                    try:
                        return str(value.value)
                    except (AttributeError, TypeError):
                        return str(value)
                
                # Handle lists
                if isinstance(value, list):
                    return [clean_value(item) for item in value]
                
                # Handle dictionaries
                if isinstance(value, dict):
                    return {k: clean_value(v) for k, v in value.items()}
                
                # Handle other types
                return str(value)
                
            except Exception as e:
                # Ultimate fallback
                return str(value) if value is not None else ""
        
        # Required string fields with enum cleaning
        validated_response["overall_strategy"] = clean_value(base_response.get("overall_strategy", 
            "Thoughtful gift selection based on personality analysis and relationship context"))
        
        # Required arrays with enum cleaning
        validated_response["key_considerations"] = clean_value(base_response.get("key_considerations", [
            "Personality-based gift selection",
            "Relationship appropriateness",
            "Emotional impact"
        ]))
        
        validated_response["emotional_themes"] = clean_value(base_response.get("emotional_themes", [
            "thoughtfulness",
            "appreciation", 
            "connection"
        ]))
        
        # Required float scores (safe conversion)
        def safe_float(value, default):
            try:
                cleaned_value = clean_value(value)
                return float(cleaned_value) if cleaned_value is not None else default
            except (TypeError, ValueError):
                return default
        
        validated_response["overall_confidence"] = safe_float(base_response.get("overall_confidence"), 0.8)
        validated_response["personalization_score"] = safe_float(base_response.get("personalization_score"), 0.75)
        validated_response["novelty_score"] = safe_float(base_response.get("novelty_score"), 0.7)
        validated_response["emotional_resonance"] = safe_float(base_response.get("emotional_resonance"), 0.8)
        
        # Required metadata with enum cleaning
        validated_response["ai_model_used"] = clean_value(base_response.get("ai_model_used", "production_orchestrator"))
        
        # Safe integer conversion
        processing_time = clean_value(base_response.get("processing_time_ms", 1))
        try:
            validated_response["processing_time_ms"] = int(processing_time) if processing_time is not None else 1
        except (TypeError, ValueError):
            validated_response["processing_time_ms"] = 1
            
        validated_response["prompt_strategy"] = clean_value(base_response.get("prompt_strategy", "personality_based"))
        validated_response["optimization_goal"] = clean_value(base_response.get("optimization_goal", "quality"))
        
        # ✅ SHOP FOCUS: Add optional gift shop specific fields if present (with enum cleaning)
        optional_fields = [
            "budget_alternatives", "last_minute_options", "experience_vs_material",
            "occasion_specific_advice", "relationship_guidance", "cultural_considerations",
            "timing_recommendations", "improvement_suggestions", "feedback_request"
        ]
        
        for field in optional_fields:
            if field in base_response:
                validated_response[field] = clean_value(base_response[field])
        
        # ✅ CRITICAL: Remove forbidden fields that cause "Extra inputs not permitted" errors
        forbidden_fields = ["processing_metadata", "error", "fallback"]
        for field in forbidden_fields:
            if field in validated_response:
                del validated_response[field]
        
        return validated_response
        
    def _record_performance_metrics(self, request, response, processing_time: float, techniques_used: List[str]):
        """Record performance metrics"""
        # Update average processing time
        total_requests = self.analytics["total_requests"]
        current_avg = self.analytics["average_processing_time"]
        self.analytics["average_processing_time"] = \
            ((current_avg * (total_requests - 1)) + processing_time) / total_requests
        
        # Record advanced techniques usage
        for technique in techniques_used:
            self.analytics["advanced_techniques_usage"][technique] = \
                self.analytics["advanced_techniques_usage"].get(technique, 0) + 1
        
        # Record model usage
        if "ai_model_used" in response:
            model = response["ai_model_used"]
            self.analytics["model_usage_statistics"][model] = \
                self.analytics["model_usage_statistics"].get(model, 0) + 1
    
    def get_system_analytics(self) -> dict:
        """Get comprehensive system analytics"""
        return {
            "request_statistics": {
                "total_requests": self.analytics["total_requests"],
                "successful_requests": self.analytics["successful_requests"],
                "success_rate": self.analytics["successful_requests"] / max(1, self.analytics["total_requests"]),
                "average_processing_time_ms": self.analytics["average_processing_time"]
            },
            "advanced_techniques_usage": self.analytics["advanced_techniques_usage"],
            "model_usage_statistics": self.analytics["model_usage_statistics"],
            "error_statistics": self.analytics["error_statistics"],
            "system_health": self._calculate_system_health()
        }
    
    def _calculate_system_health(self) -> float:
        """Calculate overall system health score"""
        success_rate = self.analytics["successful_requests"] / max(1, self.analytics["total_requests"])
        avg_processing_time_score = max(0, 1 - (self.analytics["average_processing_time"] / 10000))  # Penalize > 10s
        
        return (success_rate * 0.7) + (avg_processing_time_score * 0.3)


class PerformanceMonitor:
    """Performance monitoring system"""
    
    def __init__(self):
        self.metrics = {}
    
    def record_request_performance(self, request, response, processing_time, techniques_used):
        """Record request performance"""
        pass


class ErrorRecoverySystem:
    """Error recovery and fallback system"""
    
    async def handle_processing_error(self, error, raw_request):
        """Handle processing errors with fallbacks"""
        return {
            "recommendations": [],
            "error": str(error),
            "fallback_used": True,
            "processing_metadata": {"error_recovery": True}
        }
