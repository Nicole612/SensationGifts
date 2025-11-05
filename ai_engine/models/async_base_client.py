"""
Async Base AI Client - Async Version der BaseAIClient für High-Performance

WARUM ASYNC?
- Mehrere AI-Calls gleichzeitig (Concurrent Processing)
- Keine Wartezeiten während API-Calls
- 10x mehr Requests in der gleichen Zeit
- Bessere User Experience (keine Blockierung)

KONZEPTE:
- async/await: Python's Art zu sagen "warte nicht, mach andere Sachen"
- Semaphore: Begrenzt wie viele Requests gleichzeitig laufen
- AsyncRateLimiter: Verhindert zu viele Requests pro Minute

INTEGRATION:
Erbt von deiner bestehenden BaseAIClient, aber mit async superpowers
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from dataclasses import dataclass

# Import von der bestehenden base_client.py
from .base_client import (
    AIRequest, AIResponse, AIModelType, ModelCapability, 
    ResponseFormat, GiftRecommendationSchema, ModelMetrics
)


class AsyncRateLimiter:
    """
    Async Rate Limiter - verhindert zu viele API-Calls
    
    KONZEPT:
    - Wie ein Türsteher im Club: "Nur 60 Leute pro Minute rein!"
    - Async = Türsteher blockiert nicht die ganze Schlange
    - Wartet intelligent, ohne andere Requests zu blockieren
    """
    
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.requests: List[float] = []  # Timestamps der letzten Requests
        self.lock = asyncio.Lock()  # Verhindert Race Conditions
    
    async def acquire(self):
        """
        Fragt: "Darf ich einen Request machen?"
        Wartet wenn nötig, blockiert aber andere Requests nicht
        """
        async with self.lock:  # Nur einer gleichzeitig hier rein
            now = time.time()
            
            # Entferne alte Requests (älter als 1 Minute)
            self.requests = [req_time for req_time in self.requests 
                           if now - req_time < 60]
            
            # Sind wir am Limit?
            if len(self.requests) >= self.requests_per_minute:
                oldest_request = min(self.requests)
                wait_time = 60 - (now - oldest_request)
                
                if wait_time > 0:
                    print(f"⏳ Rate limit reached, waiting {wait_time:.1f} seconds")
                    await asyncio.sleep(wait_time)
            
            # Request registrieren
            self.requests.append(now)
            print(f"✅ Request approved, {len(self.requests)}/{self.requests_per_minute} used")


class AsyncBaseAIClient(ABC):
    """
    Async Version der BaseAIClient
    
    HAUPTUNTERSCHIEDE zu BaseAIClient:
    - Alle Methoden sind async (mit async def)
    - Verwendet asyncio für Concurrency
    - Kann mehrere Requests gleichzeitig verarbeiten
    - Nutzt aiohttp für HTTP-Calls (nicht requests)
    
    VERWENDUNG:
    # Statt:
    response = client.generate_text("Hello")
    
    # Jetzt:
    response = await client.generate_text_async("Hello")
    """
    
    def __init__(self, 
                 api_key: str,
                 model_type: AIModelType,
                 rate_limit_per_minute: int = 60,
                 max_concurrent_requests: int = 10):
        
        self.api_key = api_key
        self.model_type = model_type
        self.rate_limit_per_minute = rate_limit_per_minute
        self.max_concurrent_requests = max_concurrent_requests
        
        # Metrics (gleich wie in BaseAIClient)
        self.metrics = ModelMetrics()
        
        # ASYNC-SPEZIFISCHE COMPONENTS:
        
        # Semaphore = "Nur 10 Requests gleichzeitig"
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        # Async Rate Limiter
        self.rate_limiter = AsyncRateLimiter(rate_limit_per_minute)
        
        # HTTP Session für alle Requests (wiederverwendbar)
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Model-spezifische Capabilities (von Subclasses definiert)
        self.capabilities = self._get_model_capabilities()
        self.pricing = self._get_model_pricing()
    
    # === CONTEXT MANAGER (für saubere Resource-Verwaltung) ===
    
    async def __aenter__(self):
        """
        Async Context Manager - wird beim 'async with' aufgerufen
        
        BEISPIEL:
        async with client:
            response = await client.generate_text_async("Hello")
        """
        print(f"🔗 Initializing async client for {self.model_type.value}")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Cleanup async resources
        """
        print(f"🔒 Closing async client for {self.model_type.value}")
    
    # === ABSTRACT METHODS (müssen von Subclasses implementiert werden) ===
    
    @abstractmethod
    async def _make_async_api_call(self, request: AIRequest) -> AIResponse:
        """
        Async API Call - muss von OpenAI/Groq Clients implementiert werden
        
        WICHTIG: Diese Methode ist async!
        Statt requests.post() verwendet sie aiohttp oder die async Client Library
        """
        pass
    
    @abstractmethod
    def _get_model_capabilities(self) -> List[ModelCapability]:
        """Gleich wie in BaseAIClient"""
        pass
    
    @abstractmethod
    def _get_model_pricing(self) -> Dict[str, float]:
        """Gleich wie in BaseAIClient"""
        pass
    
    # === PUBLIC ASYNC API ===
    
    async def generate_text_async(self, 
                                 prompt: str,
                                 system_prompt: Optional[str] = None,
                                 temperature: float = 0.7,
                                 max_tokens: Optional[int] = None) -> AIResponse:
        """
        Async Text Generation
        
        VERWENDUNG:
        response = await client.generate_text_async("Was ist ein gutes Geschenk?")
        """
        request = AIRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            response_format=ResponseFormat.TEXT,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return await self._execute_async_request(request)
    
    async def generate_json_async(self,
                                 prompt: str,
                                 json_schema: Dict,
                                 system_prompt: Optional[str] = None,
                                 temperature: float = 0.3) -> AIResponse:
        """
        Async JSON Generation
        
        VERWENDUNG:
        schema = {"type": "object", "properties": {"gift": {"type": "string"}}}
        response = await client.generate_json_async("Empfehle ein Geschenk", schema)
        """
        request = AIRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            response_format=ResponseFormat.JSON,
            json_schema=json_schema,
            temperature=temperature
        )
        
        return await self._execute_async_request(request)
    
    async def recommend_gift_async(self, 
                                  personality_profile: Dict,
                                  occasion: str,
                                  budget_range: str,
                                  relationship: str) -> GiftRecommendationSchema:
        """
        Async Gift Recommendation
        
        VERWENDUNG:
        profile = {"personality_scores": {"openness": 0.8}}
        gift = await client.recommend_gift_async(profile, "birthday", "50-100", "friend")
        """
        prompt = self._build_gift_recommendation_prompt(
            personality_profile, occasion, budget_range, relationship
        )
        
        schema = GiftRecommendationSchema.model_json_schema()
        
        response = await self.generate_json_async(
            prompt=prompt,
            json_schema=schema,
            system_prompt=self._get_gift_expert_system_prompt(),
            temperature=0.4
        )
        
        if not response.success:
            raise Exception(f"Async gift recommendation failed: {response.error}")
        
        try:
            return GiftRecommendationSchema.model_validate(response.parsed_json)
        except Exception as e:
            raise Exception(f"Invalid async response format: {e}")
    
    # === BATCH PROCESSING (Das ist der große Vorteil von Async!) ===
    
    async def generate_batch_async(self, 
                                  requests: List[AIRequest],
                                  max_concurrent: int = 5) -> List[AIResponse]:
        """
        Verarbeitet mehrere Requests GLEICHZEITIG
        
        KONZEPT:
        - Statt nacheinander alle Requests abarbeiten
        - Starte alle gleichzeitig und warte auf alle Ergebnisse
        - Nutzt asyncio.gather() für Parallelverarbeitung
        
        BEISPIEL:
        requests = [
            AIRequest(prompt="Geschenk für Mama"),
            AIRequest(prompt="Geschenk für Papa"),
            AIRequest(prompt="Geschenk für Schwester")
        ]
        responses = await client.generate_batch_async(requests)
        # Alle 3 Empfehlungen in der Zeit von 1!
        """
        
        # Semaphore für Concurrency Control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_request(request: AIRequest) -> AIResponse:
            """Wrapper für einzelnen Request mit Semaphore"""
            async with semaphore:
                return await self._execute_async_request(request)
        
        print(f"🚀 Starting batch processing of {len(requests)} requests...")
        
        # Alle Requests gleichzeitig starten
        tasks = [process_single_request(req) for req in requests]
        
        # Warten bis alle fertig sind
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        print(f"✅ Batch processing completed: {len(responses)} responses")
        
        return responses
    
    async def batch_gift_recommendations_async(self,
                                             recommendations_data: List[Dict]) -> List[GiftRecommendationSchema]:
        """
        Erstellt mehrere Geschenkempfehlungen gleichzeitig
        
        VERWENDUNG:
        data = [
            {"personality_profile": profile1, "occasion": "birthday", ...},
            {"personality_profile": profile2, "occasion": "christmas", ...},
            {"personality_profile": profile3, "occasion": "anniversary", ...}
        ]
        gifts = await client.batch_gift_recommendations_async(data)
        """
        
        # Erstelle AIRequests für alle Empfehlungen
        requests = []
        for data in recommendations_data:
            prompt = self._build_gift_recommendation_prompt(
                data["personality_profile"],
                data["occasion"],
                data["budget_range"],
                data["relationship"]
            )
            
            request = AIRequest(
                prompt=prompt,
                system_prompt=self._get_gift_expert_system_prompt(),
                response_format=ResponseFormat.JSON,
                json_schema=GiftRecommendationSchema.model_json_schema(),
                temperature=0.4
            )
            requests.append(request)
        
        # Verarbeite alle gleichzeitig
        responses = await self.generate_batch_async(requests)
        
        # Konvertiere zu GiftRecommendationSchema
        recommendations = []
        for response in responses:
            if isinstance(response, AIResponse) and response.success:
                try:
                    gift = GiftRecommendationSchema.model_validate(response.parsed_json)
                    recommendations.append(gift)
                except Exception as e:
                    print(f"❌ Failed to parse gift recommendation: {e}")
                    # Fallback Empfehlung
                    fallback_gift = GiftRecommendationSchema(
                        gift_name="Gift Card",
                        reasoning="Fallback when parsing failed",
                        match_score=0.5,
                        emotional_appeal="choice",
                        personalization_ideas=["Custom amount"],
                        price_range="flexible",
                        alternative_gifts=["Cash"],
                        confidence=0.3
                    )
                    recommendations.append(fallback_gift)
        
        return recommendations
    
    # === INTERNAL ASYNC METHODS ===
    
    async def _execute_async_request(self, request: AIRequest) -> AIResponse:
        """
        Führt einen async Request aus mit Rate Limiting und Concurrency Control
        
        ABLAUF:
        1. Warte auf Semaphore (max_concurrent_requests)
        2. Warte auf Rate Limiter
        3. Mache API Call
        4. Update Metrics
        5. Gib Semaphore frei
        """
        
        async with self.semaphore:  # Nur N Requests gleichzeitig
            # Rate Limiting
            await self.rate_limiter.acquire()
            
            # Metrics Update
            self.metrics.total_requests += 1
            start_time = time.time()
            
            try:
                # Der tatsächliche API Call
                response = await self._make_async_api_call(request)
                
                # Success Metrics
                response_time = time.time() - start_time
                response.response_time = response_time
                
                if response.success:
                    self.metrics.successful_requests += 1
                    self.metrics.total_tokens += response.tokens_used
                    self.metrics.total_cost += response.cost
                    self._update_avg_response_time(response_time)
                else:
                    self.metrics.failed_requests += 1
                
                self.metrics.last_request_time = datetime.now()
                
                return response
                
            except Exception as e:
                # Failure Metrics
                self.metrics.failed_requests += 1
                
                return AIResponse(
                    content="",
                    model_type=self.model_type,
                    tokens_used=0,
                    cost=0.0,
                    response_time=time.time() - start_time,
                    timestamp=datetime.now(),
                    success=False,
                    error=f"Async API Error: {e}"
                )
    
    def _update_avg_response_time(self, response_time: float):
        """Updates average response time (gleich wie in BaseAIClient)"""
        if self.metrics.avg_response_time == 0:
            self.metrics.avg_response_time = response_time
        else:
            # Exponential moving average
            self.metrics.avg_response_time = (
                0.9 * self.metrics.avg_response_time + 0.1 * response_time
            )
    
    # === HELPER METHODS (von BaseAIClient übernommen) ===
    
    def _build_gift_recommendation_prompt(self,
                                        personality_profile: Dict,
                                        occasion: str,
                                        budget_range: str,
                                        relationship: str) -> str:
        """Gleich wie in BaseAIClient"""
        return f"""
        Als Geschenk-Experte empfiehl ein perfektes Geschenk für:
        
        Personality: {personality_profile}
        Anlass: {occasion}
        Budget: {budget_range}€
        Beziehung: {relationship}
        
        Berücksichtige die Persönlichkeit und gib eine durchdachte Empfehlung.
        """
    
    def _get_gift_expert_system_prompt(self) -> str:
        """Gleich wie in BaseAIClient"""
        return """
        Du bist ein Experte für personalisierte Geschenke mit 20 Jahren Erfahrung.
        Du verstehst Persönlichkeitspsychologie und emotionale Trigger.
        Deine Empfehlungen sind immer durchdacht, kreativ und passend.
        Antworte ausschließlich im angegebenen JSON-Format.
        """
    
    # === UTILITY METHODS ===
    
    def get_metrics(self) -> ModelMetrics:
        """Gibt aktuelle Metriken zurück (gleich wie BaseAIClient)"""
        return self.metrics
    
    def reset_metrics(self):
        """Reset Metriken für Testing"""
        self.metrics = ModelMetrics()
    
    async def test_async_connection(self) -> bool:
        """
        Testet async Verbindung
        
        VERWENDUNG:
        is_working = await client.test_async_connection()
        """
        try:
            response = await self.generate_text_async(
                prompt="Test async connection",
                max_tokens=10
            )
            return response.success
        except Exception as e:
            print(f"❌ Async connection test failed: {e}")
            return False
    
    def __repr__(self):
        return f"<Async{self.__class__.__name__} {self.model_type.value} (Success: {self.metrics.success_rate:.2%})>"


# === ASYNC UTILITY FUNCTIONS ===

async def compare_sync_vs_async_performance(sync_client, async_client, test_prompts: List[str]) -> Dict[str, Any]:
    """
    Vergleicht Performance zwischen Sync und Async Client
    
    VERWENDUNG:
    sync_client = OpenAIClient(api_key)
    async_client = AsyncOpenAIClient(api_key)
    
    results = await compare_sync_vs_async_performance(
        sync_client, async_client, 
        ["Geschenk 1", "Geschenk 2", "Geschenk 3"]
    )
    """
    
    results = {
        "sync_time": 0.0,
        "async_time": 0.0,
        "speedup_factor": 0.0,
        "sync_responses": [],
        "async_responses": []
    }
    
    # Test Sync Client (sequenziell)
    print("🐌 Testing sync client (sequential)...")
    sync_start = time.time()
    
    for prompt in test_prompts:
        response = sync_client.generate_text(prompt, max_tokens=50)
        results["sync_responses"].append(response.success)
    
    results["sync_time"] = time.time() - sync_start
    
    # Test Async Client (parallel)
    print("🚀 Testing async client (parallel)...")
    async_start = time.time()
    
    requests = [
        AIRequest(prompt=prompt, max_tokens=50) 
        for prompt in test_prompts
    ]
    
    async_responses = await async_client.generate_batch_async(requests)
    results["async_responses"] = [r.success for r in async_responses if isinstance(r, AIResponse)]
    
    results["async_time"] = time.time() - async_start
    
    # Calculate speedup
    if results["async_time"] > 0:
        results["speedup_factor"] = results["sync_time"] / results["async_time"]
    
    print(f"📊 Results: Sync={results['sync_time']:.2f}s, Async={results['async_time']:.2f}s, Speedup={results['speedup_factor']:.2f}x")
    
    return results