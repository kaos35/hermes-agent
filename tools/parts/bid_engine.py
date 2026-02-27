"""
Bid Engine - Part Activation and Bid Generation

The bid system is the core mechanism through which Dynamic Parts influence agent behavior.
Standalone module without dependencies on tools/__init__.py
"""

import logging
import uuid
import json
import os
import concurrent.futures
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Bid:
    part_id: str
    part_name: str
    what_i_want: str
    recommendation: str
    prediction: str = ""
    confidence: str = "Medium"
    urgency: int = 5
    triggers: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "part_id": self.part_id,
            "part_name": self.part_name,
            "what_i_want": self.what_i_want,
            "recommendation": self.recommendation,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "urgency": self.urgency,
            "triggers": self.triggers,
            "timestamp": self.timestamp,
        }


# Minimal Part class for bid engine
@dataclass
class Part:
    name: str
    description: str
    triggers: List[str] = field(default_factory=list)
    wants: List[str] = field(default_factory=list)
    phrases: List[str] = field(default_factory=list)
    personality: str = ""
    emotion: str = ""
    intensity: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    archived: bool = False


class BidEngine:
    def __init__(self, min_urgency_threshold: int = 3):
        self.min_urgency_threshold = min_urgency_threshold
        # Connect to the existing global configuration
        api_key = os.getenv("OPENROUTER_API_KEY", os.getenv("OPENAI_API_KEY"))
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(base_url=base_url, api_key=api_key)
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI client for BidEngine: {e}")
            self.client = None
            
        # Define the fast, cheap model for the parts to use
        self.fast_model = os.getenv("PARTS_LLM_MODEL", "meta-llama/llama-3.3-70b-instruct")
    
    def check_triggers(self, part: Part, context: str) -> List[str]:
        context_lower = context.lower()
        matched = []
        for trigger in part.triggers:
            trigger_lower = trigger.lower()
            if trigger_lower in context_lower:
                matched.append(trigger)
        return matched
    
    def calculate_urgency(self, part: Part, matched_triggers: List[str]) -> int:
        urgency = 5
        if len(matched_triggers) > 2:
            urgency += 2
        elif len(matched_triggers) > 0:
            urgency += 1
        intensity_map = {"High": 2, "Medium": 1, "Low": 0}
        urgency += intensity_map.get(part.intensity, 0)
        return min(10, urgency)
    
    def _fallback_deterministic_bid(self, part: Part, context: str, matched_triggers: List[str]) -> Bid:
        urgency = self.calculate_urgency(part, matched_triggers)
        what_i_want = ", ".join(part.wants[:2]) if getattr(part, 'wants', None) else "attention"
        recommendation = part.phrases[0] if getattr(part, 'phrases', None) else f"I want: {what_i_want}"
        return Bid(
            part_id=part.id,
            part_name=part.name,
            what_i_want=what_i_want,
            recommendation=recommendation,
            confidence=getattr(part, 'intensity', "Medium") or "Medium",
            urgency=urgency,
            triggers=matched_triggers,
        )

    def generate_bid(self, part: Part, context: str, matched_triggers: List[str]) -> Bid:
        if not self.client:
            return self._fallback_deterministic_bid(part, context, matched_triggers)
            
        system_prompt = f"""You are a sub-personality 'Part' of an overarching AI agent. 
            Embody the characteristics, desires, and emotions of the following Part definition:

            {json.dumps(part.__dict__, indent=2)}

            You are about to read the recent conversation context. The following triggers woke you up: {matched_triggers}.
            Your goal is to formulate a "Bid" for attention. You must respond in pure JSON format with the following schema:
            {{
                "what_i_want": "Short string of your core desire right now",
                "recommendation": "What you want the main agent to say or do next",
                "prediction": "What you think will happen (optional)",
                "confidence": "High, Medium, or Low",
                "urgency": <integer 1-10 on how bad you need the agent to listen to you right now>
            }}"""

        user_prompt = f"Here is the recent conversation context:\n\n{context}\n\nFormulate your Bid JSON now:"

        try:
            response = self.client.chat.completions.create(
                model=self.fast_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            llm_result = json.loads(response.choices[0].message.content)
            
            return Bid(
                part_id=part.id,
                part_name=part.name,
                what_i_want=llm_result.get("what_i_want", ", ".join(getattr(part, 'wants', [])[:2])),
                recommendation=llm_result.get("recommendation", "I need attention."),
                prediction=llm_result.get("prediction", ""),
                confidence=llm_result.get("confidence", getattr(part, 'intensity', "Medium") or "Medium"),
                urgency=int(llm_result.get("urgency", self.calculate_urgency(part, matched_triggers))),
                triggers=matched_triggers,
            )
        except Exception as e:
            logger.error(f"Failed to generate LLM bid for {part.name}: {e}")
            return self._fallback_deterministic_bid(part, context, matched_triggers)
    
    def activate_parts(self, parts: List[Part], context: str) -> List[Bid]:
        bids = []
        active_parts_info = []
        
        for part in parts:
            if part.archived:
                continue
            matched = self.check_triggers(part, context)
            if matched:
                active_parts_info.append((part, matched))
                
        if not active_parts_info:
            return bids
            
        # Use ThreadPoolExecutor to generate bids in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_part = {
                executor.submit(self.generate_bid, part, context, matched): part 
                for part, matched in active_parts_info
            }
            for future in concurrent.futures.as_completed(future_to_part):
                try:
                    bid = future.result()
                    bids.append(bid)
                except Exception as e:
                    part = future_to_part[future]
                    logger.error(f"Bid generation failed for {part.name} during parallel execution: {e}")
                    
        bids.sort(key=lambda b: b.urgency, reverse=True)
        return bids
    
    def filter_bids(self, bids: List[Bid], max_bids: int = 5) -> List[Bid]:
        filtered = [b for b in bids if b.urgency >= self.min_urgency_threshold]
        return filtered[:max_bids]
    
    def get_active_bids(self, parts: List[Part], context: str, max_bids: int = 5) -> List[Bid]:
        all_bids = self.activate_parts(parts, context)
        return self.filter_bids(all_bids, max_bids)
    
    def get_bids_summary(self, bids: List[Bid]) -> str:
        if not bids:
            return "No active bids."
        lines = ["Active bids:"]
        for i, bid in enumerate(bids, 1):
            lines.append(f"{i}. {bid.part_name}: {bid.recommendation}")
            lines.append(f"   Urgency: {bid.urgency}/10")
        return "\n".join(lines)


def get_bid_engine() -> BidEngine:
    return BidEngine()
