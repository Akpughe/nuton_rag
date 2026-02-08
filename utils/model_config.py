"""
Model configuration and switching for Course Generation.
Centralized model management following DRY principle.
"""

from typing import Dict, Any, Optional
from enum import Enum


class ModelProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GROQ = "groq"


# Model configurations
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "claude-haiku-4-5": {
        "provider": ModelProvider.ANTHROPIC,
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 16000,
        "cost_per_1k_input": 0.001,
        "cost_per_1k_output": 0.005,
        "supports_search": True,
        "temperature": 0.7
    },
    "claude-sonnet-4-5": {
        "provider": ModelProvider.ANTHROPIC,
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 16000,
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
        "supports_search": True,
        "temperature": 0.7
    },
    "claude-opus-4-6": {
        "provider": ModelProvider.ANTHROPIC,
        "model": "claude-opus-4-6",
        "max_tokens": 16000,
        "cost_per_1k_input": 0.005,
        "cost_per_1k_output": 0.025,
        "supports_search": True,
        "temperature": 0.7
    },
    "gpt-4o": {
        "provider": ModelProvider.OPENAI,
        "model": "gpt-4o",
        "max_tokens": 16000,
        "cost_per_1k_input": 0.0025,
        "cost_per_1k_output": 0.01,
        "supports_search": True,
        "temperature": 0.7
    },
    "gpt-5-mini": {
        "provider": ModelProvider.OPENAI,
        "model": "gpt-5-mini",
        "max_tokens": 16000,
        "cost_per_1k_input": 0.00025,
        "cost_per_1k_output": 0.002,
        "supports_search": True,
        "temperature": 0.7
    },
    "gpt-5.2": {
        "provider": ModelProvider.OPENAI,
        "model": "gpt-5.2",
        "max_tokens": 16000,
        "cost_per_1k_input": 0.00175,
        "cost_per_1k_output": 0.014,
        "supports_search": True,
        "temperature": 0.7
    },
    "llama-4-scout": {
        "provider": ModelProvider.GROQ,
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "max_tokens": 8192,
        "cost_per_1k_input": 0.00011,
        "cost_per_1k_output": 0.00034,
        "supports_search": False,
        "temperature": 0.7
    },
    "llama-4-maverick": {
        "provider": ModelProvider.GROQ,
        "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "max_tokens": 8192,
        "cost_per_1k_input": 0.0002,
        "cost_per_1k_output": 0.0006,
        "supports_search": False,
        "temperature": 0.7
    }
}

DEFAULT_MODEL = "claude-haiku-4-5"


def get_search_mode(model_key: Optional[str] = None) -> str:
    """
    Determine the search mode for a given model.
    Returns:
        "native" - Claude/GPT models with built-in web search
        "perplexity" - Groq/Llama models that need Perplexity as search fallback
        "none" - No search available
    """
    key = model_key or DEFAULT_MODEL
    config = MODEL_CONFIGS.get(key)
    if not config:
        return "none"

    if config.get("supports_search"):
        return "native"

    # Groq models use Perplexity as search fallback
    if config["provider"] == ModelProvider.GROQ:
        return "perplexity"

    return "none"


class ModelConfig:
    """Model configuration manager"""

    @staticmethod
    def get_config(model_key: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration for specified model or default"""
        key = model_key or DEFAULT_MODEL

        if key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {key}. Available: {list(MODEL_CONFIGS.keys())}")

        return MODEL_CONFIGS[key]

    @staticmethod
    def get_available_models() -> list:
        """List all available models"""
        return list(MODEL_CONFIGS.keys())

    @staticmethod
    def estimate_cost(model_key: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a request"""
        config = ModelConfig.get_config(model_key)

        input_cost = (input_tokens / 1000) * config["cost_per_1k_input"]
        output_cost = (output_tokens / 1000) * config["cost_per_1k_output"]

        return round(input_cost + output_cost, 4)

    @staticmethod
    def supports_web_search(model_key: Optional[str] = None) -> bool:
        """Check if model supports native web search"""
        config = ModelConfig.get_config(model_key)
        return config.get("supports_search", False)


# Course generation cost estimates
COURSE_COST_ESTIMATES = {
    "outline_generation": {
        "claude-haiku-4-5": 0.008,
        "claude-sonnet-4-5": 0.02,
        "claude-opus-4-6": 0.04,
        "gpt-4o": 0.02,
        "gpt-5-mini": 0.003,
        "gpt-5.2": 0.02,
        "llama-4-scout": 0.002,
        "llama-4-maverick": 0.003
    },
    "chapter_1000_words": {
        "claude-haiku-4-5": 0.008,
        "claude-sonnet-4-5": 0.02,
        "claude-opus-4-6": 0.04,
        "gpt-4o": 0.02,
        "gpt-5-mini": 0.003,
        "gpt-5.2": 0.02,
        "llama-4-scout": 0.002,
        "llama-4-maverick": 0.003
    },
    "search_perplexity": 0.01
}


def estimate_course_cost(model_key: str, num_chapters: int = 4) -> float:
    """Estimate total cost for course generation"""
    outline = COURSE_COST_ESTIMATES["outline_generation"].get(model_key, 0.02)
    chapter = COURSE_COST_ESTIMATES["chapter_1000_words"].get(model_key, 0.02)
    search = COURSE_COST_ESTIMATES["search_perplexity"]

    return round(outline + (chapter * num_chapters) + search, 2)
