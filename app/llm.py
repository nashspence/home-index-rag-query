from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline

from .config import settings

_cached_llm = None


def load_llm(model_name: str | None = None) -> HuggingFacePipeline:
    """Load a HuggingFace model as a LangChain LLM.

    Downloads the model from Hugging Face if it is not present locally.
    """
    global _cached_llm
    if _cached_llm is not None and model_name == settings.llm_model_name:
        return _cached_llm
    model_name = model_name or settings.llm_model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_safetensors=True,
    )
    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
    )
    llm = HuggingFacePipeline(pipeline=gen_pipeline)
    _cached_llm = llm
    return llm
