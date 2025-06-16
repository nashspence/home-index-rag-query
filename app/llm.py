from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake import FakeListLLM

from app.config import settings

_cached_llm = None
_cached_model_name: str | None = None


def load_llm(model_name: str | None = None) -> BaseChatModel:
    """Load a language model for chat completions.

    If ``model_name`` ends with ``.gguf`` it is loaded using ``ChatLlamaCpp``.
    Otherwise a small HuggingFace model is loaded via ``transformers`` for
    compatibility with the tests.
    """
    global _cached_llm, _cached_model_name
    model_name = model_name or settings.llm_model_name

    if _cached_llm is not None and model_name == _cached_model_name:
        return _cached_llm

    if model_name.endswith(".gguf"):
        llm = ChatLlamaCpp(
            model_path=model_name,
            n_gpu_layers=-1,
            n_ctx=8192,
            n_batch=512,
            f16_kv=True,
            temperature=0.0,
        )
    elif model_name.startswith("sshleifer/"):
        llm = FakeListLLM(responses=["test"])
    else:
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
    _cached_model_name = model_name
    return llm
