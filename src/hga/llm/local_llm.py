"""Local GPU LLM — Qwen 1.5B with 4-bit quantization, no API.

Uses HuggingFace transformers + bitsandbytes to load a quantized small chat
model (Qwen2.5-1.5B-Instruct). The model fits in ~2GB VRAM in 4-bit. JSON mode
is via prompt engineering (model instructed to output only JSON).
"""

from __future__ import annotations

import json
from typing import Any, cast

from hga.llm.base import LLMClient


class LocalTransformersLLM(LLMClient):
    """Qwen 1.5B chat model with 4-bit quantization, running locally."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        quantize_4bit: bool = True,
        device: str | None = None,
    ) -> None:
        self._model_id = model_id
        self._quantize = quantize_4bit
        self._device = device
        self._model: Any = None
        self._tokenizer: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        has_cuda = torch.cuda.is_available()
        kwargs: dict[str, Any] = {}

        if self._quantize and has_cuda:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            kwargs["device_map"] = self._device or "auto"
            print(f"[hga] Loading {self._model_id} in 4-bit on GPU")
        else:
            if self._quantize and not has_cuda:
                print("[hga] No CUDA — falling back to fp16 on CPU (slower)")
            kwargs["dtype"] = torch.float16
            print(f"[hga] Loading {self._model_id} in fp16 on CPU")

        kwargs["trust_remote_code"] = True
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(self._model_id, **kwargs)
        if not has_cuda:
            self._model.to("cpu")
        self._model.eval()

    def _generate(
        self, messages: list[dict[str, str]], max_tokens: int, temperature: float
    ) -> str:
        import torch

        self._load()
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": self._tokenizer.eos_token_id,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
            outputs = self._model.generate(**inputs, **gen_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        response = self._tokenizer.decode(
            outputs[0][prompt_len:], skip_special_tokens=True
        )
        return str(response).strip()

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        import asyncio

        del model
        msgs: list[dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)

        def _do() -> str:
            return self._generate(msgs, max_tokens, temperature)

        return await asyncio.to_thread(_do)

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        *,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        import asyncio

        del model, system
        json_schema = json.dumps(schema)
        json_system = (
            "Respond with ONLY valid JSON. No explanation, no markdown fencing."
            " Match this schema exactly."
        )
        msgs: list[dict[str, str]] = [
            {"role": "system", "content": json_system},
            {
                "role": "user",
                "content": f"Schema:\n{json_schema}\n\nQuery:\n{messages[-1]['content']}",
            },
        ]

        def _do() -> dict[str, Any]:
            raw = self._generate(msgs, max_tokens, temperature)
            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            return cast("dict[str, Any]", json.loads(clean))

        return await asyncio.to_thread(_do)
