"""
Abstrações de backend LLM (Ollama e Gemini).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import requests

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover - lib opcional
    genai = None

from .config import BackendType


@dataclass
class LLMResponse:
    """Encapsula a resposta retornada por um provedor de LLM junto com a latência da requisição."""

    text: str
    latency: float


class LLMBackend:
    """
    Abstração unificada para interagir com diferentes provedores de LLM (Ollama, Gemini, etc).
    Gerencia configurações como temperatura, modelo, timeouts e o roteamento da requisição.
    """

    def __init__(
        self,
        backend: BackendType,
        model: str,
        temperature: float,
        logger: logging.Logger,
        base_url: str = "http://localhost:11434",
        request_timeout: int = 120,
        gemini_api_key: Optional[str] = None,
        repeat_penalty: float | None = None,
        num_predict: int = 768,
        num_ctx: int | None = None,
        keep_alive: str | int | None = "30m",
        api_mode: str = "generate",
        think: bool | None = None,
        seed: int | None = None,
    ) -> None:
        """Processamento interno auxiliar."""
        self.backend = backend
        self.model = model
        self.temperature = temperature
        self.base_url = base_url.rstrip("/")
        self.logger = logger
        self.request_timeout = request_timeout
        self.gemini_api_key = gemini_api_key
        self.repeat_penalty = repeat_penalty
        self.num_predict = num_predict
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.api_mode = api_mode
        self.think = think
        self.seed = seed

    def generate(self, prompt: str) -> LLMResponse:
        """Envia o prompt para o backend configurado e retorna a resposta formatada."""
        start = time.perf_counter()
        if self.backend == "ollama":
            text = self._call_ollama(prompt)
        elif self.backend == "gemini":
            text = self._call_gemini(prompt)
        else:
            raise ValueError(f"Backend não suportado: {self.backend}")
        latency = time.perf_counter() - start
        return LLMResponse(text=text, latency=latency)

    def _call_ollama(self, prompt: str) -> str:
        """Processamento interno auxiliar."""
        if self.api_mode == "chat":
            return self._call_ollama_chat(prompt)
        if self.api_mode != "generate":
            raise ValueError(f"Modo de API Ollama não suportado: {self.api_mode}")

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }
        if self.repeat_penalty is not None:
            payload["options"]["repeat_penalty"] = self.repeat_penalty
        if self.num_ctx is not None:
            payload["options"]["num_ctx"] = self.num_ctx
        if self.seed is not None:
            payload["options"]["seed"] = self.seed
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        try:
            resp = requests.post(url, json=payload, timeout=self.request_timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            self.logger.error("Erro ao chamar Ollama: %s", exc)
            raise

        if "response" not in data:
            raise ValueError(f"Resposta inválida do Ollama: {json.dumps(data)[:200]}")
        return data["response"].strip()

    def _call_ollama_chat(self, prompt: str) -> str:
        """Processamento interno auxiliar."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }
        if self.repeat_penalty is not None:
            payload["options"]["repeat_penalty"] = self.repeat_penalty
        if self.num_ctx is not None:
            payload["options"]["num_ctx"] = self.num_ctx
        if self.seed is not None:
            payload["options"]["seed"] = self.seed
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        if self.think is not None:
            payload["think"] = self.think
        try:
            resp = requests.post(url, json=payload, timeout=self.request_timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            self.logger.error("Erro ao chamar Ollama chat: %s", exc)
            raise

        message = data.get("message")
        if not isinstance(message, dict) or "content" not in message:
            raise ValueError(
                f"Resposta inválida do Ollama chat: {json.dumps(data)[:200]}"
            )
        return (message.get("content") or "").strip()

    def _call_gemini(self, prompt: str) -> str:
        """Processamento interno auxiliar."""
        if genai is None:
            raise RuntimeError("google-generativeai não instalado.")
        api_key = self.gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY não configurada.")
        genai.configure(api_key=api_key)
        try:
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(
                prompt, generation_config={"temperature": self.temperature}
            )
        except Exception as exc:
            self.logger.error("Erro ao chamar Gemini: %s", exc)
            raise
        text = (response.text or "").strip()
        if not text:
            raise ValueError("Gemini retornou resposta vazia.")
        return text
