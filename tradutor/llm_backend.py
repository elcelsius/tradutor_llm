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
    text: str
    latency: float


class LLMBackend:
    def __init__(
        self,
        backend: BackendType,
        model: str,
        temperature: float,
        logger: logging.Logger,
        base_url: str = "http://localhost:11434",
        request_timeout: int = 60,
        gemini_api_key: Optional[str] = None,
    ) -> None:
        self.backend = backend
        self.model = model
        self.temperature = temperature
        self.base_url = base_url.rstrip("/")
        self.logger = logger
        self.request_timeout = request_timeout
        self.gemini_api_key = gemini_api_key

    def generate(self, prompt: str) -> LLMResponse:
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
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
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

    def _call_gemini(self, prompt: str) -> str:
        if genai is None:
            raise RuntimeError("google-generativeai não instalado.")
        api_key = self.gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY não configurada.")
        genai.configure(api_key=api_key)
        try:
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(prompt, generation_config={"temperature": self.temperature})
        except Exception as exc:
            self.logger.error("Erro ao chamar Gemini: %s", exc)
            raise
        text = (response.text or "").strip()
        if not text:
            raise ValueError("Gemini retornou resposta vazia.")
        return text
