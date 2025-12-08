# -*- coding: utf-8 -*-
"""
Tradutor de Livros PDF com IA - Versão Híbrida (Gemini / Ollama)

Recursos principais
-------------------
- Backend GEMINI (nuvem) ou OLLAMA (local), selecionável via CLI.
- Pipeline completo:
  - leitura e extração de texto de PDFs (pasta data/)
  - pré-processamento:
      * remoção de cabeçalho/rodapé (ex.: "Page X Goldenagato | mp4directs.com")
      * remoção de hifenização entre linhas
      * reconstrução de parágrafos
      * marcação de títulos/capítulos em Markdown (Prologue, Chapter X, etc.)
  - chunking inteligente (~N caracteres, cortando em parágrafos/fim de frase)
  - tradução em lotes
  - segunda passada de revisão em PT-BR (refine) OPCIONAL
  - geração de .md + .pdf na pasta saida/

Defaults (Qualidade Máxima Local)
---------------------------------
- backend        = "ollama"
- modelo tradução= "qwen3:14b"
- refine         = DESLIGADO por padrão (use --refine para ativar)
- refine_model   = "auto" -> cadeia de modelos PT-BR no Ollama:

    1) cnmoro/gemma3-gaia-ptbr-4b:q4_k_m      (principal)
    2) cnmoro/Qwen2.5-0.5B-Portuguese-v2:fp16
    3) cnmoro/Qwen2.5-0.5B-Portuguese-v1:q4_k_m
    4) cnmoro/Qwen2.5-0.5B-Portuguese-v1:q8_0
    5) cnmoro/Qwen2.5-0.5B-Portuguese-v1:fp16

Observação importante sobre "refine"
------------------------------------
A segunda passada (refine) pode, em alguns modelos, tentar "resumir" ou
reescrever demais o texto. O prompt foi reforçado para desencorajar isso,
mas ainda assim, para segurança, o refine vem DESLIGADO por padrão.
Use apenas se quiser testar melhoria de estilo e estiver pronto para conferir
o resultado com atenção, ou para trechos menores (capítulos, cenas, etc.).
"""

import os
import sys
import time
import re
import logging
import argparse
from typing import List, Optional, Any

import requests
from fpdf import FPDF
import PyPDF2


# --- 1. LOGGING ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


# --- 2. CONFIGURAÇÕES GLOBAIS ---

class Config:
    # Diretórios
    DATA_DIR = "data"
    SAIDA_DIR = "saida"

    # Backends
    DEFAULT_BACKEND = "ollama"  # "ollama" ou "gemini"

    # GEMINI
    DEFAULT_GEMINI_MODEL = "gemini-3-pro-preview"
    GEMINI_GENERATION_CONFIG = {"temperature": 0.3}
    GEMINI_SAFETY_SETTINGS = {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    }

    # OLLAMA
    OLLAMA_HOST = "http://localhost:11434"
    DEFAULT_OLLAMA_MODEL = "qwen3:14b"

    # Cadeia de modelos PT-BR para refine local (na ordem de prioridade)
    OLLAMA_REFINE_MODEL_CHAIN = [
        # 1) Gaia como refinador principal
        "cnmoro/gemma3-gaia-ptbr-4b:q4_k_m",
        # 2) Qwen2.5 como fallbacks
        "cnmoro/Qwen2.5-0.5B-Portuguese-v2:fp16",
        "cnmoro/Qwen2.5-0.5B-Portuguese-v1:q4_k_m",
        "cnmoro/Qwen2.5-0.5B-Portuguese-v1:q8_0",
        "cnmoro/Qwen2.5-0.5B-Portuguese-v1:fp16",
    ]

    # Chunking / retry
    CHARS_PER_CHUNK = 9000
    MAX_RETRIES = 3
    INITIAL_BACKOFF = 2

    # Fonte para PDF
    FONT_URL = (
        "https://github.com/dejavu-fonts/dejavu-fonts/raw/main/ttf/DejaVuSans.ttf"
    )
    FONT_NAME = "DejaVu"
    FONT_FILE = "DejaVuSans.ttf"


# --- 3. PRÉ-PROCESSAMENTO DE TEXTO ---


_HEADER_FOOTER_CONTAINS = [
    "Goldenagato | mp4directs.com",
    "mp4directs.com",
    "zerobooks",
    "jnovels.com",
    "download all your favorite light novels",
    "stay up to date on light novels",
    "join our discord",
    "newsletter",
]

_HEADER_FOOTER_REGEXES = [
    re.compile(r"^page\s+\d+\s+goldenagato\s*\|\s*mp4directs\.com$", re.IGNORECASE),
    re.compile(r"^table of contents$", re.IGNORECASE),
]


def _remove_headers_footers(text: str) -> str:
    """
    Remove padrões típicos de cabeçalho/rodapé do PDF original:
    - linhas tipo "Page 5 Goldenagato | mp4directs.com"
    - propagandas de app/site (Zerobooks, Jnovels, etc.)
    - números de página isolados
    - linhas curtas TODAS em caixa alta (candidatos a cabeçalho)

    A ideia é preservar ao máximo o conteúdo do romance, mas
    limpar lixo visual recorrente.
    """
    linhas_filtradas = []
    removidas = 0

    for line in text.splitlines():
        l = line.strip()
        if not l:
            linhas_filtradas.append(line)
            continue

        skip = False

        # Número de página isolado
        if re.fullmatch(r"\d{1,4}", l):
            skip = True

        # Linha curta TODA em maiúsculas (provável cabeçalho genérico)
        if not skip and len(l) <= 30 and l.isupper():
            skip = True

        # Padrões específicos (regex)
        if not skip:
            for rgx in _HEADER_FOOTER_REGEXES:
                if rgx.match(l):
                    skip = True
                    break

        # Trechos que contêm domínios/propagandas
        if not skip:
            low = l.lower()
            for frag in _HEADER_FOOTER_CONTAINS:
                if frag in low:
                    skip = True
                    break

        if skip:
            removidas += 1
            logging.debug(f"[PRE] Removendo cabeçalho/rodapé: {repr(l)}")
            continue

        linhas_filtradas.append(line)

    logging.info(f"Removidas {removidas} linhas de cabeçalho/rodapé.")
    return "\n".join(linhas_filtradas)


def _remove_hyphenation(text: str) -> str:
    """Remove hifenação em quebra de linha: 'infor-\\nmação' -> 'informação'."""
    return re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)


def _join_broken_lines(text: str) -> str:
    """
    Junta linhas quebradas no meio de frases em parágrafos coerentes.
    Mantém parágrafos separados por linha em branco.
    """
    lines = text.split("\n")
    reconstituted_lines: List[str] = []
    buffer = ""

    for line in lines:
        stripped_line = line.strip()

        if not stripped_line:
            if buffer:
                reconstituted_lines.append(buffer)
                buffer = ""
            continue

        buffer += (" " + stripped_line) if buffer else stripped_line

        # Heurística simples: fim de frase ou linha muito curta (título)
        if stripped_line.endswith((".", "!", "?", '"', ":")) or len(
            stripped_line.split()
        ) < 5:
            reconstituted_lines.append(buffer)
            buffer = ""

    if buffer:
        reconstituted_lines.append(buffer)

    return "\n\n".join(reconstituted_lines)


def _mark_headings(text: str) -> str:
    """
    Converte títulos de capítulos em headings Markdown para leitura melhor
    em celular (ex.: "Prologue", "Chapter 1: The Forbidden Witch", "Epilogue").
    """
    linhas = text.split("\n")
    novas_linhas: List[str] = []

    chapter_re = re.compile(r"^(Prologue|Epilogue|Afterword)$", re.IGNORECASE)
    chapter_full_re = re.compile(r"^Chapter\s+\d+[:.].*$", re.IGNORECASE)

    for line in linhas:
        stripped = line.strip()
        if chapter_re.match(stripped) or chapter_full_re.match(stripped):
            # Evita duplicar '#' se já tiver
            if not stripped.startswith("#"):
                new_line = "## " + stripped
            else:
                new_line = stripped
            novas_linhas.append(new_line)
        else:
            novas_linhas.append(line)

    return "\n".join(novas_linhas)


def preprocess_text(raw_text: str) -> str:
    logging.info("Iniciando pré-processamento do texto...")
    text = _remove_headers_footers(raw_text)
    logging.debug("Após _remove_headers_footers (500 chars): %r", text[:500])

    text = _remove_hyphenation(text)
    logging.debug("Após _remove_hyphenation (500 chars): %r", text[:500])

    text = _join_broken_lines(text)
    logging.debug("Após _join_broken_lines (500 chars): %r", text[:500])

    text = _mark_headings(text)
    logging.debug("Após _mark_headings (500 chars): %r", text[:500])

    logging.info("Pré-processamento concluído.")
    return text


# --- 4. PROMPTS E INTERAÇÃO COM LLM ---


def _create_translation_prompt(chunk: str, chunk_num: int, total_chunks: int) -> str:
    return f"""
Você é um TRADUTOR LITERÁRIO SÊNIOR especializado em LIGHT NOVELS JAPONESAS,
traduzindo do INGLÊS para o PORTUGUÊS DO BRASIL (PT-BR).

TAREFA:
Traduza o TRECHO abaixo de forma FIEL, NATURAL e COMPLETA.

REGRAS GERAIS:
- Traduza TODO o conteúdo. NÃO RESUMA, NÃO CORTE e NÃO ADICIONE nada.
- NÃO transforme o trecho em sinopse, análise ou comentário sobre a história.
- NÃO escreva em primeira pessoa sobre "o romance", "a obra", "a história" etc.
- NÃO mude a ordem das frases ou parágrafos.
- A saída deve ser APENAS o texto traduzido, sem qualquer texto adicional.

ESTILO:
- Português do Brasil fluido e natural, com ritmo de light novel.
- Preserve o tom emocional, a personalidade dos personagens e o clima da cena.
- Adapte expressões idiomáticas para algo NATURAL em PT-BR.
- Diálogos devem soar como fala natural de personagens.

FORMATAÇÃO:
- Preserve as quebras de parágrafo: se houver uma linha em branco, mantenha um novo parágrafo.
- Preserve marcadores como #, ##, ** e * quando existirem.
- Mantenha travessões, aspas e itálicos compatíveis com uso em português.
- Se houver onomatopeias, mantenha ou adapte de forma natural (ex.: "Ugh", "Hmph", etc.).

NOMES E TERMOS:
- Mantenha nomes de personagens, lugares, magias e golpes no idioma original,
  a menos que uma tradução seja amplamente estabelecida em PT-BR.
- Não traduza nomes próprios se isso soará estranho.

CONTEXTO:
- Este é o trecho {chunk_num} de {total_chunks} de um livro maior.
- NÃO faça suposições sobre partes anteriores ou futuras. Trabalhe SOMENTE com o trecho fornecido.

AGORA TRADUZA fielmente o texto entre as linhas --- para PT-BR:

---
{chunk}
---
"""


def _create_refine_prompt(translated_markdown: str) -> str:
    return f"""
Você é um REVISOR LITERÁRIO PROFISSIONAL de português do Brasil.

TAREFA:
Revise o texto abaixo, que já está em PT-BR, corrigindo apenas:
- coesão e fluidez,
- pequenos deslizes de concordância,
- pontuação ou escolha de palavras que fiquem pouco naturais.

REGRAS IMPORTANTES:
- NÃO RESUMA e NÃO CORTE cenas, frases ou parágrafos.
- NÃO transforme o texto em resumo, sinopse ou análise crítica.
- NÃO escreva comentários externos sobre "o romance", "a obra" ou "a história".
- Mantenha a MESMA estrutura do texto: número aproximado de parágrafos e falas.
- Mantenha nomes próprios, magias, golpes e termos específicos como estão,
  a menos que haja um claro erro de português.
- Se tiver dúvida, PREFIRA manter o trecho quase igual ao original, apenas
  ajustando pontuação e detalhes de estilo.

ESTILO:
- Light novel em PT-BR: envolvente, claro e natural.
- Diálogos devem continuar soando como fala natural de personagens.

SAÍDA:
- Entregue APENAS o texto revisado, sem explicações, sem comentários e sem
  nenhum texto adicional antes ou depois.

TEXTO A REVISAR:
---
{translated_markdown}
---
"""


# --- 4.1 OLLAMA ---


def call_ollama(model_name: str, prompt: str) -> str:
    """
    Chama o Ollama local (API /api/generate) e retorna o campo 'response'.
    """
    url = f"{Config.OLLAMA_HOST}/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
    }
    resp = requests.post(url, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    result = data.get("response", "")
    return result.strip()


def translate_chunk_ollama(
    model_name: str, text_chunk: str, chunk_num: int, total_chunks: int
) -> str:
    prompt = _create_translation_prompt(text_chunk, chunk_num, total_chunks)
    retries = 0
    last_err: Optional[Exception] = None

    while retries < Config.MAX_RETRIES:
        try:
            logging.info(
                f"[OLLAMA] Enviando lote {chunk_num}/{total_chunks} "
                f"({len(text_chunk)} caracteres) para o modelo '{model_name}'..."
            )
            logging.debug(
                f"[OLLAMA] Prévia do lote {chunk_num}: {repr(text_chunk[:300])}"
            )
            translated = call_ollama(model_name, prompt)
            if not translated:
                raise ValueError("Resposta vazia do Ollama.")
            logging.info(f"[OLLAMA] Lote {chunk_num} traduzido com sucesso.")
            return translated
        except Exception as e:
            last_err = e
            retries += 1
            logging.warning(
                f"[OLLAMA] Erro ao traduzir lote {chunk_num} com '{model_name}': "
                f"{e}. Tentativa {retries}/{Config.MAX_RETRIES}."
            )
            if retries < Config.MAX_RETRIES:
                wait_time = Config.INITIAL_BACKOFF**retries
                logging.info(f"Aguardando {wait_time} segundos antes de tentar novamente...")
                time.sleep(wait_time)

    raise RuntimeError(
        f"[OLLAMA] Falha ao traduzir lote {chunk_num} com '{model_name}' após "
        f"{Config.MAX_RETRIES} tentativas. Último erro: {last_err}"
    )


def refine_text_ollama(refine_model_spec: str, markdown_text: str) -> str:
    """
    Segunda passada local usando Ollama.
    - Se refine_model_spec != "auto": usa apenas esse modelo.
    - Se refine_model_spec == "auto": tenta a cadeia Config.OLLAMA_REFINE_MODEL_CHAIN
      na ordem, parando no primeiro modelo que responder sem erro.
    """
    prompt = _create_refine_prompt(markdown_text)

    if refine_model_spec and refine_model_spec != "auto":
        candidates = [refine_model_spec]
    else:
        candidates = Config.OLLAMA_REFINE_MODEL_CHAIN

    last_err: Optional[Exception] = None

    for model_name in candidates:
        retries = 0
        logging.info(f"[OLLAMA] Tentando refine com modelo '{model_name}'...")
        while retries < Config.MAX_RETRIES:
            try:
                logging.info(
                    f"[OLLAMA] Revisão com '{model_name}', tentativa "
                    f"{retries + 1}/{Config.MAX_RETRIES}..."
                )
                refined = call_ollama(model_name, prompt)
                if not refined:
                    raise ValueError("Resposta vazia do Ollama na revisão.")
                logging.info(
                    f"[OLLAMA] Revisão concluída com sucesso com o modelo '{model_name}'."
                )
                return refined
            except Exception as e:
                last_err = e
                retries += 1
                logging.warning(
                    f"[OLLAMA] Erro na revisão com '{model_name}': {e} "
                    f"(tentativa {retries}/{Config.MAX_RETRIES})."
                )
                if retries < Config.MAX_RETRIES:
                    wait_time = Config.INITIAL_BACKOFF**retries
                    logging.info(
                        f"Aguardando {wait_time} segundos antes de tentar novamente..."
                    )
                    time.sleep(wait_time)
        logging.warning(
            f"[OLLAMA] Desistindo do modelo '{model_name}' após múltiplas falhas."
        )

    raise RuntimeError(
        "[OLLAMA] Falha na revisão: nenhum dos modelos de refine respondeu com sucesso. "
        f"Último erro: {last_err}"
    )


# --- 4.2 GEMINI ---


def init_gemini_model(model_name: str, temperature: float) -> Any:
    """
    Inicializa o modelo Gemini apenas quando o backend 'gemini' é usado.
    Faz import tardio para evitar dependência quando rodar só com Ollama.
    """
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError as e:
        logging.critical(
            "O pacote 'google-generativeai' não está instalado. "
            "Instale com: pip install google-generativeai"
        )
        raise e

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "A variável de ambiente GEMINI_API_KEY não foi definida para uso do backend 'gemini'."
        )

    genai.configure(api_key=api_key)
    gem_model = genai.GenerativeModel(
        model_name,
        generation_config={"temperature": temperature},
        safety_settings=Config.GEMINI_SAFETY_SETTINGS,
    )
    return gem_model


def translate_chunk_gemini(
    gem_model: Any, text_chunk: str, chunk_num: int, total_chunks: int
) -> str:
    prompt = _create_translation_prompt(text_chunk, chunk_num, total_chunks)
    retries = 0
    last_err: Optional[Exception] = None

    while retries < Config.MAX_RETRIES:
        try:
            logging.info(
                f"[GEMINI] Enviando lote {chunk_num}/{total_chunks} "
                f"({len(text_chunk)} caracteres)..."
            )
            logging.debug(
                f"[GEMINI] Prévia do lote {chunk_num}: {repr(text_chunk[:300])}"
            )
            response = gem_model.generate_content(prompt, request_options={"timeout": 1000})
            if not getattr(response, "text", None):
                raise ValueError("A resposta da API Gemini está vazia ou mal formada.")
            logging.info(f"[GEMINI] Lote {chunk_num} traduzido com sucesso.")
            return response.text
        except Exception as e:
            last_err = e
            retries += 1
            logging.warning(
                f"[GEMINI] Erro ao traduzir lote {chunk_num}: {e}. "
                f"Tentativa {retries}/{Config.MAX_RETRIES}."
            )
            if retries < Config.MAX_RETRIES:
                wait_time = Config.INITIAL_BACKOFF**retries
                logging.info(
                    f"[GEMINI] Aguardando {wait_time} segundos antes de tentar novamente..."
                )
                time.sleep(wait_time)

    raise RuntimeError(
        f"[GEMINI] Falha ao traduzir lote {chunk_num} após "
        f"{Config.MAX_RETRIES} tentativas. Último erro: {last_err}"
    )


def refine_text_gemini(gem_model: Any, markdown_text: str) -> str:
    prompt = _create_refine_prompt(markdown_text)
    retries = 0
    last_err: Optional[Exception] = None

    while retries < Config.MAX_RETRIES:
        try:
            logging.info("[GEMINI] Iniciando revisão (segunda passada)...")
            resp = gem_model.generate_content(prompt, request_options={"timeout": 1000})
            if not getattr(resp, "text", None):
                raise ValueError("Resposta vazia na revisão Gemini.")
            logging.info("[GEMINI] Revisão concluída com sucesso.")
            return resp.text
        except Exception as e:
            last_err = e
            retries += 1
            logging.warning(
                f"[GEMINI] Erro na revisão: {e} "
                f"(tentativa {retries}/{Config.MAX_RETRIES})."
            )
            if retries < Config.MAX_RETRIES:
                wait_time = Config.INITIAL_BACKOFF**retries
                logging.info(
                    f"[GEMINI] Aguardando {wait_time} segundos antes de tentar novamente..."
                )
                time.sleep(wait_time)

    raise RuntimeError(
        "[GEMINI] Falha na revisão após múltiplas tentativas. "
        f"Último erro: {last_err}"
    )


# --- 5. EXTRAÇÃO E CHUNKING ---


def extract_text_from_pdf(pdf_path: str) -> str:
    logging.info(f"Iniciando extração de texto de '{pdf_path}'...")
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
        logging.info(f"Texto extraído com sucesso de {len(reader.pages)} páginas.")
        logging.debug("Primeiros 800 caracteres do texto bruto: %r", text[:800])
        return text
    except Exception as e:
        logging.error(f"Falha ao ler o arquivo PDF '{pdf_path}'. Erro: {e}")
        raise


def chunk_text(text: str, chunk_size: int) -> List[str]:
    """
    Divide o texto em lotes tentando:
    1) cortar em quebra de parágrafo (\n\n),
    2) se não der, cortar próximo ao fim de frase (. ? !),
    3) fallback no limite bruto.
    """
    logging.info(f"Dividindo o texto em lotes de ~{chunk_size} caracteres...")
    chunks: List[str] = []
    current_pos = 0
    text_len = len(text)

    while current_pos < text_len:
        end_pos = min(current_pos + chunk_size, text_len)
        corte = end_pos

        # Tenta quebra de parágrafo
        bp_par = text.rfind("\n\n", current_pos, end_pos)
        if bp_par > current_pos:
            corte = bp_par
        else:
            # Tenta fim de frase
            candidatos = [
                text.rfind(".", current_pos, end_pos),
                text.rfind("?", current_pos, end_pos),
                text.rfind("!", current_pos, end_pos),
            ]
            bp_frase = max(candidatos)
            if bp_frase > current_pos:
                corte = bp_frase + 1

        if corte <= current_pos:
            corte = end_pos

        chunk = text[current_pos:corte].strip()
        if chunk:
            chunks.append(chunk)
        current_pos = corte

    logging.info(f"Texto dividido em {len(chunks)} lotes.")
    logging.debug(
        "Tamanhos dos 5 primeiros lotes: %s",
        [len(c) for c in chunks[:5]],
    )
    return chunks


# --- 6. GERAÇÃO DE PDF A PARTIR DE MARKDOWN ---


def markdown_to_pdf(markdown_text: str, pdf_path: str):
    font_path = os.path.join(os.path.dirname(__file__), Config.FONT_FILE)

    if not os.path.exists(font_path):
        logging.info("Baixando fonte DejaVu para suporte a caracteres Unicode...")
        try:
            r = requests.get(Config.FONT_URL, timeout=30)
            r.raise_for_status()
            with open(font_path, "wb") as f:
                f.write(r.content)
            logging.info("Fonte baixada com sucesso.")
        except Exception as e:
            logging.error(f"Falha ao baixar a fonte. PDF não será gerado. Erro: {e}")
            return

    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.add_font(Config.FONT_NAME, "", font_path, uni=True)
        pdf.set_font(Config.FONT_NAME, "", 12)

        for line in markdown_text.split("\n"):
            if line.startswith("# "):
                pdf.set_font(Config.FONT_NAME, "B", 18)
                pdf.multi_cell(0, 10, line[2:], 0, 1)
            elif line.startswith("## "):
                pdf.set_font(Config.FONT_NAME, "B", 14)
                pdf.multi_cell(0, 8, line[3:], 0, 1)
            elif line.startswith("**") and line.endswith("**"):
                pdf.set_font(Config.FONT_NAME, "B", 12)
                pdf.multi_cell(0, 7, line[2:-2], 0, 1)
            else:
                pdf.set_font(Config.FONT_NAME, "", 12)
                pdf.multi_cell(0, 5, line)
            if not line.startswith(("**", "# ")):
                pdf.ln(3)

        pdf.output(pdf_path)
        logging.info(f"PDF final salvo em: {pdf_path}")
    except Exception as e:
        logging.error(f"Não foi possível gerar o PDF. Erro: {e}")


# --- 7. PIPELINE PRINCIPAL POR PDF ---


def process_pdf_translation(
    pdf_path: str,
    backend: str,
    translate_handle: Any,
    refine_enabled: bool,
    refine_model_spec: str,
    chunk_size: int,
):
    pdf_filename = os.path.basename(pdf_path)
    base_name, _ = os.path.splitext(pdf_filename)
    paths = {
        "md": os.path.join(Config.SAIDA_DIR, f"{base_name}_pt.md"),
        "pdf": os.path.join(Config.SAIDA_DIR, f"{base_name}_pt.pdf"),
    }

    try:
        raw_text = extract_text_from_pdf(pdf_path)
        clean_text = preprocess_text(raw_text)
        text_chunks = chunk_text(clean_text, chunk_size)

        # Arquivo MD inicial
        with open(paths["md"], "w", encoding="utf-8") as f:
            f.write(f"# Tradução de: {pdf_filename}\n\n")

        # Tradução em lotes
        for i, chunk in enumerate(text_chunks, 1):
            if backend == "ollama":
                translated_chunk = translate_chunk_ollama(
                    translate_handle, chunk, i, len(text_chunks)
                )
            elif backend == "gemini":
                translated_chunk = translate_chunk_gemini(
                    translate_handle, chunk, i, len(text_chunks)
                )
            else:
                raise ValueError(f"Backend desconhecido: {backend}")

            with open(paths["md"], "a", encoding="utf-8") as f:
                f.write(translated_chunk.strip() + "\n\n")

        logging.info(f"Tradução completa salva em '{paths['md']}'.")

        # Lê o markdown completo
        with open(paths["md"], "r", encoding="utf-8") as f:
            full_markdown = f.read()

        # Segunda passada (refine) – opcional
        final_markdown = full_markdown
        if refine_enabled:
            logging.info("Iniciando segunda passada de revisão da tradução (refine)...")
            if backend == "ollama":
                final_markdown = refine_text_ollama(refine_model_spec, full_markdown)
            elif backend == "gemini":
                final_markdown = refine_text_gemini(translate_handle, full_markdown)
            else:
                raise ValueError(f"Backend desconhecido: {backend}")

        # Sobrescreve com a versão final
        with open(paths["md"], "w", encoding="utf-8") as f:
            f.write(final_markdown)

        # Gera PDF
        logging.info(f"Gerando PDF final em '{paths['pdf']}'...")
        markdown_to_pdf(final_markdown, paths["pdf"])

    except Exception as e:
        logging.critical(
            f"Processo interrompido para '{pdf_filename}' por um erro fatal: {e}"
        )


# --- 8. MAIN ---


def main():
    parser = argparse.ArgumentParser(
        description="Tradutor de Livros PDF com IA (Gemini / Ollama)."
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["gemini", "ollama"],
        default=Config.DEFAULT_BACKEND,
        help="Backend de LLM a usar: 'gemini' (nuvem) ou 'ollama' (local). "
        "Padrão: ollama.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Modelo principal de TRADUÇÃO. "
        "Exemplos: gemini-3-pro-preview, qwen3:14b, qwen2.5:14b, aya:8b...",
    )
    parser.add_argument(
        "--refine-model",
        type=str,
        default="auto",
        help="Modelo de REVISÃO em PT-BR (segunda passada). "
        "Para backend=ollama, use 'auto' para tentar cadeia de modelos PT-BR "
        "recomendados. Para backend=gemini, é ignorado (usa o mesmo modelo).",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=Config.CHARS_PER_CHUNK,
        help=f"Tamanho de cada lote em caracteres (padrão: {Config.CHARS_PER_CHUNK}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=Config.GEMINI_GENERATION_CONFIG["temperature"],
        help="Temperatura da geração (usada para Gemini; Ollama costuma ignorar).",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help=(
            "Ativa a segunda passada de revisão da tradução. "
            "DESLIGADO por padrão (recomendado deixar off para livros inteiros; "
            "use apenas para trechos/capítulos se quiser testar)."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Ativa logs detalhados (DEBUG) sobre pré-processamento e chunking.",
    )

    args = parser.parse_args()

    # Ajusta nível de log se --debug
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Modo DEBUG ativado.")

    backend = args.backend
    refine_enabled = args.refine

    # Garante diretórios
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    os.makedirs(Config.SAIDA_DIR, exist_ok=True)

    # Inicializa backend
    translate_handle: Any = None

    if backend == "gemini":
        model_name = args.model or Config.DEFAULT_GEMINI_MODEL
        logging.info(f"Inicializando backend GEMINI com modelo '{model_name}'...")
        gem_model = init_gemini_model(model_name, args.temperature)
        translate_handle = gem_model
        logging.info("Backend GEMINI pronto.")
    elif backend == "ollama":
        model_name = args.model or Config.DEFAULT_OLLAMA_MODEL
        translate_handle = model_name
        logging.info(
            f"Backend OLLAMA selecionado. Modelo de tradução: '{model_name}'. "
            f"Refine: {'ativo' if refine_enabled else 'desativado'}; "
            f"refine_model='{args.refine_model}'."
        )
    else:
        raise ValueError(f"Backend desconhecido: {backend}")

    # Localiza PDFs
    pdf_files = [f for f in os.listdir(Config.DATA_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        logging.warning(
            f"Nenhum arquivo PDF encontrado no diretório '{Config.DATA_DIR}'. "
            "Coloque seus PDFs lá e rode novamente."
        )
        return

    logging.info(f"Encontrados {len(pdf_files)} arquivos PDF para processar.")
    for pdf_file in pdf_files:
        logging.info(f"--- Iniciando processamento de '{pdf_file}' ---")
        process_pdf_translation(
            os.path.join(Config.DATA_DIR, pdf_file),
            backend=backend,
            translate_handle=translate_handle,
            refine_enabled=refine_enabled,
            refine_model_spec=args.refine_model,
            chunk_size=args.chunk_size,
        )
        logging.info(f"--- Finalizado processamento de '{pdf_file}' ---")


if __name__ == "__main__":
    main()
