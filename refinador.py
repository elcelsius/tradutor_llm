# -*- coding: utf-8 -*-
"""
Refinador de Traduções - Revisão capítulo a capítulo em PT-BR

Uso típico:
-----------
1) Primeiro, rode o tradutor normal para gerar os arquivos *_pt.md:

    python tradutor.py

2) Depois, rode o refinador para revisar os .md traduzidos:

    # Refina TODOS os *_pt.md da pasta 'saida' usando Ollama (Gaia + Qwen2.5)
    python refinador.py

    # Refina um arquivo específico
    python refinador.py --input "saida/Vol 05 - PDF Room_unlocked_pt.md"

3) A saída será sempre:
   - <nome>_refinado.md
   - <nome>_refinado.pdf

O original *_pt.md NUNCA é sobrescrito.
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


# --- 1. LOGGING ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


# --- 2. CONFIGURAÇÕES GLOBAIS ---

class Config:
    SAIDA_DIR = "saida"

    # Backend principal para refine
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

    # Tamanho padrão dos chunks para refine (em caracteres)
    REFINE_CHARS_PER_CHUNK = 5000

    # Retry
    MAX_RETRIES = 3
    INITIAL_BACKOFF = 2

    # Fonte para PDF
    FONT_URL = (
        "https://github.com/dejavu-fonts/dejavu-fonts/raw/main/ttf/DejaVuSans.ttf"
    )
    FONT_NAME = "DejaVu"
    FONT_FILE = "DejaVuSans.ttf"


# --- 3. SUPORTE A LLM (PROMPTS + CHAMADAS) ---

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


# --- 3.1 OLLAMA ---

def call_ollama(model_name: str, prompt: str) -> str:
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


def refine_chunk_ollama(refine_model_spec: str, markdown_text: str) -> str:
    """
    Refina UM chunk de texto usando Ollama.
    Se refine_model_spec == 'auto', tenta a cadeia OLLAMA_REFINE_MODEL_CHAIN.
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


# --- 3.2 GEMINI ---

def init_gemini_model(model_name: str, temperature: float) -> Any:
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


def refine_chunk_gemini(gem_model: Any, markdown_text: str) -> str:
    prompt = _create_refine_prompt(markdown_text)
    retries = 0
    last_err: Optional[Exception] = None

    while retries < Config.MAX_RETRIES:
        try:
            logging.info("[GEMINI] Iniciando revisão (chunk)...")
            resp = gem_model.generate_content(prompt, request_options={"timeout": 1000})
            if not getattr(resp, "text", None):
                raise ValueError("Resposta vazia na revisão Gemini.")
            logging.info("[GEMINI] Revisão de chunk concluída com sucesso.")
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


# --- 4. SPLIT DE DOCUMENTO EM SEÇÕES / CHUNKS ---

def split_md_by_sections(md_text: str) -> List[dict]:
    """
    Divide o .md em seções baseadas em headings '## ' (capítulos/partes).

    Retorna uma lista de dicts:
        [
          {"title": None ou "## X", "body": "texto..."},
          ...
        ]

    A primeira seção (antes do primeiro '##') fica com title=None.
    """
    lines = md_text.splitlines()
    sections: List[dict] = []
    current_title: Optional[str] = None
    current_lines: List[str] = []

    for line in lines:
        if line.startswith("## "):
            # fecha seção anterior
            if current_lines or current_title is not None:
                sections.append(
                    {
                        "title": current_title,
                        "body": "\n".join(current_lines).strip(),
                    }
                )
            current_title = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    # última seção
    if current_lines or current_title is not None:
        sections.append(
            {
                "title": current_title,
                "body": "\n".join(current_lines).strip(),
            }
        )

    return sections


def chunk_text(text: str, chunk_size: int) -> List[str]:
    """
    Divide um texto em blocos menores, tentando cortar em parágrafo ou fim de frase.
    """
    chunks: List[str] = []
    current_pos = 0
    text_len = len(text)

    while current_pos < text_len:
        end_pos = min(current_pos + chunk_size, text_len)
        corte = end_pos

        bp_par = text.rfind("\n\n", current_pos, end_pos)
        if bp_par > current_pos:
            corte = bp_par
        else:
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

    return chunks


# --- 5. GERAÇÃO DE PDF ---

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
        logging.info(f"PDF refinado salvo em: {pdf_path}")
    except Exception as e:
        logging.error(f"Não foi possível gerar o PDF refinado. Erro: {e}")


# --- 6. PIPELINE DE REFINO POR ARQUIVO .MD ---

def refine_markdown_file(
    md_path: str,
    backend: str,
    refine_handle: Any,
    refine_model_spec: str,
    chunk_size: int,
):
    base_name = os.path.basename(md_path)
    logging.info(f"=== Iniciando refine de '{base_name}' ===")

    with open(md_path, "r", encoding="utf-8") as f:
        full_markdown = f.read()

    sections = split_md_by_sections(full_markdown)
    logging.info(f"Documento dividido em {len(sections)} seções (capítulos/partes).")

    refined_sections: List[str] = []

    for idx, sec in enumerate(sections, 1):
        title = sec["title"]
        body = sec["body"]

        if title:
            logging.info(f"[Seção {idx}] Heading: {title}")
        else:
            logging.info(f"[Seção {idx}] (sem heading, provavelmente cabeçalho inicial)")

        if not body.strip():
            # Sem conteúdo relevante; apenas replica heading (se tiver)
            if title:
                refined_sections.append(title)
            continue

        # Divide o corpo em chunks menores para evitar estourar contexto
        body_chunks = chunk_text(body, chunk_size)
        logging.info(
            f"[Seção {idx}] corpo dividido em {len(body_chunks)} chunk(s) para refine."
        )

        refined_body_parts: List[str] = []
        for c_idx, c_text in enumerate(body_chunks, 1):
            logging.info(
                f"[Seção {idx}] Refinando chunk {c_idx}/{len(body_chunks)} "
                f"({len(c_text)} caracteres)..."
            )
            if backend == "ollama":
                refined_chunk = refine_chunk_ollama(refine_model_spec, c_text)
            elif backend == "gemini":
                refined_chunk = refine_chunk_gemini(refine_handle, c_text)
            else:
                raise ValueError(f"Backend desconhecido: {backend}")
            refined_body_parts.append(refined_chunk.strip())

        refined_body = "\n\n".join(refined_body_parts)

        if title:
            refined_sections.append(title)
        refined_sections.append(refined_body)
        refined_sections.append("")  # linha em branco entre seções

    final_markdown = "\n".join(refined_sections).strip() + "\n"

    # Caminhos de saída
    root, ext = os.path.splitext(md_path)
    md_out = root + "_refinado.md"
    pdf_out = root + "_refinado.pdf"

    with open(md_out, "w", encoding="utf-8") as f:
        f.write(final_markdown)

    logging.info(f"Markdown refinado salvo em: {md_out}")
    logging.info("Gerando PDF refinado...")
    markdown_to_pdf(final_markdown, pdf_out)


# --- 7. MAIN ---

def main():
    parser = argparse.ArgumentParser(
        description="Refinador de traduções em PT-BR (capítulo a capítulo)."
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["gemini", "ollama"],
        default=Config.DEFAULT_BACKEND,
        help="Backend de LLM a usar para o refine. Padrão: ollama.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Modelo de refine. Para backend=ollama, use 'auto' ou um modelo específico. "
             "Para backend=gemini, nome do modelo Gemini.",
        default="auto",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=Config.REFINE_CHARS_PER_CHUNK,
        help=f"Tamanho dos chunks de refine em caracteres (padrão: {Config.REFINE_CHARS_PER_CHUNK}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=Config.GEMINI_GENERATION_CONFIG["temperature"],
        help="Temperatura da geração (usada para Gemini).",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Caminho para um arquivo .md específico. "
             "Se omitido, o refinador processa todos os *_pt.md em 'saida/'.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Ativa logs em nível DEBUG.",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Modo DEBUG ativado.")

    backend = args.backend
    refine_model_spec = args.model

    # Inicializa handle do backend (se necessário)
    refine_handle: Any = None

    if backend == "gemini":
        model_name = (
            refine_model_spec
            if refine_model_spec != "auto"
            else Config.DEFAULT_GEMINI_MODEL
        )
        logging.info(f"Inicializando backend GEMINI para refine com modelo '{model_name}'...")
        refine_handle = init_gemini_model(model_name, args.temperature)
        logging.info("Backend GEMINI pronto para refine.")
    elif backend == "ollama":
        logging.info(
            f"Backend OLLAMA selecionado para refine. "
            f"Modelo: '{refine_model_spec}'. (use 'auto' para cadeia Gaia + Qwen2.5)"
        )
    else:
        raise ValueError(f"Backend desconhecido: {backend}")

    # Descobre quais arquivos .md refinar
    md_files: List[str] = []
    if args.input:
        if not os.path.isfile(args.input):
            logging.error(f"Arquivo especificado em --input não existe: {args.input}")
            sys.exit(1)
        md_files = [args.input]
    else:
        # Todos *_pt.md da pasta 'saida'
        if not os.path.isdir(Config.SAIDA_DIR):
            logging.error(
                f"Diretório '{Config.SAIDA_DIR}' não existe. Rode primeiro o tradutor."
            )
            sys.exit(1)
        for fname in os.listdir(Config.SAIDA_DIR):
            if fname.lower().endswith("_pt.md"):
                md_files.append(os.path.join(Config.SAIDA_DIR, fname))

    if not md_files:
        logging.warning(
            "Nenhum arquivo *_pt.md encontrado para refinar. "
            "Certifique-se de rodar o tradutor antes."
        )
        return

    logging.info(f"Encontrados {len(md_files)} arquivo(s) .md para refinar.")
    for md in md_files:
        refine_markdown_file(
            md_path=md,
            backend=backend,
            refine_handle=refine_handle,
            refine_model_spec=refine_model_spec,
            chunk_size=args.chunk_size,
        )


if __name__ == "__main__":
    main()
