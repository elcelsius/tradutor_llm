# -*- coding: utf-8 -*-
# Versão 4.1
import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests

# --- Constantes de Configuração ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma3:12b"

CHUNK_SIZE_DEFAULT = 25
LOOKAHEAD_SIZE_DEFAULT = 15
GLOSSARY_FILE_DEFAULT = "glossario.txt"
CACHE_FILE_DEFAULT = "desquebrar_cache.json"

# --- Constantes de Robustez ---
MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 2

# Validação: tamanho mínimo e máximo do texto de saída
VALIDATION_MIN_RATIO = 0.70   # saída não pode ser < 70% do tamanho da entrada
VALIDATION_MAX_RATIO = 1.50   # nem > 150% (protege contra lero-lero)
FOREIGN_CHAR_THRESHOLD = 0.20  # 20%


@dataclass
class Stats:
    """Classe para coletar estatísticas de execução."""
    total_chunks: int = 0
    cache_hits: int = 0
    api_calls: int = 0
    api_errors: int = 0
    validation_failures: int = 0
    fallback_chunks: int = 0
    total_time: float = 0.0
    input_chars: int = 0
    output_chars: int = 0

    def print_report(self) -> None:
        """Imprime o relatório final de execução no console."""
        logging.info("--- Relatório Final de Execução ---")
        logging.info(f"Tempo Total: {self.total_time:.2f} segundos")
        logging.info(f"Total de Blocos: {self.total_chunks}")
        logging.info(f"Recuperados do Cache: {self.cache_hits}")
        logging.info(f"Chamadas à API: {self.api_calls}")
        logging.info(f"Falhas de Validação: {self.validation_failures}")
        logging.info(f"Erros de API (após retries): {self.api_errors}")
        logging.info(f"Blocos em Fallback: {self.fallback_chunks}")
        logging.info(f"Tamanho entrada: {self.input_chars} chars")
        logging.info(f"Tamanho saída:   {self.output_chars} chars")
        if self.input_chars > 0:
            ratio = self.output_chars / self.input_chars
            logging.info(f"Razão saída/entrada: {ratio:.3f}")
        logging.info("------------------------------------")


# --- Funções auxiliares ---
def normalize_md_paragraphs(md_text: str) -> str:
    """
    Normaliza parágrafos de Markdown juntando linhas internas, preservando
    blocos especiais (headings, listas, blocos de código, citações e diálogos
    iniciados com travessão/hífen).
    """
    if not md_text:
        return md_text

    text = md_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    normalized: list[str] = []
    buffer: list[str] = []
    in_fence = False
    fence_marker = ""

    def flush_buffer() -> None:
        nonlocal buffer
        if buffer:
            normalized.append(" ".join(buffer).strip())
            buffer = []

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        if in_fence:
            normalized.append(raw_line)
            if stripped.startswith(fence_marker):
                in_fence = False
                fence_marker = ""
            continue

        if stripped.startswith("```") or stripped.startswith("~~~"):
            flush_buffer()
            in_fence = True
            fence_marker = stripped[:3]
            normalized.append(raw_line)
            continue

        if stripped == "":
            flush_buffer()
            normalized.append("")
            continue

        if (
            re.match(r"^#{1,6}\s", stripped)
            or re.match(r"^>\s", stripped)
            or re.match(r"^[-*+]\s", stripped)
            or re.match(r"^\d+\.\s", stripped)
        ):
            flush_buffer()
            normalized.append(stripped)
            continue

        if stripped.startswith(("-", "–", "—")):
            flush_buffer()
            normalized.append(stripped)
            continue

        if buffer:
            if buffer[-1].endswith("-"):
                buffer[-1] = buffer[-1][:-1]
                buffer.append(stripped.lstrip())
            else:
                buffer.append(stripped)
        else:
            buffer.append(stripped)

    flush_buffer()

    compact: list[str] = []
    prev_blank = False
    for ln in normalized:
        if ln == "":
            if not prev_blank:
                compact.append("")
            prev_blank = True
        else:
            compact.append(ln)
            prev_blank = False

    return "\n".join(compact).strip()


# --- Glossário e Prompt do Sistema ---
def load_glossary(path: str) -> list[str]:
    """Carrega termos de glossário (um por linha)."""
    if not os.path.exists(path):
        logging.info(
            f"Arquivo de glossário '{path}' não encontrado. "
            "Continuando sem glossário."
        )
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            terms = [line.strip() for line in f if line.strip()]
        logging.info(f"Glossário carregado com {len(terms)} termos de '{path}'.")
        return terms
    except IOError as e:
        logging.error(f"Erro ao ler o arquivo de glossário '{path}': {e}")
        return []


def get_system_prompt(glossary_terms: list[str]) -> str:
    base_prompt = """Você é um assistente especializado em formatar textos de livros. Sua ÚNICA tarefa é corrigir quebras de linha incorretas dentro do texto fornecido entre as tags <INPUT_TEXT>.
REGRAS ESTRITAS:
1. Analise o texto dentro de <INPUT_TEXT> e junte apenas frases que foram cortadas ao meio.
2. Mantenha diálogos (linhas que começam com —, –, \", \') em linhas separadas.
3. Mantenha títulos (como 'Prólogo', 'Capítulo X') em linhas separadas.
4. NÃO altere nenhuma palavra, NÃO resuma, NÃO adicione comentários.
5. Retorne APENAS o texto corrigido, sem as tags <INPUT_TEXT> ou qualquer outra explicação."""
    if glossary_terms:
        glossary_str = ", ".join(glossary_terms)
        glossary_rule = (
            "\n6. GLOSSÁRIO (Termos que você DEVE manter exatamente como estão): "
            f"{glossary_str}"
        )
        return base_prompt + glossary_rule
    return base_prompt


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_cache(cache_path: str) -> dict:
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        logging.warning("Cache corrompido ou ilegível. Reiniciando cache.")
        return {}


def save_cache(cache_path: str, cache_data: dict) -> None:
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
    except IOError as e:
        logging.warning(f"Não foi possível salvar o cache em {cache_path}: {e}")


# --- Funções de Processamento e Validação ---
def clean_ia_response(text: str) -> str:
    """Limpa a resposta da IA, removendo artefatos, CJK e ajustando formatação."""
    lines = text.split("\n")
    cjk_pattern = re.compile(r"[\u4e00-\u9fff]")
    cleaned_lines: list[str] = []

    for line in lines:
        # Filtro anti-leaking de instruções
        upper = line.upper()
        if "CONTEXTO ANTERIOR" in upper or "SYSTEM PROMPT" in upper:
            continue

        # Filtro de caracteres não-latinos (CJK)
        stripped = line.strip()
        line_len = len(stripped)
        if line_len > 0 and (len(cjk_pattern.findall(stripped)) / line_len) > FOREIGN_CHAR_THRESHOLD:
            logging.warning(
                "Removendo linha com excesso de caracteres não-latinos: "
                f"'{stripped[:50]}...'"
            )
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Garante espaço após travessão de diálogo: "—Palavra" -> "— Palavra"
    text = re.sub(r"^(—)(\\S)", r"\\1 \\2", text, flags=re.MULTILINE)

    return text.strip()

def remove_repetitive_loops(text: str) -> str:
    """Usa Regex para remover agressivamente padrões de loop (linhas repetidas)."""
    loop_pattern = re.compile(r"^(.*)(\\s*\\n\\1){2,}", re.MULTILINE)
    cleaned_text = loop_pattern.sub(r"\\1", text)
    if cleaned_text != text:
        logging.warning("Filtro de loop removeu repetições.")
    return cleaned_text

def validate_response(input_text: str, output_text: str, stats: Stats) -> bool:
    """Executa validação de tamanho na resposta da IA.

Garante que o texto gerado não seja absurdamente menor ou maior que o texto
de entrada. Em caso de falha, incrementa contadores e informa o caller para
acionar fallback.
    """
    if not input_text:
        # Nada para validar
        return True

    in_len = len(input_text)
    out_len = len(output_text)
    ratio = out_len / in_len if in_len else 1.0

    if ratio < VALIDATION_MIN_RATIO:
        logging.warning(
            "Falha na validação: texto de saída muito menor que o de entrada "
            f"({ratio:.3f} < {VALIDATION_MIN_RATIO:.3f})."
        )
        stats.validation_failures += 1
        return False

    if ratio > VALIDATION_MAX_RATIO:
        logging.warning(
            "Falha na validação: texto de saída muito maior que o de entrada "
            f"({ratio:.3f} > {VALIDATION_MAX_RATIO:.3f})."
        )
        stats.validation_failures += 1
        return False

    return True

def process_chunk_with_ia(
    prompt_text: str,
    original_chunk_text: str,
    model_name: str,
    cache: dict,
    stats: Stats,
    system_prompt: str,
    chunk_index: int,
    total_chunks: int,
    use_cache: bool,
) -> str:
    """Processa um chunk com IA, usando cache + retries + validação."""
    cache_key = hashlib.md5(
        (prompt_text + model_name + system_prompt).encode("utf-8")
    ).hexdigest()

    if use_cache and cache_key in cache:
        logging.info(f"Bloco {chunk_index}/{total_chunks}: recuperado do cache.")
        stats.cache_hits += 1
        return cache[cache_key]

    final_prompt = f"<INPUT_TEXT>\n{prompt_text}\n</INPUT_TEXT>"
    stats.api_calls += 1

    payload = {
        "model": model_name,
        "system": system_prompt,
        "prompt": final_prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_ctx": 4096,
            "repeat_penalty": 1.2,
        },
    }

    start_time = time.monotonic()

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                OLLAMA_API_URL,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()

            data = response.json()
            corrected_text = data.get("response", "") or ""
            corrected_text = clean_ia_response(corrected_text)
            corrected_text = remove_repetitive_loops(corrected_text)

            if validate_response(original_chunk_text, corrected_text, stats):
                duration = time.monotonic() - start_time
                logging.info(
                    f"Bloco {chunk_index}/{total_chunks}: OK "
                    f"({duration:.2f}s, {len(corrected_text)} chars)"
                )
                if use_cache:
                    cache[cache_key] = corrected_text
                return corrected_text

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            logging.warning(
                f"Bloco {chunk_index}/{total_chunks}: Erro de rede "
                f"({type(e).__name__}, tentativa {attempt + 1})."
            )
            if attempt == MAX_RETRIES - 1:
                stats.api_errors += 1
        except Exception as e:  # noqa: BLE001
            logging.warning(
                f"Bloco {chunk_index}/{total_chunks}: Erro inesperado "
                f"({e}, tentativa {attempt + 1})."
            )
            if attempt == MAX_RETRIES - 1:
                stats.api_errors += 1

        if attempt < MAX_RETRIES - 1:
            # Backoff exponencial simples
            sleep_time = RETRY_BACKOFF_FACTOR**attempt
            logging.debug(f"Aguardando {sleep_time}s antes do retry...")
            time.sleep(sleep_time)

    logging.error(
        f"Bloco {chunk_index}/{total_chunks}: FALHA. "
        "Usando texto original (fallback)."
    )
    stats.fallback_chunks += 1
    return original_chunk_text

def create_smart_chunks(
    lines: list[str],
    target_size: int,
    max_lookahead: int,
) -> list[str]:
    """Divide o texto em blocos de linhas respeitando pausas 'naturais'."""
    chunks: list[str] = []
    current_pos = 0
    line_count = len(lines)

    # Considera como corte seguro: fim de frase / diálogo / parágrafo
    safe_end_re = re.compile(r'(?:[.?!]|”|"|–|—)\s*$')

    while current_pos < line_count:
        end_pos = min(current_pos + target_size, line_count)

        if end_pos == line_count:
            chunks.append("\n".join(lines[current_pos:end_pos]))
            break

        last_line = lines[end_pos - 1].strip()
        if not last_line or safe_end_re.search(last_line):
            # Já terminou em ponto razoável
            pass
        else:
            found_safe_cut = False
            for lookahead_pos in range(
                end_pos,
                min(end_pos + max_lookahead, line_count),
            ):
                candidate = lines[lookahead_pos].strip()
                if not candidate or safe_end_re.search(candidate):
                    end_pos = lookahead_pos + 1
                    found_safe_cut = True
                    break

            if not found_safe_cut:
                end_pos = min(end_pos + max_lookahead, line_count)

        chunks.append("\n".join(lines[current_pos:end_pos]))
        current_pos = end_pos

    return chunks

def fix_line_breaks_ia(
    text: str,
    model_name: str,
    glossary_terms: list[str],
    chunk_size: int,
    lookahead_size: int,
    cache_path: str,
    use_cache: bool,
) -> tuple[str, Stats]:
    """Pipeline principal: chunking + IA + cache + estatísticas."""
    lines = text.splitlines()
    cache = load_cache(cache_path) if use_cache else {}
    system_prompt = get_system_prompt(glossary_terms)

    logging.info(
        "Dividindo o texto em blocos inteligentes "
        f"(alvo de {chunk_size} linhas, tolerância de {lookahead_size})..."
    )
    chunks = create_smart_chunks(lines, chunk_size, lookahead_size)

    stats = Stats(total_chunks=len(chunks), input_chars=len(text))
    logging.info(f"Texto dividido em {stats.total_chunks} blocos.")

    processed_chunks: list[str] = []
    previous_context_sentence = ""

    for i, chunk_text in enumerate(chunks, start=1):
        if not chunk_text.strip():
            continue

        if previous_context_sentence:
            prompt_for_ia = (
                "CONTEXTO ANTERIOR "
                "(Apenas para referência, NÃO REPITA): "
                f"'{previous_context_sentence}'\n\n---\n\n{chunk_text}"
            )
        else:
            prompt_for_ia = chunk_text

        corrected_chunk = process_chunk_with_ia(
            prompt_for_ia,
            chunk_text,
            model_name,
            cache,
            stats,
            system_prompt,
            chunk_index=i,
            total_chunks=stats.total_chunks,
            use_cache=use_cache,
        )

        processed_chunks.append(corrected_chunk)

        if corrected_chunk.strip():
            # Pega a última linha não-vazia como contexto
            last_line = next(
                (line for line in reversed(corrected_chunk.split("\n")) if line.strip()),
                "",
            )
            previous_context_sentence = last_line

        # Salva o cache periodicamente
        if use_cache and (i % 5 == 0):
            save_cache(cache_path, cache)

    if use_cache:
        save_cache(cache_path, cache)

    corrected_content = "\n".join(processed_chunks)
    stats.output_chars = len(corrected_content)

    return corrected_content, stats

def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_ia{input_path.suffix}")

def process_file(
    input_path: Path,
    output_path: Path,
    model_name: str,
    glossary_path: str,
    chunk_size: int,
    lookahead_size: int,
    cache_path: str,
    use_cache: bool,
) -> None:
    if not input_path.is_file():
        logging.error(f"Arquivo de entrada não encontrado em '{input_path}'")
        sys.exit(1)

    start_time = time.monotonic()

    glossary_terms = load_glossary(glossary_path)

    logging.info(f"Modelo: {model_name}")
    logging.info(f"Lendo arquivo de entrada: {input_path}")

    content = input_path.read_text(encoding="utf-8")

    corrected_content, stats = fix_line_breaks_ia(
        content,
        model_name,
        glossary_terms,
        chunk_size,
        lookahead_size,
        cache_path,
        use_cache,
    )

    stats.total_time = time.monotonic() - start_time
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(corrected_content, encoding="utf-8")

    logging.info(f"Processamento concluído. Arquivo salvo em: {output_path}")
    stats.print_report()

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Corrige quebras de linha em textos de Markdown usando "
            "um modelo de IA local via Ollama."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Caminho para o arquivo de entrada (.md, .txt).",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Caminho para o arquivo de saída. Padrão: <nome>_ia.extensão.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Nome do modelo Ollama a ser usado (padrão: {DEFAULT_MODEL}).",
    )

    # Parametrização do pipeline
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE_DEFAULT,
        help=(
            f"Número alvo de linhas por bloco (padrão: {CHUNK_SIZE_DEFAULT}). "
            "Valores menores = mais chamadas à IA."
        ),
    )
    parser.add_argument(
        "--lookahead-size",
        type=int,
        default=LOOKAHEAD_SIZE_DEFAULT,
        help=(
            "Número máximo de linhas extras para buscar um ponto de corte "
            f"mais natural (padrão: {LOOKAHEAD_SIZE_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--glossary",
        type=str,
        default=GLOSSARY_FILE_DEFAULT,
        help=(
            "Caminho para o arquivo de glossário (um termo por linha). "
            f"Padrão: {GLOSSARY_FILE_DEFAULT}."
        ),
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default=CACHE_FILE_DEFAULT,
        help=(
            "Caminho do arquivo de cache JSON "
            f"(padrão: {CACHE_FILE_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Desativa o uso de cache (não lê nem grava o cache).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Ativa log detalhado (nível DEBUG).",
    )

    return parser.parse_args(argv)

def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    setup_logging(verbose=args.verbose)

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else _default_output_path(input_path)

    try:
        process_file(
            input_path=input_path,
            output_path=output_path,
            model_name=args.model,
            glossary_path=args.glossary,
            chunk_size=args.chunk_size,
            lookahead_size=args.lookahead_size,
            cache_path=args.cache_file,
            use_cache=not args.no_cache,
        )
    except KeyboardInterrupt:
        logging.error("Execução interrompida pelo usuário (Ctrl+C).")
        sys.exit(130)


if __name__ == "__main__":
    main()
