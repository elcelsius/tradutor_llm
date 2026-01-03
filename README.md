# Tradutor de Light Novels (EN → PT-BR)

Pipeline em Python 3.12 para converter PDFs/Markdown em PT-BR com LLM (Ollama ou Gemini): extrai, limpa, desquebra linhas, traduz em chunks, refina e gera relatórios/PDF. Tudo roda local, priorizando Windows (mas funciona em Linux).

## Requisitos e instalação
- Python 3.12.
- Backend LLM:
  - Ollama (padrão). Ajustes em `config.yaml`.
  - Gemini: defina `GEMINI_API_KEY`.
- Instalação mínima:
  ```bash
  pip install -r requirements.txt
  ```
- Ambiente de desenvolvimento:
  ```bash
  pip install -r requirements-dev.txt
  pytest -q
  pre-commit install
  pre-commit run --all-files
  ```
Todas as fontes são UTF-8 (ver testes e hook de mojibake). Use editor com UTF-8.

## Guia rápido
### 1) Traduzir PDF (com desquebrar + refine)
```bash
python -m tradutor.main traduz --input "data/meu_livro.pdf"
```
Saídas em `saida/`: `meu_livro_pt.md`, `meu_livro_pt_refinado.md`, relatórios JSON e caches (ver Outputs).

### 2) Traduzir Markdown já desquebrado
```bash
python -m tradutor.main traduz-md --input "saida/meu_texto_desquebrado.md"
```

### 3) Refine separado
```bash
python -m tradutor.main refina --input "saida/meu_livro_pt.md"
```

### 4) Gerar PDF a partir do MD
```bash
python -m tradutor.main pdf --input "saida/meu_livro_pt_refinado.md"
```

## Pipeline v2 (como o código executa)
1) **Extração e pré-processo** (`tradutor/preprocess.py::extract_text_from_pdf`, `preprocess_text`):
   - Normaliza quebras, remove rodapés/ruído e front-matter/TOC se `skip_front_matter` estiver ativo (padrão vindo do config).
2) **Desquebrar** (`tradutor/desquebrar.py::desquebrar_text`):
   - LLM ou modo seguro (`desquebrar_safe`) junta linhas, corrige hifenização/aspas e valida saída.
   - Controlado por `--use-desquebrar/--no-use-desquebrar` e `--desquebrar-mode llm|safe`.
3) **Chunking e tradução** (`tradutor/translate.py::translate_document`):
   - Divide por seções se `split_by_sections` ativo (usa `tradutor/section_splitter.py`).
   - Tradução chunk a chunk com glossário manual por chunk, guardrails de diálogo, retries e sanitização.
   - Saídas: `_pt.md`, métricas/relatórios JSON, progress para resume.
4) **Cleanup opcional pré-refine** (`tradutor/cleanup.py::cleanup_before_refine`), controlado por `--cleanup-before-refine {off,auto,on}`.
5) **Refine** (`tradutor/refine.py::refine_markdown_file`):
   - Chunking do PT, guardrails, glossário manual/dinâmico, normalizadores estruturais, anti-colapso.
   - Saídas: `_pt_refinado.md`, métricas/relatórios JSON, progress.
6) **PDF** (`tradutor/pdf.py::convert_markdown_to_pdf` via CLI) se `--pdf-enabled` ou configuração.

Resumos, métricas e progress são escritos em `saida/` (ver Outputs).

## CLI e opções principais (ver `tradutor/main.py`)
### Subcomando `traduz` (PDF → PT)
- `--input <pdf>`: PDF específico (senão pega todos de `data/`).
- `--backend {ollama,gemini}`, `--model <nome>`, `--num-predict <int>`.
- `--no-refine`: pula refine.
- `--desquebrar-mode {llm,safe}` e `--use-desquebrar/--no-use-desquebrar`.
- `--resume`: usa `<slug>_pt_progress.json` para retomar.
- `--use-glossary` / `--manual-glossary <json>`: glossário manual (apenas termos presentes no chunk são injetados; limite configurável).
- `--translate-allow-adaptation`: habilita bloco de adaptação no prompt.
- `--split-by-sections` / `--skip-front-matter`: controle de headings/TOC.
- `--cleanup-before-refine {off,auto,on}`: limpeza determinística antes do refine.
- `--debug`: salva `_raw_extracted.md`, `_preprocessed.md` (e `_raw_desquebrado.md` se aplicável).
- `--debug-chunks`: JSONL detalhado por chunk (tradução/refine).
- `--fail-on-chunk-error`: aborta na primeira falha (senão marca placeholders).
- `--pdf-enabled`: gera PDF após refine.
- `--clear-cache {all,translate,refine,desquebrar}`: limpa caches em `saida/cache_*`.

### Subcomando `traduz-md` (MD → PT)
Mesmas opções de tradução/refine relevantes; inclui `--normalize-paragraphs` para normalizar o Markdown antes de traduzir.

### Subcomando `refina` (PT → PT refinado)
- `--input <*_pt.md>` (senão refina todos em `saida/`).
- Glossário manual/dinâmico: `--use-glossary`, `--manual-glossary`, `--dynamic-glossary`, `--auto-glossary-dir`.
- `--resume`: usa `<slug>_pt_refinado_progress.json`.
- `--normalize-paragraphs`, `--cleanup-before-refine`, `--debug-refine`, `--debug-chunks`.
- Editores opcionais (`editor_*`) aplicados pós-refine (ver `tradutor/main.py` e `tradutor/editor.py`).

### Subcomando `pdf`
Converte um `.md` em PDF com as configs de fonte/margem do `config.yaml`.

## Glossário (fonte única em código: `tradutor/glossary_utils.py`)
- Manual: JSON com `terms: [{key, pt, aliases?, category?, notes?, locked?}]`. Use `--manual-glossary`. Glossário dinâmico salvo em `saida/glossario_dinamico.json` quando habilitado no refine.
- Injeção por chunk: só termos que aparecem no chunk entram no prompt (`select_terms_for_chunk`, limite `translate_glossary_match_limit`; fallback de até `translate_glossary_fallback_limit` termos quando nada casa).
- Enforcement: termos com `enforce=true` são forçados no texto traduzido (após o LLM) apenas para os termos selecionados naquele chunk (`translate.enforce_canonical_terms`).

## Cleanup antes do refine
- `cleanup_before_refine` (off/auto/on) aplica dedupe e fix de diálogos colados (`tradutor/cleanup.py::cleanup_before_refine`). Quando aplicado, gera `<slug>_pre_refine_cleanup.md` para inspeção.

## Resume e progress
- Tradução: `<slug>_pt_progress.json` inclui hashes e chunks para retomar (`translate.translate_document` via `run_translate`/`run_translate_md`).
- Refine: `<slug>_pt_refinado_progress.json` para retomar (`refine.refine_markdown_file`).
- Estados rápidos: `saida/state_traducao.json`, `saida/state_refine.json`.

## Outputs, caches e debug (ver também docs/OUTPUTS.md)
- Tradução: `saida/<slug>_pt.md`, `<slug>_translate_report.json`, `<slug>_translate_metrics.json`, progress (`_pt_progress.json`), debug opcional (`debug_traducao/`, `*_pt_chunks_debug.jsonl`).
- Refine: `saida/<slug>_pt_refinado.md`, `<slug>_refine_report.json`, `<slug>_refine_metrics.json`, progress (`_pt_refinado_progress.json`), debug opcional (`debug_refine*/`).
- Desquebrar: métricas em `<slug>_desquebrar_metrics.json` se rodar com LLM; debug raw/preprocess quando `--debug`.
- PDF: `saida/pdf/<slug>_pt_refinado.pdf` se `--pdf-enabled`.
- Caches: `saida/cache_traducao`, `saida/cache_refine`, `saida/cache_desquebrar` (`tradutor/cache_utils.py`).

## Testes e qualidade
- Testes locais: `pytest -q`.
- Hooks locais: `pre-commit run --all-files` (mojibake, EOF, whitespace; usa `scripts/check_mojibake.py` que importa tokens de `tradutor/mojibake.py`).
- CI: GitHub Actions (`.github/workflows/ci.yml`) roda `pytest -q` e `pre-commit run --all-files` em Ubuntu/Windows com Python 3.12, cache de pip.

## Troubleshooting
- **UTF-8/mojibake:** fontes/tokens inválidos são bloqueados por testes/hook. Use editor em UTF-8.
- **TOC ou headings vazios virando capítulos:** `skip_front_matter` e `split_by_sections` controlam; o splitter ignora stubs (`tradutor/section_splitter.py`).
- **Diálogos curtos colados ou omitidos:** guardrails/retries no translate (`needs_retry`, `dialogue_guardrails`); normalizadores de aspas em `translate` e `refine`.
- **Truncamento de chunk:** retries automáticos; se `fail_on_chunk_error` off, placeholders `[CHUNK_*]` aparecem no MD e logs indicam falha.
- **Caches contaminados:** use `--clear-cache all` ao mudar modelo/prompt; caches ficam em `saida/cache_*`.
- **Retomar execução:** use `--resume` em `traduz`/`traduz-md`/`refina` para aproveitar progress.
- **PDF não gerado:** confira `--pdf-enabled` e fonte configurada em `config.yaml`.

## Roadmap curto (doc)
- Confirmar/atualizar exemplos de modelos recomendados (dependem do host).
- Adicionar exemplos de glossários reais (sem dados sensíveis) em `glossario/`.

