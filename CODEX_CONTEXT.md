# Contexto do Projeto (humano) ÔÇö Tradutor LLM

> Este documento foi montado ap├│s varredura de `README.md`, `tradutor/main.py`, `tradutor/translate.py`, `tradutor/refine.py`, `tradutor/cache_utils.py`, `tradutor/config.py` e diret├│rios `tests/`.
> Se algo parecer incompleto, confira os arquivos acima.

## Arquitetura (componentes e responsabilidades)
- **CLI principal**: `tradutor/main.py`
  - Subcomandos: `traduz`, `traduz-md`, `refina`, `pdf`.
  - Orquestra o pipeline e aplica flags/configs.
- **Config**: `tradutor/config.py`
  - Define defaults e carrega `config.yaml`.
  - Define `data_dir`, `output_dir`, modelos/temperaturas/ctx, PDF settings.
- **Tradu├º├úo**: `tradutor/translate.py`
  - Pr├®-processamento, chunking, gloss├írio por chunk, chamadas LLM, sanitiza├º├úo.
  - Grava m├®tricas/report e manifesto de progresso.
- **Desquebrar**: `tradutor/desquebrar.py`, `tradutor/desquebrar_safe.py`
  - Une linhas quebradas (LLM ou heur├¡stico ÔÇ£safeÔÇØ).
- **Refine**: `tradutor/refine.py`
  - Revis├úo em chunks, guardrails, cleanup determin├¡stico e p├│s-processo.
  - Grava m├®tricas/report e manifesto de progresso.
- **PDF**: `tradutor/pdf.py`, `tradutor/pdf_export.py`
  - Exporta Markdown ÔåÆ PDF usando ReportLab.
- **Utilit├írios**:
  - `tradutor/cache_utils.py` (cache por chunk), `tradutor/preprocess.py`, `tradutor/sanitizer.py` etc.
- **Wrappers legados**: `tradutor.py`, `refinador.py`, `desquebrar.py`.

## Fluxo de dados (alto n├¡vel)
1. **Extra├º├úo**: PDF ÔåÆ texto (`tradutor/pdf_reader.py`).
2. **Pr├®-processamento**: limpeza b├ísica + remo├º├úo de front-matter/TOC (`tradutor/preprocess.py`).
3. **Desquebrar** (opcional): une linhas quebradas (LLM ou safe).
4. **Tradu├º├úo**: chunking + LLM + sanitiza├º├úo + recomposi├º├úo (`tradutor/translate.py`).
5. **Cleanup antes do refine** (opcional): heur├¡sticas determin├¡sticas.
6. **Refine**: revis├úo por chunk com guardrails (`tradutor/refine.py`).
7. **P├│s-processamento**: corre├º├Áes finais e normalizadores.
8. **PDF** (opcional): exporta `*_pt_refinado.md` ÔåÆ PDF.

## Artefatos/JSONs principais (schemas resumidos)
> Campos listados abaixo foram coletados dos writers no c├│digo. Outros campos podem aparecer.

### `*_translate_report.json` (em `saida/`)
- `mode`: `"translate"`
- `status`: `"ok"` ou `"failed"`
- `input`: slug do documento
- `total_chunks`, `cache_hits`, `fallbacks`, `failed_chunks`, `collapse_detected`, `duplicates_reused`
- `pipeline_version`, `timestamp`
- `effective_translate_chunk_chars`, `max_chunk_chars_observed`
- `dialogue_splits`, `triple_quotes_removed`
- `paragraph_mismatch` (opcional)

### `*_translate_metrics.json` (em `saida/`)
- `total_chunks`, `cache_hits`, `duplicates_reused`, `fallbacks`, `failed_chunks`, `collapse_detected`
- `chunks`: m├®tricas por chunk (ver `tradutor/translate.py`)
- `effective_translate_chunk_chars`, `max_chunk_chars_observed`
- `dialogue_splits`, `triple_quotes_removed`

### `*_progress.json` (tradu├º├úo)
- `total_chunks`
- `translated_chunks`: lista de ├¡ndices
- `failed_chunks`: lista de ├¡ndices
- `timestamp`
- `doc_hash`
- `chunk_hashes`: mapa `{idx: hash}`
- `chunks`: mapa `{idx: texto traduzido}`

### `*_refine_report.json` (em `saida/`)
- `mode`: `"refine"`
- `input`
- `total_chunks`, `cache_hits`, `fallbacks`, `collapse_detected`, `duplicates_reused`
- `pipeline_version`, `timestamp`
- `refine_guardrails`, `effective_refine_chunk_chars`, `max_chunk_chars_observed`
- `dialogue_splits`, `triple_quotes_removed`

### `*_refine_metrics.json` (em `saida/`)
- `total_blocks`, `cache_hits`, `duplicates`, `fallbacks`, `collapse`
- `blocks`: m├®tricas por bloco
- `guardrails_mode`
- `cleanup_mode`, `cleanup_applied`, `cleanup_stats`, `cleanup_preview_hash_before/after`
- `effective_refine_chunk_chars`, `max_chunk_chars_observed`
- `dialogue_splits`, `triple_quotes_removed`

### `*_progress.json` (refine)
- `total_blocks`
- `refined_blocks`: lista de ├¡ndices
- `error_blocks`: lista de ├¡ndices
- `timestamp`
- `chunks`: mapa `{idx: texto refinado}`

### `state_refine.json`
- `input_file`, `hash`, `timestamp`, `total_chunks`, `refine_guardrails`

### Debugs
- `*_pt_chunks_debug.jsonl`: debug detalhado por chunk (tradu├º├úo).
- `*_chunks_debug.jsonl`: debug por chunk (refine).
- `saida/debug_refine*/`: arquivos `orig/raw/final` quando `--debug-refine`.

## Debug playbook (chunks/merge/cache)
1. **Reproduzir bug de chunking**
   - Rode com `--debug-chunks`.
   - Inspecione `saida/*_pt_chunks_debug.jsonl` (tradu├º├úo) ou `*_chunks_debug.jsonl` (refine).
2. **Verificar cache indevido**
   - Limpe com `--clear-cache all` ou apague `saida/cache_*`.
   - Rode novamente e compare `cache_hits` nos reports.
3. **Problemas de merge/recomposi├º├úo**
   - Compare `*_progress.json` (mapa `chunks`) com o arquivo final.
   - Procure discrep├óncias em `paragraph_mismatch` do report.
4. **Sa├¡das colapsadas ou contaminadas**
   - Verifique `collapse_detected` nos reports.
   - Procure entradas suspeitas nos JSONLs de debug.

## FAQ de manuten├º├úo
### Como adicionar um novo volume?
- Coloque o PDF em `data/`.
- Ajuste `config.yaml` se necess├írio (modelos, chunk sizes, pdf settings).
- Rode: `python -m tradutor.main traduz --input "data/arquivo.pdf"`.

### Como ajustar chunking?
- Em `config.yaml`, ajuste `translate_chunk_chars` e/ou `refine_chunk_chars`.
- Re-execute com `--clear-cache all` para evitar reuso de chunks antigos.

### Como trocar modelo/backend?
- Em `config.yaml`: `translate_backend`, `translate_model`, `refine_backend`, `refine_model`.
- Para Gemini, configure apenas a vari├ível `GEMINI_API_KEY` no ambiente.

### Como alterar pipeline (desquebrar/refine/pdf)?
- Use flags: `--no-refine`, `--use-desquebrar/--no-use-desquebrar`, `--pdf-enabled`.
- Para desquebrar ÔÇ£safeÔÇØ (sem LLM): `--desquebrar-mode safe`.

## Observa├º├Áes de incerteza
- N├úo encontrei configura├º├úo formal de lint/format (`pyproject.toml`, `setup.cfg`, `tox.ini`).
- N├úo encontrei pol├¡ticas de branch/commit. Caso existam, documente em `AGENTS.md`.
