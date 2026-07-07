# Contexto do Projeto (humano) — Tradutor LLM

> Este documento foi montado após varredura de `README.md`, `tradutor/main.py`, `tradutor/translate.py`, `tradutor/refine.py`, `tradutor/cache_utils.py`, `tradutor/config.py` e diretórios `tests/`.
> Se algo parecer incompleto, confira os arquivos acima.

## Arquitetura (componentes e responsabilidades)
- **CLI principal**: `tradutor/main.py`
  - Subcomandos: `traduz`, `traduz-md`, `refina`, `pdf`.
  - Orquestra o pipeline e aplica flags/configs.
- **Config**: `tradutor/config.py`
  - Define defaults e carrega `config.yaml`.
  - Define `data_dir`, `output_dir`, modelos/temperaturas/ctx, PDF settings.
- **Tradução**: `tradutor/translate.py`
  - Pré-processamento, chunking, glossário por chunk, chamadas LLM, sanitização, métricas/report e manifesto de progresso.
- **Desquebrar**: `tradutor/desquebrar.py`, `tradutor/desquebrar_safe.py`
  - Une linhas quebradas (LLM ou heurística "safe").
- **Refine**: `tradutor/refine.py`
  - Revisão em chunks, guardrails, cleanup determinístico e pós-processo.
- **PDF**: `tradutor/pdf.py`, `tradutor/pdf_export.py`
  - Exporta Markdown para PDF usando ReportLab.
- **Utilitários**:
  - `tradutor/cache_utils.py` (cache por chunk), `tradutor/preprocess.py`, `tradutor/sanitizer.py` etc.
- **Wrappers legados**: `tradutor.py`, `refinador.py`, `desquebrar.py`.

## Fluxo de dados (alto nível)
1. **Extração**: PDF -> texto (`tradutor/pdf_reader.py`).
2. **Pré-processamento**: limpeza básica + remoção de front-matter/TOC (`tradutor/preprocess.py`).
3. **Desquebrar** (opcional): une linhas quebradas (LLM ou safe).
4. **Tradução**: chunking + LLM + sanitização + recomposição (`tradutor/translate.py`).
5. **Cleanup antes do refine** (opcional): heurísticas determinísticas.
6. **Refine**: revisão por chunk com guardrails (`tradutor/refine.py`).
7. **Pós-processamento**: correções finais e normalizadores.
8. **PDF** (opcional): exporta `*_pt_refinado.md` para PDF.

## Artefatos/JSONs principais (schemas resumidos)
> Campos listados abaixo foram coletados dos writers no código. Outros campos podem aparecer.

### `*_translate_report.json` (em `saida/`)
- `mode`: "translate"
- `status`: "ok" ou "failed"
- `input`: slug do documento
- `total_chunks`, `cache_hits`, `fallbacks`, `failed_chunks`, `collapse_detected`, `duplicates_reused`
- `pipeline_version`, `timestamp`
- `effective_translate_chunk_chars`, `max_chunk_chars_observed`
- `dialogue_splits`, `triple_quotes_removed`
- `paragraph_mismatch` (opcional)

### `*_translate_metrics.json` (em `saida/`)
- `total_chunks`, `cache_hits`, `duplicates_reused`, `fallbacks`, `failed_chunks`, `collapse_detected`
- `chunks`: métricas por chunk (ver `tradutor/translate.py`)
- `effective_translate_chunk_chars`, `max_chunk_chars_observed`
- `dialogue_splits`, `triple_quotes_removed`

### `*_progress.json` (tradução)
- `total_chunks`
- `translated_chunks`: lista de índices
- `failed_chunks`: lista de índices
- `timestamp`
- `doc_hash`
- `chunk_hashes`: mapa `{idx: hash}`
- `chunks`: mapa `{idx: texto traduzido}`

### `*_refine_report.json` (em `saida/`)
- `mode`: "refine"
- `input`
- `total_chunks`, `cache_hits`, `fallbacks`, `collapse_detected`, `duplicates_reused`
- `pipeline_version`, `timestamp`
- `refine_guardrails`, `effective_refine_chunk_chars`, `max_chunk_chars_observed`
- `dialogue_splits`, `triple_quotes_removed`

### `*_refine_metrics.json` (em `saida/`)
- `total_blocks`, `cache_hits`, `duplicates`, `fallbacks`, `collapse`
- `blocks`: métricas por bloco
- `guardrails_mode`
- `cleanup_mode`, `cleanup_applied`, `cleanup_stats`, `cleanup_preview_hash_before/after`
- `effective_refine_chunk_chars`, `max_chunk_chars_observed`
- `dialogue_splits`, `triple_quotes_removed`

### `*_progress.json` (refine)
- `total_blocks`
- `refined_blocks`: lista de índices
- `error_blocks`: lista de índices
- `timestamp`
- `chunks`: mapa `{idx: texto refinado}`

### `state_refine.json`
- `input_file`, `hash`, `timestamp`, `total_chunks`, `refine_guardrails`

### Debugs
- `*_pt_chunks_debug.jsonl`: debug detalhado por chunk (tradução).
- `*_chunks_debug.jsonl`: debug por chunk (refine).
- `saida/debug_refine*/`: arquivos `orig/raw/final` quando `--debug-refine`.
- `saida/debug_runs/<slug>/<timestamp>/40_translate/translate_manifest.json`: inclui metadados de glossário por chunk; `debug_traducao/chunkNNN_glossary.txt` mostra o bloco enviado ao prompt.

## Debug playbook (chunks/merge/cache)
1. **Auditar preprocess antes de tradução longa**
   - Se já houver `saida/<slug>_raw_extracted.md`, rode `python scripts/audit_preprocess.py --input "saida/<slug>_raw_extracted.md"`.
   - Inspecione `saida/preprocess_audit/<slug>_audit.md`: `suspicious_removed` e `raw_missing_candidates` devem ser propaganda/TOC, não narrativa.
2. **Reproduzir bug de chunking**
   - Rode com `--debug-chunks`.
   - Inspecione `saida/*_pt_chunks_debug.jsonl` (tradução) ou `*_chunks_debug.jsonl` (refine).
3. **Verificar cache indevido**
   - Limpe com `--clear-cache all` ou apague `saida/cache_*`.
   - Rode novamente e compare `cache_hits` nos reports.
4. **Problemas de merge/recomposição**
   - Compare `*_progress.json` (mapa `chunks`) com o arquivo final.
   - Procure discrepâncias em `paragraph_mismatch` do report.
5. **Saídas colapsadas ou contaminadas**
   - Verifique `collapse_detected` nos reports.
   - Procure entradas suspeitas nos JSONLs de debug.

## FAQ de manutenção
### Como adicionar um novo volume?
- Coloque o PDF em `data/`.
- Ajuste `config.yaml` se necessário (modelos, chunk sizes, pdf settings).
- Rode: `python -m tradutor.main traduz --input "data/arquivo.pdf"`.

### Como ajustar chunking?
- Em `config.yaml`, ajuste `translate_chunk_chars` e/ou `refine_chunk_chars`.
- Re-execute com `--clear-cache all` para evitar reuso de chunks antigos.

### Como trocar modelo/backend?
- Em `config.yaml`: `translate_backend`, `translate_model`, `refine_backend`, `refine_model`.
- Para Gemini, configure apenas a variável `GEMINI_API_KEY` no ambiente.

### Como alterar pipeline (desquebrar/refine/pdf)?
- Use flags: `--no-refine`, `--use-desquebrar/--no-use-desquebrar`, `--pdf-enabled`.
- Para desquebrar "safe" (sem LLM): `--desquebrar-mode safe`.

## Atualizações recentes (pipeline v2)
- Desquebrar: pós-processo determinístico junta hard-wraps seguros, corrige hífen silábico quando a forma sem hífen domina no texto e isola `***`; métricas extras nos stats.
- Cleanup pré-refine: dedupe não remove falas/onomatopeias curtas (Crack!, "— ?", "— …") a menos que haja glitch claro (>=4 repetições consecutivas).
- Tradução/QA: retries para aspas desbalanceadas ou repetições extras de linhas curtas; pós-processo remove aspas curvas sobrando em falas iniciadas por travessão.
- Glossário: suporte a `enforce` por termo; "Lord of the Flies" -> "Senhor das Moscas" aplicado automaticamente quando o termo aparece no chunk.
- Baseline local atual: tradução com `mistral-small3.2:24b-instruct-2506-q4_K_M`, desquebrar com `gemma3:27b-it-q4_K_M`, refine com `gemma4:26b-a4b-it-q4_K_M`, Ollama em modo `chat` e `ollama_think: false`.
- Glossário: loader separa `source_aliases`/`aliases` de busca, `bad_aliases` proibidos e `allowed_target_aliases`; `--use-glossary` resolve `glossario/glossario_manual.json` ou `glossario/glossario_geral.json`.
- Glossário: `scripts/audit_glossary.py` audita duplicatas, aliases ambíguos, aliases PT-BR em busca e redundâncias; `scripts/migrate_glossary_aliases.py --write` limpa o glossário local sem versionar dados reais.
- Revisão pós-tradução determinística: `scripts/review_translation.py` restaura headings a partir de `sections.json` e aplica correções conservadoras de glossário/resíduos sem reescrever o estilo.
- QA objetiva: `tradutor/quality_checks.py` checa termos canônicos, aliases ruins, inglês residual, marcadores internos, aspas e sinais simples de gênero.
- Benchmarks: `bench_llms`, `bench_refine_llms` e `bench_e2e_llms` usam o pipeline real, geram `*_qa.json` e aceitam glossário.
- Normalizadores: interjeições comuns (`Phew`, `Geez`, `Huh`, `Ugh`) são normalizadas para PT-BR; atribuições quebradas em travessão (`— perguntou...`) são reunidas ao diálogo anterior.
- Preprocess: `scripts/audit_preprocess.py` gera auditoria isolada do raw extraído, incluindo remoções suspeitas e linhas do raw que não aparecem no texto limpo.

## Observações de incerteza
- Não encontrei configuração formal de lint/format (`pyproject.toml`, `setup.cfg`, `tox.ini`).
- Não encontrei políticas de branch/commit. Caso existam, documente em `AGENTS.md`.
- Revisão literária inicial sugeriu que `mistral-small3.1` pode preservar diálogo melhor em alguns trechos, apesar da baseline objetiva/e2e favorecer `mistral-small3.2`. Não trocar o default sem nova matriz literária.
