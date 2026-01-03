# Outputs e Caches

## Arquivos principais (por etapa)
- Tradução (`tradutor/translate.py`):
  - `saida/<slug>_pt.md`
  - `<slug>_translate_report.json` (status + contagens)
  - `<slug>_translate_metrics.json` (por chunk)
  - `<slug>_pt_progress.json` (resume)
- Refine (`tradutor/refine.py`):
  - `saida/<slug>_pt_refinado.md`
  - `<slug>_refine_report.json`
  - `<slug>_refine_metrics.json`
  - `<slug>_pt_refinado_progress.json` (resume)
  - Opcional: `<slug>_pre_refine_cleanup.md` quando `cleanup_before_refine` aplica.
- Desquebrar (`tradutor/desquebrar.py`):
  - `<slug>_desquebrar_metrics.json` (quando LLM é usado)
  - Arquivos `_raw_extracted.md`, `_preprocessed.md`, `_raw_desquebrado.md` se `--debug`.
- PDF:
  - `saida/pdf/<slug>_pt_refinado.pdf` (quando `--pdf-enabled` ou config).

## Caches (`tradutor/cache_utils.py`)
- `saida/cache_traducao`
- `saida/cache_refine`
- `saida/cache_desquebrar`

Use `--clear-cache {all,translate,refine,desquebrar}` para limpar.

## Debug / estado
- Tradução: `*_pt_chunks_debug.jsonl` (se `--debug-chunks`), `debug_traducao/` para falhas.
- Refine: `*_pt_refinado_chunks_debug.jsonl` (se `--debug-chunks`), `debug_refine*` quando `--debug-refine`.
- Estados rápidos: `saida/state_traducao.json`, `saida/state_refine.json`.
