# Outputs e Caches

## Arquivos principais (por etapa)
- Tradução (`tradutor/translate.py`):
  - `saida/<slug>_pt.md`
  - `<slug>_translate_report.json` (status + contagens)
  - `<slug>_translate_metrics.json` (por chunk, inclui `chunk_profile` e tamanho do contexto usado)
  - `<slug>_pt_progress.json` (resume)
  - `<slug>_source_sections.json` (metadados de estrutura usados na revisão final)
  - `<slug>_pt_review_report.json` (revisão final automática + QA)
- Repair seletivo (`tradutor/repair.py`, chamado pela tradução):
  - `<slug>_repair_report.json` (totais da etapa)
  - `<slug>_repair_metrics.json` (por chunk)
  - Com `--debug`: `debug_runs/<slug>/<run>/45_repair/repair_manifest.json`
  - Com `--debug`: `45_repair/debug_repair/chunkNNN_before_pt.txt` e `chunkNNN_after_pt.txt` para chunks reparados.
- Revisão bilíngue conservadora (`tradutor/bilingual_review.py`, chamada depois da tradução/repair):
  - `<slug>_bilingual_review_metrics.json` (tempo, chunks alterados, propostas aceitas, recusadas e focos de inspeção por chunk)
  - Com `--debug`: `debug_runs/<slug>/<run>/46_bilingual_review/bilingual_review_manifest.json`
  - Com `--debug`: `46_bilingual_review/chunkNNN_raw.txt` e `chunkNNN_review.json` para auditoria da proposta do revisor.
- Refine (`tradutor/refine.py`):
  - `saida/<slug>_pt_refinado.md`
  - `<slug>_refine_report.json`
  - `<slug>_refine_metrics.json`
  - `<slug>_pt_refinado_progress.json` (resume)
  - Opcional: `<slug>_pre_refine_cleanup.md` quando `cleanup_before_refine` aplica.
  - `<slug>_pt_refinado_review_report.json` (revisão final automática + QA)
- Desquebrar (`tradutor/desquebrar.py`):
  - `<slug>_desquebrar_metrics.json` (quando LLM é usado)
  - Arquivos `_raw_extracted.md`, `_preprocessed.md`, `_raw_desquebrado.md` se `--debug`.
- PDF:
  - `saida/pdf/<slug>_pt_refinado.pdf` (quando `--pdf-enabled` ou config).
- Tempos:
  - `saida/<slug>_timings.json` (sempre ao final de `traduz`/`traduz-md`, inclusive em falha após início do processamento)
  - Com `--debug`: `debug_runs/<slug>/<run>/99_reports/timings.json`
  - `stages.translate` inclui o repair seletivo e a revisão bilíngue; seus tempos aparecem também em `nested_stages.translation_repair` e `nested_stages.bilingual_review`, como detalhe sem dupla contagem.
  - `stages.post_translate_review` registra a revisão determinística após a tradução; `stages.post_refine_normalize` inclui a revisão final após o refine.

## Caches (`tradutor/cache_utils.py`)
- `saida/cache_traducao`
- `saida/cache_repair`
- `saida/cache_revisao_bilingue`
- `saida/cache_refine`
- `saida/cache_desquebrar`

Use `--clear-cache {all,translate,repair,review,refine,desquebrar}` para limpar.

## Debug / estado
- Tradução: `*_pt_chunks_debug.jsonl` (se `--debug-chunks`), `debug_traducao/` para falhas.
- Debug completo: `saida/debug_runs/<slug>/<timestamp>/40_translate/translate_manifest.json` inclui metadados do glossário por chunk; `debug_traducao/chunkNNN_glossary.txt` guarda o bloco de glossário enviado ao prompt.
- Repair: `saida/debug_runs/<slug>/<timestamp>/45_repair/repair_manifest.json` inclui problemas detectados, tentativas, cache, suspeitas e se o chunk foi alterado.
- Revisão bilíngue: `saida/debug_runs/<slug>/<timestamp>/46_bilingual_review/bilingual_review_manifest.json` inclui propostas, rejeições e justificativas de validação.
- Refine: `*_pt_refinado_chunks_debug.jsonl` (se `--debug-chunks`), `debug_refine*` quando `--debug-refine`.
- Estados rápidos: `saida/state_traducao.json`, `saida/state_refine.json`.
