# AGENTS.md ÔÇö Contexto r├ípido para o Codex

## Objetivo do projeto (5ÔÇô10 linhas)
- Pipeline completo para traduzir Light Novels de **EN ÔåÆ PT-BR** usando LLMs (Ollama ou Gemini).
- Fluxo cobre extra├º├úo de PDF/MD, limpeza, ÔÇ£desquebrarÔÇØ linhas, tradu├º├úo em chunks, refine/revis├úo e gera├º├úo de PDF.
- Configura├º├úo central em `config.yaml`, com overrides por flags de CLI.
- Sa├¡das e auditoria gravadas em `saida/` (markdown final, m├®tricas, manifests e PDFs).
- Gloss├írios manuais/din├ómicos s├úo suportados e injetados por chunk.
- Projeto prioriza uso em Windows, mas funciona em qualquer ambiente Python 3.10+.
- N├úo commitamos dados reais (gloss├írios, PDFs, chaves); use arquivos de exemplo.

## Mapa de pastas (o que ├® o qu├¬)
- `tradutor/` ÔÇö c├│digo principal do pipeline (CLI, tradu├º├úo, refine, PDF, utils).
- `data/` ÔÇö PDFs de entrada (n├úo versionar conte├║do real).
- `saida/` ÔÇö sa├¡das, caches e artefatos de debug (gerado em runtime).
- `glossario/` ÔÇö gloss├írios manuais (mant├®m s├│ exemplos no Git).
- `benchmark/` ÔÇö dados e scripts de benchmark.
- `tests/` ÔÇö testes unit├írios/smoke.
- `config.yaml` ÔÇö config principal (modelos, caminhos, flags padr├úo).
- `config.example.yaml` ÔÇö exemplo comentado.
- `desquebrar.py`, `tradutor.py`, `refinador.py` ÔÇö wrappers legados para CLI.

## Principais entrypoints/CLIs (com exemplos)
- CLI principal (traduz/refina/pdf):
  ```bash
  python -m tradutor.main traduz --input "data/meu_livro.pdf"
  python -m tradutor.main traduz-md --input "saida/meu_texto_desquebrado.md"
  python -m tradutor.main refina --input "saida/meu_livro_pt.md"
  python -m tradutor.main pdf --input "saida/meu_livro_pt_refinado.md"
  ```
- Desquebrar direto (wrapper legado):
  ```bash
  python desquebrar.py --input "arquivo.md" --output "arquivo_desquebrado.md" --config config.yaml
  ```
- Wrappers legados: `python tradutor.py ...`, `python refinador.py ...` (chamam `tradutor.main`).

## Testes, lint, checks
- Testes (smoke/unit):
  ```bash
  pytest -q
  ```
- Lint/format: **n├úo foi encontrado** no repo (procurei por `pyproject.toml`, `setup.cfg`, `tox.ini` e scripts no README; n├úo h├í). Se adicionar, documente aqui.

## Pipeline principal end-to-end (com flags ├║teis)
```bash
python -m tradutor.main traduz \
  --input "data/meu_livro.pdf" \
  --pdf-enabled \
  --translate-allow-adaptation \
  --request-timeout 180 \
  --num-predict 3072
```
- Flags comuns: `--skip-front-matter`, `--split-by-sections`, `--debug`, `--debug-chunks`, `--clear-cache {all,translate,refine,desquebrar}`.
- Modelos/ctx/num_predict podem ser configurados no `config.yaml` ou sobrescritos por flags.

## Caches, progresso, estado (e como limpar)
- Caches por chunk (em `saida/` por padr├úo):
  - `saida/cache_traducao`, `saida/cache_refine`, `saida/cache_desquebrar` (ver `tradutor/cache_utils.py`).
- Progress/state:
  - `*_progress.json` (tradu├º├úo/refine), `state_refine.json` (refine).
  - Debug: `*_pt_chunks_debug.jsonl`, `*_chunks_debug.jsonl`.
- Limpeza segura:
  - Prefer├¡vel: `--clear-cache all` na CLI.
  - Manual: apagar `saida/cache_*` e `saida/*_progress.json` quando iniciar um run limpo.
- Para evitar ÔÇ£cross-run contaminationÔÇØ:
  - Use `--clear-cache` ao mudar modelos/prompts.
  - Garanta que `output_dir` e `data_dir` (em `config.yaml`) sejam espec├¡ficos por projeto/volume.

## Conven├º├Áes do projeto
- Estilo: n├úo h├í guia formal; siga PEP 8 e o estilo existente (fun├º├Áes pequenas, logs com `logging`).
- Logs: use `logging.getLogger(__name__)` (padr├úo no c├│digo).
- Nomes de arquivo: sa├¡das seguem `<slug>_pt.md`, `<slug>_pt_refinado.md`, `*_metrics.json` etc. (ver `tradutor/translate.py`, `tradutor/refine.py`).
- Padr├úo de commit/branch: **n├úo encontrado** no repo (busquei em README/arquivos de config). Se definir, documente aqui.

## Regras de seguran├ºa
- **N├âO** inclua segredos (tokens, chaves, `.env`) em commits.
- Se encontrar `.env`, **n├úo** copie valores; apenas documente as vari├íveis esperadas.
- Vari├íveis esperadas observadas:
  - `GEMINI_API_KEY` (quando backend Gemini ├® usado).
