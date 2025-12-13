# Tradutor de Light Novels (EN → PT-BR) – Windows-friendly

Pipeline completo para traduzir PDFs/Markdown usando LLMs (Ollama/Gemini), com etapas automáticas de limpeza, desquebrar, tradução, refine e PDF final.

> Para iniciantes: siga a seção **Passo a passo rápido**. Tudo é configurado pelo `config.yaml`.

---

## Passo a passo rápido
1) Instale dependências:
```bash
pip install -r requirements.txt
```
2) Ajuste o `config.yaml` (modelos, caminhos, fonte do PDF). Padrão: Ollama rodando localmente.
3) Coloque seus PDFs em `data/`.
4) Rode a tradução completa (com refine; PDF é opcional e só sai se estiver habilitado):
```bash
python -m tradutor.main traduz --input "data/meu_livro.pdf"
```
5) Saídas em `saida/`:
   - `<slug>_pt.md` (tradução)
   - `<slug>_pt_refinado.md` (refine)
   - `pdf/<slug>_pt_refinado.pdf` (apenas se `pdf_enabled: true` ou flag `--pdf-enabled`)
   - métricas/manifestos para auditoria.

---

## O que o pipeline faz
- **Pré-processa** o PDF (limpa lixo básico).
- **Desquebra** linhas (usa LLM configurado em `desquebrar_*`).
- **Traduz** EN → PT-BR com contexto leve e glossário opcional.
- **Cleanup antes do refine** (remove duplicatas/glued, etc).
- **Refina** o PT-BR com guardrails.
- **Gera PDF** com fonte configurável (ReportLab).

---

## Configuração (config.yaml)
Principais chaves (padrões já preenchidos):
- `translate_backend`, `translate_model` (ex.: `gemma3:27b-it-q4_K_M`), `translate_temperature`, `translate_repeat_penalty`, `translate_chunk_chars`, `translate_num_predict`.
- `use_desquebrar` (true/false) e `desquebrar_*` (backend/model/temp/repeat_penalty/chunk/num_predict).
- `refine_backend`, `refine_model` (ex.: `mistral-small3.1:24b-instruct-2503-q4_K_M`), `refine_temperature`, `refine_guardrails`, `cleanup_before_refine` (off/auto/on).
- PDF: `pdf_enabled` (padrão false; habilite no config ou com `--pdf-enabled`), `pdf_font.file/size/leading`, `pdf_font_fallbacks`, `pdf_margin`, `pdf_author`, `pdf_language`.
- Caminhos: `data_dir`, `output_dir`.

> O desquebrar usa exatamente o modelo/backend definidos em `config.yaml`; nada hardcoded.

---

## Comandos principais e flags

### Traduzir PDF → PT-BR (com refine e PDF, se habilitado)
```bash
python -m tradutor.main traduz --input "data/meu_livro.pdf"
```
Flags (todas opcionais):
- `--backend {ollama,gemini}` / `--model <nome>`: override de backend/modelo de tradução.
- `--num-predict <int>`: tokens máximos por chunk na tradução.
- `--no-refine`: pula o refine (gera só `<slug>_pt.md`).
- `--resume`: retoma a partir do manifesto de progresso da tradução.
- `--use-glossary`: injeta glossário manual (JSON) na tradução.
- `--manual-glossary <path>`: caminho do glossário manual (default `glossario/glossario_manual.json`).
- `--parallel <n>`: workers paralelos (tradução força ordem; >1 pode ser limitado).
- `--preprocess-advanced`: limpeza extra antes de traduzir.
- `--cleanup-before-refine {off,auto,on}`: força/auto/desliga cleanup antes do refine.
- `--use-desquebrar` / `--no-use-desquebrar`: ativa/desativa desquebrar pré-tradução (default vem do config).
- `--desquebrar-backend/model/temperature/repeat-penalty/chunk-chars/num-predict`: overrides específicos do desquebrar.
- `--debug`: salva artefatos intermediários (`*_raw_extracted.md`, `*_preprocessed.md`, `*_raw_desquebrado.md`).
- `--debug-chunks`: JSONL detalhado por chunk.
- `--pdf-enabled` / `--no-pdf-enabled`: liga/desliga PDF automático após refine (se refine estiver ativo).
- `--request-timeout <s>`: timeout por chamada de modelo.

### Refine separado em um Markdown PT-BR
```bash
python -m tradutor.main refina --input "saida/meu_livro_pt.md"
```
Flags:
- `--backend {ollama,gemini}` / `--model <nome>`: override de refine.
- `--num-predict <int>`: tokens máximos por chunk no refine.
- `--resume`: retoma a partir do manifesto de refine.
- `--normalize-paragraphs`: normaliza parágrafos antes de refinar.
- `--use-glossary`: ativa glossário manual/dinâmico.
- `--manual-glossary <path>` / `--dynamic-glossary <path>` / `--auto-glossary-dir <dir>`: fontes de glossário.
- `--debug-refine`: salva debug dos primeiros chunks de refine.
- `--parallel <n>`: workers paralelos (ordem preservada na montagem).
- `--preprocess-advanced`: limpeza extra antes do refine.
- `--cleanup-before-refine {off,auto,on}`: modo de cleanup determinístico.
- `--debug-chunks`: JSONL detalhado por chunk.
- Editor opcional pós-refine: `--editor-lite`, `--editor-consistency`, `--editor-voice`, `--editor-strict`, `--editor-report` (gera `editor_report.json`).
- `--request-timeout <s>`: timeout por chamada.

### Gerar PDF a partir de um Markdown existente
```bash
python -m tradutor.main pdf --input "saida/meu_livro_pt_refinado.md"
```
Usa as configs de fonte/margem do `config.yaml`. Sem flags adicionais além de `--debug` (para logs verbosos).

### Usar desquebrar direto em um arquivo
```bash
python desquebrar.py --input "arquivo.md" --output "arquivo_desquebrado.md" --config config.yaml
```
Flags: `--config` (opcional), `--debug` (logs). As demais configs vêm do `config.yaml`.
---

## Estrutura de pastas
```
tradutor/
  main.py             # CLI principal (traduz/refina/pdf)
  translate.py        # pipeline de tradução em chunks
  desquebrar.py       # função de desquebrar usada no pipeline
  refine.py           # refine e cleanup determinístico
  pdf.py              # conversor Markdown → PDF (ReportLab)
  config.py           # carrega/mescla config.yaml
  cleanup.py          # heurísticas determinísticas (dedupe, prefixos)
  preprocess.py       # pré-processo de PDFs e chunking seguro
  advanced_preprocess.py # limpeza opcional extra
  sanitizer.py        # sanitização de saída LLM
  anti_hallucination.py # filtros AAA anti-alucinação/repetição
  cache_utils.py      # cache/hash por chunk, resume
  glossary_utils.py   # carga/merge/glossário dinâmico
  pdf_reader.py       # extração de texto de PDF (fitz)
  pdf_export.py       # exportador PDF legado (ReportLab)
  postprocess.py      # ajustes finais em PT-BR
  structure_normalizer.py # normaliza títulos/cabeçalhos
  editor.py           # modos editor opcionais (lite/consistency/voice/strict)
  llm_backend.py      # cliente LLM (Ollama/Gemini)
  benchmark.py        # benchmark BLEU/chrF
  bench_llms.py       # benchmark rápido de tradução
  bench_refine_llms.py# benchmark rápido de refine
  VERSION             # versão interna do pipeline

data/                 # PDFs de entrada
saida/
  cache_*             # caches de tradução/refine/desquebrar
  pdf/                # PDFs finais gerados
  *_pt.md             # tradução
  *_pt_refinado.md    # refine
  *metrics.json       # métricas de cada etapa
  *progress.json      # manifestos de progresso
  glossario_dinamico.json # se glossário dinâmico estiver ativo

glossario/            # glossários manuais por volume
benchmark/            # insumos para benchmarks
tests/                # testes (smoke e unitários)
config.yaml           # configuração central (modelos, fontes, caminhos)
config.example.yaml   # exemplo de configuração comentado
desquebrar.py         # wrapper CLI para desquebrar direto
tradutor.py / refinador.py # wrappers legados (chamam main)
```

---

## Saídas e auditoria
- Tradução: `saida/<slug>_pt.md` + métricas `*_translate_metrics.json` + `report.json`.
- Refine: `saida/<slug>_pt_refinado.md` + `*_refine_metrics.json`.
- Desquebrar (se debug): `*_raw_extracted.md`, `*_raw_desquebrado.md`, métricas `*_desquebrar_metrics.json`.
- PDF: `saida/pdf/<slug>_pt_refinado.pdf` (quando `pdf_enabled: true`).
- Manifestos de progresso: `*_progress.json` (trad/refine).

---

## Requisitos
- Windows 11 (prioritário), Python 3.10+.
- Dependências: `pip install -r requirements.txt`.
- Backend: Ollama (padrão) ou Gemini (`GEMINI_API_KEY` no ambiente).

---

## Dicas rápidas
- Modelos sugeridos (Ollama):
  - Tradução: `gemma3:27b-it-q4_K_M`
  - Desquebrar: use o mesmo ou outro em `config.yaml`.
  - Refine: `mistral-small3.1:24b-instruct-2503-q4_K_M`
- Para evitar truncamentos: mantenha `translate_chunk_chars` em ~2400.
- Se fonte do PDF não existir, ajuste `pdf_font.file` ou use um fallback válido (ex.: `C:/Windows/Fonts/Arial.ttf`).

---

## Testes
- Smoke: `pytest -q` (usa Fakes/stubs; não chama LLM real).
- Benchmarks opcionais em `benchmark/` e comandos `bench_llms`/`bench_refine_llms`.
