# Tradutor de Light Novels (EN → PT-BR) – Windows-friendly

Este projeto é um **pipeline completo** para traduzir Light Novels em PDF/Markdown usando LLMs (Ollama ou Gemini). Ele cuida de tudo: extração do texto, limpeza, “desquebrar” (unir linhas quebradas), tradução, refine (revisão automática) e geração de PDF final.

> **Para iniciantes:** siga a seção **Passo a passo rápido**. Toda a configuração central fica no `config.yaml`.

---

## Passo a passo rápido
1) Instale dependências:
```bash
pip install -r requirements.txt
```
2) Ajuste o `config.yaml` (modelos, caminhos, fonte do PDF). **Padrão:** Ollama rodando localmente.
3) Coloque seus PDFs em `data/`.
4) Rode a tradução completa (com refine; o PDF só sai se estiver habilitado).
   - Flags úteis para PDFs longos: `--skip-front-matter` (padrão), `--split-by-sections` (padrão), `--translate-allow-adaptation`, `--request-timeout`, `--num-predict`.
   - Se usar Ollama, ajuste também `translate_num_ctx`/`refine_num_ctx`/`desquebrar_num_ctx` e `ollama_keep_alive` no `config.yaml`.
```bash
python -m tradutor.main traduz --input "data/meu_livro.pdf"
```
5) Confira as saídas em `saida/`:
   - `<slug>_pt.md` (tradução)
   - `<slug>_pt_refinado.md` (refine)
   - `pdf/<slug>_pt_refinado.pdf` (se `pdf_enabled: true` ou `--pdf-enabled`)
   - métricas/manifestos para auditoria.

---

## O que o pipeline faz (em linguagem simples)
 - **Pré-processa** o PDF (limpa lixo básico, remove front-matter/TOC quando habilitado).
 - **Desquebra** linhas (une linhas quebradas; usa LLM se configurado).
 - **Traduz** EN → PT-BR com contexto leve e glossário opcional.
 - **Limpa o texto** antes do refine (remove duplicatas, colagens e artefatos).
 - **Refina** o PT-BR com guardrails (revisão automática).
 - **Gera PDF** com fonte configurável (ReportLab).

---

## Configuração (config.yaml)
Principais chaves (com valores padrão já preenchidos):
- `translate_*`: backend/model (ex.: `gemma3:27b-it-q4_K_M`), temperatura/repeat_penalty/chunk_chars/num_predict/num_ctx e guardrails de diálogo (`translate_dialogue_guardrails`, `translate_dialogue_retry_temps`, `translate_dialogue_split_fallback`). Glossário contextual: `translate_glossary_match_limit`/`translate_glossary_fallback_limit`. `translate_allow_adaptation` deixa exemplos de adaptação de piadas no prompt.
- `use_desquebrar` (true/false) e `desquebrar_*` (backend/model/temp/repeat_penalty/chunk/num_predict/num_ctx) + `desquebrar_mode` (`safe` usa desquebrar_safe sem LLM).
- `refine_backend`, `refine_model` (ex.: `mistral-small3.1:24b-instruct-2503-q4_K_M`), `refine_temperature`, `refine_guardrails`, `cleanup_before_refine` (off/auto/on).
- `fail_on_chunk_error`: se true, aborta a tradução na primeira falha de chunk; se false (default), salva placeholders e segue.
- `ollama_keep_alive`: mantém o modelo carregado entre chamadas (ex.: `30m`).
- PDF: `pdf_enabled` (padrão false; habilite no config ou com `--pdf-enabled`), `pdf_font.file/size/leading`, `pdf_font_fallbacks`, `pdf_margin`, `pdf_author`, `pdf_language`.
- Caminhos: `data_dir`, `output_dir`.
- Robustez contra TOC: `skip_front_matter: true` (default) ativa heurística de remoção de sumário inicial (`strip_toc`); headings vazios (# Prologue/# Chapter N) são mesclados/pulados antes de chamar o LLM.

> O desquebrar usa exatamente o modelo/backend definidos em `config.yaml`; nada hardcoded.

---

## 📘 Glossários
- Glossários são dados editoriais específicos de cada projeto/obra. Não versione glossários reais.
- Exemplo de referência: `glossario/glossario_exemplo.json` (15 termos genéricos com campos `term`, `translation`, `type`, `locked`, `notes`, `aliases`).
- Estrutura básica (JSON):
  - `term`: termo de origem.
  - `translation`: tradução fixa.
  - `type`: categoria (ex.: creature, place, magic, title, organization, item, event).
  - `locked`: true/false para fixar a tradução.
  - `notes`: observação opcional.
  - `aliases`: lista opcional de variações do termo.
- Para criar o seu: copie `glossario_exemplo.json`, edite os termos e aponte no CLI com `--manual-glossary <seu_glossario.json>`.
- Todos os `glossario/*.json` são ignorados no Git, exceto `glossario_exemplo.json`.
- Na tradução, só são injetados no prompt os termos que aparecem no chunk (limite configurável por `translate_glossary_match_limit`; fallback para `translate_glossary_fallback_limit` quando nada casa).

---

## Comandos principais e flags
> Dica para iniciantes: copie e cole os comandos abaixo e mude apenas o caminho do arquivo.

### Traduzir PDF → PT-BR (com refine e PDF, se habilitado)
```bash
python -m tradutor.main traduz --input "data/meu_livro.pdf"
```
Flags (todas opcionais):
- `--backend {ollama,gemini}` / `--model <nome>`: override de backend/modelo de tradução.
- `--num-predict <int>`: tokens máximos por chunk na tradução.
- `--no-refine`: pula o refine (gera só `<slug>_pt.md`).
- `--desquebrar-mode {llm,safe}`: `safe` usa desquebrar_safe (sem LLM) no passo de desquebrar, preservando layout. Alias legado: `--refine-mode`.
- `--resume`: retoma a partir do manifesto de progresso da tradução.
- `--clear-cache {all,translate,refine,desquebrar}`: limpa caches antes de rodar.
- `--use-glossary`: injeta glossário manual (JSON) na tradução.
- `--manual-glossary <path>`: caminho do glossário manual (default `glossario/glossario_manual.json`).
- `--parallel <n>`: workers paralelos (tradução força ordem; >1 pode ser limitado).
- `--preprocess-advanced`: limpeza extra antes de traduzir.
- `--cleanup-before-refine {off,auto,on}`: força/auto/desliga cleanup antes do refine.
- `--use-desquebrar` / `--no-use-desquebrar`: ativa/desativa desquebrar pré-tradução (default vem do config).
- `--desquebrar-backend/model/temperature/repeat-penalty/chunk-chars/num-predict`: overrides específicos do desquebrar.
- `--debug`: salva artefatos intermediários (`*_raw_extracted.md`, `*_preprocessed.md`, `*_raw_desquebrado.md`).
- `--debug-chunks`: JSONL detalhado por chunk.
- `--fail-on-chunk-error`: aborta na primeira falha de chunk (padrão é continuar com placeholders).
- `--pdf-enabled` / `--no-pdf-enabled`: liga/desliga PDF automático após refine (se refine estiver ativo).
- `--request-timeout <s>`: timeout por chamada de modelo.

### Traduzir Markdown já desquebrado (pula extração e desquebrar)
```bash
python -m tradutor.main traduz-md --input "saida/meu_texto_desquebrado.md"
```
Flags principais (opcionais):
- `--backend {ollama,gemini}` / `--model <nome>` / `--num-predict <int>`
- `--no-refine` para só gerar `<slug>_pt.md`
- `--use-glossary` / `--manual-glossary <path>`
- `--normalize-paragraphs` (normaliza parágrafos do MD antes de traduzir)
- `--translate-allow-adaptation` / `--debug-chunks` / `--pdf-enabled`
- `--fail-on-chunk-error` (interrompe se algum chunk falhar)

### Refine separado em um Markdown PT-BR
```bash
python -m tradutor.main refina --input "saida/meu_livro_pt.md"
```
Flags:
- `--backend {ollama,gemini}` / `--model <nome>`: override de refine.
- `--num-predict <int>`: tokens máximos por chunk no refine.
- `--desquebrar-mode {llm,safe}`: compatível com `traduz` (safe usa desquebrar_safe sem LLM). No comando `refina`, não altera o fluxo. Alias legado: `--refine-mode`.
- `--resume`: retoma a partir do manifesto de refine.
- `--clear-cache {all,translate,refine,desquebrar}`: limpa caches antes de refinar.
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
- Limpezas extras/robustez: `--skip-front-matter/--no-skip-front-matter` (pula front-matter antes de Prologue/Chapter 1), `--split-by-sections/--no-split-by-sections` (tradução por seção), `--translate-allow-adaptation/--no-translate-allow-adaptation` (permite exemplos de adaptação de piadas no prompt).

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
- Backend:
  - **Ollama (padrão):** precisa estar rodando localmente.
  - **Gemini:** defina `GEMINI_API_KEY` no ambiente.

---

## Dicas rápidas
- Modelos sugeridos (Ollama):
  - Tradução: `gemma3:27b-it-q4_K_M`
  - Desquebrar: use o mesmo ou outro em `config.yaml`.
  - Refine: `mistral-small3.1:24b-instruct-2503-q4_K_M`
- Para PDFs longos: mantenha `translate_chunk_chars` em ~2000 e `translate_num_predict` em ~3000 (Ollama). Guardrails de diálogo (`translate_dialogue_guardrails`) ajudam a evitar omissão de falas.
- Se fonte do PDF não existir, ajuste `pdf_font.file` ou use um fallback válido (ex.: `C:/Windows/Fonts/Arial.ttf`).
- Se o debug mostrar `chunk001_original_en.txt` só com `# Prologue`, habilite `skip_front_matter` (já é padrão); o pipeline agora remove TOC curto e não envia headings vazios para o modelo.

---

## Testes
- Smoke: `pytest -q` (usa Fakes/stubs; não chama LLM real).
- Benchmarks opcionais em `benchmark/` e comandos `bench_llms`/`bench_refine_llms`.
