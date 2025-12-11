# Tradutor Literário com LLMs (Ollama/Gemini)

Pipeline completo EN → PT-BR com foco em previsibilidade e anti-alucinação para novels e PDFs. Compatível com Windows 11. Fluxo principal:
- **Passo 1 – Tradução** (`tradutor/main.py traduz`): lê PDFs, pré-processa (opcional avançado), chunking inteligente (fim de frase), contexto leve entre chunks, glossário manual opcional, guarda cache e gera `_pt.md`.
- **Passo 2 – Refine opcional** (`tradutor/main.py refina`): lê `_pt.md`, refina capítulo a capítulo com prompt minimalista, glossário manual/dinâmico opcional, guardrails e gera `_pt_refinado.md` + PDF.

Wrappers legados (`tradutor.py`, `refinador.py`) apenas chamam o CLI.

---

## Arquitetura
```
tradutor/
  main.py                # CLI traduz/refina + flags (debug, glossary, parallel, editor, preprocess advanced)
  config.py              # defaults (chunk=2400, timeout=60s, num_predict=768, repeat_penalty, paths)
  preprocess.py          # pré-processo básico + chunking por frase (lookahead)
  advanced_preprocess.py # limpeza opcional (hífens, tags estranhas, espaços)
  translate.py           # tradução em lotes, contexto leve, glossário manual opcional, cache
  refine.py              # refine minimalista por seção, glossário manual/dinâmico, guardrails, cache
  llm_backend.py         # Ollama/Gemini, retries, timeout, repeat_penalty
  sanitizer.py           # sanitização agressiva (trad) e leve (refine)
  anti_hallucination.py  # AAA shield anti-alucinação/repetição/estrutura (não altera semântica correta)
  cache_utils.py         # hash/cache por chunk, deduplicação, detecção de colapso
  glossary_utils.py      # carga manual/dinâmico, merge para prompt, parser de sugestões
  glossary/
    merge.py             # CLI para unir glossários em MASTER_GLOSSARIO.json
  intervolume.py         # QA opcional de consistência inter-volume (termos/gênero/voz/timeline)
  pdf_export.py          # PDF com ReportLab (fontes locais Aptos/Segoe/Calibri/Arial, fallback Helvetica)
  postprocess.py         # pós-processo PT-BR (travessões, espaços, marcadores)
  structure_normalizer.py# normaliza títulos/capítulos antes do PDF
  editor.py              # modos editor opcionais (lite/consistency/voice/strict) + relatório
  benchmark.py           # benchmark BLEU/chrF entre modelos
  bench_llms.py          # benchmark rápido no prompt de tradução
  bench_refine_llms.py   # benchmark rápido no prompt de refine
  pdf_reader.py          # extração de texto com PyMuPDF (fitz)
  utils.py               # logging, IO, helpers
  desquebrar.py          # utilitário para corrigir quebras de parágrafos
  VERSION                # versão interna do pipeline
data/                    # PDFs de entrada
saida/                   # saídas (_pt.md, _pt_refinado.md, glossário dinâmico, cache/state/report)
glossario/               # glossários manuais por volume
benchmark/               # amostras para benchmark
tests/                   # smoke tests
tradutor.py / refinador.py# wrappers legados
```

---

## Requisitos
- Python 3.10+
- Instalação: `pip install -r requirements.txt` (google-generativeai, PyMuPDF, reportlab, pyphen, requests, sacrebleu, PyYAML)
- Ollama (padrão) ou `GEMINI_API_KEY` para Gemini.

## Configuração
- Opcional `config.yaml` (exemplo em `config.example.yaml`): chunk sizes, timeouts, modelos, caminhos.
- Principais parâmetros: `translate_chunk_chars` / `refine_chunk_chars` (padrão 2400), `request_timeout` (60s), `translate_repeat_penalty` (1.1), `dump_chunks` (debug).

## Modelos padrão
- Tradução: backend `ollama`, modelo `brunoconterato/Gemma-3-Gaia-PT-BR-4b-it:f16`, temperatura `0.15`, chunk 2400.
- Refine: backend `ollama`, modelo `cnmoro/gemma3-gaia-ptbr-4b:q4_k_m`, temperatura `0.30`, chunk 2400.
- Retry: tradução 3 tentativas; refine 1 (fallback usa chunk original).

---

## Guardrails e segurança
- **Prompts minimalistas** (sem molduras): saídas delimitadas por `### TEXTO_TRADUZIDO_INICIO/FIM` e `### TEXTO_REFINADO_INICIO/FIM`; texto fora é ignorado.
- **Sanitização**: remove `<think>`, molduras, glossário sugerido, marcadores residuais, repetições e lixo; pós-processo PT ajusta travessões/espaços.
- **Anti-alucinação (AAA)**: checa idioma indevido, repetição, estrutura; na tradução não faz fallback para o inglês; no refine pode manter chunk original se suspeito.
- **Cache/estado/resume**: cache por hash em `saida/cache_traducao` e `saida/cache_refine`; `state_traducao.json` / `state_refine.json` para retomar; reutiliza duplicatas próximas.
- **Contexto leve**: usa a última frase do chunk anterior como referência (não traduzida) para dar continuidade.
- **Glossário**: manual opcional na tradução (injeta até 30 pares EN→PT); refine usa manual+dinâmico (sem duplicar `pt` do manual, filtros contra termos longos/descritivos).
- **Editor opcional**: modos lite/consistency/voice/strict pós-refine; gera `saida/editor_report.json` se solicitado (padrão OFF).

---

## Uso
### Tradução (PDF → PT-BR)
```bash
python -m tradutor.main traduz --input "data/meu_livro.pdf"
# com glossário manual
python -m tradutor.main traduz --use-glossary --manual-glossary "glossario/glossario_manual.json" --input "data/meu_livro.pdf"
# opções úteis
python -m tradutor.main traduz --backend gemini --model gemini-3-pro-preview
python -m tradutor.main traduz --resume --input "data/meu_livro.pdf"
python -m tradutor.main traduz --preprocess-advanced --debug
```
Saídas: `saida/<nome>_pt.md`, manifesto `<nome>_pt_progress.json`, cache/state/report em `saida/`. Debug salva raw/preprocessed.

### Refine (Markdown PT-BR)
```bash
python -m tradutor.main refina --input "saida/meu_livro_pt.md"
python -m tradutor.main refina --normalize-paragraphs --preprocess-advanced
python -m tradutor.main refina --use-glossary --manual-glossary "glossario/glossario_manual.json" --dynamic-glossary "saida/glossario_dinamico.json"
python -m tradutor.main refina --debug-refine
# editor opcional
python -m tradutor.main refina --editor-lite --editor-report
```
Saídas: `saida/<nome>_pt_refinado.md`, PDF correspondente, manifesto `<nome>_pt_refinado_progress.json`, cache/state/report em `saida/`, `glossario_dinamico.json` se habilitado.

### Mesclar glossários
```bash
python -m tradutor.glossario.merge --input glossario/ --output MASTER_GLOSSARIO.json
```
Conflitos em `saida/glossario_conflicts.log` (prioriza locked/manual).

### Checagem inter-volume (QA opcional)
```bash
python -m tradutor.intervolume --volumes "saida/" --glossario-dir "glossario/" --master-glossario "glossario/MASTER_GLOSSARIO.json" --output "saida/consistencia_intervolume.json"
# desativar checagens específicas
python -m tradutor.intervolume --volumes "saida/" --glossario-dir "glossario/" --no-check-gender --no-check-voice
```
Gera `saida/consistencia_intervolume.json` com inconsistências de termos, gênero, voz e timeline.

### Utilitário desquebrar (opcional)
```bash
python desquebrar.py --input "arquivo.md" --output "arquivo_corrigido.md" --model "llama3:8b"
```
Corrige quebras de parágrafo com cache de progresso e fallback.

---

## Sanitização e robustez adicionais
- Chunking seguro (2400 alvo) com lookahead para não cortar frases; contexto leve para transições.
- Guardrails de tamanho: tradução alerta se <70% do original (não faz fallback); refine troca por original se <80% ou >200%.
- Repetição e colapso: detecta loops/repetição; cache evita reprocessar; resume continua de onde parou.
- Anti-alucinação: bloqueia marcadores indevidos, CJK/fr/esp indevidos; no refine pode usar chunk original como fallback.

---

## Testes e benchmark
- Smoke tests: `pytest -q` (sem LLM real).
- Benchmark tradução: `python -m tradutor.benchmark` (usa `tests/benchmark_samples.json`).
- Benchmarks rápidos: `python -m tradutor.bench_llms --input benchmark/teste_traducao_en.md --max-chars 1500 --out-dir benchmark/traducao` e `python -m tradutor.bench_refine_llms --input benchmark/teste_refine_pt.md --max-chars 1500 --out-dir benchmark/refine`.

---

## Modelos recomendados (Ollama)
- Tradução: `brunoconterato/Gemma-3-Gaia-PT-BR-4b-it:f16`
- Refine: `cnmoro/gemma3-gaia-ptbr-4b:q4_k_m` (pode ser usado também para tradução se quiser saída mais conservadora).
