# Tradutor Literario com LLMs (Ollama/Gemini)

Pipeline de traducao/refine para novels adultas e PDFs em PT-BR, com foco em **zero alucinacao**. Fluxo principal:
- **Passo 1 - Traducao** (`tradutor/main.py` subcomando `traduz`): le PDFs, pre-processa, faz chunking seguro, traduz em lotes e gera Markdown.
- **Passo 2 - Refine opcional** (`tradutor/main.py` subcomando `refina`): le `*_pt.md` em `saida/`, detecta capitulos (`## `) e refina capitulo a capitulo gerando `*_pt_refinado.md` sem sobrescrever o original.

Compatibilidade: Windows 11. Wrappers legados (`tradutor.py`, `refinador.py`) apenas redirecionam para o novo CLI.

Prompts especializados para novels:
- Traducao: prompt adulto (dark fantasy, violencia/blasfemia), preserva tom, genero, numero, sem suavizar.
- Refine: prompt adulto para fluidez/coerencia, sem cortar conteudo, corrige genero/plural/naturalidade.
- Sanitizacao: remove `<think>` e meta-texto em PT/EN (ex.: "as an ai language model").

---

## Arquitetura
```
tradutor/
  __init__.py
  main.py          # CLI com subcomandos traduz/refina
  config.py        # constantes e paths padrao
  preprocess.py    # extracao de PDF, limpeza, chunking seguro
  translate.py     # prompt e pipeline de traducao por lotes
  refine.py        # refine capitulo a capitulo em Markdown
  llm_backend.py   # chamadas Ollama/Gemini com retry
  sanitizer.py     # sanitizacao agressiva contra alucinacao/loops/meta
  pdf_export.py    # conversao Markdown -> PDF (desativada no fluxo principal)
  pdf_reader.py    # extracao de texto com PyMuPDF (fitz)
  benchmark.py     # benchmark BLEU/chrF entre modelos
  bench_llms.py    # benchmark rapido de modelos Ollama no prompt de traducao
  bench_refine_llms.py # benchmark rapido de modelos Ollama no prompt de refine
  utils.py         # logging, IO, helpers
tests/
  benchmark_samples.json  # amostras para benchmark
  test_smoke_translation.py
```

---

## Requisitos
- Python 3.10+
- Instale dependencias:
  ```bash
  pip install -r requirements.txt
  ```
 Principais libs: `google-generativeai`, `PyMuPDF`, `fpdf`, `requests`, `sacrebleu`.
- Ollama instalado (padrao). Para Gemini, defina `GEMINI_API_KEY`.

---

## Modelos e parametros padrao
- Traducao: backend `ollama`, modelo `huihui_ai/qwen3-abliterated:14b-q4_K_M`, temperatura `0.15`, chunk `3800` caracteres.
- Refine: backend `ollama`, modelo `gemma3-gaia-ptbr-4b:q4_k_m`, temperatura `0.30`, chunk `10000` caracteres.
- Retry: 3 tentativas, backoff exponencial.
- Sanitizacao: remove `<think>...</think>`, meta-comentarios (PT/EN), repeticoes/loops e respostas vazias ou contaminadas (falha e re-tenta).

---

## Como usar

### Passo 1 - Traduzir PDFs
```bash
python -m tradutor.main traduz
# ou um PDF especifico
python -m tradutor.main traduz --input "data/meu_livro.pdf"
# modo debug (salva raw/preprocessed)
python -m tradutor.main traduz --debug --input "data/meu_livro.pdf"
# usando Gemini
python -m tradutor.main traduz --backend gemini --model gemini-3-pro-preview
# pular refine automatico ao final
python -m tradutor.main traduz --no-refine

# modo legados (wrappers):
python tradutor.py --input "data/meu_livro.pdf"
```
Saidas na pasta `saida/` (sem PDF):
- Sempre: `<nome>_pt.md`
- Se debug: `<nome>_raw_extract.md`, `<nome>_preprocessed.md`
- Se refine habilitado: `<nome>_pt_refinado.md`

### Passo 2 - Refine opcional (capitulo a capitulo)
```bash
# refina todos os *_pt.md em saida/
python -m tradutor.main refina

# refina um arquivo especifico
python -m tradutor.main refina --input "saida/MEU_ARQUIVO_pt.md"

# modo legados (wrappers):
python refinador.py --input "saida/MEU_ARQUIVO_pt.md"
```
Saidas:
```
saida/MEU_ARQUIVO_pt_refinado.md
saida/MEU_ARQUIVO_pt_refinado.pdf
```
O original `*_pt.md` nunca e sobrescrito.

---

## Sanitizacao e robustez
- Remove `<think>`/`</think>`, meta-texto (inclusive ingles), loops e blocos repetidos.
- Falha e re-tenta em caso de contaminacao ou resposta vazia.
- Chunking seguro (3800/10000 chars) com cortes duros se necessario; evita chunks gigantes.
- Logs detalhados por chunk, sanitizacao e tempo de processamento.
- Logs informam backend/model/temperatura/chunk usados em traducao e refine (inclusive opcional).

---

## Testes e benchmark
- Smoke test sem LLM real (backend fake e stub de PyMuPDF): `pytest -q`
- Benchmark (custa tempo/tokens, chama LLM real): `python -m tradutor.benchmark`
  - Usa `tests/benchmark_samples.json`.
  - Calcula BLEU/chrF (sacrebleu) e latencia media por modelo; ajuste lista em `DEFAULT_MODELS`.
- Benchmark rapido de LLMs no prompt de traducao: `python -m tradutor.bench_llms --input data/meu_texto.md [--models ...] [--max-chars 1500]`
  - Gera uma traducao por modelo (Ollama) em `benchmark/` + um `resumo_<slug>.md` com tempos.
- Benchmark rapido de LLMs no prompt de refine (texto em PT): `python -m tradutor.bench_refine_llms --input data/meu_texto_pt.md [--models ...] [--max-chars 1500]`
  - Gera uma revisao por modelo (Ollama) em `benchmark/` + um `resumo_refine_<slug>.md` com tempos.

---

## Modelos recomendados para Ollama
- Traducao: `qwen3:14b-q4_K_M`
- Refine: `gemma3-gaia-ptbr-4b:q4_k_m` (principal). Ajuste manualmente em `--model` se quiser um fallback (ex.: Qwen2.5 PT-BR).
