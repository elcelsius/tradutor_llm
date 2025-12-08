# Tradutor Literário com LLMs (Ollama/Gemini)

Arquitetura modular para tradução e refine de light novels e PDFs para PT-BR com foco em **zero alucinação**. Pipeline em dois passos:
- **Passo 1 – Tradução** (`tradutor/main.py` subcomando `traduz`): lê PDFs, pré-processa, faz chunking seguro, traduz em lotes e gera Markdown/PDF.
- **Passo 2 – Refine opcional** (`tradutor/main.py` subcomando `refina`): lê `*_pt.md` da pasta `saida/`, detecta capítulos (`## `), refina capítulo a capítulo e gera `*_pt_refinado.md` e PDF sem sobrescrever o original.

Compatível com Windows 11.

---

## Arquitetura
```
tradutor/
  __init__.py
  main.py          # CLI com subcomandos traduz/refina
  config.py        # constantes e paths padrão
  preprocess.py    # extração de PDF, limpeza, chunking seguro
  translate.py     # prompt e pipeline de tradução por lotes
  refine.py        # refine capítulo a capítulo em Markdown
  llm_backend.py   # chamadas Ollama/Gemini com retry
  sanitizer.py     # sanitização agressiva contra alucinação/loops
  pdf_export.py    # conversão Markdown -> PDF com fonte Unicode
  utils.py         # logging, IO, helpers
```

---

## Requisitos
- Python 3.10+
- Bibliotecas: `google-generativeai`, `PyPDF2`, `fpdf`, `requests`
  ```bash
  pip install google-generativeai PyPDF2 fpdf requests
  ```
- Ollama instalado (padrão). Para Gemini, defina `GEMINI_API_KEY`.

---

## Modelos e parâmetros padrão
- Tradução: backend `ollama`, modelo `qwen3:14b-q4_K_M`, temperatura `0.15`, chunk `3800` caracteres.
- Refine: backend `ollama`, modelo `gemma3-gaia-ptbr-4b:q4_k_m`, temperatura `0.30`, chunk `10000` caracteres.
- Retry: 3 tentativas, backoff exponencial.
- Sanitização: remove `<think>...</think>`, meta-comentários de LLM, repetições/loops e respostas vazias ou contaminadas (falha e re-tenta).

---

## Como usar

### Passo 1 – Traduzir PDFs
```bash
python -m tradutor.main traduz
# ou um PDF específico
python -m tradutor.main traduz --input "data/meu_livro.pdf"
# usando Gemini
python -m tradutor.main traduz --backend gemini --model gemini-3-pro-preview
# pular refine automático ao final
python -m tradutor.main traduz --no-refine
```
Saídas na pasta `saida/`:
```
<nome>_pt.md
<nome>_pt.pdf
<nome>_pt_refinado.md   (se refine automático estiver ligado)
<nome>_pt_refinado.pdf  (se refine automático estiver ligado)
```

### Passo 2 – Refine opcional (capítulo a capítulo)
```bash
# refina todos os *_pt.md em saida/
python -m tradutor.main refina

# refina um arquivo específico
python -m tradutor.main refina --input "saida/MEU_ARQUIVO_pt.md"
```
Saídas:
```
saida/MEU_ARQUIVO_pt_refinado.md
saida/MEU_ARQUIVO_pt_refinado.pdf
```
O original `*_pt.md` nunca é sobrescrito.

---

## Sanitização e robustez
- Remove `<think>`/`</think>`, meta-texto (“parece que você está...”), loops e blocos repetidos.
- Falha e re-tenta em caso de contaminação ou resposta vazia.
- Chunking seguro (3800/10000 chars) com cortes duros se necessário; proíbe chunks gigantes.
- Logs detalhados por chunk, sanitização e tempo de processamento.

---

## Boas práticas
- Rode tradução primeiro, refine depois (em capítulos).
- Mantenha sempre uma cópia do `*_pt.md` original.
- Ajuste `data/` e `saida/` conforme necessário; ambos são criados automaticamente.
- Monitore VRAM/CPU ao usar modelos grandes no Ollama.

---

## Exemplos rápidos
```bash
# Tradução padrão local (Ollama)
python -m tradutor.main traduz

# Tradução usando Gemini
python -m tradutor.main traduz --backend gemini --model gemini-3-pro-preview

# Refine de todos os arquivos *_pt.md
python -m tradutor.main refina

# Refine de um arquivo específico
python -m tradutor.main refina --input "saida/MEU_ARQUIVO_pt.md"
```

---

## Modelos recomendados para Ollama
- Tradução: `qwen3:14b-q4_K_M`
- Refine: `gemma3-gaia-ptbr-4b:q4_k_m` (principal). Ajuste manualmente em `--model` se quiser um fallback (ex.: Qwen2.5 PT-BR).
