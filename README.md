# Tradutor Literário com LLMs (Ollama/Gemini)

Pipeline de tradução/refine (EN → PT-BR) para novels adultas e PDFs em PT-BR, com foco em **zero alucinação**. Fluxo principal:
- **Passo 1 - Tradução** (`tradutor/main.py` subcomando `traduz`): lê PDFs, pré-processa, faz chunking seguro, traduz em lotes e gera Markdown.
- **Passo 2 - Refine opcional** (`tradutor/main.py` subcomando `refina`): lê `*_pt.md` em `saida/`, detecta capítulos (`## `) e refina capítulo a capítulo gerando `*_pt_refinado.md` sem sobrescrever o original.

Compatibilidade: Windows 11. Wrappers legados (`tradutor.py`, `refinador.py`) apenas redirecionam para o novo CLI.

Prompts especializados para novels:
- Tradução: prompt adulto (dark fantasy, violência/blasfêmia), preserva tom, gênero, número, sem suavizar.
- Refine: prompt adulto para fluidez/coerência, sem cortar conteúdo, corrige gênero/plural/naturalidade.
- Sanitização: remove `<think>` e meta-texto em PT/EN (ex.: "as an ai language model").

---

## Arquitetura
```
.
├── tradutor/
│   ├── main.py               # CLI com subcomandos traduz/refina
│   ├── config.py             # constantes, caminhos padrão e ensure_paths
│   ├── preprocess.py         # extração de PDF, limpeza e chunking seguro
│   ├── translate.py          # prompt e pipeline de tradução por lotes
│   ├── refine.py             # refine capítulo a capítulo em Markdown
│   ├── llm_backend.py        # chamadas Ollama/Gemini com retry
│   ├── sanitizer.py          # sanitização agressiva contra alucinação/loops/meta
│   ├── glossary_utils.py     # carregamento manual/dinâmico e formatação para prompt
│   ├── pdf_reader.py         # extração de texto com PyMuPDF (fitz)
│   ├── pdf_export.py         # conversão Markdown -> PDF (opcional)
│   ├── benchmark.py          # benchmark BLEU/chrF entre modelos
│   ├── bench_llms.py         # benchmark rápido no prompt de tradução
│   ├── bench_refine_llms.py  # benchmark rápido no prompt de refine
│   └── utils.py              # logging, IO, helpers
├── glossario/
│   └── glossario_manual.json # glossário manual padrão (somente leitura)
├── data/                     # insumos (PDFs)
├── saida/                    # saídas (_pt.md, _pt_refinado.md, glossario_dinamico.json)
├── benchmark/                # entradas e saídas de benchmark
├── tests/                    # testes (smoke + amostras)
├── tradutor.py / refinador.py# wrappers legados do CLI
└── desquebrar.py             # utilitário para desquebrar parágrafos de Markdown
```

---

## Requisitos
- Python 3.10+
- Instale dependências:
  ```bash
  pip install -r requirements.txt
  ```
  Principais libs: `google-generativeai`, `PyMuPDF`, `fpdf`, `requests`, `sacrebleu`, `PyYAML`.
- Ollama instalado (padrão). Para Gemini, defina `GEMINI_API_KEY`.
- Hardware sugerido: GPU com ~16GB VRAM para rodar os modelos padrão com folga local.

## Configuração
- Opcional: crie um `config.yaml` na raiz (há um `config.example.yaml` de referência) para ajustar parâmetros sem editar código.
- Parâmetros relevantes:
  - `translate_chunk_chars` / `refine_chunk_chars`: tamanho máximo do chunk (padrão 3200 para ambos).
  - `request_timeout`: timeout HTTP (segundos) para chamadas ao LLM (padrão 60; aumente manualmente se precisar).

---

## Modelos e parâmetros padrão
- Tradução: backend `ollama`, modelo `brunoconterato/Gemma-3-Gaia-PT-BR-4b-it:f16`, temperatura `0.15`, chunk `3200` caracteres.
- Refine: backend `ollama`, modelo `cnmoro/gemma3-gaia-ptbr-4b:q4_k_m`, temperatura `0.30`, chunk `3200` caracteres.
- Retry: 3 tentativas, backoff exponencial.
- Timeout HTTP padrão: `60s` (usado nas chamadas LLM).
- Sanitização: remove `<think>...</think>`, meta-comentários (PT/EN), repetições/loops e respostas vazias ou contaminadas (falha e re-tenta).

---

## Como usar

### Passo 1 - Traduzir PDFs
```bash
python -m tradutor.main traduz
# ou um PDF específico
python -m tradutor.main traduz --input "data/meu_livro.pdf"
# modo debug (salva raw/preprocessed)
python -m tradutor.main traduz --debug --input "data/meu_livro.pdf"
# retomar tradução do ponto salvo (usa manifesto *_pt_progress.json)
python -m tradutor.main traduz --resume --input "data/meu_livro.pdf"
# usando Gemini
python -m tradutor.main traduz --backend gemini --model gemini-3-pro-preview
# pular refine automático ao final
python -m tradutor.main traduz --no-refine

# modo legado (wrappers):
python tradutor.py --input "data/meu_livro.pdf"
```
Saídas na pasta `saida/` (sem PDF):
- Sempre: `<nome>_pt.md`
- Se debug: `<nome>_raw_extract.md`, `<nome>_preprocessed.md`
- Se refine habilitado: `<nome>_pt_refinado.md`

### Passo 2 - Refine opcional (capítulo a capítulo)
```bash
# refina todos os *_pt.md em saida/
python -m tradutor.main refina

# refina um arquivo específico
python -m tradutor.main refina --input "saida/MEU_ARQUIVO_pt.md"
# retomar refine do ponto salvo (usa manifesto *_pt_refinado_progress.json)
python -m tradutor.main refina --resume --input "saida/MEU_ARQUIVO_pt.md"
# normalizar parágrafos do .md antes de refinar (remove quebras internas)
python -m tradutor.main refina --input "saida/MEU_ARQUIVO_pt.md" --normalize-paragraphs
# modo glossário (manual + dinâmico); dinâmico é salvo em saida/glossario_dinamico.json por padrão
python -m tradutor.main refina --use-glossary --manual-glossary "glossario/glossario_manual.json" --dynamic-glossary "saida/glossario_dinamico.json"
# carregar todos os JSONs de glossário manual dentro de um diretório (útil para quebrar glossários grandes em vários arquivos)
python -m tradutor.main refina --use-glossary --auto-glossary-dir "glossario/minha_obra" --dynamic-glossary "saida/glossario_dinamico.json"

# modo legado (wrappers):
python refinador.py --input "saida/MEU_ARQUIVO_pt.md"
```
Saídas:
```
saida/MEU_ARQUIVO_pt_refinado.md
saida/MEU_ARQUIVO_pt_refinado.pdf
```
O original `*_pt.md` nunca é sobrescrito.
Manifestos de progresso:
- Tradução: `saida/<nome>_pt_progress.json`
- Refine: `saida/<nome>_pt_refinado_progress.json`
- Glossário dinâmico (opcional): `saida/glossario_dinamico.json` (criado/atualizado somente quando `--use-glossary` é utilizado; o glossário manual é apenas lido e nunca alterado). Se passar `--auto-glossary-dir`, todos os JSONs dentro do diretório informado são carregados como glossário manual agregado (primeira ocorrência de cada key prevalece).

### Glossário (manual + dinâmico)
- Manual padrão: `glossario/glossario_manual.json` (somente leitura; pode ser fragmentado em vários arquivos usando `--auto-glossary-dir glossario/`).
- Dinâmico padrão: `saida/glossario_dinamico.json` (criado/atualizado apenas quando `--use-glossary` é passado).
- Comando típico usando os padrões:
  ```bash
  python -m tradutor.main refina --use-glossary --manual-glossary "glossario/glossario_manual.json" --dynamic-glossary "saida/glossario_dinamico.json"
  ```

### Apenas extrair raw/preprocessed (sem traduzir/refinar)
Para gerar somente os arquivos brutos e pré-processados de todos os PDFs em `data/`:

```powershell
@'
from pathlib import Path
from tradutor.utils import setup_logging, write_text
from tradutor.preprocess import extract_text_from_pdf, preprocess_text

logger = setup_logging()
data_dir = Path("data")
out_dir = Path("saida")
out_dir.mkdir(exist_ok=True)

for pdf in data_dir.glob("*.pdf"):
    raw = extract_text_from_pdf(pdf, logger)
    pre = preprocess_text(raw, logger)
    write_text(out_dir / f"{pdf.stem}_raw_extract.md", raw)
    write_text(out_dir / f"{pdf.stem}_preprocessed.md", pre)
    logger.info("Gerado: %s_raw_extract.md e %s_preprocessed.md", pdf.stem, pdf.stem)
'@ | python -
```

Isso cria, para cada `*.pdf` em `data/`, os arquivos `saida/<nome>_raw_extract.md` (texto bruto) e `saida/<nome>_preprocessed.md` (texto limpo/normalizado em EN), sem acionar tradução nem refine. Útil para montar glossários manuais antes de traduzir.

---

## Sanitização e robustez
- Remove `<think>`/`</think>`, meta-texto (inclusive inglês), loops e blocos repetidos.
- Falha e re-tenta em caso de contaminação ou resposta vazia.
- Chunking seguro (3200/3200 chars) com cortes duros se necessário; evita chunks gigantes.
- Logs detalhados por chunk, sanitização e tempo de processamento.
- Logs informam backend/model/temperatura/chunk usados em tradução e refine (inclusive opcional).

---

## Testes e benchmark
- Smoke test sem LLM real (backend fake e stub de PyMuPDF): `pytest -q`
- Benchmark (custa tempo/tokens, chama LLM real): `python -m tradutor.benchmark`
  - Usa `tests/benchmark_samples.json`.
  - Calcula BLEU/chrF (sacrebleu) e latência média por modelo; ajuste lista em `DEFAULT_MODELS`.
- Benchmark rápido de LLMs no prompt de tradução: `python -m tradutor.bench_llms --input benchmark/teste_traducao_en.md --max-chars 1500 --out-dir benchmark/traducao`
  - Gera uma tradução por modelo (Ollama) em `benchmark/traducao/` + um `resumo_<slug>.md` com tempos.
  - Sem `--models`, detecta dinamicamente os modelos instalados (`ollama list`/`/api/tags`); passe `--models <m1> <m2>` para limitar.
- Benchmark rápido de LLMs no prompt de refine (texto em PT): `python -m tradutor.bench_refine_llms --input benchmark/teste_refine_pt.md --max-chars 1500 --out-dir benchmark/refine`
  - Gera uma revisão por modelo (Ollama) em `benchmark/refine/` + um `resumo_refine_<slug>.md` com tempos.
  - Mesma detecção automática de modelos via `ollama list`/`/api/tags` quando `--models` não é informado.

---

## Modelos recomendados para Ollama
- Tradução: `brunoconterato/Gemma-3-Gaia-PT-BR-4b-it:f16`
- Refine: `cnmoro/gemma3-gaia-ptbr-4b:q4_k_m` (principal). Como backup mais conservador, pode usar o mesmo modelo também para tradução se quiser menos variação.
