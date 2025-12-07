# Tradutor Automático de PDFs com IA (Gemini / Ollama)

Script em Python para traduzir livros e documentos PDF do inglês para o português do Brasil (PT-BR) usando modelos de linguagem (LLMs). Funciona com dois backends:

- Ollama (local, custo zero) – padrão
- Gemini (Google, via API) – opcional

Foco em qualidade literária alta, ideal para light novels, romances e manuais longos.

---

## Principais recursos

- Leitura automática de PDFs na pasta `data/`
- Pré-processamento avançado:
  - remove cabeçalhos/rodapés típicos de livro
  - remove hifenização de quebra de linha (`infor-\nmação` -> `informação`)
  - reconstrói parágrafos quebrados
- Divisão inteligente em lotes (chunks):
  - tenta cortar em quebras de parágrafo
  - se não der, corta em fim de frase (`. ? !`)
  - fallback no limite bruto de caracteres
- Tradução com LLM:
  - prompt ajustado para tradução literária de light novel
  - não resume, não altera ordem, não inventa conteúdo
- Segunda passada de revisão em PT-BR (refine):
  - melhora fluidez, coesão e pontuação
  - preserva sentido e nomes próprios
  - roda localmente (Ollama) ou via Gemini
- Geração automática:
  - `saida/<nome>_pt.md` – texto traduzido em Markdown
  - `saida/<nome>_pt.pdf` – versão PDF pronta para leitura

---

## Estrutura do projeto

```
seu-projeto/
  data/        PDFs de entrada (originais em inglês)
  saida/       Arquivos traduzidos (.md e .pdf)
  tradutor.py  Script principal de tradução
  README.md
```

Pipeline por PDF:
1. Extrai texto com `PyPDF2`.
2. Pré-processa (limpeza + parágrafos).
3. Divide em chunks (`chunk_size` caracteres).
4. Tradução lote a lote com LLM.
5. Junta tudo em um `.md`.
6. (Opcional) Refine: revisão global em PT-BR.
7. Converte o `.md` final em `.pdf` usando `FPDF` + fonte Unicode (DejaVu).

---

## Requisitos

### Python
- Python 3.10+ recomendado

### Bibliotecas Python
Instale:
```bash
pip install google-generativeai PyPDF2 fpdf requests
```

Se usar apenas Ollama (local) e não quiser Gemini, `google-generativeai` só é necessária para o backend em nuvem.

---

## Backends suportados

### 1. Ollama (local – padrão)
Site: https://ollama.com

Modelos recomendados:
- Tradutor principal: `ollama pull qwen3:14b`
- Revisor PT-BR (segunda passada): `ollama pull cnmoro/Qwen2.5-0.5B-Portuguese-v2:fp16`
- Opções de fallback:  
  `cnmoro/Qwen2.5-0.5B-Portuguese-v1:q4_k_m`  
  `cnmoro/Qwen2.5-0.5B-Portuguese-v1:q8_0`  
  `cnmoro/Qwen2.5-0.5B-Portuguese-v1:fp16`  
  `cnmoro/gemma3-gaia-ptbr-4b:q4_k_m` (revisão “de luxo” opcional)

### 2. Gemini (Google – opcional)
1. Crie/pegue sua API key no painel Google AI.
2. Defina a variável de ambiente:
   - Linux/macOS: `export GEMINI_API_KEY="SUA_CHAVE_AQUI"`
   - Windows (PowerShell): `setx GEMINI_API_KEY "SUA_CHAVE_AQUI"`
3. Instale a biblioteca: `pip install google-generativeai`

Modelo padrão: `gemini-3-pro-preview` (ajustável via `--model`).

---

## Como usar

1) Coloque os PDFs na pasta `data/`.
```
data/
  failure_frame_vol4.pdf
  light_novel_x.pdf
```

2) Rode o script no modo padrão (qualidade máxima local):
```bash
python tradutor.py
```
Equivale a:
- `--backend ollama`
- `--model qwen3:14b`
- `--refine` ligado
- `--refine-model auto` (tenta a cadeia de modelos PT-BR listada acima)

Saída esperada para `meulivro.pdf`:
```
saida/
  meulivro_pt.md
  meulivro_pt.pdf
```

### Somente tradução (sem refine)
```bash
python tradutor.py --no-refine
```

### Mudar o modelo local
```bash
python tradutor.py --backend ollama --model qwen2.5:14b
```

### Forçar modelo específico de refine
```bash
python tradutor.py \
  --backend ollama \
  --model qwen3:14b \
  --refine-model "cnmoro/Qwen2.5-0.5B-Portuguese-v2:fp16"
```

### Usar Gemini como backend
```bash
python tradutor.py --backend gemini --model gemini-3-pro-preview
```
Requer `GEMINI_API_KEY` configurada. O mesmo modelo é usado no refine, a menos que `--no-refine` seja informado.

### Ajustar tamanho dos chunks
```bash
python tradutor.py --chunk_size 7000   # textos longos
python tradutor.py --chunk_size 12000  # se o modelo aguentar
```

### Ajustar temperatura (Gemini)
```bash
python tradutor.py --backend gemini --temperature 0.2
```
Temperaturas baixas (0.1–0.3) são mais fiéis; mais altas (0.4–0.6) dão mais liberdade estilística.

---

## Parâmetros da linha de comando

```bash
python tradutor.py [opções]
```

- `--backend {gemini,ollama}`: backend de LLM (padrão: ollama)
- `--model NOME`: modelo principal de tradução (ex.: gemini-3-pro-preview, qwen3:14b, qwen2.5:14b, aya:8b)
- `--refine-model NOME|auto`: modelo de revisão PT-BR. Para Ollama:
  - `auto`: usa a cadeia de modelos recomendados
  - `NOME`: usa apenas o modelo indicado  
  Para Gemini, o mesmo modelo da tradução é usado no refine.
- `--chunk_size N`: tamanho de cada lote em caracteres (padrão: 9000)
- `--temperature X`: temperatura (usada no backend Gemini; padrão: 0.3)
- `--no-refine`: desativa a segunda passada de revisão

---

## Resumo técnico do código

1) `extract_text_from_pdf`: lê texto página a página com `PyPDF2.PdfReader`.
2) `preprocess_text`: remove cabeçalhos/rodapés, desfaz hifenização e junta linhas em parágrafos.
3) `chunk_text`: corta por parágrafo, depois por fim de frase, depois pelo limite bruto.
4) `translate_chunk_ollama` / `translate_chunk_gemini`: traduz cada chunk com prompt de tradutor literário; inclui retry com backoff.
5) `refine_text_ollama` / `refine_text_gemini`: revisão global em PT-BR, preservando sentido e nomes.
6) `markdown_to_pdf`: garante a fonte DejaVuSans.ttf e converte Markdown simples em PDF.

---

## Dicas práticas

- Para livros longos, deixe rodando; modelos locais grandes podem consumir VRAM/CPU.
- Se o modelo começar a se perder, reduza `--chunk_size`.
- Se quiser apenas `.md` (por exemplo, para exportar via Calibre/EPUB), ignore o `.pdf`.

---

## Licença / uso

Uso pessoal, estudo e adaptação livres. Para fins comerciais ou integração em sistemas maiores, revise as licenças:
- dos modelos (Qwen, Gemini, Gemma etc.)
- do Ollama
- das obras que você traduz (direitos autorais).

