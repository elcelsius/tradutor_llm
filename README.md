# Tradutor e Refinador de PDFs com IA (Gemini / Ollama)

Pipeline em Python para traduzir light novels e outros PDFs do inglês para PT-BR usando LLMs. Agora há **dois passos distintos**:
- **Passo 1 – Tradução**: `tradutor.py` gera `<nome>_pt.md` e `<nome>_pt.pdf`.
- **Passo 2 – Refine opcional**: `refinador.py` revisa capítulo a capítulo o `.md` e gera `<nome>_pt_refinado.md` e `<nome>_pt_refinado.pdf` sem sobrescrever o original.

Tom técnico, pensado para usuários avançados/devs (Python + Ollama + Gemini).

---

## Visão geral
- Lê PDFs em `data/`.
- Pré-processa (remove cabeçalhos/rodapés, desfaz hifenização, junta linhas).
- Faz chunking (~9000 caracteres para tradução).
- Tradução por LLM (padrão `qwen3:14b` via Ollama).
- Gera Markdown e PDF em `saida/`.
- Refine opcional capítulo a capítulo (~5000 caracteres) usando headings `## ` como delimitadores.

---

## Requisitos
- Python 3.10+.
- Bibliotecas: `google-generativeai`, `PyPDF2`, `fpdf`, `requests`.
  ```bash
  pip install google-generativeai PyPDF2 fpdf requests
  ```
- Ollama instalado para uso local (padrão). Para Gemini, definir `GEMINI_API_KEY`.

---

## Como traduzir PDFs (tradutor.py)

Padrão local (Ollama, qualidade máxima):
```bash
python tradutor.py
```
- Backend: `ollama`
- Modelo: `qwen3:14b`
- Refine interno: ligado por padrão (`--refine-model auto` usa cadeia PT-BR interna)

Gemini:
```bash
python tradutor.py --backend gemini --model gemini-3-pro-preview
```
- Requer `GEMINI_API_KEY`.
- O mesmo modelo é usado na passada de refine, exceto se `--no-refine`.

Flags principais:
- `--no-refine` desativa a revisão interna do tradutor.
- `--chunk_size N` ajusta o tamanho dos lotes (padrão 9000).
- `--temperature X` (Gemini) controla fidelidade/variação (padrão 0.3).
- `--model NOME` troca o modelo principal (ex.: `qwen2.5:14b`).
- `--refine-model NOME|auto` fixa ou alterna a cadeia de revisão interna (Ollama).

Saída típica para `livro.pdf`:
```
saida/
  livro_pt.md
  livro_pt.pdf
```

---

## Como refinar as traduções (refinador.py)

Objetivo: revisão em PT-BR capítulo a capítulo sobre um `.md` já traduzido.
- Lê `*_pt.md` da pasta `saida/` (ou um arquivo único via `--input`).
- Separa por headings `## ` (capítulos/partes).
- Refina cada seção em chunks ~5000 caracteres.
- Backend padrão: Ollama, com cadeia auto (Gaia principal + Qwen2.5 PT-BR de fallback).
- Opcional: `--backend gemini` para usar Gemini.
- Não sobrescreve o arquivo original; escreve `_pt_refinado`.

Refinar todos os arquivos `*_pt.md`:
```bash
python refinador.py
```

Refinar só um arquivo específico:
```bash
python refinador.py --input "saida/MEU_ARQUIVO_pt.md"
```

Saída típica para `livro_pt.md`:
```
saida/
  livro_pt_refinado.md
  livro_pt_refinado.pdf
```

---

## Modelos recomendados (Ollama)
- **Tradução**: `qwen3:14b`
- **Refine** (cadeia auto, nessa ordem):
  1) `cnmoro/gemma3-gaia-ptbr-4b:q4_k_m` (principal)
  2) Qwen2.5 PT-BR como fallback:
     - `cnmoro/Qwen2.5-0.5B-Portuguese-v2:fp16`
     - `cnmoro/Qwen2.5-0.5B-Portuguese-v1:q4_k_m`
     - `cnmoro/Qwen2.5-0.5B-Portuguese-v1:q8_0`
     - `cnmoro/Qwen2.5-0.5B-Portuguese-v1:fp16`

---

## Boas práticas
- Rode sempre o tradutor primeiro; refine depois.
- Para o refine, priorize capítulos/partes (headings `## `) em vez do livro inteiro de uma vez.
- Mantenha uma cópia do `*_pt.md` original antes de refinar.
- Ajuste `--chunk_size` se notar perda de contexto: diminua para estabilidade; aumente só se o modelo suportar.
- Monitore VRAM/CPU no primeiro uso de modelos grandes.

---

## Exemplos rápidos
```bash
# Tradução padrão local (Ollama, qwen3:14b)
python tradutor.py

# Tradução usando Gemini
python tradutor.py --backend gemini --model gemini-3-pro-preview

# Refine de todos os arquivos *_pt.md
python refinador.py

# Refine de um arquivo específico
python refinador.py --input "saida/MEU_ARQUIVO_pt.md"
```
