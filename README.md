# tradutor_llm — Tradução e refine previsíveis (Ollama/Gemini)

Pipeline completo EN→PT-BR com foco em estabilidade, anti-alucinação e automação. Compatível com Windows 11.

- **Traduzir** (`python -m tradutor.main traduz`): extrai PDF, pré-processa, chunk inteligente por frase (2400 alvo), contexto leve entre chunks, glossário manual opcional, guardrails e cache.
- **Refinar** (`python -m tradutor.main refina`): polidor minimalista em PT-BR com glossário manual/dinâmico (opcional), guardrails, cache e pós-processamento final.
- **Saídas**: `_pt.md`, `_pt_refinado.md`, PDF otimizado para leitura digital, manifestos de progresso e relatórios em `saida/`.

---

## Arquitetura resumida
```
tradutor/
  main.py              # CLI traduz/refina (+ flags debug, glossario, parallel, editor, preprocess avancado)
  config.py            # defaults (chunk=2400, timeout=60s, num_predict=768, repeat_penalty, paths)
  preprocess.py        # pré-processo básico e chunking por frase (lookahead)
  advanced_preprocess.py # limpeza opcional (hifens, tags estranhas, espaços)
  translate.py         # pipeline de tradução com prompt minimalista, contexto e cache
  refine.py            # pipeline de refine com guardrails, cache e glossário
  llm_backend.py       # cliente Ollama/Gemini, retries, timeout, repeat_penalty
  sanitizer.py         # sanitização agressiva (trad) e leve (refine)
  anti_hallucination.py# AAA shield: detecção de idioma/estrutura/repetição/deriva
  cache_utils.py       # hash, cache por chunk, deduplicação, detecção de colapso
  glossary_utils.py    # carga manual/dinâmico, merge para prompt, parser de sugestões
  glossary/merge.py    # utilitário CLI para unir glossários em MASTER_GLOSSARIO.json
  pdf_export.py        # ReportLab com fontes locais (Aptos/Segoe/Calibri/Arial, fallback Helvetica)
  postprocess.py       # pós-processo final PT-BR (travessões, espaços, marcadores)
  structure_normalizer.py # normaliza capítulos/títulos antes do PDF
  editor.py            # modos editor opcionais (lite/consistency/voice/strict) + relatório
  desquebrar.py        # normalização opcional de parágrafos
  VERSION              # versão interna do pipeline
```

---

## Principais garantias e guardrails
- **Chunk**: alvo 2400 chars (trad/refine), lookahead para fim de frase; contexto leve usa última frase do chunk anterior (não traduzido).
- **Timeout LLM**: 60s; **num_predict**: 768; **repeat_penalty** suportado (trad default 1.1).
- **Retries**: tradução 3 tentativas; refine 1 (fallback para chunk original).
- **Sanitização**: remove `<think>`, molduras, glossários sugeridos e marcadores residuais; pós-processo PT-BR ajusta travessões/espaços e remove resíduos.
- **Anti-alucinação (AAA)**: checa idioma indevido, repetição, estrutura, deriva semântica; se anômalo, cai para chunk original.
- **Cache/estado**: cache por hash em `saida/cache_traducao` e `saida/cache_refine`; states `state_traducao.json` / `state_refine.json`; resume automático e reuso de duplicatas próximas.
- **Glossário**: manual opcional na tradução (injeção de até 30 pares EN→PT); refine usa manual+dinâmico (sem duplicar pt do manual, filtra termos longos/descritivos).
- **Editor opcional**: modos lite/consistency/voice/strict pós-refine; gera `saida/editor_report.json` se habilitado; padrão OFF.
- **PDF**: ReportLab com fontes locais (Aptos > AptosDisplay > Segoe UI > Calibri > Arial > Helvetica), margens 42/52 pt, corpo 11.5 pt, diálogos com estilo próprio, hyphenation pt-BR se pyphen disponível.
- **Relatórios**: `saida/report.json` resume execuções (chunks, cache hits, fallbacks, colapsos, duplicatas, versão).

---

## Prompts minimalistas (inalterados)
- **Refine (PT-BR)**: saída apenas
  ```
  ### TEXTO_REFINADO_INICIO
  <texto refinado>
  ### TEXTO_REFINADO_FIM
  ```
- **Tradução (EN → PT-BR)**: saída apenas
  ```
  ### TEXTO_TRADUZIDO_INICIO
  <texto traduzido para PT-BR>
  ### TEXTO_TRADUZIDO_FIM
  ```
Texto fora dos marcadores é ignorado; fallback usa sanitização se marcadores não vierem.

---

## Uso básico
### Traduzir PDF
```bash
python -m tradutor.main traduz --input "data/meu_livro.pdf"
# opções úteis
python -m tradutor.main traduz --use-glossary --manual-glossary "glossario/glossario_manual.json" --input "data/meu_livro.pdf"
python -m tradutor.main traduz --backend gemini --model gemini-3-pro-preview
python -m tradutor.main traduz --resume --input "data/meu_livro.pdf"
python -m tradutor.main traduz --parallel 1          # paralelismo seguro (contexto exige 1)
python -m tradutor.main traduz --preprocess-advanced # limpeza extra antes de chunking
```
Saídas: `saida/<nome>_pt.md`, manifesto `<nome>_pt_progress.json`, cache/state/report em `saida/`.

### Refine Markdown
```bash
python -m tradutor.main refina --input "saida/meu_livro_pt.md"
python -m tradutor.main refina --normalize-paragraphs
python -m tradutor.main refina --use-glossary --manual-glossary "glossario/glossario_manual.json" --dynamic-glossary "saida/glossario_dinamico.json"
python -m tradutor.main refina --debug-refine
python -m tradutor.main refina --parallel 1          # paralelismo opcional (ordem preservada)
python -m tradutor.main refina --preprocess-advanced # limpeza extra do .md antes de refinar
# modos editor opcionais (pós-refine, padrão OFF)
python -m tradutor.main refina --editor-lite --editor-report
```
Saídas: `saida/<nome>_pt_refinado.md`, `<nome>_pt_refinado.pdf`, manifesto `<nome>_pt_refinado_progress.json`, cache/state/report em `saida/`, `glossario_dinamico.json` se habilitado.

### Mesclar glossários
```bash
python -m tradutor.glossario.merge --input glossario/ --output MASTER_GLOSSARIO.json
```
Conflitos em `saida/glossario_conflicts.log`. Prioriza locked/manual.

---

## Requisitos
- Python 3.10+
- `pip install -r requirements.txt` (google-generativeai, PyMuPDF, reportlab, pyphen, requests, sacrebleu, PyYAML)
- Ollama instalado (padrão) ou `GEMINI_API_KEY` para Gemini.

---

## Comportamento seguro esperado
- Sem molduras ou glossário sugerido no texto final; blocos indesejados removidos.
- Guardrails de tamanho/repetição/colapso + anti-alucinação garantem fallback para chunk original quando necessário.
- Cache + resume evitam retrabalho; duplicatas reutilizadas automaticamente.
- PDF offline com fontes locais e layout para leitura confortável.
- Editor e preprocess avançado são opcionais e não alteram semântica se desativados.
