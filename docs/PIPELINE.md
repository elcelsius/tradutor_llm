# Pipeline de Tradução

Este documento descreve a regra atual do pipeline EN -> PT-BR. A lógica é genérica para qualquer obra; o que muda por projeto é glossário, configuração e modelos.

## Diagrama

```mermaid
flowchart TD
    A[PDF ou Markdown] --> B[Extração / leitura]
    B --> C[Preprocess determinístico]
    C --> D[Desquebrar linhas]
    D --> E1[Contexto deslizante + perfil diálogo/narração]
    E1 --> E[Tradução por chunk com glossário]
    E --> F[QA da tradução]
    F -->|sem problema| H[Markdown PT]
    F -->|problema objetivo| G[Repair seletivo]
    G --> F
    H --> I[Cleanup pré-refine]
    I --> J[Refine literário]
    J --> K[Revisor determinístico final opcional]
    K --> L[QA final / relatórios / PDF]
```

## Regra Por Etapa

1. **Extração / leitura**
   - PDF vira texto bruto.
   - Markdown/texto já preparado pode entrar por `traduz-md`.

2. **Preprocess determinístico**
   - Remove ruído de PDF, TOC, URLs promocionais e front matter quando configurado.
   - Não usa LLM.

3. **Desquebrar**
   - Junta linhas quebradas e corrige hifenização de extração.
   - Pode usar LLM (`llm`) ou modo determinístico (`safe`).

4. **Tradução**
   - Traduz EN -> PT-BR por chunk.
   - Injeta apenas termos do glossário que aparecem no chunk.
   - Usa uma janela deslizante de contexto recente, somente leitura, com últimos parágrafos do original e opcionalmente da tradução PT-BR.
   - Reseta a janela ao mudar de capítulo/seção e quando um chunk começa ou termina com separador de cena (`***`, `---`, etc.).
   - Classifica o chunk como diálogo, narração ou misto e injeta instruções específicas para esse perfil.
   - Preserva nomes, honoríficos, ordem narrativa e diálogos.
   - Tem retries para truncamento, omissão de diálogo e inglês residual.

5. **QA da tradução**
   - Roda logo após a tradução de cada chunk.
   - Detecta problemas objetivos:
     - frase ou parágrafo ainda em inglês;
     - possível omissão de diálogo;
     - chunk curto demais;
     - termo fonte vazando quando existe termo PT canônico;
     - alias proibido no texto final.

6. **Repair seletivo**
   - Só chama LLM quando o QA encontra problema.
   - Recebe original em inglês, tradução atual, lista de problemas e glossário do chunk.
   - Corrige apenas os problemas listados.
   - Não deve reescrever o trecho inteiro nem alterar estrutura narrativa.
   - Rejeita reparos que encurtam demais o chunk, removem parágrafos/falas ou apagam o começo já traduzido.
   - O resultado reparado vira a entrada do refine.

7. **Pós-processamento determinístico**
   - Roda no `*_pt.md` antes do refine e novamente no `*_pt_refinado.md`.
   - Corrige artefatos simples como aspas duplicadas, falsos literais recorrentes e erros gramaticais determinísticos.

8. **Cleanup pré-refine**
   - Remove duplicações óbvias e corrige diálogos colados antes do refine.
   - Controlado por `cleanup_before_refine: off|auto|on`.

9. **Refine literário**
   - Edita o PT-BR para fluidez, pontuação, ritmo e naturalidade.
   - Não deve retraduzir a obra nem mudar estrutura.
   - Mantém uma rede de segurança para inglês residual, mas essa não é sua responsabilidade principal.

10. **Revisor determinístico final opcional**
   - Corrige problemas mecânicos:
     - `bad_aliases` do glossário;
     - duplicações de nomes canônicos;
     - artigos/gênero conhecidos;
     - pequenas substituições editoriais conservadoras.
   - Atualmente é executado via `scripts/review_translation.py`; não roda automaticamente no `traduz`.

11. **QA final / relatórios**
    - Gera métricas de tradução, repair e refine.
    - Com `--debug`, grava manifests e arquivos por chunk em `saida/debug_runs/`.

## Modularidade Por Obra

- Troque o glossário via `--manual-glossary` ou `glossario/glossario_geral.json`.
- Use `source_aliases` apenas para busca no original.
- Use `bad_aliases` para formas proibidas no texto final.
- Use `allowed_target_aliases` para formas aceitas que não devem gerar falso positivo.
- O algoritmo de QA/repair não depende de personagens ou termos de uma obra específica.

## Artefatos Principais

- Tradução reparada: `saida/<slug>_pt.md`.
- Repair report: `saida/<slug>_repair_report.json`.
- Repair metrics: `saida/<slug>_repair_metrics.json`.
- Debug do repair: `saida/debug_runs/<slug>/<run>/45_repair/`.
- Refine final: `saida/<slug>_pt_refinado.md`.
- Tempos por etapa: `saida/<slug>_timings.json` e, com `--debug`, `99_reports/timings.json`.
