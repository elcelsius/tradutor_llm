# Benchmarks rápidos de tradução/refine (Ollama)

Siga sempre o padrão abaixo, separando resultados de tradução e de refine.

Tradução (inglês → PT-BR):
```bash
python -m tradutor.bench_llms \
  --input benchmark/teste_traducao_en.md \
  --max-chars 1500 \
  --out-dir benchmark/traducao
```

Refine (texto já em PT-BR):
```bash
python -m tradutor.bench_refine_llms \
  --input benchmark/teste_refine_pt.md \
  --max-chars 1500 \
  --out-dir benchmark/refine
```

Sem `--models`, os scripts usam todos os modelos retornados por `ollama list`. Use `--models <m1> <m2>` para limitar a um subconjunto.

Estrutura sugerida de arquivos:
```
benchmark/
  README.md
  teste.pdf                # opcional, entrada em PDF
  teste.md                 # opcional, entrada simples
  teste_traducao_en.md     # entrada padrão para tradução
  teste_refine_pt.md       # entrada padrão para refine
  traducao/
    resumo_teste_traducao_en.md
    teste_traducao_en_<modelo>.md
    ...
  refine/
    resumo_teste_refine_pt.md
    teste_refine_pt_<modelo>_refine.md
    ...
```
