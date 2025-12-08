# Benchmarks rápidos de tradução/refine (Ollama)

Use o arquivo `benchmark/teste.md` (ou qualquer outro texto) como entrada.

Tradução (inglês -> PT-BR):
```bash
python -m tradutor.bench_llms --input benchmark/teste.md --max-chars 1500
```

Refine (texto já em PT-BR):
```bash
python -m tradutor.bench_refine_llms --input benchmark/teste.md --max-chars 1500
```

Saídas padrão em `benchmark/`:
- Tradução: um `.md` por modelo + `resumo_<slug>.md`
- Refine: um `.md` por modelo + `resumo_refine_<slug>.md`
