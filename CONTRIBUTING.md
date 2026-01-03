# Contribuindo

## Ambiente
1. Requer Python 3.12.
2. Crie venv e instale dependências:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate  # Windows
   source .venv/bin/activate # Linux
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

## Checks locais
- Testes: `pytest -q`
- Hooks: `pre-commit run --all-files`

## Dicas
- Tudo é UTF-8 (há testes/hook de mojibake).
- Caches ficam em `saida/cache_*`; evite commitar saídas reais.
- CLI principal em `tradutor/main.py` (subcomandos `traduz`, `traduz-md`, `refina`, `pdf`).
