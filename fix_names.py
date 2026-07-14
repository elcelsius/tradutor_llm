import re

file_path = 'tradutor/translate.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Rename 'h' to 'chunk_hash_val' when assigning and checking hash
content = content.replace('h = chunk_hash(chunk)', 'chunk_hash_val = chunk_hash(chunk)')
content = content.replace('cache_exists("translate", h)', 'cache_exists("translate", chunk_hash_val)')
content = content.replace('load_cache("translate", h)', 'load_cache("translate", chunk_hash_val)')
content = content.replace('save_cache(\n                    "translate",\n                    h,', 'save_cache(\n                    "translate",\n                    chunk_hash_val,')

# 2. Rename iq, oq, iql, oql
content = content.replace('iq = _count_quotes(chunk)', 'input_quotes_count = _count_quotes(chunk)')
content = content.replace('oq = _count_quotes(parsed_clean)', 'output_quotes_count = _count_quotes(parsed_clean)')
content = content.replace('iql = count_quote_lines(chunk)', 'input_quote_lines = count_quote_lines(chunk)')
content = content.replace('oql = count_quote_lines(parsed_clean)', 'output_quote_lines = count_quote_lines(parsed_clean)')

content = content.replace('input_quotes=iq,', 'input_quotes=input_quotes_count,')
content = content.replace('output_quotes=oq,', 'output_quotes=output_quotes_count,')
content = content.replace('input_quote_lines=iql,', 'input_quote_lines=input_quote_lines,')
content = content.replace('output_quote_lines=oql,', 'output_quote_lines=output_quote_lines,')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
