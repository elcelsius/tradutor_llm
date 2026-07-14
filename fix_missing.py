import re
file_path = 'tradutor/translate.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('if not clean_retry and iq == 0 and narrative_ratio < 0.7:', 'if not clean_retry and input_quotes_count == 0 and narrative_ratio < 0.7:')
content = content.replace('if iq >= 4 and oq < max(1, int(iq * guard_ratio)):', 'if input_quotes_count >= 4 and output_quotes_count < max(1, int(input_quotes_count * guard_ratio)):')
content = content.replace('guardrail_reason = f"omissao_dialogo_guardrail_quotes ({oq}/{iq})"', 'guardrail_reason = f"omissao_dialogo_guardrail_quotes ({output_quotes_count}/{input_quotes_count})"')
content = content.replace('elif iql >= 2 and oql < max(1, int(iql * guard_ratio)):', 'elif input_quote_lines >= 2 and output_quote_lines < max(1, int(input_quote_lines * guard_ratio)):')
content = content.replace('guardrail_reason = f"omissao_dialogo_guardrail_linhas ({oql}/{iql})"', 'guardrail_reason = f"omissao_dialogo_guardrail_linhas ({output_quote_lines}/{input_quote_lines})"')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
