import glob
import re

for f in glob.glob("tests/*.py"):
    with open(f, "r", encoding="utf-8") as file:
        content = file.read()

    new_content = re.sub(
        r"\n\s*\"\"\"Processamento interno auxiliar\.\"\"\"\n(\s*\))", r"\n\1", content
    )

    if new_content != content:
        with open(f, "w", encoding="utf-8") as file:
            file.write(new_content)
        print(f"Fixed {f}")
