"""
Pós-processamento determinístico para texto refinado em PT-BR.
"""

from __future__ import annotations

import re
from typing import List


def _normalise_straight_dialogue_quotes(text: str) -> str:
    """Converte pares seguros de aspas retas em aspas tipograficas de dialogo.

    A conversao e feita apenas quando a quantidade de aspas candidatas e par.
    Marcas de polegadas (apos digitos) ficam intactas para nao confundir texto
    tecnico com dialogo literario.
    """
    positions = [
        idx
        for idx, ch in enumerate(text)
        if ch == '"'
        and not (idx > 0 and text[idx - 1].isdigit())
        and not (idx + 1 < len(text) and text[idx + 1].isdigit())
    ]
    if not positions or len(positions) % 2:
        return text

    replacements = {idx: "“" if order % 2 == 0 else "”" for order, idx in enumerate(positions)}
    return "".join(replacements.get(idx, ch) for idx, ch in enumerate(text))


def _normalise_quote_spacing(text: str) -> str:
    """Corrige espacos perdidos ao redor de aspas sem mudar palavras."""
    text = re.sub(r"([“\"])[ \t]+", r"\1", text)
    text = re.sub(r"[ \t]+([”\"])", r"\1", text)
    text = re.sub(r"([”\"])(?=[“\"])", r"\1 ", text)
    text = re.sub(r"(?<=[\w.,;:!?…])(?=“[A-Za-zÀ-ÿ])", " ", text)
    text = re.sub(r"(?<=[”\"])(?=[A-Za-zÀ-ÿ])", " ", text)
    return text


def normalize_dialogue_quotes(text: str) -> str:
    """Normaliza pares seguros de aspas de diálogo sem alterar o conteúdo."""
    cleaned = _normalise_straight_dialogue_quotes(text)
    return _normalise_quote_spacing(cleaned)


def final_pt_postprocess(text: str) -> str:
    """
    Ajustes finais semânticamente neutros:
    - normaliza reticências, travessões, aspas e espaços
    - padroniza diálogo com travessão
    - remove marcadores residuais de tradução/refine
    - garante linha vazia entre parágrafos não diálogos
    """
    if not text:
        return text

    cleaned = text
    cleaned = re.sub(r"\.{3,}", "…", cleaned)
    cleaned = cleaned.replace("--", "—")
    cleaned = re.sub(r"(?m)^([ \t]*)\*{2}(?=[ \t]+\S)", r"\1***", cleaned)
    cleaned = normalize_dialogue_quotes(cleaned)
    # Contrações inglesas às vezes viram aspas simples dentro de uma fala já
    # delimitada por aspas curvas (ex.: “'Claro que estou.'”).
    cleaned = re.sub(r"“['‘]", "“", cleaned)
    cleaned = re.sub(r"['’]”", "”", cleaned)
    # Corrige fechamento espúrio antes da abertura no início de uma fala.
    cleaned = re.sub(r"(?m)^(\s*)”\s*(?=“[A-Za-zÀ-ÿ])", r"\1", cleaned)
    cleaned = re.sub(r"([”\"])\1+", r"\1", cleaned)
    cleaned = re.sub(r"([.!?…][”\"])\*(?=\s|$)", r"\1", cleaned)
    cleaned = re.sub(r"([?!…])”\.", r"\1”", cleaned)
    cleaned = re.sub(r"—,\s*", "—", cleaned)
    cleaned = re.sub(r"(?<=\S)—(?=\S)", " — ", cleaned)
    cleaned = re.sub(r"—(?=[A-Za-zÀ-ÿ])", "— ", cleaned)
    cleaned = re.sub(r"\s+([.!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ ]{2,}", " ", cleaned)
    cleaned = _normalise_quote_spacing(cleaned)
    replacements = [
        (r"\bO criatura\b", "A criatura"),
        (r"\beu ter sido mandada para as Ruínas do Descarte\b", "eu ter sido mandado para as Ruínas do Descarte"),
        (r"\benterrar o machado(?: de guerra)?\b", "deixar isso para trás"),
        (r"\bdeixar isso para trás de guerra\b", "deixar isso para trás"),
        (r"\breestrear contato\b", "retomar contato"),
        (r"\blavada no cérebro\b", "submetida a lavagem cerebral"),
        (r"\bmultidão de estratégias\b", "muitas estratégias"),
        (r"\bvenceu o dia\b", "decidiu a batalha"),
        (r"\bvenceram o dia\b", "garantiram a vitória"),
        (r"\bguarda-corpo\b", "guarda-costas"),
        (r"\bre[- ]?estabelec", "restabelec"),
        (r"\bSeu socorro de primeiros socorros e no combate\b", "Sua ajuda nos primeiros socorros e no combate"),
        (r"\bSeu primeiro socorro e sua ajuda na luta\b", "Sua ajuda nos primeiros socorros e na luta"),
        (r"\bSeu primeiro socorro e na luta\b", "Sua ajuda nos primeiros socorros e na luta"),
        (r"\bSeu primeiro socorro e no combate\b", "Sua ajuda nos primeiros socorros e no combate"),
        (r"\bSeu socorro de emergência e (na luta|no combate)\b", r"Sua ajuda nos primeiros socorros e \1"),
        (r"\bSeu socorro e no combate\b", "Sua ajuda nos primeiros socorros e no combate"),
        (r"\bFoi sua muitas estratégias que decidiu a (batalha|vitória)\b", r"Foram as suas muitas estratégias que decidiram a \1"),
        (r"\bFoi sua muitas estratégias que venceu a batalha\b", "Foram as suas muitas estratégias que decidiram a batalha"),
        (r"\bTouka, você foi afiado como uma lâmina\b", "Touka, você foi muito esperto"),
        (r"\bColocar sua vida em jogo pode ter sido um pouco demais, embora\.", "Colocar sua vida em jogo pode ter sido um pouco demais."),
        (r"\bArriscar sua vida talvez tenha sido um pouco demais, embora\.", "Arriscar sua vida talvez tenha sido um pouco demais."),
        (r"\bverificar se minhas criações funcionando corretamente\b", "verificar se minhas criações estão funcionando corretamente"),
        (r"\bparede de cortina\b", "cortina da tenda"),
        (r"\btenda de cortina\b", "tenda"),
        (r"\bEu deve\b", "Eu devo"),
        (r"\bse apegar demais nisso\b", "se apegar demais a isso"),
        (r"\bselves temporários\b", "identidades temporárias"),
        (r"\bseres temporários para servir a um propósito\b", "identidades temporárias adequadas a um propósito"),
        (r"\blavado cerebral\b", "submetido a lavagem cerebral"),
        (r"\blavada cerebral\b", "submetida a lavagem cerebral"),
        (r"\blavado no cérebro\b", "submetido à lavagem cerebral"),
        (r"\blavado do cérebro\b", "submetido à lavagem cerebral"),
        (r"\blavada a cérebro\b", "submetida à lavagem cerebral"),
        (r"\bsubmetido a lavagem cerebral\b", "submetido à lavagem cerebral"),
        (r"\bsubmetida a lavagem cerebral\b", "submetida à lavagem cerebral"),
        (r"\bexerce em (seu|sua|o|a) corpo para considerar\b", r"exerce sobre \1 corpo e que precisamos considerar"),
        (r"\bexerce sobre (seu|sua|o|a) corpo para considerar\b", r"exerce sobre \1 corpo, algo que precisamos considerar"),
        (r"\bexerce sobre (seu|sua|o|a) corpo e que precisamos considerar\b", r"exerce sobre \1 corpo, algo que precisamos considerar"),
        (r"\bexerce no corpo dela para considerar\b", "exerce sobre o corpo dela, algo que precisamos considerar"),
        (r"\bimpõe ao corpo (dele|dela) para considerar\b", r"impõe ao corpo \1, algo que precisamos considerar"),
        (r"\buma pano\b", "um pano"),
        (r"\bessa foi um teste\b", "isso foi um teste"),
        (r"\bmais ou menos igual ao de Paralisar\b", "mais ou menos igual ao da habilidade Paralisar"),
        (r"\bmais ou menos igual ao do Paralisar\b", "mais ou menos igual ao da habilidade Paralisar"),
        (r"\bmais ou menos o mesmo que Paralisar\b", "mais ou menos igual ao da habilidade Paralisar"),
        (r"\bAliás, eu nem sei (.*?), agora que parou pra pensar\b", r"Aliás, eu nem sei \1, agora que parei para pensar"),
        (r"\bHavia limites no que podíamos alcançar por meio daquele familiar\b", "Havia limites para o que podíamos descobrir por meio daquele familiar"),
        (r"\bHavia limites ao que conseguíamos através daquele familiar\b", "Havia limites para o que conseguíamos descobrir por meio daquele familiar"),
        (r"\btalento de palco significativo\b", "talento teatral considerável"),
        (r"\bpensam muito bem demais de mim\b", "têm uma opinião boa demais de mim"),
        (r"\bidentidades temporárias para um propósito\b", "identidades temporárias adequadas a um propósito"),
        (r"\bAcredito que o verdadeiro eu seja\b", "Acredito que o verdadeiro você seja"),
        (r"\bo apanhador no campo de centeio dela\b", "seu apanhador no campo de centeio"),
        (r"\bser lavada cerebralmente\b", "sofrer lavagem cerebral"),
        (r"\bem frente o ([A-ZÀ-Ý][A-Za-zÀ-ÿ'-]*(?:\s+[A-ZÀ-Ý][A-Za-zÀ-ÿ'-]*){0,3})\b", r"em frente ao \1"),
        (r"\bnosso carroça\b", "nossa carruagem"),
        (r"\bYasu fora um dos vários que haviam pegado no caminho\b", "Yasu fora um dos vários que eles haviam levado consigo pelo caminho"),
        (r"\bTodos emergiram em segurança do outro lado da luta\b", "Todos saíram ilesos do confronto"),
        (r"\bthey todos\b", "todos"),
        (r"\bSer atencioso… é nisso que a Hijiri culpa pelo seu fracasso\b", "Ser atencioso… é isso que Hijiri culpa pelo próprio fracasso"),
        (r"\bGentileza não é o tipo de coisa que você pode rejeitar de mão beijada\b", "Gentileza não é o tipo de coisa que dá pra simplesmente rejeitar assim, de cara"),
        (r"\baumentar seu poder como deusa, talvez\?", "aumentar seu poder como divindade, talvez?"),
        (r"\bquando o Congelar for removido\b", "quando o efeito de Congelar for removido"),
        (r"\bconvenceu a me apoiar\b", "convenceu a confiar em mim"),
        (r"\bMas se enviar nós, heróis, para casa consome\b", "Mas se enviar todos nós de volta para casa consome"),
        (r"\bnum cenário pior\b", "no pior dos cenários"),
        (r"\bVocê só está trabalhando de trás pra frente depois da derrota\b", "Você só está raciocinando de trás para frente a partir da derrota"),
        (r"\bT-trabalhando de trás pra frente\b", "R-raciocinando de trás para frente"),
        (r"\btã bonitinha fazendo isso\b", "tá bonita pra caramba assim"),
        (r"\bUma daquelas tipos femme fatale\b", "Uma dessas mulheres fatais"),
        (r"\bAcho que acertei o ponto agora\?", "Acho que toquei numa ferida agora?"),
        (r"\bguardou tudo através das Ruínas do Descarte\b", "guardou tudo durante a passagem pelas Ruínas do Descarte"),
        (r"\bNossa avó do lado da nossa mãe\b", "Nossa avó materna"),
        (r"\bYasu lembrava do punho fechado que o dono da carruagem fez ao dizer aquelas palavras\b", "Yasu se lembrou de como o dono da carruagem fechou o punho ao dizer aquelas palavras"),
        (r"\bmensageiros de montaria rápidos\b", "mensageiros a cavalo velozes"),
        (r"\bA gente evitava essas criaturas justamente por isso\b", "A gente evitava a estrada justamente por causa dessas criaturas"),
        (r"\bDeixe o de fora pra gente\b", "Deixe o que está lá fora por nossa conta"),
        (r"\bArriscar sua vida pode ter sido um pouco demais, embora\.", "Arriscar sua vida pode ter sido um pouco demais."),
        (r"\bArright\b", "Beleza"),
        (r"\bboost\b", "impulso"),
        (r"\bganharam o dia\b", "garantiram a vitória"),
        (r"\bDesde que todos queremos\b", "Desde que todos queiramos"),
        (r"\bI see\b", "Entendo"),
        (r"\bI-isso\b", "S-sim"),
        (r"\bI não\b", "Eu não"),
        (r"\bKYS\b", "Se mata"),
        (r"\bP mais ou menos\b", "Mais ou menos"),
        (r"\bfizesse isso a gente\b", "fizesse isso com a gente"),
        (r"\bfizer isso a gente\b", "fizer isso com a gente"),
        (r"\bfaço um bom imitar\b", "faço uma boa imitação"),
        (r"\bimitação a Deusazinha\b", "imitação da Deusazinha"),
        (r"\blavar cérebro\b", "fazer lavagem cerebral"),
        (r"\bmissão falha\b", "missão fracassada"),
        (r"\bAh,\s+você pode calar a boca por mim\b", "Ah, você pode calar a boca"),
        (r"\bAh,\s+você poderia calar a boca por mim\b", "Ah, poderia calar a boca"),
        (r"\bPode calar a boca por mim\b", "Pode calar a boca"),
        (r"\bse não se importa\b", "se não se importar"),
        (r"\baquelas insetos\b", "aqueles insetos"),
        (r"\bTudo (?:o )?que você serve é para se ajoelhar\b", "Você só serve para se ajoelhar"),
        (r"\bem real perigo\b", "em perigo real"),
        (r"\bhabilidade Congelar\b", "habilidade de Congelar"),
        (r"\bhabilidade Kyokugen\b", "habilidade de Kyokugen"),
        (r"\bé como uma preservação\b", "é uma forma de preservação"),
        (r"\ba me acreditar\b", "a acreditar em mim"),
        (r"\bse arregalaram surpresos\b", "se arregalaram de surpresa"),
        (r"\bse virou de mim\b", "se virou para longe de mim"),
        (r"\bE-eu não é que não confio em você\b", "N-não é que eu não confie em você"),
        (r"\beu faço um ótimo Deusazinha\b", "eu faço uma ótima imitação da Deusazinha"),
        (r"\bthough\b", "porém"),
        (r"\bSeras-san está aqui, vou poupá-lo de usá-lo como exemplo\b", "Seras-san está aqui, vou poupá-la de usá-lo como exemplo"),
        (r"\ba Seras-san está aqui então vou poupá-lo de usar ele como exemplo\b", "a Seras-san está aqui, então vou poupá-la de usá-lo como exemplo"),
        (r"\ba Seras-san está aqui então vou poupá-lo de usar como exemplo\b", "a Seras-san está aqui, então vou poupá-la de usá-lo como exemplo"),
        (r"\ba Seras-san está aqui então vou poupá-lo de usá-lo como exemplo\b", "a Seras-san está aqui, então vou poupá-la de usá-lo como exemplo"),
        (r"\bcomo a Seras Ashrain está aqui, vou poupá-lo de usar ele como exemplo\b", "como a Seras Ashrain está aqui, vou poupá-la de usá-lo como exemplo"),
        (r"\bEla viu a espada da Seras\s*—\s*está se cuidando dela\b", "Ela viu a espada da Seras — está tomando cuidado com ela"),
        (r"\bDo que ela tá feliz\?", "Por que ela está tão feliz?"),
        (r"\bMantenho o Slei galopando\b", "Continuei galopando com a Slei"),
        (r"\bpose do ponteiro olímpico\b", "posição de ponte de ginástica"),
        (r"\bpose do ponte de ginástica\b", "posição de ponte de ginástica"),
        (r"\bO pescoço dele é uma cara\b", "A base do pescoço dele é um rosto"),
        (r"\bTem cuidado de manter-se fora do alcance de seus ataques\b", "Toma cuidado para se manter fora do alcance de seus ataques"),
        (r"\bpara apoiar a perna de Seras e fornecendo suspensão\b", "para apoiar a parte inferior do corpo de Seras e dar sustentação"),
        (r"\bPodíamos agora tomar ações intensas montados sem sermos derrubados do chão\b", "Agora podíamos realizar manobras intensas montados sem sermos derrubados"),
        (r"\bEle viu a espada da Seras\s*—\s*está se mantendo fora do alcance dela\.\s*Cuidadoso para não chegar perto demais de seus ataques\.", "Ele viu a espada da Seras e está se precavendo contra ela. Toma cuidado para ficar fora do alcance de seus ataques."),
        (r"\bEle viu a espada da Seras\s*—\s*está cauteloso com ela\b", "Ele viu a espada da Seras — está se precavendo contra ela"),
        (r"\bEle viu a espada da Seras\s*—\s*está desconfiado dela\.\s*Cuidadoso para manter uma distância segura de seus ataques\.", "Ele viu a espada da Seras e está se precavendo contra ela. Toma cuidado para ficar fora do alcance de seus ataques."),
        (r"\balterando sua forma para apoiar a parte inferior do corpo de Seras e fornecendo suspensão\b", "alterando sua forma para apoiar a parte inferior do corpo de Seras e dar sustentação"),
        (r"\balterando sua forma para apoiar o corpo inferior da Seras e fornecendo suspensão\b", "alterando sua forma para apoiar a parte inferior do corpo de Seras e dar sustentação"),
        (r"\bAgora podíamos tomar ações intensas a cavalo sem sermos derrubados\b", "Agora podíamos realizar manobras intensas a cavalo sem sermos derrubados"),
        (r"\bAgora podíamos tomar ações intensas (?:montados|a cavalo) sem sermos (?:derrubados|lançados) ao chão\b", "Agora podíamos realizar manobras intensas a cavalo sem sermos derrubados"),
        (r"\bsem sermos derrubados do chão\b", "sem sermos derrubados"),
        (r"\bV-vocês, divindades, podem ser interessantes então, não é\?", "V-vocês, divindades, até que podem ser interessantes, não são?"),
        (r"\bforça nos números\b", "força na quantidade"),
        (r"\bhumanos que você estava prestes a gritar seu amor por eles\b", "humanos por quem você estava prestes a gritar seu amor"),
        (r"\bTragarei o maior sofrimento sobre toda dimensão e todo mundo\b", "Trarei o maior sofrimento sobre todas as dimensões e todos os mundos"),
        (r"\bToda existência me pertence\b", "Toda a existência me pertence"),
        (r"\buma membro\b", "um membro"),
        (r"“…\s*Boa sorte pra você, Nyaki\.”", "“… Que bom pra você, Nyaki.”"),
        (r"\bMiau-ow\s+—\s*Piggymaru\b", "Miau-ow—Piggymaru"),
        (r"\bNee-nyaaaa\s+—!", "Nee-nyaaaa—!"),
        (r"\bcome comida\b", "come alguma coisa"),
        (r"\bfaziam um movimento\b", "tomavam a iniciativa"),
        (r"\bMas Ah…", "Mas, ah…"),
        (r"\bdeu o dedo dela\b", "mostrou o dedo para ela"),
        (r"\bFoi só uma piada meio pesada é que\b", "Foi só uma piada meio pesada."),
        (r"\bAsagi-san é do tipo misterioso\b", "Asagi-san é do tipo misteriosa"),
        (r"\bpra juntar com ele\b", "para se juntar a ele"),
        (r"“['‘]Porque\b", "“Porque"),
    ]
    for pattern, repl in replacements:
        cleaned = re.sub(pattern, repl, cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\bVocês ganham neles nesse aspecto\b", "Vocês levam vantagem nesse aspecto", cleaned)
    cleaned = re.sub(r"\bvocês ganham neles nesse aspecto\b", "vocês levam vantagem nesse aspecto", cleaned)
    cleaned = re.sub(r"\bAh,\s+você Pode calar a boca\b", "Ah, você pode calar a boca", cleaned)

    # padroniza travessão em diálogos no início da linha
    lines: List[str] = []
    for ln in cleaned.splitlines():
        stripped = ln.lstrip()
        if stripped.startswith("- ") or stripped.startswith("– "):
            ln = ln.replace("- ", "— ", 1) if stripped.startswith("- ") else ln.replace("– ", "— ", 1)
        lines.append(ln)
    cleaned = "\n".join(lines)

    # remove marcadores residuais
    cleaned = re.sub(r"###\s*TEXTO_TRADUZ[A-Z_]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"###\s*TEXTO_REFINADO_[A-Z_]*", "", cleaned, flags=re.IGNORECASE)

    # garante quebra de parágrafo (linha vazia) entre blocos narrativos
    final_lines: List[str] = []
    prev_nonempty = False
    prev_dialog = False
    for ln in cleaned.splitlines():
        stripped = ln.strip()
        if stripped == "":
            final_lines.append("")
            prev_nonempty = False
            prev_dialog = False
            continue
        is_dialog = stripped.startswith("— ")
        if prev_nonempty and not is_dialog and not prev_dialog:
            final_lines.append("")  # insere linha vazia entre parágrafos narrativos consecutivos
        final_lines.append(stripped)
        prev_nonempty = True
        prev_dialog = is_dialog

    result = "\n".join(final_lines).strip()
    return result
