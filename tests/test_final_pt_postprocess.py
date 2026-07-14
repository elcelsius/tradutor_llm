from tradutor.postprocess import final_pt_postprocess


def test_final_pt_postprocess_fixes_small_editorial_artifacts() -> None:
    text = (
        "O criatura caiu.””\n\n"
        "Eu fui mandada para as Ruínas do Descarte? Não: eu ter sido mandada para as Ruínas do Descarte.\n\n"
        "Precisamos enterrar o machado de guerra e reestrear contato.\n\n"
        "Foi sua multidão de estratégias que venceu o dia. Seu socorro e no combate foi essencial.\n\n"
        "Foi sua muitas estratégias que decidiu a vitória. Seu socorro de primeiros socorros e no combate foi essencial.\n\n"
        "Arriscar sua vida talvez tenha sido um pouco demais, embora. Também gostaria de verificar se minhas criações funcionando corretamente.\n\n"
        "Seu primeiro socorro e no combate foi essencial. Eu deve ter esquecido disso. Não acho bom se apegar demais nisso.\n\n"
        "Seu socorro de emergência e na luta contra Kirihara foi essencial. Seu socorro de emergência e no combate foi essencial.\n\n"
        "As máscaras eram selves temporários. Eram seres temporários para servir a um propósito. Ela foi lavada cerebral. O peso que a habilidade exerce em seu corpo para considerar era grande.\n\n"
        "O peso que a habilidade exerce sobre seu corpo para considerar era grande. Ela foi submetida a lavagem cerebral. Usei uma pano. Essa foi um teste.\n\n"
        "O peso que a habilidade exerce sobre seu corpo e que precisamos considerar era grande.\n\n"
        "A carga que sua habilidade Kyokugen impõe ao corpo dela para considerar era grande. Ela tentou convencer a Sogou a me acreditar. Os olhos dela se arregalaram surpresos. Ela se virou de mim. E-eu não é que não confio em você. Eu faço um ótimo Deusazinha. Não faltaram pretendentes?”. Seras-san está aqui, vou poupá-lo de usá-lo como exemplo —, você entende? A Seras-san está aqui então vou poupá-lo de usar ele como exemplo. A Seras-san está aqui então vou poupá-lo de usar como exemplo. A Seras-san está aqui então vou poupá-lo de usá-lo como exemplo. Como a Seras Ashrain está aqui, vou poupá-lo de usar ele como exemplo. Ela come comida e faziam um movimento. Mas Ah… sei disso.\n\n"
        "O alcance é mais ou menos igual ao de Paralisar. Aliás, eu nem sei quanto posso confiar nela, agora que parou pra pensar.\n\n"
        "O alcance é mais ou menos igual ao do Paralisar.\n\n"
        "Havia limites no que podíamos alcançar por meio daquele familiar. Isso exigiria um talento de palco significativo. Todos pensam muito bem demais de mim.\n\n"
        "A habilidade exerce no corpo dela para considerar. Ela foi lavada a cérebro. As identidades temporárias para um propósito bastam.\n\n"
        "Arriscar sua vida pode ter sido um pouco demais, embora. As estratégias ganharam o dia. Desde que todos queremos a mesma coisa. “I see”, disse Hijiri.\n\n"
        "O alcance é mais ou menos o mesmo que Paralisar. Havia limites ao que conseguíamos através daquele familiar. Acredito que o verdadeiro eu seja diferente.\n\n"
        "Ela precisava ser o apanhador no campo de centeio dela, não ser lavada cerebralmente.\n\n"
        "I-isso? I não me importo. KYS. P mais ou menos. Se ela fizesse isso a gente e fizer isso a gente, eu faço um bom imitar. Ela sabe lavar cérebro.\n\n"
        "Ele foi lavado no cérebro e lavado do cérebro. Você deu o dedo dela. Foi só uma piada meio pesada é que. Asagi-san é do tipo misterioso e vai pra juntar com ele. “'Porque isso importa?'”*\n\n"
        "Eu faço uma boa imitação a Deusazinha. Isso é uma missão falha.\n\n"
        "Pode calar a boca por mim, Worm? Tudo o que você serve é para se ajoelhar. Tudo que você serve é para se ajoelhar.\n\n"
        "Vamos estar em real perigo. Use sua habilidade Congelar. Congelar é como uma preservação.\n\n"
        "”“Ah, tudo bem.”\n\n"
        "A tenda de cortina ficava perto da parede de cortina.\n\n"
        "As estratégias venceram o dia; o guarda-corpo decidiu re-estabelecer contato."
    )

    result = final_pt_postprocess(text)

    assert "A criatura caiu.”" in result
    assert "””" not in result
    assert "eu ter sido mandado para as Ruínas do Descarte" in result
    assert "deixar isso para trás" in result
    assert "deixar isso para trás de guerra" not in result
    assert "retomar contato" in result
    assert "muitas estratégias" in result
    assert "Foram as suas muitas estratégias que decidiram a batalha" in result
    assert "Sua ajuda nos primeiros socorros e no combate" in result
    assert "Foram as suas muitas estratégias que decidiram a vitória" in result
    assert "Arriscar sua vida talvez tenha sido um pouco demais." in result
    assert "criações estão funcionando corretamente" in result
    assert "Sua ajuda nos primeiros socorros e no combate foi essencial" in result
    assert "Sua ajuda nos primeiros socorros e na luta contra Kirihara foi essencial" in result
    assert "Sua ajuda nos primeiros socorros e no combate foi essencial" in result
    assert "Eu devo ter esquecido disso" in result
    assert "se apegar demais a isso" in result
    assert "identidades temporárias" in result
    assert "identidades temporárias adequadas a um propósito" in result
    assert "exerce sobre seu corpo e que precisamos considerar" not in result
    assert "exerce sobre seu corpo, algo que precisamos considerar" in result
    assert result.count("exerce sobre seu corpo, algo que precisamos considerar") >= 2
    assert "habilidade de Kyokugen impõe ao corpo dela, algo que precisamos considerar" in result
    assert "a acreditar em mim" in result
    assert "se arregalaram de surpresa" in result
    assert "se virou para longe de mim" in result
    assert "N-não é que eu não confie em você" in result
    assert "eu faço uma ótima imitação da Deusazinha" in result
    assert "Não faltaram pretendentes?”" in result
    assert "poupá-la de usá-lo como exemplo — você entende?" in result
    assert "a Seras-san está aqui, então vou poupá-la de usá-lo como exemplo" in result
    assert "como a Seras Ashrain está aqui, vou poupá-la de usá-lo como exemplo" in result
    assert "come alguma coisa" in result
    assert "tomavam a iniciativa" in result
    assert "Mas, ah… sei disso" in result
    assert "submetida à lavagem cerebral" in result
    assert "um pano" in result
    assert "isso foi um teste" in result
    assert "igual ao da habilidade Paralisar" in result
    assert "agora que parei para pensar" in result
    assert "limites para o que podíamos descobrir por meio daquele familiar" in result
    assert "talento teatral considerável" in result
    assert "têm uma opinião boa demais de mim" in result
    assert "exerce sobre o corpo dela, algo que precisamos considerar" in result
    assert result.count("submetida à lavagem cerebral") >= 2
    assert "identidades temporárias adequadas a um propósito" in result
    assert "Arriscar sua vida pode ter sido um pouco demais." in result
    assert "As estratégias garantiram a vitória" in result
    assert "Desde que todos queiramos a mesma coisa" in result
    assert "“Entendo”, disse Hijiri" in result
    assert "limites para o que conseguíamos descobrir por meio daquele familiar" in result
    assert "Acredito que o verdadeiro você seja diferente" in result
    assert "seu apanhador no campo de centeio" in result
    assert "sofrer lavagem cerebral" in result
    assert "S-sim? Eu não me importo. Se mata. Mais ou menos." in result
    assert "fizesse isso com a gente" in result
    assert "faço uma boa imitação" in result
    assert "fazer lavagem cerebral" in result
    assert "submetido à lavagem cerebral" in result
    assert "fizer isso com a gente" in result
    assert "mostrou o dedo para ela" in result
    assert "Foi só uma piada meio pesada." in result
    assert "Asagi-san é do tipo misteriosa" in result
    assert "para se juntar a ele" in result
    assert "“Porque isso importa?”*" not in result
    assert "“Porque isso importa?”" in result
    assert "imitação da Deusazinha" in result
    assert "missão fracassada" in result
    assert "Pode calar a boca, Worm?" in result
    assert result.count("Você só serve para se ajoelhar") == 2
    assert "em perigo real" in result
    assert "habilidade de Congelar" in result
    assert "Congelar é uma forma de preservação" in result
    assert "”“" not in result
    assert "“Ah, tudo bem.”" in result
    assert "cortina da tenda" in result
    assert "As estratégias garantiram a vitória" in result
    assert "guarda-costas" in result
    assert "restabelecer contato" in result


def test_final_pt_postprocess_fixes_action_calques() -> None:
    result = final_pt_postprocess(
        "Ela viu a espada da Seras — está se cuidando dela. Do que ela tá feliz? "
        "Mantenho o Slei galopando. O monstro estava na pose do ponteiro olímpico. "
        "O pescoço dele é uma cara. Tem cuidado de manter-se fora do alcance de seus ataques. "
        "Alterava sua forma para apoiar a perna de Seras e fornecendo suspensão. "
        "Podíamos agora tomar ações intensas montados sem sermos derrubados do chão. "
        "Ele viu a espada da Seras — está se mantendo fora do alcance dela. Cuidadoso para não chegar perto demais de seus ataques. "
        "Slei estava alterando sua forma para apoiar a parte inferior do corpo de Seras e fornecendo suspensão. "
        "Agora podíamos tomar ações intensas a cavalo sem sermos derrubados. "
        "Ele viu a espada da Seras — está cauteloso com ela. "
        "Agora podíamos tomar ações intensas montados sem sermos lançados ao chão."
    )

    assert result == (
        "Ela viu a espada da Seras — está tomando cuidado com ela. Por que ela está tão feliz? "
        "Continuei galopando com a Slei. O monstro estava na posição de ponte de ginástica. "
        "A base do pescoço dele é um rosto. Toma cuidado para se manter fora do alcance de seus ataques. "
        "Alterava sua forma para apoiar a parte inferior do corpo de Seras e dar sustentação. "
        "Agora podíamos realizar manobras intensas montados sem sermos derrubados. "
        "Ele viu a espada da Seras e está se precavendo contra ela. Toma cuidado para ficar fora do alcance de seus ataques. "
        "Slei estava alterando sua forma para apoiar a parte inferior do corpo de Seras e dar sustentação. "
        "Agora podíamos realizar manobras intensas a cavalo sem sermos derrubados. "
        "Ele viu a espada da Seras — está se precavendo contra ela. "
        "Agora podíamos realizar manobras intensas a cavalo sem sermos derrubados."
    )


def test_final_pt_postprocess_normalizes_dashes_and_action_variants() -> None:
    result = final_pt_postprocess(
        "Ele viu a espada da Seras—está desconfiado dela. Cuidadoso para manter uma distância segura de seus ataques. "
        "O monstro estava na pose do ponte de ginástica. Slei estava alterando sua forma para apoiar o corpo inferior da Seras e fornecendo suspensão. "
        "Agora podíamos realizar manobras intensas a cavalo sem sermos derrubados do chão. Ele respondeu—mas hesitou."
    )

    assert result == (
        "Ele viu a espada da Seras e está se precavendo contra ela. Toma cuidado para ficar fora do alcance de seus ataques. "
        "O monstro estava na posição de ponte de ginástica. Slei estava alterando sua forma para apoiar a parte inferior do corpo de Seras e dar sustentação. "
        "Agora podíamos realizar manobras intensas a cavalo sem sermos derrubados. Ele respondeu — mas hesitou."
    )


def test_final_pt_postprocess_fixes_goddess_calques() -> None:
    result = final_pt_postprocess(
        "Ah, você poderia calar a boca por mim, Worm? Agora então, ajoelhe-se se não se importa. "
        "Ah, você pode calar a boca por mim, Worm? Ainda há aquelas insetos no oeste. "
        "Ah, você Pode calar a boca, Worm? "
        "V-vocês, divindades, podem ser interessantes então, não é? Vocês ganham neles nesse aspecto. "
        "Os humanos ainda têm força nos números. Vou massacrar humanos que você estava prestes a gritar seu amor por eles. "
        "Tragarei o maior sofrimento sobre toda dimensão e todo mundo. Toda existência me pertence."
    )

    assert result == (
        "Ah, poderia calar a boca, Worm? Agora então, ajoelhe-se se não se importar. "
        "Ah, você pode calar a boca, Worm? Ainda há aqueles insetos no oeste. "
        "Ah, você pode calar a boca, Worm? "
        "V-vocês, divindades, até que podem ser interessantes, não são? Vocês levam vantagem nesse aspecto. "
        "Os humanos ainda têm força na quantidade. Vou massacrar humanos por quem você estava prestes a gritar seu amor. "
        "Trarei o maior sofrimento sobre todas as dimensões e todos os mundos. Toda a existência me pertence."
    )


def test_final_pt_postprocess_fixes_nyaki_reunion_artifacts() -> None:
    result = final_pt_postprocess(
        "Você é uma membro importante. “… Boa sorte pra você, Nyaki.” "
        "“Miau-ow — Piggymaru! Slei!” “Nee-nyaaaa —! Waaahn!”"
    )

    assert result == (
        "Você é um membro importante. “… Que bom pra você, Nyaki.” "
        "“Miau-ow—Piggymaru! Slei!” “Nee-nyaaaa—! Waaahn!”"
    )


def test_final_pt_postprocess_fixes_scene_marker_and_late_editorial_artifacts() -> None:
    result = final_pt_postprocess(
        "** Os carros seguiram para o norte.\n\n"
        "Foi sua muitas estratégias que venceu a batalha.\n\n"
        "Ela estava sentada em frente o Yasu.\n\n"
        "E assim nosso carroça chegou. Yasu fora um dos vários que haviam pegado no caminho.\n\n"
        "Ninguém o encarava com raiva — they todos pareciam felizes. Todos emergiram em segurança do outro lado da luta.\n\n"
        "Touka, você foi afiado como uma lâmina. Ser atencioso… é nisso que a Hijiri culpa pelo seu fracasso.\n\n"
        "Gentileza não é o tipo de coisa que você pode rejeitar de mão beijada. Eles precisavam de um boost extra.\n\n"
        "Arright!"
    )

    assert "*** Os carros seguiram para o norte." in result
    assert "Foram as suas muitas estratégias que decidiram a batalha." in result
    assert "em frente ao Yasu" in result
    assert "nossa carruagem chegou" in result
    assert "Yasu fora um dos vários que eles haviam levado consigo pelo caminho" in result
    assert "todos pareciam felizes" in result
    assert "Todos saíram ilesos do confronto" in result
    assert "Touka, você foi muito esperto" in result
    assert "Ser atencioso… é isso que Hijiri culpa pelo próprio fracasso" in result
    assert "Gentileza não é o tipo de coisa que dá pra simplesmente rejeitar assim, de cara" in result
    assert "impulso extra" in result
    assert "Beleza!" in result


def test_final_pt_postprocess_fixes_reviewed_chapter_calques() -> None:
    result = final_pt_postprocess(
        "Vicius pode aumentar seu poder como deusa, talvez? Quando o Congelar for removido, ela pode perder a confiança em Hijiri, que foi quem a convenceu a me apoiar. Mas se enviar nós, heróis, para casa consome muito.\n\n"
        "Num cenário pior, isso basta. Você só está trabalhando de trás pra frente depois da derrota. T-trabalhando de trás pra frente…?\n\n"
        "A presidente da classe está dormindo feito uma princesa e tã bonitinha fazendo isso. Uma daquelas tipos femme fatale, né? Acho que acertei o ponto agora?\n\n"
        "Você guardou tudo através das Ruínas do Descarte? Nossa avó do lado da nossa mãe gostava disso.\n\n"
        "Yasu lembrava do punho fechado que o dono da carruagem fez ao dizer aquelas palavras. Havia mensageiros de montaria rápidos.\n\n"
        "A gente evitava essas criaturas justamente por isso. Deixe o de fora pra gente."
    )

    assert "poder como divindade" in result
    assert "quando o efeito de Congelar for removido" in result
    assert "convenceu a confiar em mim" in result
    assert "enviar todos nós de volta para casa" in result
    assert "no pior dos cenários" in result
    assert "raciocinando de trás para frente" in result
    assert "tá bonita pra caramba assim" in result
    assert "Uma dessas mulheres fatais" in result
    assert "toquei numa ferida agora" in result
    assert "durante a passagem pelas Ruínas do Descarte" in result
    assert "Nossa avó materna" in result
    assert "Yasu se lembrou de como o dono da carruagem fechou o punho" in result
    assert "mensageiros a cavalo velozes" in result
    assert "evitava a estrada justamente por causa dessas criaturas" in result
    assert "Deixe o que está lá fora por nossa conta" in result
