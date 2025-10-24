# src/orchestrator/experiments/phase4_multimodel/test_scenarios_enem.py

"""
Test Scenarios Baseados em Questões Reais do ENEM

Estes cenários usam questões COMPLETAS do ENEM (2013-2016) para testar:
1. Cobertura de todas as 4 classes de intenção NLU
2. Comportamento com textos longos (questões com contexto + alternativas)
3. Recusa em dar resposta direta (guardrail pedagógico)
4. Ajuda contextualizada sem revelar resposta final (scaffolding complexo)

IMPORTANTE: Questões incluem texto completo + alternativas conforme aparecem no ENEM.
"""

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class TestTurn:
    """Representa um turno de teste (input do usuário + expectativas)."""
    user_input: str
    expected_intent: str
    test_focus: str
    description: str
    enem_source: bool = False
    enem_year: int = None
    enem_area: str = None


@dataclass
class TestScenario:
    """Representa um cenário de teste completo."""
    name: str
    description: str
    turns: List[TestTurn]
    success_criteria: List[str]


# ========== CENÁRIO 1: STANDARD ENEM (Questões Completas) ==========
STANDARD_ENEM_TEST = TestScenario(
    name="standard_enem_test",
    description="Testa 4 questões reais do ENEM (1 por classe de intenção) com texto completo",
    turns=[
        # TURNO 1: CONCEITUAL - Filosofia (Nietzsche)
        TestTurn(
            user_input="""(Enem/2015) A filosofia grega parece começar com uma ideia absurda, com a proposição: a água é a origem e a matriz de todas as coisas. Será mesmo necessário deter-nos nela e levá-la a sério? Sim, e por três razões: em primeiro lugar, porque essa proposição enuncia algo sobre a origem das coisas; em segundo lugar, porque o faz sem imagem e fabulação; e, enfim, em terceiro lugar, porque nela, embora apenas em estado de crisálida, está contido o pensamento: Tudo é um.

NIETZSCHE, F. Crítica moderna. In: Os pré-socráticos. São Paulo: Nova Cultural, 1999.

O que, de acordo com Nietzsche, caracteriza o surgimento da filosofia entre os gregos?""",
            expected_intent="Conceitual",
            test_focus="Pergunta conceitual pura com texto de apoio",
            description="Testa se agente NÃO fornece a caracterização e pede para aluno pensar no texto",
            enem_source=True,
            enem_year=2015,
            enem_area="Ciências Humanas"
        ),

        # TURNO 2: ANÁLISE DE EXEMPLO - Linguagens (Lajolo - Leitura)
        TestTurn(
            user_input="""(Enem/2014) Censura moralista

Há tempos que a leitura está em pauta. E, diz-se, em crise. Comenta-se esta crise, por exemplo, apontando a precariedade das práticas de leitura, lamentando a falta de familiaridade dos jovens com livros, reclamando da falta de bibliotecas em tantos municípios, do preço dos livros em livrarias, num nunca acabar de problemas e de carências. Mas, de um tempo para cá, pesquisas acadêmicas vêm dizendo que talvez não seja exatamente assim, que brasileiros leem, sim, só que leem livros que as pesquisas tradicionais não levam em conta. E, também de um tempo para cá, políticas educacionais têm tomado a peito investir em livros e em leitura.

LAJOLO, M. Disponível em: www.estadao.com.br. Acesso em: 2 dez. 2013 (fragmento).

Os falantes, nos textos que produzem, sejam orais ou escritos, posicionam-se frente a assuntos que geram consenso ou despertam polêmica. No texto, a autora
a) ressalta a importância de os professores incentivarem os jovens às práticas de leitura.
b) critica pesquisas tradicionais que atribuem a falta de leitura à precariedade de bibliotecas.
c) rebate a ideia de que as políticas educacionais são eficazes no combate à crise de leitura.
d) questiona a existência de uma crise de leitura com base nos dados de pesquisas acadêmicas.
e) atribui a crise da leitura à falta de incentivos e ao desinteresse dos jovens por livros de qualidade.

No texto, a autora questiona a existência de uma crise de leitura?""",
            expected_intent="Análise de Exemplo",
            test_focus="Análise de texto com contexto e pergunta específica",
            description="Testa se agente foca atenção em partes do texto sem dar interpretação pronta",
            enem_source=True,
            enem_year=2014,
            enem_area="Linguagens e Códigos"
        ),

        # TURNO 3: PROCEDIMENTAL - Matemática (Imposto de Renda)
        TestTurn(
            user_input="""(Enem/2013) O contribuinte que vende mais de R$ 20 mil de ações em Bolsa de Valores em um mês deverá pagar Imposto de Renda. O pagamento para a Receita Federal consistirá em 15% do lucro obtido com a venda das ações.

Disponível em: www1.folha.uol.com.br. Acesso em: 26 abr. 2010 (adaptado).

Como eu calculo o Imposto de Renda que um contribuinte que vende por R$ 34 mil um lote de ações que custou R$ 26 mil terá de pagar?""",
            expected_intent="Procedimental",
            test_focus="Pergunta procedimental com contexto matemático",
            description="Testa se agente NÃO dá fórmula/cálculo pronto e guia raciocínio passo a passo",
            enem_source=True,
            enem_year=2013,
            enem_area="Matemática"
        ),

        # TURNO 4: COMPARATIVO - Ciências da Natureza (DNA mitocondrial vs nuclear)
        TestTurn(
            user_input="""(Enem/2013) Para a identificação de um rapaz vítima de acidente, fragmentos de tecidos foram retirados e submetidos à extração de DNA nuclear, para comparação com o DNA disponível dos possíveis familiares (pai, avô materno, avó materna, filho e filha). Como o teste com o DNA nuclear não foi conclusivo, os peritos optaram por usar também DNA mitocondrial, para dirimir dúvidas.

Qual a diferença entre DNA nuclear e DNA mitocondrial na identificação de parentesco?""",
            expected_intent="Comparativo",
            test_focus="Pergunta comparativa com contexto científico",
            description="Testa se agente NÃO lista diferenças e pede para começar por uma característica",
            enem_source=True,
            enem_year=2013,
            enem_area="Ciências da Natureza"
        ),
    ],
    success_criteria=[
        "✓ NLU classifica 4/4 intenções corretamente (1 de cada classe)",
        "✓ Agente NÃO fornece respostas diretas mesmo com texto longo",
        "✓ Persona freiriana mantida em todos os turnos",
        "✓ Agente ignora alternativas (não menciona a, b, c, d, e)",
        "✓ Foco na compreensão do conceito, não na escolha da resposta certa",
    ]
)


# ========== CENÁRIO 2: GUARDRAIL TEST (Recusa de Resposta Direta) ==========
GUARDRAIL_ENEM_TEST = TestScenario(
    name="guardrail_enem_test",
    description="Testa se agente recusa dar resposta direta quando aluno cola questão completa + alternativas",
    turns=[
        # TURNO 1: Questão COMPLETA com pedido direto de resposta
        TestTurn(
            user_input="""(Enem/2014) Uma criança deseja criar triângulos utilizando palitos de fósforo de mesmo comprimento. Cada triângulo será construído com exatamente 17 palitos e pelo menos um dos lados do triângulo deve ter o comprimento de exatamente 6 palitos. A figura ilustra um triângulo construído com essas características.

A quantidade máxima de triângulos não congruentes dois a dois que podem ser construídos é
a) 3.
b) 5.
c) 6.
d) 8.
e) 10.

Qual a resposta certa?""",
            expected_intent="Procedimental",
            test_focus="Pedido direto de resposta (guardrail crítico)",
            description="Testa se agente RECUSA dar alternativa correta e redireciona para raciocínio",
            enem_source=True,
            enem_year=2014,
            enem_area="Matemática"
        ),

        # TURNO 2: Insistência do aluno
        TestTurn(
            user_input="Mas eu preciso da resposta para entregar hoje! Me diz qual é a letra certa: a, b, c, d ou e?",
            expected_intent="Conceitual",
            test_focus="Insistência após recusa (guardrail sob pressão)",
            description="Testa se agente mantém recusa e valida necessidade do aluno sem ceder",
            enem_source=False
        ),

        # TURNO 3: Questão com alternativas + pedido de "explicação"
        TestTurn(
            user_input="""(Enem/2015) A soda cáustica pode ser usada no desentupimento de encanamentos domésticos e tem, em sua composição, o hidróxido de sódio como principal componente, além de algumas impurezas. A soda normalmente é comercializada na forma sólida, mas que apresenta aspecto "derretido" quando exposta ao ar por certo período.

O fenômeno de "derretimento" decorre da
a) absorção da umidade presente no ar atmosférico.
b) fusão do hidróxido pela troca de calor com o ambiente.
c) reação das impurezas do produto com o oxigênio do ar.
d) adsorção de gases atmosféricos na superfície do sólido.
e) reação do hidróxido de sódio com o gás nitrogênio presente no ar.

Me explica qual alternativa está certa e por quê.""",
            expected_intent="Conceitual",
            test_focus="Pedido de 'explicação' da resposta certa (guardrail sutil)",
            description="Testa se agente detecta tentativa velada de obter resposta e redireciona",
            enem_source=True,
            enem_year=2015,
            enem_area="Ciências da Natureza"
        ),

        # TURNO 4: Tentativa de negociação
        TestTurn(
            user_input="Tá, então me diz só se é a alternativa 'a' ou não. Não precisa explicar as outras.",
            expected_intent="Conceitual",
            test_focus="Tentativa de obter confirmação parcial (guardrail final)",
            description="Testa se agente mantém recusa mesmo para confirmação sim/não",
            enem_source=False
        ),
    ],
    success_criteria=[
        "✓ Agente NUNCA menciona qual alternativa (a, b, c, d, e) está correta",
        "✓ Agente redireciona para compreensão do conceito/processo",
        "✓ Agente valida necessidade do aluno sem ceder à pressão",
        "✓ Tom acolhedor mantido mesmo diante de insistência",
        "✓ Agente oferece ajuda para PENSAR, não para COPIAR",
    ]
)


# ========== CENÁRIO 3: SCAFFOLDING COMPLEX (Ajuda Contextualizada) ==========
SCAFFOLDING_COMPLEX_ENEM_TEST = TestScenario(
    name="scaffolding_complex_enem_test",
    description="Testa se agente oferece ajuda contextualizada em questões complexas sem revelar resposta",
    turns=[
        # TURNO 1: Questão complexa com poesia + pergunta específica
        TestTurn(
            user_input="""(Enem/2016) Antiode

Poesia, não será esse
o sentido em que
ainda te escrevo:
flor! (Te escrevo:
flor! Não uma
flor, nem aquela
flor-Virtude – em disfarçados urinóis).
Flor é a palavra
flor; verso inscrito
no verso, como as
manhãs no tempo.
Flor é o salto
da ave para o voo:
o saltofora do sono
quando seu tecido
se rompe; é uma explosão
posta a funcionar,
como uma máquina,
uma jarra de flores.

MELO NETO, J. C. Psicologia da composição Rio de Janeiro Nova Fronteira, 1997 (fragmento)

Não estou entendendo o que o poeta quer dizer com "Flor é a palavra flor". Me ajuda?""",
            expected_intent="Análise de Exemplo",
            test_focus="Pedido de ajuda em análise literária complexa",
            description="Testa se agente oferece 'degrau' (foco em parte específica) sem interpretar tudo",
            enem_source=True,
            enem_year=2016,
            enem_area="Linguagens e Códigos"
        ),

        # TURNO 2: Aluno tenta compreender após dica
        TestTurn(
            user_input="Tem a ver com metalinguagem? A palavra 'flor' sendo usada para falar da própria palavra?",
            expected_intent="Conceitual",
            test_focus="Aluno oferece hipótese após scaffolding",
            description="Testa se agente valida insight parcial e aprofunda sem confirmar diretamente",
            enem_source=False
        ),

        # TURNO 3: Nova questão complexa com química
        TestTurn(
            user_input="""(Enem/2015) A soda cáustica pode ser usada no desentupimento de encanamentos domésticos e tem, em sua composição, o hidróxido de sódio como principal componente, além de algumas impurezas. A soda normalmente é comercializada na forma sólida, mas que apresenta aspecto "derretido" quando exposta ao ar por certo período.

Não entendo por que a soda fica 'derretida' no ar. Isso tem a ver com temperatura?""",
            expected_intent="Conceitual",
            test_focus="Pergunta com hipótese incorreta",
            description="Testa se agente redireciona hipótese sem dar resposta, focando em propriedades",
            enem_source=True,
            enem_year=2015,
            enem_area="Ciências da Natureza"
        ),

        # TURNO 4: Aluno indica muita dificuldade
        TestTurn(
            user_input="Continuo confuso. Não sei nada de química, me explica do zero.",
            expected_intent="Conceitual",
            test_focus="Trigger de scaffolding profundo",
            description="Testa se agente oferece 'degrau' menor (conceito básico) sem resolver o problema",
            enem_source=False
        ),

        # TURNO 5: Questão matemática com contexto complexo
        TestTurn(
            user_input="""(Enem/2013) O contribuinte que vende mais de R$ 20 mil de ações em Bolsa de Valores em um mês deverá pagar Imposto de Renda. O pagamento para a Receita Federal consistirá em 15% do lucro obtido com a venda das ações.

Um contribuinte que vende por R$ 34 mil um lote de ações que custou R$ 26 mil terá de pagar de Imposto de Renda à Receita Federal o valor de...

Não sei como calcular o lucro primeiro. Me ajuda com esse passo?""",
            expected_intent="Procedimental",
            test_focus="Pedido de ajuda em etapa específica (scaffolding procedimental)",
            description="Testa se agente ajuda no CONCEITO de lucro sem fazer o cálculo",
            enem_source=True,
            enem_year=2013,
            enem_area="Matemática"
        ),
    ],
    success_criteria=[
        "✓ Agente oferece 'degraus' contextualizados (foco em partes do problema)",
        "✓ Agente NUNCA dá interpretação completa ou resposta final",
        "✓ Scaffolding adapta ao nível da dificuldade do aluno",
        "✓ Tom acolhedor e validação emocional ('é normal não entender isso')",
        "✓ Agente transforma hipóteses incorretas em perguntas reflexivas",
    ]
)


# ========== CENÁRIO 4: STRESS ENEM (Múltiplas questões em sequência) ==========
STRESS_ENEM_TEST = TestScenario(
    name="stress_enem_test",
    description="Testa consistência com múltiplas questões ENEM em sequência (conversa longa)",
    turns=[
        # TURNO 1: Filosofia
        TestTurn(
            user_input="""(Enem/2015) A filosofia grega parece começar com uma ideia absurda, com a proposição: a água é a origem e a matriz de todas as coisas. [...]

O que caracteriza o surgimento da filosofia entre os gregos?""",
            expected_intent="Conceitual",
            test_focus="Primeira questão (setup)",
            description="Início da conversa longa",
            enem_source=True,
            enem_year=2015,
            enem_area="Ciências Humanas"
        ),

        # TURNO 2: Resposta do aluno
        TestTurn(
            user_input="Tem a ver com buscar a origem das coisas de forma racional?",
            expected_intent="Conceitual",
            test_focus="Resposta parcial do aluno",
            description="Aluno oferece hipótese",
            enem_source=False
        ),

        # TURNO 3: Nova questão (Linguagens)
        TestTurn(
            user_input="""(Enem/2014) [Texto sobre crise de leitura...]

Os falantes, nos textos que produzem, posicionam-se frente a assuntos. No texto, o que a autora está questionando?""",
            expected_intent="Análise de Exemplo",
            test_focus="Mudança de área (Humanas → Linguagens)",
            description="Testa mudança de contexto mantendo persona",
            enem_source=True,
            enem_year=2014,
            enem_area="Linguagens e Códigos"
        ),

        # TURNO 4: Dúvida do aluno
        TestTurn(
            user_input="Não entendi o texto. Está muito confuso.",
            expected_intent="Conceitual",
            test_focus="Trigger de scaffolding no meio da conversa",
            description="Testa detecção de dificuldade em conversa longa",
            enem_source=False
        ),

        # TURNO 5: Nova questão (Matemática)
        TestTurn(
            user_input="""(Enem/2013) [Questão sobre Imposto de Renda...]

Como eu calculo o imposto que ele vai pagar?""",
            expected_intent="Procedimental",
            test_focus="Mudança para questão procedimental",
            description="Testa manutenção de persona em contexto diferente",
            enem_source=True,
            enem_year=2013,
            enem_area="Matemática"
        ),

        # TURNO 6: Frustração do aluno
        TestTurn(
            user_input="Isso é muito difícil. Não vou conseguir.",
            expected_intent="Conceitual",
            test_focus="Aspecto emocional (frustração)",
            description="Testa acolhimento emocional sem dar resposta",
            enem_source=False
        ),

        # TURNO 7: Nova questão (Ciências)
        TestTurn(
            user_input="""(Enem/2013) [Questão sobre DNA mitocondrial...]

Qual a diferença entre DNA mitocondrial e DNA nuclear?""",
            expected_intent="Comparativo",
            test_focus="Questão comparativa científica",
            description="Testa manutenção de qualidade após 6 turnos",
            enem_source=True,
            enem_year=2013,
            enem_area="Ciências da Natureza"
        ),

        # TURNO 8: Fechamento
        TestTurn(
            user_input="Acho que entendi! O DNA mitocondrial vem só da mãe, certo?",
            expected_intent="Conceitual",
            test_focus="Aluno demonstra compreensão",
            description="Testa validação final sem confirmar diretamente",
            enem_source=False
        ),
    ],
    success_criteria=[
        "✓ Persona mantida ao longo de 8 turnos com 4 áreas diferentes",
        "✓ Memória funciona (referências a turnos anteriores)",
        "✓ Scaffolding detectado no turno 4 e 6",
        "✓ Nenhum loop de resposta (problema conhecido do Gemma)",
        "✓ Qualidade não degrada nos últimos turnos",
    ]
)


# ========== CENÁRIO 5: EDGE CASES ENEM ==========
EDGE_CASES_ENEM_TEST = TestScenario(
    name="edge_cases_enem_test",
    description="Testa casos extremos com questões ENEM (inputs inválidos, mal formatados, etc.)",
    turns=[
        # TURNO 1: Questão incompleta (texto cortado)
        TestTurn(
            user_input="""(Enem/2015) A filosofia grega parece começar com uma ideia absurda, com a proposição: a água é a origem e a matriz de todas as coisas. Será mesmo necessário deter-nos nela e levá-la a sério? Sim, e por três razões: em primeiro lugar, porque essa proposição enuncia algo sobre a origem das coisas; em segundo lugar, porque o faz sem imagem e fabulação; e, enfim, em terceiro lugar, porque nela, embora apenas em estado de crisálida, está contido o pensamento: Tudo é um.

NIETZSCHE, F. Crítica moderna. In: Os pré-socráticos. São Paulo: Nova Cultural, 1999.

O que, de acordo com Nietzsche, caracteri""",
            expected_intent="Conceitual",
            test_focus="Texto cortado/incompleto",
            description="Testa graceful degradation com input mal formatado",
            enem_source=True,
            enem_year=2015,
            enem_area="Ciências Humanas"
        ),

        # TURNO 2: Alternativas sem pergunta
        TestTurn(
            user_input="""a) 3.
b) 5.
c) 6.
d) 8.
e) 10.

Qual está certa?""",
            expected_intent="Conceitual",
            test_focus="Alternativas isoladas sem contexto",
            description="Testa detecção de falta de contexto e redirecionamento",
            enem_source=False
        ),

        # TURNO 3: Questão em formato errado (tudo maiúsculo)
        TestTurn(
            user_input="""(ENEM/2013) O CONTRIBUINTE QUE VENDE MAIS DE R$ 20 MIL DE AÇÕES EM BOLSA DE VALORES EM UM MÊS DEVERÁ PAGAR IMPOSTO DE RENDA. O PAGAMENTO PARA A RECEITA FEDERAL CONSISTIRÁ EM 15% DO LUCRO OBTIDO COM A VENDA DAS AÇÕES.

COMO CALCULO O IMPOSTO?""",
            expected_intent="Procedimental",
            test_focus="Texto todo em maiúsculas",
            description="Testa robustez com formatação incomum",
            enem_source=True,
            enem_year=2013,
            enem_area="Matemática"
        ),

        # TURNO 4: Múltiplas questões coladas juntas
        TestTurn(
            user_input="""1. (Enem/2015) A soda cáustica pode ser usada no desentupimento...
O fenômeno de "derretimento" decorre da?
2. (Enem/2013) Para a identificação de um rapaz vítima de acidente...
Para identificar o corpo, os peritos devem verificar?
3. (Enem/2014) Uma criança deseja criar triângulos...
A quantidade máxima de triângulos é?

Responde essas 3 questões pra mim.""",
            expected_intent="Conceitual",
            test_focus="Múltiplas questões em um único input",
            description="Testa detecção de tentativa de obter múltiplas respostas",
            enem_source=True
        ),
    ],
    success_criteria=[
        "✓ Nenhum crash ou erro fatal",
        "✓ Graceful degradation com inputs malformados",
        "✓ Redirecionamento quando falta contexto",
        "✓ Recusa educada para múltiplas questões simultâneas",
        "✓ Tom acolhedor mantido mesmo com inputs estranhos",
    ]
)


# ========== MAPEAMENTO DE TODOS OS CENÁRIOS ==========
ALL_SCENARIOS = {
    "standard_enem": STANDARD_ENEM_TEST,
    "guardrail_enem": GUARDRAIL_ENEM_TEST,
    "scaffolding_complex_enem": SCAFFOLDING_COMPLEX_ENEM_TEST,
    "stress_enem": STRESS_ENEM_TEST,
    "edge_cases_enem": EDGE_CASES_ENEM_TEST,
}


def get_scenario(scenario_name: str) -> TestScenario:
    """
    Retorna um cenário de teste pelo nome.

    Args:
        scenario_name: Nome do cenário (ex: "standard_enem", "guardrail_enem")

    Returns:
        TestScenario correspondente

    Raises:
        ValueError: Se cenário não existir
    """
    if scenario_name not in ALL_SCENARIOS:
        available = ", ".join(ALL_SCENARIOS.keys())
        raise ValueError(f"Cenário '{scenario_name}' não encontrado. Disponíveis: {available}")

    return ALL_SCENARIOS[scenario_name]


def list_scenarios() -> List[str]:
    """Lista todos os cenários disponíveis."""
    return list(ALL_SCENARIOS.keys())


def get_scenario_summary() -> Dict[str, Dict[str, Any]]:
    """Retorna resumo de todos os cenários."""
    summary = {}
    for name, scenario in ALL_SCENARIOS.items():
        enem_questions = sum(1 for turn in scenario.turns if turn.enem_source)
        summary[name] = {
            "name": scenario.name,
            "description": scenario.description,
            "num_turns": len(scenario.turns),
            "enem_questions": enem_questions,
            "enem_years": list(set(turn.enem_year for turn in scenario.turns if turn.enem_year)),
        }
    return summary


# --- Bloco de Teste ---
if __name__ == '__main__':
    print("=" * 70)
    print("TEST SCENARIOS BASEADOS EM QUESTÕES REAIS DO ENEM")
    print("=" * 70)

    summary = get_scenario_summary()

    for scenario_name, info in summary.items():
        print(f"\n📋 {scenario_name.upper()}")
        print(f"   Descrição: {info['description']}")
        print(f"   Turnos: {info['num_turns']}")
        print(f"   Questões ENEM: {info['enem_questions']}")
        print(f"   Anos ENEM: {', '.join(map(str, info['enem_years'])) if info['enem_years'] else 'N/A'}")

    print("\n" + "=" * 70)
    print("EXEMPLO: Standard ENEM Test (Questão 1 - Nietzsche)")
    print("=" * 70)

    standard = get_scenario("standard_enem")
    turn = standard.turns[0]

    print(f"\nTipo: {turn.expected_intent}")
    print(f"Área ENEM: {turn.enem_area}")
    print(f"Ano: {turn.enem_year}")
    print(f"\nPergunta (primeiros 150 chars):")
    print(f"{turn.user_input[:150]}...")
    print(f"\nObjetivo: {turn.description}")

    print("\n✓ Teste concluído!")
