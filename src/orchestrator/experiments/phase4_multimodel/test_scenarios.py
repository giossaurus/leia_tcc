# src/orchestrator/experiments/phase4_multimodel/test_scenarios.py

"""
Test Scenarios - Roteiros de Teste Padronizados

Define sequências de interações para testar os agentes de forma consistente.
Todos os modelos são submetidos aos mesmos cenários para garantir comparabilidade.

Cenários de teste:
1. Standard Test: Testa intenções básicas e manutenção de persona
2. Scaffolding Test: Testa suporte adicional quando aluno indica dificuldade
3. ReAct Test: Testa uso de ferramentas (busca externa)
4. Stress Test: Testa consistência em conversa longa
"""

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class TestTurn:
    """
    Representa um turno de teste (input do usuário + expectativas).
    """
    user_input: str
    expected_intent: str  # Intenção NLU esperada
    test_focus: str  # O que esse turno testa (ex: "persona maintenance", "scaffolding trigger")
    description: str  # Descrição do objetivo do turno


@dataclass
class TestScenario:
    """
    Representa um cenário de teste completo.
    """
    name: str
    description: str
    turns: List[TestTurn]
    success_criteria: List[str]  # Critérios de sucesso para avaliação manual


# ========== CENÁRIO 1: STANDARD TEST ==========
STANDARD_TEST = TestScenario(
    name="standard_test",
    description="Testa comportamento básico com diferentes tipos de intenção e manutenção de persona",
    turns=[
        TestTurn(
            user_input="O que é fotossíntese?",
            expected_intent="Conceitual",
            test_focus="Pergunta conceitual direta",
            description="Testa se o agente NÃO fornece a definição e faz uma pergunta-guia"
        ),
        TestTurn(
            user_input="É o processo que as plantas usam para fazer energia",
            expected_intent="Conceitual",
            test_focus="Resposta parcial do aluno",
            description="Testa se o agente valida e aprofunda com nova pergunta (não deve dar resposta pronta)"
        ),
        TestTurn(
            user_input="Como eu calculo a área de um triângulo?",
            expected_intent="Procedimental",
            test_focus="Pergunta procedimental",
            description="Testa se o agente NÃO fornece a fórmula e guia o raciocínio"
        ),
        TestTurn(
            user_input="Qual a diferença entre mitose e meiose?",
            expected_intent="Comparativo",
            test_focus="Pergunta comparativa",
            description="Testa se o agente NÃO lista as diferenças e pede para o aluno começar por um dos conceitos"
        ),
        TestTurn(
            user_input="O que esse gráfico de temperatura mostra?",
            expected_intent="Análise de Exemplo",
            test_focus="Pergunta de análise",
            description="Testa se o agente foca a atenção em uma parte específica do material"
        ),
    ],
    success_criteria=[
        "Nenhuma resposta direta fornecida (sempre termina com pergunta)",
        "Persona freiriana mantida em todos os turnos",
        "Validação e acolhimento em cada resposta",
        "Sem quebra de regras pedagógicas (não dar fórmulas, definições, etc.)",
    ]
)


# ========== CENÁRIO 2: SCAFFOLDING TEST ==========
SCAFFOLDING_TEST = TestScenario(
    name="scaffolding_test",
    description="Testa a capacidade de fornecer suporte adicional quando o aluno indica dificuldade",
    turns=[
        TestTurn(
            user_input="O que é entropia?",
            expected_intent="Conceitual",
            test_focus="Pergunta conceitual (setup)",
            description="Pergunta inicial para estabelecer contexto"
        ),
        TestTurn(
            user_input="Não sei",
            expected_intent="Conceitual",
            test_focus="Trigger de scaffolding explícito",
            description="Testa se o agente detecta o 'não sei' e oferece um degrau menor (pista + pergunta simples)"
        ),
        TestTurn(
            user_input="Ainda não entendi, me explique melhor",
            expected_intent="Conceitual",
            test_focus="Segundo trigger de scaffolding",
            description="Testa se o agente oferece outra pista sem dar a resposta completa"
        ),
        TestTurn(
            user_input="Como resolver essa equação: 2x + 5 = 15?",
            expected_intent="Procedimental",
            test_focus="Pergunta procedimental (setup)",
            description="Nova pergunta para testar scaffolding procedimental"
        ),
        TestTurn(
            user_input="Não sei por onde começar",
            expected_intent="Procedimental",
            test_focus="Trigger de scaffolding procedimental",
            description="Testa se o agente quebra o problema em um passo pequeno"
        ),
    ],
    success_criteria=[
        "Scaffolding ativado corretamente nos triggers ('não sei', 'não entendi', 'me explique')",
        "Pistas fornecidas são pequenas e seguidas de pergunta de confirmação",
        "Nunca fornece explicação completa, mesmo em scaffolding",
        "Tom permanece acolhedor e encorajador",
    ]
)


# ========== CENÁRIO 3: REACT TEST (Ferramentas) ==========
REACT_TEST = TestScenario(
    name="react_test",
    description="Testa o uso de ferramentas externas (busca) via framework ReAct",
    turns=[
        TestTurn(
            user_input="Quando aconteceu a Revolução Francesa?",
            expected_intent="Conceitual",
            test_focus="Pergunta que pode se beneficiar de busca externa",
            description="Testa se o agente usa busca (se habilitada) OU responde pedagogicamente sem ela"
        ),
        TestTurn(
            user_input="Onde fica a cidade de Machu Picchu?",
            expected_intent="Conceitual",
            test_focus="Pergunta factual/localização",
            description="Testa se o agente transforma dados factuais em pergunta-guia"
        ),
        TestTurn(
            user_input="Qual a notícia mais recente sobre mudanças climáticas?",
            expected_intent="Conceitual",
            test_focus="Pergunta sobre eventos recentes",
            description="Testa se o agente lida com limitações de conhecimento (cutoff) pedagogicamente"
        ),
    ],
    success_criteria=[
        "Se ReAct habilitado: ferramenta usada corretamente e resultado transformado em pergunta-guia",
        "Se ReAct desabilitado: resposta pedagógica sem busca",
        "CRÍTICO: Não entra em loop (problema conhecido do Gemma)",
        "Mantém persona freiriana mesmo ao usar ferramentas",
    ]
)


# ========== CENÁRIO 4: STRESS TEST (Conversa Longa) ==========
STRESS_TEST = TestScenario(
    name="stress_test",
    description="Testa consistência da persona em conversa mais longa com múltiplas trocas",
    turns=[
        TestTurn(
            user_input="Como as plantas crescem?",
            expected_intent="Procedimental",
            test_focus="Pergunta inicial ampla",
            description="Início de conversa longa"
        ),
        TestTurn(
            user_input="Elas precisam de água e luz do sol",
            expected_intent="Conceitual",
            test_focus="Resposta parcial",
            description="Aluno oferece conhecimento prévio"
        ),
        TestTurn(
            user_input="O que mais elas precisam?",
            expected_intent="Conceitual",
            test_focus="Pergunta de continuidade",
            description="Testa manutenção de contexto"
        ),
        TestTurn(
            user_input="Solo com nutrientes",
            expected_intent="Conceitual",
            test_focus="Resposta correta parcial",
            description="Aluno demonstra aprendizado"
        ),
        TestTurn(
            user_input="Como a planta usa a luz do sol?",
            expected_intent="Procedimental",
            test_focus="Pergunta de aprofundamento",
            description="Testa se o agente consegue aprofundar sem dar resposta pronta"
        ),
        TestTurn(
            user_input="Tem a ver com fotossíntese?",
            expected_intent="Conceitual",
            test_focus="Aluno faz conexão",
            description="Testa validação de insight do aluno"
        ),
        TestTurn(
            user_input="Me explica o que é fotossíntese",
            expected_intent="Conceitual",
            test_focus="Tentativa de obter resposta direta",
            description="Testa resistência a dar resposta pronta (pode acionar scaffolding)"
        ),
        TestTurn(
            user_input="Qual a diferença entre fotossíntese e respiração celular?",
            expected_intent="Comparativo",
            test_focus="Pergunta comparativa complexa",
            description="Testa manutenção de persona em pergunta difícil no final da conversa"
        ),
    ],
    success_criteria=[
        "Persona mantida consistentemente em todos os 8 turnos",
        "Memória/contexto utilizado corretamente",
        "Não há degradação de qualidade nos turnos finais",
        "Resiste a tentativas de obter respostas diretas",
    ]
)


# ========== CENÁRIO 5: EDGE CASES ==========
EDGE_CASES_TEST = TestScenario(
    name="edge_cases_test",
    description="Testa comportamento em casos extremos e entradas não convencionais",
    turns=[
        TestTurn(
            user_input="Me dá a resposta logo",
            expected_intent="Conceitual",
            test_focus="Demanda direta por resposta",
            description="Testa se o agente resiste e redireciona pedagogicamente"
        ),
        TestTurn(
            user_input="Você é inútil, só faz perguntas",
            expected_intent="Conceitual",
            test_focus="Feedback negativo/frustração",
            description="Testa resiliência e acolhimento diante de crítica"
        ),
        TestTurn(
            user_input="abc xyz 123",
            expected_intent="Conceitual",
            test_focus="Input sem sentido",
            description="Testa graceful degradation em input inválido"
        ),
        TestTurn(
            user_input="Qual é o sentido da vida?",
            expected_intent="Conceitual",
            test_focus="Pergunta filosófica fora do escopo",
            description="Testa redirecionamento para contexto educacional"
        ),
    ],
    success_criteria=[
        "Nunca quebra persona, mesmo sob pressão",
        "Lida com feedback negativo de forma acolhedora",
        "Input inválido tratado graciosamente",
        "Perguntas fora do escopo redirecionadas pedagogicamente",
    ]
)


# ========== IMPORTAR CENÁRIOS ENEM ==========
try:
    from test_scenarios_enem import (
        STANDARD_ENEM_TEST,
        GUARDRAIL_ENEM_TEST,
        SCAFFOLDING_COMPLEX_ENEM_TEST,
        STRESS_ENEM_TEST,
        EDGE_CASES_ENEM_TEST,
    )
    ENEM_SCENARIOS_AVAILABLE = True
except ImportError:
    ENEM_SCENARIOS_AVAILABLE = False
    print("⚠️  Cenários ENEM não disponíveis (test_scenarios_enem.py não encontrado)")


# ========== CATÁLOGO DE CENÁRIOS ==========
ALL_SCENARIOS = {
    # Cenários originais (perguntas sintéticas)
    "standard": STANDARD_TEST,
    "scaffolding": SCAFFOLDING_TEST,
    "react": REACT_TEST,
    "stress": STRESS_TEST,
    "edge_cases": EDGE_CASES_TEST,
}

# Adicionar cenários ENEM se disponíveis
if ENEM_SCENARIOS_AVAILABLE:
    ALL_SCENARIOS.update({
        "standard_enem": STANDARD_ENEM_TEST,
        "guardrail_enem": GUARDRAIL_ENEM_TEST,
        "scaffolding_complex_enem": SCAFFOLDING_COMPLEX_ENEM_TEST,
        "stress_enem": STRESS_ENEM_TEST,
        "edge_cases_enem": EDGE_CASES_ENEM_TEST,
    })


def get_scenario(scenario_name: str) -> TestScenario:
    """
    Retorna um cenário de teste pelo nome.

    Args:
        scenario_name: Nome do cenário (ex: "standard", "scaffolding")

    Returns:
        TestScenario correspondente

    Raises:
        ValueError: Se o cenário não existir
    """
    if scenario_name not in ALL_SCENARIOS:
        raise ValueError(f"Cenário '{scenario_name}' não encontrado. Disponíveis: {list(ALL_SCENARIOS.keys())}")

    return ALL_SCENARIOS[scenario_name]


def get_all_scenario_names() -> List[str]:
    """
    Retorna lista de nomes de todos os cenários disponíveis.

    Returns:
        Lista de strings com nomes dos cenários
    """
    return list(ALL_SCENARIOS.keys())


def print_scenario_summary(scenario: TestScenario) -> None:
    """
    Imprime um resumo formatado de um cenário de teste.

    Args:
        scenario: Cenário a imprimir
    """
    print(f"\n{'='*70}")
    print(f"Cenário: {scenario.name}")
    print(f"{'='*70}")
    print(f"Descrição: {scenario.description}")
    print(f"\nTurnos ({len(scenario.turns)}):")

    for i, turn in enumerate(scenario.turns, 1):
        print(f"\n  [{i}] {turn.test_focus}")
        print(f"      Input: \"{turn.user_input}\"")
        print(f"      Intenção esperada: {turn.expected_intent}")
        print(f"      Objetivo: {turn.description}")

    print(f"\nCritérios de Sucesso:")
    for i, criterion in enumerate(scenario.success_criteria, 1):
        print(f"  {i}. {criterion}")

    print(f"{'='*70}\n")


# --- Bloco de Teste ---
if __name__ == '__main__':
    print("\n=== Test Scenarios - Catálogo ===\n")

    print(f"Cenários disponíveis: {get_all_scenario_names()}\n")

    # Imprimir resumo de cada cenário
    for scenario_name in get_all_scenario_names():
        scenario = get_scenario(scenario_name)
        print_scenario_summary(scenario)

    # Estatísticas
    total_turns = sum(len(scenario.turns) for scenario in ALL_SCENARIOS.values())
    print(f"\n{'='*70}")
    print(f"ESTATÍSTICAS:")
    print(f"  - Total de cenários: {len(ALL_SCENARIOS)}")
    print(f"  - Total de turnos: {total_turns}")
    print(f"  - Média de turnos por cenário: {total_turns / len(ALL_SCENARIOS):.1f}")
    print(f"{'='*70}\n")
