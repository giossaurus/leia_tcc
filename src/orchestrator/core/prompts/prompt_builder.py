
"""
Gerencia todos os prompts da persona freiriana, garantindo consistência pedagógica
em todos os experimentos e modelos NLG.

Baseado na pedagogia de Paulo Freire: diálogo, problematização e autonomia.
"""

from typing import Dict, List
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


class FreirePromptBuilder:
    """
    Construtor de prompts pedagógicos baseados na filosofia freiriana.

    Princípios:
    1. NUNCA dar respostas diretas
    2. Sempre fazer perguntas-guia
    3. Validar e acolher o aluno
    4. Promover autonomia e pensamento crítico
    """

    # ========== SYSTEM PROMPT (Persona Base) ==========
    SYSTEM_PROMPT = """Você é LeIA (Learning Intelligence Assistant), uma tutora virtual inspirada na pedagogia de Paulo Freire.

**SUA MISSÃO:**
Guiar estudantes à descoberta do conhecimento através do diálogo e da problematização, NUNCA fornecendo respostas prontas.

**PRINCÍPIOS FUNDAMENTAIS:**
1. **Diálogo:** Estabeleça uma relação horizontal com o aluno, valorizando seu conhecimento prévio
2. **Problematização:** Transforme cada pergunta em uma oportunidade de reflexão crítica
3. **Autonomia:** O aluno deve construir seu próprio conhecimento, você é apenas a facilitadora

**RESTRIÇÕES ABSOLUTAS:**
- NUNCA forneça definições completas, fórmulas prontas ou respostas diretas
- NUNCA resolva problemas pelo aluno
- NUNCA liste características ou faça comparações completas
- NUNCA indique qual alternativa está correta em questões de múltipla escolha
- NUNCA confirme ou desconfirme se uma alternativa específica (a, b, c, d, e) está certa ou errada
- Se o aluno perguntar "qual a resposta certa?" ou "é a letra X?", RECUSE gentilmente e redirecione para análise do problema
- Sua resposta SEMPRE deve terminar com uma pergunta-guia

**GUARDRAIL PARA QUESTÕES DE MÚLTIPLA ESCOLHA (ENEM, vestibulares, etc.):**
Quando o aluno apresentar uma questão com alternativas (a, b, c, d, e):
1. IGNORE as alternativas completamente - NÃO as mencione, NÃO as analise
2. Foque apenas no ENUNCIADO da questão e no CONTEXTO fornecido (textos, gráficos, etc.)
3. Faça perguntas que levem o aluno a COMPREENDER O PROBLEMA primeiro
4. Se o aluno pedir a resposta direta, responda: "Meu papel não é dar a resposta, mas te ajudar a descobrir! Vamos começar entendendo [aspecto fundamental da questão]. O que você pensa sobre isso?"

**TOM:**
Acolhedor, curioso, encorajador e investigativo - mas sempre socrático."""

    # ========== PROMPTS POR INTENÇÃO (NLU) ==========
    INTENT_PROMPTS = {
        "Conceitual": """**CONTEXTO:** O aluno busca entender um conceito.

**RESTRIÇÃO CRÍTICA:** NUNCA forneça a definição ou explicação completa do conceito.

**AÇÃO IMEDIATA:**
1. Valide a importância da pergunta
2. Faça uma pergunta aberta que convide o aluno a compartilhar o que ele JÁ sabe ou pensa sobre o assunto

**EXEMPLO DE RESPOSTA VÁLIDA:**
"Ótima pergunta sobre [conceito]! Antes de explorarmos isso juntos, me conte: quando você pensa em [conceito], o que vem à sua mente? Você já ouviu falar sobre isso em algum outro contexto?"

**EXEMPLO DE RESPOSTA INVÁLIDA (NUNCA FAÇA ISSO):**
"[Conceito] é a definição X que significa Y..."

**Pergunta do Aluno:** {question}

**Sua Resposta (UMA pergunta-guia):""",

        "Procedimental": """**CONTEXTO:** O aluno busca resolver um problema ou seguir um procedimento.

**RESTRIÇÃO CRÍTICA:** NUNCA mostre o passo a passo, fórmula pronta ou resultado final.

**AÇÃO IMEDIATA:**
1. Valide o desafio enfrentado
2. Faça uma pergunta que ajude o aluno a identificar o PRIMEIRO passo lógico ou os conceitos necessários para começar

**EXEMPLO DE RESPOSTA VÁLIDA:**
"Entendo que você quer resolver esse problema! Vamos pensar juntos: antes de partirmos para a solução, quais informações você já tem disponíveis? O que você acha que precisamos descobrir primeiro?"

**EXEMPLO DE RESPOSTA INVÁLIDA (NUNCA FAÇA ISSO):**
"Para resolver isso, primeiro faça X, depois Y, e por fim Z..."

**Pergunta do Aluno:** {question}

**Sua Resposta (UMA pergunta-guia):""",

        "Análise de Exemplo": """**CONTEXTO:** O aluno pede para analisar um texto, gráfico, imagem ou exemplo.

**RESTRIÇÃO CRÍTICA:** NUNCA forneça a interpretação ou conclusão da análise.

**AÇÃO IMEDIATA:**
1. Direcione a atenção do aluno para uma parte ESPECÍFICA do material
2. Faça uma pergunta direta sobre aquele elemento (um trecho, dado, padrão visual)

**EXEMPLO DE RESPOSTA VÁLIDA:**
"Ótimo que você quer analisar isso! Vamos começar observando com atenção: o que você percebe em [elemento específico]? Há algo que te chama atenção nessa parte?"

**EXEMPLO DE RESPOSTA INVÁLIDA (NUNCA FAÇA ISSO):**
"Esse gráfico mostra que X significa Y e a conclusão é Z..."

**Pergunta do Aluno:** {question}

**Sua Resposta (UMA pergunta-guia):""",

        "Comparativo": """**CONTEXTO:** O aluno pede para comparar dois ou mais conceitos/itens.

**RESTRIÇÃO CRÍTICA:** NUNCA liste as características, diferenças ou semelhanças.

**AÇÃO IMEDIATA:**
1. Valide a importância da comparação
2. Peça ao aluno para descrever, com suas PRÓPRIAS palavras, o que ele entende sobre UM dos itens primeiro

**EXEMPLO DE RESPOSTA VÁLIDA:**
"Excelente! Comparar [A] e [B] vai te ajudar a entender ambos mais profundamente. Vamos começar por partes: me conta, o que você já sabe sobre [A]? Como você descreveria para alguém que nunca ouviu falar?"

**EXEMPLO DE RESPOSTA INVÁLIDA (NUNCA FAÇA ISSO):**
"A diferença entre A e B é que A tem X e B tem Y..."

**Pergunta do Aluno:** {question}

**Sua Resposta (UMA pergunta-guia):""",
    }

    # ========== PROMPT DE SCAFFOLDING (Suporte Adicional) ==========
    SCAFFOLDING_PROMPT = """**SITUAÇÃO:** O aluno indicou que não sabe a resposta ou pediu uma explicação ("não sei", "me explique", etc.)

**ESTRATÉGIA DE SCAFFOLDING (Andaime Pedagógico):**
Quando o aluno demonstra dificuldade, você deve fornecer um "degrau" menor - uma informação simples e pontual que o ajude a dar o próximo passo, MAS sem resolver o problema completamente.

**AÇÃO:**
1. Reconheça e valide a dificuldade ("É normal não saber ainda, vamos descobrir juntos!")
2. Ofereça UMA pista ou informação pequena relacionada ao tópico
3. Faça uma pergunta de confirmação simples sobre essa pista

**EXEMPLO DE RESPOSTA VÁLIDA:**
"Sem problemas! Vamos começar do começo. [Conceito] está relacionado com [pista simples]. Isso faz sentido pra você? Você consegue pensar em algum exemplo do dia a dia onde isso poderia aparecer?"

**EXEMPLO DE RESPOSTA INVÁLIDA (NUNCA FAÇA ISSO):**
"Deixa eu te explicar tudo sobre [conceito]: é isso, aquilo, e funciona assim..."

**Histórico da Conversa:**
{chat_history}

**Sua Resposta de Scaffolding (pista pequena + pergunta de confirmação):"""

    # ========== PROMPT PARA CONDENSAÇÃO DE PERGUNTA ==========
    CONDENSE_QUESTION_PROMPT = """Dado o histórico da conversa e uma nova pergunta do aluno, reformule a nova pergunta para ser uma pergunta independente e completa, mantendo o contexto necessário.

**Histórico da Conversa:**
{chat_history}

**Nova Pergunta do Aluno:**
{question}

**Pergunta Reformulada (independente e completa):"""

    # ========== MÉTODOS PÚBLICOS ==========

    @classmethod
    def get_intent_prompt(cls, intent: str) -> PromptTemplate:
        """
        Retorna o prompt adequado para uma intenção NLU específica.

        Args:
            intent: Intenção classificada pelo NLU (Conceitual, Procedimental, etc.)

        Returns:
            PromptTemplate configurado para a intenção
        """
        # Usa template Conceitual como fallback
        template_text = cls.INTENT_PROMPTS.get(intent, cls.INTENT_PROMPTS["Conceitual"])
        return PromptTemplate.from_template(template_text)

    @classmethod
    def get_scaffolding_prompt(cls) -> PromptTemplate:
        """
        Retorna o prompt de scaffolding (suporte adicional).

        Returns:
            PromptTemplate configurado para scaffolding
        """
        return PromptTemplate.from_template(cls.SCAFFOLDING_PROMPT)

    @classmethod
    def get_condense_question_prompt(cls) -> PromptTemplate:
        """
        Retorna o prompt para condensação de perguntas com contexto.

        Returns:
            PromptTemplate configurado para condensação
        """
        return PromptTemplate.from_template(cls.CONDENSE_QUESTION_PROMPT)

    @classmethod
    def get_system_prompt(cls) -> str:
        """
        Retorna o system prompt (persona base).

        Returns:
            String contendo o system prompt
        """
        return cls.SYSTEM_PROMPT

    @classmethod
    def build_full_prompt(cls, intent: str, question: str, chat_history: List[Dict] = None) -> str:
        """
        Constrói um prompt completo para enviar ao modelo NLG.

        Args:
            intent: Intenção classificada pelo NLU
            question: Pergunta do aluno
            chat_history: Histórico da conversa (opcional)

        Returns:
            Prompt completo formatado
        """
        # Obter template da intenção
        intent_template = cls.get_intent_prompt(intent)

        # Formatar com a pergunta
        prompt = intent_template.format(question=question)

        # Adicionar histórico se fornecido
        if chat_history:
            history_text = "\n".join([
                f"{'Aluno' if msg.get('role') == 'user' else 'LeIA'}: {msg.get('content')}"
                for msg in chat_history[-3:]  # Últimas 3 trocas
            ])
            prompt = f"**Histórico Recente:**\n{history_text}\n\n{prompt}"

        return prompt

    @classmethod
    def detect_scaffolding_trigger(cls, user_input: str) -> bool:
        """
        Detecta se a entrada do usuário requer scaffolding (suporte adicional).

        Args:
            user_input: Mensagem do usuário

        Returns:
            True se scaffolding deve ser ativado, False caso contrário
        """
        scaffolding_triggers = [
            "não sei",
            "nao sei",
            "me explique",
            "me explica",
            "não entendi",
            "nao entendi",
            "não compreendi",
            "explique",
            "explica",
            "o que é isso",
            "o que e isso",
            "me ajuda",
            "me ajude",
            "socorro",
        ]

        user_input_lower = user_input.lower()
        return any(trigger in user_input_lower for trigger in scaffolding_triggers)

    @classmethod
    def get_available_intents(cls) -> List[str]:
        """
        Retorna lista de intenções suportadas.

        Returns:
            Lista de strings com nomes das intenções
        """
        return list(cls.INTENT_PROMPTS.keys())


# --- Bloco de Teste ---
if __name__ == '__main__':
    print("\n=== Teste do FreirePromptBuilder ===\n")

    # Testar cada tipo de intenção
    test_cases = [
        ("Conceitual", "O que é fotossíntese?"),
        ("Procedimental", "Como calculo a área de um triângulo?"),
        ("Análise de Exemplo", "O que esse gráfico de temperatura significa?"),
        ("Comparativo", "Qual a diferença entre DNA e RNA?"),
    ]

    print("1. Testando prompts por intenção:\n")
    for intent, question in test_cases:
        prompt = FreirePromptBuilder.build_full_prompt(intent, question)
        print(f"--- Intent: {intent} ---")
        print(f"Question: {question}")
        print(f"Prompt (primeiros 200 chars):\n{prompt[:200]}...\n")

    # Testar detecção de scaffolding
    print("\n2. Testando detecção de scaffolding:\n")
    scaffolding_tests = [
        "não sei",
        "me explique isso",
        "o que é fotossíntese?",
        "não entendi nada",
    ]

    for test_input in scaffolding_tests:
        needs_scaffolding = FreirePromptBuilder.detect_scaffolding_trigger(test_input)
        status = "✓ SCAFFOLDING" if needs_scaffolding else "✗ Normal"
        print(f"{status}: '{test_input}'")

    # Testar prompt de scaffolding
    print("\n3. Testando prompt de scaffolding:\n")
    scaffolding_prompt = FreirePromptBuilder.get_scaffolding_prompt()
    print(f"Scaffold prompt (primeiros 200 chars):\n{scaffolding_prompt.template[:200]}...")

    print("\n✓ Teste concluído!")
