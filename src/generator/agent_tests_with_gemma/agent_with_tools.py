# Loop detectado entre Action e Observation. Incompatível com RESTRIÇÃO CRÍTICA do LeIA.
import sys
from pathlib import Path

# --- Configuração de Caminho e Imports ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))
from src.classifier.classifier import NLUClassifier

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, create_react_agent

class LeIAAgentWithTools:
    def __init__(self, nlu_model_path: str, nlg_model_name: str = "google/gemma-3-1b-it"):
        print("Inicializando o Agente LeIA com Ferramentas...")

        # --- Carregar Módulos NLU e NLG (como antes) ---
        self.nlu_classifier = NLUClassifier(model_path=nlu_model_path)
        nlg_pipeline = pipeline(
            "text-generation", model=nlg_model_name, max_new_tokens=250,
            do_sample=True, temperature=0.7, top_k=50, top_p=0.95, return_full_text=False
        )
        self.llm = HuggingFacePipeline(pipeline=nlg_pipeline)

        # --- Criar o Agente com Memória e Ferramentas ---
        self.agent_with_memory = self._create_agent_with_memory()

        print("Agente LeIA com Ferramentas inicializado com sucesso.")

    def _get_prompt_template_for_intent(self, intent: str):
        # Seleciona a string de template correta com base na intenção
        base_templates = {
            "Conceitual": """**PERSONA:** Você é LeIA, um tutor que guia os alunos à descoberta. Você nunca dá a resposta direta, mas faz perguntas inteligentes que os ajudam a pensar por si mesmos. Seu tom é acolhedor e curioso.

**CONTEXTO:** A pergunta do aluno é sobre {disciplina} e busca entender um conceito.
**Pergunta do Aluno:** "{user_question}"

**REGRAS DE COMPORTAMENTO:**
1.  **Restrição Crítica:** NUNCA forneça a definição ou a explicação completa.
2.  **Ação Imediata:** Inicie sua resposta validando a pergunta do aluno e, em seguida, faça uma pergunta aberta que o convide a compartilhar o que ele já sabe ou pensa sobre o assunto.
3.  **Objetivo:** Sua primeira resposta deve abrir um diálogo, não encerrá-lo com uma explicação.
4.  **Uso de Ferramentas:** Você tem acesso a ferramentas de busca. Use-as se precisar de informações específicas ou atuais, mas NUNCA entregue o resultado direto da busca.
**Sua Resposta:**""",
            "Procedimental": """**PERSONA:** Você é LeIA, um tutor que guia os alunos à descoberta. Você nunca dá a solução de um problema, mas os ajuda a encontrar o caminho para resolvê-lo. Seu tom é encorajador e colaborativo.

**CONTEXTO:** A pergunta do aluno é sobre {disciplina} e busca um passo a passo para resolver um problema.
**Pergunta do Aluno:** "{user_question}"

**REGRAS DE COMPORTAMENTO:**
1.  **Restrição Crítica:** NUNCA mostre o passo a passo, a fórmula pronta ou o resultado final.
2.  **Ação Imediata:** Inicie sua resposta validando o desafio e, em seguida, faça uma pergunta que ajude o aluno a identificar o primeiro passo lógico ou os conceitos necessários para começar.
3.  **Objetivo:** Sua resposta deve funcionar como um "andaime", dando ao aluno apenas o suporte necessário para que ele mesmo construa a solução.
4.  **Uso de Ferramentas:** Use ferramentas de busca apenas se precisar verificar fórmulas ou conceitos específicos, mas transforme qualquer informação encontrada em perguntas-guia.
**Sua Resposta:**""",
            "Análise de Exemplo": """**PERSONA:** Você é LeIA, um tutor que guia os alunos à descoberta. Você não interpreta textos ou gráficos para os alunos, mas os ajuda a desenvolverem sua própria capacidade de análise. Seu tom é investigativo.

**CONTEXTO:** A pergunta do aluno é sobre {disciplina} e envolve a análise de um texto, gráfico ou imagem.
**Pergunta do Aluno:** "{user_question}"

**REGRAS DE COMPORTAMENTO:**
1.  **Restrição Crítica:** NUNCA forneça a interpretação ou a conclusão da análise.
2.  **Ação Imediata:** Inicie sua resposta focando a atenção do aluno em uma parte específica do material de apoio. Faça uma pergunta direta sobre aquele trecho, dado ou imagem.
3.  **Objetivo:** Sua resposta deve transformar o aluno em um detetive, guiando-o a encontrar as pistas no material fornecido.
4.  **Uso de Ferramentas:** Use ferramentas de busca para contexto adicional sobre o material, mas transforme qualquer informação em pistas para o aluno descobrir.
**Sua Resposta:**""",
            "Comparativo": """**PERSONA:** Você é LeIA, um tutor que guia os alunos à descoberta. Você não lista as diferenças e semelhanças, mas ajuda o aluno a construir as pontes entre os conceitos. Seu tom é relacional.

**CONTEXTO:** A pergunta do aluno é sobre {disciplina} e pede para comparar dois ou mais itens.
**Pergunta do Aluno:** "{user_question}"

**REGRAS DE COMPORTAMENTO:**
1.  **Restrição Crítica:** NUNCA liste as características de cada item para o aluno.
2.  **Ação Imediata:** Inicie sua resposta validando a importância da comparação. Em seguida, peça ao aluno para descrever, com suas próprias palavras, o que ele entende sobre *um* dos itens primeiro.
3.  **Objetivo:** Sua resposta deve estruturar o raciocínio, abordando um lado da comparação de cada vez para que o próprio aluno possa, ao final, enxergar as conexões.
4.  **Uso de Ferramentas:** Use ferramentas de busca para verificar informações sobre os itens sendo comparados, mas apresente os achados como perguntas para reflexão.
**Sua Resposta:**""",
        }
        # Usa o template 'Conceitual' como fallback
        return base_templates.get(intent, base_templates["Conceitual"])

    def _create_agent_with_memory(self):
        # 1. Definir as ferramentas que o agente pode usar
        tools = [DuckDuckGoSearchRun()]

        # 2. Criar o Prompt do Agente adaptado para usar templates baseados em intenção
        # Template base que será dinamicamente preenchido com o template específico da intenção
        agent_template = """Você tem acesso às seguintes ferramentas:

{tools}

Use o seguinte formato:

Question: a pergunta de entrada que você deve responder
Thought: você deve sempre pensar sobre o que fazer
Action: a ação a tomar, deve ser uma das [{tool_names}]
Action Input: a entrada para a ação
Observation: o resultado da ação
... (este Thought/Action/Action Input/Observation pode se repetir N vezes)
Thought: Agora eu sei a resposta final
Final Answer: a resposta final para a pergunta original

{agent_instruction}

**Histórico da Conversa:**
{chat_history}

Question: {input}
Thought: {agent_scratchpad}"""

        prompt = ChatPromptTemplate.from_template(agent_template)

        # 3. Criar o Agente ReAct (Reasoning and Acting)
        agent = create_react_agent(self.llm, tools, prompt)

        # 4. Criar o Executor do Agente
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

        # 5. Adicionar a Memória ao Executor
        chat_history_store = {}
        def get_session_history(session_id: str):
            if session_id not in chat_history_store:
                chat_history_store[session_id] = ChatMessageHistory()
            return chat_history_store[session_id]

        agent_with_memory = RunnableWithMessageHistory(
            agent_executor,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        return agent_with_memory

    def run_query(self, user_question: str, discipline: str = "Assunto Geral", session_id: str = "default"):
        print(f"\n[Sessão: {session_id}] [ALUNO]: {user_question}")

        # 1. Classificar a intenção usando o NLU
        nlu_result = self.nlu_classifier.predict(user_question)
        intent = nlu_result['label']
        print(f"--> Intenção identificada pelo NLU: '{intent}'")

        # 2. Obter o template pedagógico específico para a intenção
        pedagogical_instruction = self._get_prompt_template_for_intent(intent)

        # 3. Formatar o template com as variáveis específicas da pergunta
        formatted_instruction = pedagogical_instruction.format(
            disciplina=discipline,
            user_question=user_question
        )

        # 4. Invocar o agente com a instrução pedagógica específica
        response = self.agent_with_memory.invoke(
            {
                "input": user_question,
                "agent_instruction": formatted_instruction
            },
            config={"configurable": {"session_id": session_id}}
        )

        final_response = response.get("output", "Ocorreu um erro ao processar a resposta.")
        print(f"\n[LEIA]: {final_response}")
        return final_response


if __name__ == '__main__':
    MODELO_NLU_FINAL = ROOT_DIR / "models" / "leia_classifier_1k_final"
    leia_agent = LeIAAgentWithTools(
        nlu_model_path=str(MODELO_NLU_FINAL),
        nlg_model_name="google/gemma-3-1b-it"
    )

    SESSION_ID = "conversa_interativa_1"
    print("\n--- Agente LeIA está pronto para conversar! ---")
    print("Digite 'sair' para terminar a conversa.")

    while True:
        user_input = input("\n[VOCÊ]: ")
        if user_input.lower() == 'sair':
            print("\n[LEIA]: Até a próxima!")
            break

        # Você pode especificar a disciplina aqui ou deixar como padrão
        response = leia_agent.run_query(
            user_question=user_input,
            discipline="Assunto Geral",
            session_id=SESSION_ID
        )