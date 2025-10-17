# Experimento negativo: memória colapsa com pipeline local HuggingFace
# Mantido para documentação no trabalho
import sys
from pathlib import Path
from transformers import pipeline

# --- Configuração de Caminho e Imports ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))
from src.classifier.classifier import NLUClassifier

# Imports do LangChain para a abordagem moderna de memória
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

class LeIAAgentWithMemory:
    def __init__(self, nlu_model_path: str, nlg_model_name: str = "google/gemma-3-1b-it"):
        print("Inicializando o Agente LeIA com Memória (v2)...")

        # Carregar Módulo NLU
        self.nlu_classifier = NLUClassifier(model_path=nlu_model_path)

        # Carregar Módulo NLG
        nlg_pipeline = pipeline(
            "text-generation", model=nlg_model_name, max_new_tokens=200,
            do_sample=True, temperature=0.7, top_k=50, top_p=0.95, return_full_text=False
        )
        self.llm = HuggingFacePipeline(pipeline=nlg_pipeline)

        # Dicionário para armazenar o histórico de cada sessão
        self.chat_history_store = {}

        # Criar a chain com memória
        self.chain_with_memory = self._create_chain_with_memory()

        print("Agente LeIA com Memória inicializado com sucesso.")

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
**Sua Resposta:**""",
            "Procedimental": """**PERSONA:** Você é LeIA, um tutor que guia os alunos à descoberta. Você nunca dá a solução de um problema, mas os ajuda a encontrar o caminho para resolvê-lo. Seu tom é encorajador e colaborativo.

**CONTEXTO:** A pergunta do aluno é sobre {disciplina} e busca um passo a passo para resolver um problema.
**Pergunta do Aluno:** "{user_question}"

**REGRAS DE COMPORTAMENTO:**
1.  **Restrição Crítica:** NUNCA mostre o passo a passo, a fórmula pronta ou o resultado final.
2.  **Ação Imediata:** Inicie sua resposta validando o desafio e, em seguida, faça uma pergunta que ajude o aluno a identificar o primeiro passo lógico ou os conceitos necessários para começar.
3.  **Objetivo:** Sua resposta deve funcionar como um "andaime", dando ao aluno apenas o suporte necessário para que ele mesmo construa a solução.
**Sua Resposta:**""",
            "Análise de Exemplo": """**PERSONA:** Você é LeIA, um tutor que guia os alunos à descoberta. Você não interpreta textos ou gráficos para os alunos, mas os ajuda a desenvolverem sua própria capacidade de análise. Seu tom é investigativo.

**CONTEXTO:** A pergunta do aluno é sobre {disciplina} e envolve a análise de um texto, gráfico ou imagem.
**Pergunta do Aluno:** "{user_question}"

**REGRAS DE COMPORTAMENTO:**
1.  **Restrição Crítica:** NUNCA forneça a interpretação ou a conclusão da análise.
2.  **Ação Imediata:** Inicie sua resposta focando a atenção do aluno em uma parte específica do material de apoio. Faça uma pergunta direta sobre aquele trecho, dado ou imagem.
3.  **Objetivo:** Sua resposta deve transformar o aluno em um detetive, guiando-o a encontrar as pistas no material fornecido.
**Sua Resposta:**""",
            "Comparativo": """**PERSONA:** Você é LeIA, um tutor que guia os alunos à descoberta. Você não lista as diferenças e semelhanças, mas ajuda o aluno a construir as pontes entre os conceitos. Seu tom é relacional.

**CONTEXTO:** A pergunta do aluno é sobre {disciplina} e pede para comparar dois ou mais itens.
**Pergunta do Aluno:** "{user_question}"

**REGRAS DE COMPORTAMENTO:**
1.  **Restrição Crítica:** NUNCA liste as características de cada item para o aluno.
2.  **Ação Imediata:** Inicie sua resposta validando a importância da comparação. Em seguida, peça ao aluno para descrever, com suas próprias palavras, o que ele entende sobre *um* dos itens primeiro.
3.  **Objetivo:** Sua resposta deve estruturar o raciocínio, abordando um lado da comparação de cada vez para que o próprio aluno possa, ao final, enxergar as conexões.
**Sua Resposta:**""",
        }
        # Usa o template 'Conceitual' como fallback
        return base_templates.get(intent, base_templates["Conceitual"])

    def _create_chain_with_memory(self):
        # Função que define a lógica da nossa chain com memória
        def get_session_history(session_id: str) -> ChatMessageHistory:
            if session_id not in self.chat_history_store:
                self.chat_history_store[session_id] = ChatMessageHistory()
            return self.chat_history_store[session_id]

        # Criar uma chain básica que será usada com memória
        # Primeiro vamos criar um template padrão que será dinâmico
        from langchain_core.runnables import RunnableLambda

        def process_with_nlu(inputs: dict):
            # 1. Classificar a intenção a partir do input do usuário
            nlu_result = self.nlu_classifier.predict(inputs["user_question"])
            intent = nlu_result["label"]
            print(f"--> Intenção identificada: '{intent}'")

            # 2. Selecionar o template de prompt correto
            template_string = self._get_prompt_template_for_intent(intent)

            # 3. Criar o prompt do LangChain com placeholders para o histórico
            prompt = ChatPromptTemplate.from_messages([
                ("system", template_string),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{user_question}")
            ])

            # 4. Criar a cadeia de execução final e invocar
            chain = prompt | self.llm
            return chain.invoke(inputs)

        # Criar um runnable a partir da função
        runnable_chain = RunnableLambda(process_with_nlu)

        # "Embrulha" nossa chain com o gerenciador de histórico
        chain_with_memory = RunnableWithMessageHistory(
            runnable_chain,
            get_session_history,
            input_messages_key="user_question",
            history_messages_key="history",
        )
        return chain_with_memory

    def run_query(self, user_question: str, discipline: str, session_id: str):
        print(f"\n[Sessão: {session_id}] [ALUNO]: {user_question}")

        # Invocar a chain, passando o ID da sessão para que ela saiba qual histórico usar
        response = self.chain_with_memory.invoke(
            {"disciplina": discipline, "user_question": user_question},
            config={"configurable": {"session_id": session_id}}
        )

        print(f"\n[LEIA]: {response.strip()}")
        return response.strip()

if __name__ == '__main__':
    MODELO_NLU_FINAL = ROOT_DIR / "models" / "leia_classifier_1k_final"
    leia_agent = LeIAAgentWithMemory(
        nlu_model_path=str(MODELO_NLU_FINAL),
        nlg_model_name="google/gemma-3-1b-it"
    )

    # Simular uma conversa com múltiplos turnos usando um ID de sessão
    SESSION_ID = "conversa_teste_123"

    leia_agent.run_query(
        "o que foi o Renascimento e quais suas principais características?",
        discipline="História",
        session_id=SESSION_ID
    )
    leia_agent.run_query(
        "e o que isso tem a ver com o humanismo?",
        discipline="História",
        session_id=SESSION_ID
    )
    leia_agent.run_query(
        "me dê um exemplo de um artista dessa época.",
        discipline="História",
        session_id=SESSION_ID
    )