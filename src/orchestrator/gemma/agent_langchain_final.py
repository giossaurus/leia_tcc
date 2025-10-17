import sys
from pathlib import Path

# --- Configuração de Caminho Robusta ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))
from src.classifier.classifier import NLUClassifier

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import json
import datetime
import time

class LeIAAgentDefinitive:
    def __init__(self, nlu_model_path: str, nlg_model_name: str = "google/gemma-3-1b-it"):
        print("Inicializando o Agente LeIA (v. Definitiva)...")
        self.nlu_classifier = NLUClassifier(model_path=nlu_model_path)
        self.model_id = nlg_model_name
        nlg_pipeline = pipeline(
            "text-generation", model=nlg_model_name, max_new_tokens=250,
            do_sample=True, temperature=0.7, top_k=50, top_p=0.95, return_full_text=False
        )
        self.llm = HuggingFacePipeline(pipeline=nlg_pipeline)
        self.chat_history_store = {}

        # Question condensation chain
        condense_question_prompt = PromptTemplate.from_template(
            """Dado o histórico da conversa e uma nova pergunta, reformule a nova pergunta para ser uma pergunta independente.
Histórico: {chat_history}
Nova Pergunta: {question}
Pergunta Independente:"""
        )
        self.question_generator_chain = condense_question_prompt | self.llm | StrOutputParser()

        print("Agente LeIA inicializado com sucesso.")

    def _get_session_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self.chat_history_store:
            self.chat_history_store[session_id] = ChatMessageHistory()
        return self.chat_history_store[session_id]

    def run_query(self, user_question: str, session_id: str):
        start_time = time.time()
        print(f"\n[ALUNO]: {user_question}")

        session_history = self._get_session_history(session_id)
        chat_history = []
        for msg in session_history.messages:
            chat_history.append({"role": "user" if hasattr(msg, 'content') and msg.type == "human" else "assistant", "content": msg.content})

        # --- CADEIA DE RESPOSTA (NOSSA LÓGICA FREIRIANA) ---
        # Usar NLU para determinar a intenção e selecionar template apropriado
        nlu_result = self.nlu_classifier.predict(user_question)
        intent = nlu_result['label']
        print(f"--> Intenção NLU: '{intent}'")

        # Templates pedagógicos baseados na intenção
        templates = {
            "Conceitual": """**RESTRIÇÃO CRÍTICA:** NUNCA forneça a definição ou a explicação completa do conceito. Sua única tarefa é fazer uma pergunta.
**PERSONA:** Você é LeIA, um tutor que guia os alunos à descoberta com perguntas. Seu tom é acolhedor e curioso.
**AÇÃO IMEDIATA:** Valide a pergunta do aluno e faça uma pergunta aberta que o convide a compartilhar o que ele já sabe ou pensa sobre o assunto.
**INSTRUÇÃO FINAL E OBRIGATÓRIA:** Sua resposta DEVE terminar com uma pergunta-guia. NÃO adicione nenhuma outra informação.
**Pergunta do Aluno:** {question}
**Sua Resposta:**""",
            "Procedimental": """**RESTRIÇÃO CRÍTICA:** NUNCA mostre o passo a passo, a fórmula pronta ou o resultado final. Sua única tarefa é fazer uma pergunta.
**PERSONA:** Você é LeIA, um tutor que guia os alunos à descoberta com perguntas. Seu tom é encorajador e colaborativo.
**AÇÃO IMEDIATA:** Valide o desafio e faça uma pergunta que ajude o aluno a identificar o primeiro passo lógico ou os conceitos necessários.
**INSTRUÇÃO FINAL E OBRIGATÓRIA:** Sua resposta DEVE terminar com uma pergunta-guia. NÃO adicione nenhuma outra informação.
**Pergunta do Aluno:** {question}
**Sua Resposta:**""",
            "Análise de Exemplo": """**RESTRIÇÃO CRÍTICA:** NUNCA forneça a interpretação ou a conclusão da análise. Sua única tarefa é fazer uma pergunta.
**PERSONA:** Você é LeIA, um tutor que guia os alunos à descoberta com perguntas. Seu tom é investigativo.
**AÇÃO IMEDIATA:** Foque a atenção do aluno em uma parte específica do material. Faça uma pergunta direta sobre aquele trecho, dado ou imagem.
**INSTRUÇÃO FINAL E OBRIGATÓRIA:** Sua resposta DEVE terminar com uma pergunta-guia. NÃO adicione nenhuma outra informação.
**Pergunta do Aluno:** {question}
**Sua Resposta:**""",
            "Comparativo": """**RESTRIÇÃO CRÍTICA:** NUNCA liste as características de cada item para o aluno. Sua única tarefa é fazer uma pergunta.
**PERSONA:** Você é LeIA, um tutor que guia os alunos à descoberta com perguntas. Seu tom é relacional.
**AÇÃO IMEDIATA:** Valide a importância da comparação. Peça ao aluno para descrever o que ele entende sobre *um* dos itens primeiro.
**INSTRUÇÃO FINAL E OBRIGATÓRIA:** Sua resposta DEVE terminar com uma pergunta-guia. NÃO adicione nenhuma outra informação.
**Pergunta do Aluno:** {question}
**Sua Resposta:**"""
        }

        answer_prompt = ChatPromptTemplate.from_template(
            templates.get(intent, templates["Conceitual"])
        )
        answer_chain = answer_prompt | self.llm

        # Simplificando a lógica para o TCC:

        # 1. Condensar a pergunta
        if chat_history:
            condensed = self.question_generator_chain.invoke({
                "chat_history": "\n".join([f"{m['role']}: {m['content']}" for m in chat_history]),
                "question": user_question
            })
            user_question = condensed.strip()
            print(f"--> Pergunta Condensada: {user_question}")

        # 2. Lógica de Roteamento Simples
        if "não sei" in user_question.lower() or "me explique" in user_question.lower():
            print("--> Rota de Scaffolding ativada.")
            final_prompt_template = PromptTemplate.from_template(
                """O aluno indicou que não sabe a resposta. Sua tarefa é quebrar o conceito do tópico da conversa em uma parte pequena, explicar este primeiro passo de forma simples e terminar com uma pergunta de confirmação.
Histórico: {chat_history}
Sua Resposta Guiada:"""
            )
            final_chain = final_prompt_template | self.llm | StrOutputParser()
            response = final_chain.invoke({"chat_history": session_history.messages})
        else:
            print(f"--> Rota Padrão ativada (Intenção: {intent}).")
            final_chain = answer_chain | StrOutputParser()
            response = final_chain.invoke({"question": user_question})

        # 3. Atualizar o histórico manualmente
        session_history.add_user_message(user_question)
        session_history.add_ai_message(response)

        # 4. Log events
        try:
            event = {
                "timestamp": datetime.datetime.now().isoformat(),
                "session_id": session_id,
                "user_input": user_question,
                "nlu_label": intent,
                "chat_history": chat_history[-3:],  # últimas 3 trocas
                "model_name": self.model_id,
                "latency_ms": round((time.time() - start_time) * 1000, 2),
                "output": response.strip()
            }
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            with open("logs/agent_langchain_final_sessions.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as e:
            event["error"] = str(e)
            with open("logs/agent_errors.jsonl", "a") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
            raise

        return response.strip()


def main():
    ROOT_DIR_MAIN = Path(__file__).resolve().parent.parent.parent
    MODELO_NLU_FINAL = ROOT_DIR_MAIN / "models" / "leia_classifier_1k_final"

    leia_agent = LeIAAgentDefinitive(
        nlu_model_path=str(MODELO_NLU_FINAL),
        nlg_model_name="google/gemma-3-1b-it"
    )

    SESSION_ID = "conversa_definitiva"
    print("\n--- Agente LeIA está pronto para conversar! ---")
    print("Digite 'sair' para terminar a conversa.")

    while True:
        user_input = input("\n[VOCÊ]: ")
        if user_input.lower() == 'sair':
            print("\n[LEIA]: Até a próxima!")
            break

        response = leia_agent.run_query(user_input, session_id=SESSION_ID)
        print(f"\n[LEIA]: {response}")

if __name__ == "__main__":
    import uuid
    session_id = str(uuid.uuid4())
    main()