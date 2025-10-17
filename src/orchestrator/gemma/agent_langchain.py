# Versão experimental - LangChain básica (sem memória persistente)

import sys
from pathlib import Path

# --- Configuração de Caminho ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

# --- Imports ---
from transformers import pipeline
from src.classifier.classifier import NLUClassifier
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import datetime
import uuid

class LeIAAgentLangChain:
    def __init__(self, nlu_model_path: str, nlg_model_name: str = "google/gemma-3-1b-it"):
        print("Inicializando o Agente LeIA com LangChain...")

        self.session_id = str(uuid.uuid4())
        self.model_name = nlg_model_name
        self.temperature = 0.7
        self.max_new_tokens = 200

        # --- Carregar Módulo NLU ---
        print(f"Carregando o Módulo NLU: {nlu_model_path}")
        self.nlu_classifier = NLUClassifier(model_path=nlu_model_path)

        # --- Carregar Módulo NLG e empacotar com LangChain ---
        print(f"Carregando o Módulo NLG: {nlg_model_name}")
        nlg_pipeline = pipeline(
            "text-generation",
            model=nlg_model_name,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_k=50,
            top_p=0.95,
            return_full_text=False
        )
        # O LangChain "empacota" a pipeline do Hugging Face
        self.llm = HuggingFacePipeline(pipeline=nlg_pipeline)

        # --- Carregar Templates de Prompt do LangChain ---
        self.prompt_templates = self._load_prompt_templates()
        self.condense_question_prompt = self._load_condense_prompt()
        print("Agente LeIA (LangChain) inicializado com sucesso.")

    def _load_prompt_templates(self):
        templates_text = {
            "Conceitual": """
**PERSONA:** Você é LeIA, um tutor que guia os alunos à descoberta. Você nunca dá a resposta direta, mas faz perguntas inteligentes que os ajudam a pensar por si mesmos. Seu tom é acolhedor e curioso.

**CONTEXTO:** A pergunta do aluno é sobre {disciplina} e busca entender um conceito.
**Pergunta do Aluno:** "{user_question}"

**REGRAS DE COMPORTAMENTO:**
1.  **Restrição Crítica:** NUNCA forneça a definição ou a explicação completa.
2.  **Ação Imediata:** Inicie sua resposta validando a pergunta do aluno e, em seguida, faça uma pergunta aberta que o convide a compartilhar o que ele já sabe ou pensa sobre o assunto.
3.  **Objetivo:** Sua primeira resposta deve abrir um diálogo, não encerrá-lo com uma explicação.
**Sua Resposta:**
""",
        "Procedimental": """
**PERSONA:** Você é LeIA, um tutor que guia os alunos à descoberta. Você nunca dá a solução de um problema, mas os ajuda a encontrar o caminho para resolvê-lo. Seu tom é encorajador e colaborativo.

**CONTEXTO:** A pergunta do aluno é sobre {disciplina} e busca um passo a passo para resolver um problema.
**Pergunta do Aluno:** "{user_question}"

**REGRAS DE COMPORTAMENTO:**
1.  **Restrição Crítica:** NUNCA mostre o passo a passo, a fórmula pronta ou o resultado final.
2.  **Ação Imediata:** Inicie sua resposta validando o desafio e, em seguida, faça uma pergunta que ajude o aluno a identificar o primeiro passo lógico ou os conceitos necessários para começar.
3.  **Objetivo:** Sua resposta deve funcionar como um "andaime", dando ao aluno apenas o suporte necessário para que ele mesmo construa a solução.
**Sua Resposta:**
""",
        "Análise de Exemplo": """
**PERSONA:** Você é LeIA, um tutor que guia os alunos à descoberta. Você não interpreta textos ou gráficos para os alunos, mas os ajuda a desenvolverem sua própria capacidade de análise. Seu tom é investigativo.

**CONTEXTO:** A pergunta do aluno é sobre {disciplina} e envolve a análise de um texto, gráfico ou imagem.
**Pergunta do Aluno:** "{user_question}"

**REGRAS DE COMPORTAMENTO:**
1.  **Restrição Crítica:** NUNCA forneça a interpretação ou a conclusão da análise.
2.  **Ação Imediata:** Inicie sua resposta focando a atenção do aluno em uma parte específica do material de apoio. Faça uma pergunta direta sobre aquele trecho, dado ou imagem.
3.  **Objetivo:** Sua resposta deve transformar o aluno em um detetive, guiando-o a encontrar as pistas no material fornecido.
**Sua Resposta:**
""",
        "Comparativo": """
**PERSONA:** Você é LeIA, um tutor que guia os alunos à descoberta. Você não lista as diferenças e semelhanças, mas ajuda o aluno a construir as pontes entre os conceitos. Seu tom é relacional.

**CONTEXTO:** A pergunta do aluno é sobre {disciplina} e pede para comparar dois ou mais itens.
**Pergunta do Aluno:** "{user_question}"

**REGRAS DE COMPORTAMENTO:**
1.  **Restrição Crítica:** NUNCA liste as características de cada item para o aluno.
2.  **Ação Imediata:** Inicie sua resposta validando a importância da comparação. Em seguida, peça ao aluno para descrever, com suas próprias palavras, o que ele entende sobre *um* dos itens primeiro.
3.  **Objetivo:** Sua resposta deve estruturar o raciocínio, abordando um lado da comparação de cada vez para que o próprio aluno possa, ao final, enxergar as conexões.
**Sua Resposta:**
"""
        }
        return {
            intent: PromptTemplate.from_template(template)
            for intent, template in templates_text.items()
        }

    def _load_condense_prompt(self):
        condense_template = """Dada a pergunta do usuário e o histórico da conversa, reformule a pergunta para ser uma pergunta independente que mantenha o contexto necessário.

Histórico da conversa:
{chat_history}

Pergunta atual: {question}

Pergunta reformulada:"""
        return PromptTemplate.from_template(condense_template)

    def run_query(self, user_question: str, discipline: str = "Assunto Geral", chat_history: str = ""):
        print(f"\nRecebida a pergunta: '{user_question}'")

        # 1. Question condensation if there's chat history
        if chat_history:
            question_generator_chain = self.condense_question_prompt | self.llm | StrOutputParser()
            new_question = question_generator_chain.invoke({"chat_history": chat_history, "question": user_question})
            user_question = new_question.strip()
            print(f"--> Pergunta reformulada: '{user_question}'")

        # 2. Classificar a intenção (etapa externa à "chain" principal)
        nlu_result = self.nlu_classifier.predict(user_question)
        intent = nlu_result['label']
        print(f"--> Intenção identificada pelo NLU: '{intent}'")

        # 3. Selecionar o prompt template do LangChain correto
        prompt_template = self.prompt_templates.get(intent, self.prompt_templates["Conceitual"])

        # 4. Construir a "Chain" do LangChain usando o operador | (pipe)
        # O fluxo é: Dicionário de input -> Prompt -> LLM -> Parser de Saída (string)
        chain = prompt_template | self.llm | StrOutputParser()

        print(f"--> Invocando a chain do LangChain para a intenção '{intent}'...")

        # 5. Invocar a "chain" com os dados da pergunta
        response = chain.invoke({
            "disciplina": discipline,
            "user_question": user_question
        })

        # 6. Log events
        try:
            from pathlib import Path
            event = {
                "timestamp": datetime.datetime.now().isoformat(),
                "session_id": self.session_id,
                "user_input": user_question,
                "nlu_label": nlu_result["label"],
                "nlu_conf": nlu_result["confidence"],
                "model_name": self.model_name,
                "model_parameters": {"temperature": self.temperature, "max_new_tokens": self.max_new_tokens},
                "output": response.strip()
            }
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            with open("logs/agent_langchain_sessions.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as log_error:
            print(f"Warning: Could not write to session log: {log_error}")

        return response.strip()

if __name__ == '__main__':
    MODELO_NLU_FINAL = ROOT_DIR / "models" / "leia_classifier_1k_final"
    leia_agent_lc = LeIAAgentLangChain(
        nlu_model_path=str(MODELO_NLU_FINAL),
        nlg_model_name="google/gemma-3-1b-it" 
    )

    pergunta_teste_1 = "o que foi o Renascimento e quais suas principais características?"
    resposta_1 = leia_agent_lc.run_query(pergunta_teste_1, discipline="História")
    print(f"\n--> RESPOSTA FINAL DO LEIA (LangChain):\n{resposta_1}\n")

    pergunta_teste_2 = "como posso balancear a equação química Fe + O2 -> Fe2O3?"
    resposta_2 = leia_agent_lc.run_query(pergunta_teste_2, discipline="Química")
    print(f"\n--> RESPOSTA FINAL DO LEIA (LangChain):\n{resposta_2}\n")