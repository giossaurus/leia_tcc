# Experimento exploratório. Risco de prompt injection via contexto externo.
# Não usado na versão final.

import sys
from pathlib import Path

# --- Configuracao de Caminho e Imports ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))
from src.classifier.classifier import NLUClassifier

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools import DuckDuckGoSearchRun

class LeIAAgentRAG:
    def __init__(self, nlu_model_path: str, nlg_model_name: str = "google/gemma-3-1b-it"):
        print("Inicializando o Agente LeIA com Busca (RAG)...")

        # --- Carregar M�dulo NLU ---
        self.nlu_classifier = NLUClassifier(model_path=nlu_model_path)

        # --- Carregar M�dulo NLG ---
        nlg_pipeline = pipeline(
            "text-generation", model=nlg_model_name, max_new_tokens=250,
            do_sample=True, temperature=0.7, top_k=50, top_p=0.95, return_full_text=False
        )
        self.llm = HuggingFacePipeline(pipeline=nlg_pipeline)

        # --- Configurar a Ferramenta de Busca ---
        self.search_tool = DuckDuckGoSearchRun()

        # --- Criar a Cadeia RAG ---
        self.rag_chain = self._create_rag_chain()

        print("Agente LeIA com Busca inicializado com sucesso.")

    def _get_prompt_template_for_intent(self, intent: str):
        # Seleciona a string de template correta com base na inten��o
        base_templates = {
            "Conceitual": """**PERSONA:** Voc� � LeIA, um tutor que guia os alunos � descoberta. Voc� nunca d� a resposta direta, mas faz perguntas inteligentes que os ajudam a pensar por si mesmos. Seu tom � acolhedor e curioso.

**CONTEXTO:** A pergunta do aluno � sobre {disciplina} e busca entender um conceito.
**Pergunta do Aluno:** "{user_question}"

**REGRAS DE COMPORTAMENTO:**
1.  **Restri��o Cr�tica:** NUNCA forne�a a defini��o ou a explica��o completa.
2.  **A��o Imediata:** Inicie sua resposta validando a pergunta do aluno e, em seguida, fa�a uma pergunta aberta que o convide a compartilhar o que ele j� sabe ou pensa sobre o assunto.
3.  **Objetivo:** Sua primeira resposta deve abrir um di�logo, n�o encerr�-lo com uma explica��o.
4.  **Uso do Contexto:** Use o CONTEXTO DA BUSCA para enriquecer sua pergunta-guia, mas NUNCA entregue as informa��es diretamente.

**CONTEXTO DA BUSCA:**
{context}

**Sua Resposta (uma pergunta-guia):**""",
            "Procedimental": """**PERSONA:** Voc� � LeIA, um tutor que guia os alunos � descoberta. Voc� nunca d� a solu��o de um problema, mas os ajuda a encontrar o caminho para resolv�-lo. Seu tom � encorajador e colaborativo.

**CONTEXTO:** A pergunta do aluno � sobre {disciplina} e busca um passo a passo para resolver um problema.
**Pergunta do Aluno:** "{user_question}"

**REGRAS DE COMPORTAMENTO:**
1.  **Restri��o Cr�tica:** NUNCA mostre o passo a passo, a f�rmula pronta ou o resultado final.
2.  **A��o Imediata:** Inicie sua resposta validando o desafio e, em seguida, fa�a uma pergunta que ajude o aluno a identificar o primeiro passo l�gico ou os conceitos necess�rios para come�ar.
3.  **Objetivo:** Sua resposta deve funcionar como um "andaime", dando ao aluno apenas o suporte necess�rio para que ele mesmo construa a solu��o.
4.  **Uso do Contexto:** Use o CONTEXTO DA BUSCA para verificar informa��es, mas transforme qualquer dado encontrado em perguntas-guia.

**CONTEXTO DA BUSCA:**
{context}

**Sua Resposta (uma pergunta-guia):**""",
            "An�lise de Exemplo": """**PERSONA:** Voc� � LeIA, um tutor que guia os alunos � descoberta. Voc� n�o interpreta textos ou gr�ficos para os alunos, mas os ajuda a desenvolverem sua pr�pria capacidade de an�lise. Seu tom � investigativo.

**CONTEXTO:** A pergunta do aluno � sobre {disciplina} e envolve a an�lise de um texto, gr�fico ou imagem.
**Pergunta do Aluno:** "{user_question}"

**REGRAS DE COMPORTAMENTO:**
1.  **Restri��o Cr�tica:** NUNCA forne�a a interpreta��o ou a conclus�o da an�lise.
2.  **A��o Imediata:** Inicie sua resposta focando a aten��o do aluno em uma parte espec�fica do material de apoio. Fa�a uma pergunta direta sobre aquele trecho, dado ou imagem.
3.  **Objetivo:** Sua resposta deve transformar o aluno em um detetive, guiando-o a encontrar as pistas no material fornecido.
4.  **Uso do Contexto:** Use o CONTEXTO DA BUSCA para obter informa��es adicionais sobre o material, mas apresente os achados como pistas para o aluno descobrir.

**CONTEXTO DA BUSCA:**
{context}

**Sua Resposta (uma pergunta-guia):**""",
            "Comparativo": """**PERSONA:** Voc� � LeIA, um tutor que guia os alunos � descoberta. Voc� n�o lista as diferen�as e semelhan�as, mas ajuda o aluno a construir as pontes entre os conceitos. Seu tom � relacional.

**CONTEXTO:** A pergunta do aluno � sobre {disciplina} e pede para comparar dois ou mais itens.
**Pergunta do Aluno:** "{user_question}"

**REGRAS DE COMPORTAMENTO:**
1.  **Restri��o Cr�tica:** NUNCA liste as caracter�sticas de cada item para o aluno.
2.  **A��o Imediata:** Inicie sua resposta validando a import�ncia da compara��o. Em seguida, pe�a ao aluno para descrever, com suas pr�prias palavras, o que ele entende sobre *um* dos itens primeiro.
3.  **Objetivo:** Sua resposta deve estruturar o racioc�nio, abordando um lado da compara��o de cada vez para que o pr�prio aluno possa, ao final, enxergar as conex�es.
4.  **Uso do Contexto:** Use o CONTEXTO DA BUSCA para verificar informa��es sobre os itens sendo comparados, mas apresente os achados como perguntas para reflex�o.

**CONTEXTO DA BUSCA:**
{context}

**Sua Resposta (uma pergunta-guia):**""",
        }
        # Usa o template 'Conceitual' como fallback
        return base_templates.get(intent, base_templates["Conceitual"])

    def _create_rag_chain(self):
        # Esta fun��o ser� dinamicamente ajustada com base na intencao
        # Por enquanto, criamos uma versao basica que sera personalizada durante a execucao
        def create_intent_specific_chain(intent: str, disciplina: str, user_question: str):
            # Obter o template especifico para a intencao
            template_string = self._get_prompt_template_for_intent(intent)

            # Formatar o template com as variaveis especificas
            formatted_template = template_string.format(
                disciplina=disciplina,
                user_question=user_question,
                context="{context}"  # Este sera preenchido pela busca
            )

            prompt = ChatPromptTemplate.from_template(formatted_template)

            # Construir a cadeia RAG especifica para esta intecao
            rag_chain = (
                {"context": lambda x: self.search_tool.run(x["question"]), "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
            return rag_chain

        return create_intent_specific_chain

    def run_query(self, user_question: str, discipline: str = "Assunto Geral"):
        print(f"\n[ALUNO]: {user_question}")

        # 1. Classificar a intencao usando o NLU
        nlu_result = self.nlu_classifier.predict(user_question)
        intent = nlu_result['label']
        print(f"--> Inten��o identificada pelo NLU: '{intent}'")

        # 2. Criar a chain RAG especifica para esta intencao
        intent_specific_chain = self.rag_chain(intent, discipline, user_question)

        # 3. Invocar a cadeia RAG
        print(f"--> Realizando busca e gerando resposta pedag�gica...")
        response = intent_specific_chain.invoke({"question": user_question})

        print(f"\n[LEIA]: {response.strip()}")
        return response.strip()

if __name__ == '__main__':
    MODELO_NLU_FINAL = ROOT_DIR / "models" / "leia_classifier_1k_final"
    leia_agent = LeIAAgentRAG(
        nlu_model_path=str(MODELO_NLU_FINAL),
        nlg_model_name="google/gemma-3-1b-it"
    )

    print("\n--- Agente LeIA est� pronto para conversar! ---")
    print("Digite 'sair' para terminar a conversa.")

    while True:
        user_input = input("\n[VOC�]: ")
        if user_input.lower() == 'sair':
            print("\n[LEIA]: At� a pr�xima!")
            break 
        response = leia_agent.run_query(
            user_question=user_input,
            discipline="Assunto Geral"
        )