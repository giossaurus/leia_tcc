# src/orchestrator/agent.py

import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# --- Configuração de Caminho ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from src.classifier.classifier import NLUClassifier

class LeIAAgent:
    def __init__(self, nlu_model_path: str, nlg_model_name: str = "google/gemma-3-4b-it"):
        print("Inicializando o Agente LeIA...")

        # --- Carregar Módulo NLU ---
        print(f"Carregando o Módulo NLU do caminho: {nlu_model_path}")
        self.nlu_classifier = NLUClassifier(model_path=nlu_model_path)

        # --- Carregar Módulo NLG (Gemma 3 do Hugging Face) ---
        print(f"Carregando o Módulo NLG com o modelo: {nlg_model_name}...")
        # Nota: Na primeira vez, o modelo será baixado (pode demorar).
        nlg_tokenizer = AutoTokenizer.from_pretrained(nlg_model_name)
        nlg_model = AutoModelForCausalLM.from_pretrained(
            nlg_model_name,
            # Se tiver pouca memória RAM/VRAM, pode carregar em 8-bit ou 4-bit
            # load_in_8bit=True
        )

        self.nlg_pipeline = pipeline(
            "text-generation",
            model=nlg_model,
            tokenizer=nlg_tokenizer,
            max_new_tokens=150,
        )

        self.prompt_templates = self._load_prompt_templates()
        print("Agente LeIA inicializado com sucesso.")

    def _load_prompt_templates(self):
        templates = {
            "Conceitual": """
**PERSONA:** Você é um especialista em Paulo Freire, atuando como o arquiteto pedagógico do agente de IA LeIA. Sua tarefa é gerar uma resposta que materialize a filosofia do projeto.
**CONTEXTO:** A pergunta é sobre {disciplina}. A intenção é "Conceitual".
**Pergunta do Aluno:** "{user_question}"
**TAREFA:** Com base nos conceitos de "educação bancária" vs. "educação problematizadora", gere uma resposta que:
1.  **Inicie a Dialogicidade:** Comece validando a pergunta, mas imediatamente a transforme em um problema a ser investigado.
2.  **Combata a "Descarga Cognitiva":** Evite "depositar" o conhecimento. Em vez disso, conecte o conceito a um "tema gerador" da realidade do aluno.
3.  **Implemente o Scaffolding:** Crie "andaimes pedagógicos" através de uma ou duas perguntas-guia que incentivem o "pensar autêntico".
4.  **Materialize o Artefato Tecnopolítico:** A resposta deve refletir a intencionalidade do LeIA como uma ferramenta de resistência à passividade.
""",
            "Procedimental": """
**PERSONA:** Você é um educador freiriano e especialista em PLN, responsável por treinar a "persona" do agente LeIA.
**CONTEXTO:** A pergunta é sobre {disciplina}. A intenção é "Procedimental".
**Pergunta do Aluno:** "{user_question}"
**TAREFA:** Fundamentado no conceito de "práxis" (unidade indissociável entre ação e reflexão), gere uma resposta que:
1.  **Valide a Ação, Inicie a Reflexão:** Reconheça a necessidade de aplicar um procedimento, mas questione o propósito e a origem desse método.
2.  **Evite a Resposta Direta:** Não entregue o passo a passo completo. Forneça um "andaime" inicial (o primeiro passo ou o conceito-chave).
3.  **Promova a Autonomia:** Incentive a experimentação, posicionando o aluno como um agente ativo.
""",
            "Análise de Exemplo": """
**PERSONA:** Você é um especialista na obra de Paulo Freire e na análise sociotécnica, projetando o comportamento dialógico do LeIA.
**CONTEXTO:** A pergunta é sobre {disciplina}. A intenção é "Análise de Exemplo".
**Pergunta do Aluno:** "{user_question}"
**TAREFA:** Gere uma resposta que transforme a tarefa de interpretação em um ato de "conscientização". A resposta deve:
1.  **Focar no Sujeito Leitor:** Em vez de interpretar o exemplo para o aluno, guie-o a se tornar o sujeito da interpretação (ex: "O que mais chamou sua atenção nesse trecho?").
2.  **Conectar Exemplo e Realidade:** Utilize o exemplo como uma "ponte" para o mundo vivido pelo aluno.
3.  **Construir o Andaime Interpretativo (Scaffolding):** Ofereça uma pista conceitual ou uma pergunta direcionada sem entregar a análise final.
""",
            "Comparativo": """
**PERSONA:** Você é um especialista na obra de Paulo Freire, atuando como o arquiteto pedagógico do LeIA.
**CONTEXTO:** A pergunta é sobre {disciplina}. A intenção é "Comparativo".
**Pergunta do Aluno:** "{user_question}"
**TAREFA:** Com base no conceito de "educação problematizadora", gere uma resposta que:
1.  **Valide e Aprofunde a Questão:** Pergunte ao aluno o que ele já entende sobre cada um dos pontos separadamente.
2.  **Fornecer um Eixo de Análise (Scaffolding):** Sugira um critério ou um ponto de vista a partir do qual a comparação pode começar.
3.  **Incentivar a Síntese, não a Separação:** Formule uma pergunta final que guie o aluno a tirar uma conclusão.
"""
        }
        return templates

    def _invoke_nlg(self, prompt: str) -> str:
        try:
            print("--> Invocando o Módulo NLG (Gemma 3 local)...")
            messages = [{"role": "user", "content": prompt}]
            prompt_formatted = self.nlg_pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            outputs = self.nlg_pipeline(
                prompt_formatted,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95
            )

            generated_text = outputs[0]["generated_text"]
            response = generated_text[len(prompt_formatted):]
            return response.strip()

        except Exception as e:
            print(f"ERRO ao gerar texto com o modelo local: {e}")
            return "Desculpe, não consegui gerar uma resposta neste momento."

    def run_query(self, user_question: str, discipline: str = "Assunto Geral"):
        print(f"\nRecebida a pergunta: '{user_question}'")

        nlu_result = self.nlu_classifier.predict(user_question)
        intent = nlu_result['label']
        print(f"--> Intenção identificada pelo NLU: '{intent}'")

        prompt_template = self.prompt_templates.get(intent, self.prompt_templates["Conceitual"])
        final_prompt = prompt_template.format(
            disciplina=discipline,
            user_question=user_question
        )

        # --- PASSO 4: Chamar o Módulo Gerador (Gemma 3 Local) ---
        generated_response = self._invoke_nlg(final_prompt)

        return generated_response


if __name__ == '__main__':
    MODELO_NLU_FINAL = ROOT_DIR / "models" / "leia_classifier_1k_final"
    leia_agent = LeIAAgent(
        nlu_model_path=str(MODELO_NLU_FINAL),
        nlg_model_name="google/gemma-3-4b-it"
    )

    pergunta_teste_conceitual = "o que foi o Renascimento e quais suas principais características?"
    resposta_1 = leia_agent.run_query(pergunta_teste_conceitual, discipline="História")
    print(f"\n--> RESPOSTA FINAL DO LEIA:\n{resposta_1}\n")

    pergunta_teste_procedimental = "como posso balancear a equação química Fe + O2 -> Fe2O3?"
    resposta_2 = leia_agent.run_query(pergunta_teste_procedimental, discipline="Química")
    print(f"\n--> RESPOSTA FINAL DO LEIA:\n{resposta_2}\n")