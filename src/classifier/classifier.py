# src/nlu/classifier.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from pathlib import Path
import sys

# Adiciona o diretório raiz do projeto ao path para permitir importações relativas se necessário
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

class NLUClassifier:
    def __init__(self, model_path: str):
        """
        Inicializa o classificador de intenção pedagógica.
        Carrega um modelo e tokenizador treinados a partir de um caminho local.

        :param model_path: Caminho para a pasta do modelo salvo (ex: './models/leia_classifier_1k_final').
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            
            # Utiliza a pipeline do Hugging Face para simplificar a predição e o processamento de softmax
            self.classifier_pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                return_all_scores=True # Retorna a probabilidade de todas as classes
            )
            print(f"Modelo NLU carregado com sucesso de '{model_path}'")
        except Exception as e:
            print(f"Erro ao carregar o modelo de '{model_path}': {e}")
            raise e

    def predict(self, text: str) -> dict:
        """
        Recebe um texto (pergunta do aluno) e retorna a intenção prevista com a maior confiança.

        :param text: A pergunta do aluno.
        :return: Um dicionário com a 'label' prevista e a 'confidence' (probabilidade).
        """
        if not text or not isinstance(text, str):
            return {"label": "N/A", "confidence": 0.0, "error": "Input inválido."}

        try:
            # A pipeline lida com a tokenização, inferência e softmax
            predictions = self.classifier_pipeline(text)[0]
            
            # Encontrar o rótulo com a maior pontuação
            best_prediction = max(predictions, key=lambda x: x['score'])
            
            return {
                "label": best_prediction['label'],
                "confidence": best_prediction['score']
            }
        except Exception as e:
            print(f"Erro durante a predição: {e}")
            return {"label": "N/A", "confidence": 0.0, "error": str(e)}

# --- Bloco de Teste ---
# Este bloco só será executado quando você rodar este arquivo diretamente
# Permite verificar se a classe está funcionando de forma independente
if __name__ == '__main__':
    # Define o caminho para o nosso melhor modelo NLU a partir da raiz do projeto
    MODELO_NLU_FINAL = ROOT_DIR / "models" / "leia_classifier_1k_final"
    
    if not MODELO_NLU_FINAL.exists():
        print(f"ERRO: O diretório do modelo não foi encontrado em '{MODELO_NLU_FINAL}'")
        print("Por favor, verifique se o modelo treinado está no local correto.")
    else:
        # Cria uma instância do nosso classificador
        classifier = NLUClassifier(model_path=str(MODELO_NLU_FINAL))
        
        # Simula algumas perguntas para teste
        pergunta1 = "o que significa o conceito de mais-valia na teoria de Karl Marx?"
        pergunta2 = "como eu faço para balancear a equação química H2 + O2 -> H2O?"
        
        resultado1 = classifier.predict(pergunta1)
        resultado2 = classifier.predict(pergunta2)
        
        print("\n--- Teste da Classe NLUClassifier ---")
        print(f"Pergunta: '{pergunta1}'\n--> Resultado: {resultado1}\n")
        print(f"Pergunta: '{pergunta2}'\n--> Resultado: {resultado2}\n")