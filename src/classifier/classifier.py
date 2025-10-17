# src/classifier/classifier.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import random
import numpy as np
import json
import datetime
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

class NLUClassifier:
    def __init__(self, model_path: str):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        try:
            device = 0 if torch.cuda.is_available() else -1
            self.pipeline = pipeline(
                "text-classification",
                model=model_path,
                tokenizer=model_path,
                device=device,
                return_all_scores=True
            )
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)

            print(f"Modelo NLU carregado com sucesso de '{model_path}' (Device: {'GPU' if device == 0 else 'CPU'})")
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
            # Run inference using the pipeline
            result = self.pipeline(text)[0]
            label = result["label"]
            score = result["score"]

            if score < 0.55:
                label = "Conceitual"  # fallback
            log = {
                "timestamp": datetime.datetime.now().isoformat(),
                "input": text,
                "label": label,
                "confidence": float(score),
                "raw_scores": result
            }

            try:
                with open("logs/nlu_inference.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(log, ensure_ascii=False) + "\n")
            except Exception as log_error:
                print(f"Warning: Could not write to log file: {log_error}")

            return {"label": label, "confidence": score}

        except Exception as e:
            print(f"Erro durante a predição: {e}")
            return {"label": "N/A", "confidence": 0.0, "error": str(e)}

# --- Bloco de Teste ---
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