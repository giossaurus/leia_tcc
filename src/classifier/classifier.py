# src/classifier/classifier.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import random
import numpy as np
import json
import datetime
import re
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
                top_k=None  # Returns all scores (replaces deprecated return_all_scores)
            )
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)

            print(f"Modelo NLU carregado com sucesso de '{model_path}' (Device: {'GPU' if device == 0 else 'CPU'})")
        except Exception as e:
            print(f"Erro ao carregar o modelo de '{model_path}': {e}")
            raise e

    def _apply_heuristics(self, text: str) -> dict:
        """
        Aplica regras heurísticas baseadas em keywords para classificação rápida.

        Esta camada corrige o viés "Conceitual" do modelo ML, garantindo que perguntas
        procedimentais, comparativas e de análise sejam detectadas corretamente.

        :param text: Pergunta do aluno
        :return: dict com label, confidence e source="Heuristic" ou None se não match
        """
        normalized_text = text.lower()

        # PROCEDIMENTAL: Como fazer, calcular, resolver
        procedimental_patterns = [
            r'\b(como|calcul[eo]|resolv[eo]|fa[çc]o|encontr[eo])\b',
            r'\b(passo a passo|etapas?|procedimento|m[ée]todo)\b',
            r'\b(quantos?|qual o valor|determine)\b.*\b(necessári[oa]s?|precis[oa])\b',
        ]
        for pattern in procedimental_patterns:
            if re.search(pattern, normalized_text):
                return {
                    "label": "Procedimental",
                    "confidence": 0.95,
                    "source": "Heuristic",
                    "pattern": pattern
                }

        # COMPARATIVO: Diferenças, semelhanças, comparar
        comparativo_patterns = [
            r'\b(diferen[çc]as?|semelhan[çc]as?|compar[eo]|contraste)\b',
            r'\b(qual a diferen[çc]a|o que difere)\b',
            r'\b(versus|vs\.?|ou)\b.*\b(qual|entre)\b',
            r'\b(melhor|pior|maior|menor)\b.*\b(entre|que)\b',
        ]
        for pattern in comparativo_patterns:
            if re.search(pattern, normalized_text):
                return {
                    "label": "Comparativo",
                    "confidence": 0.95,
                    "source": "Heuristic",
                    "pattern": pattern
                }

        # ANÁLISE DE EXEMPLO: Gráficos, trechos, exemplos, interpretação
        analise_patterns = [
            r'\b(gr[áa]fico|tabela|imagem|figura|diagrama)\b',
            r'\b(trecho|texto|poema|exemplo|caso)\b',
            r'\b(analis[eo]|interprete?|observ[eo]|identifiqu[eo])\b',
            r'\b(o que.*mostra|o que.*indica|o que.*revela)\b',
            r'\b(ao ler|ao analisar|considerando|nesse)\b.*\b(texto|exemplo)\b',
        ]
        for pattern in analise_patterns:
            if re.search(pattern, normalized_text):
                return {
                    "label": "Análise de Exemplo",
                    "confidence": 0.95,
                    "source": "Heuristic",
                    "pattern": pattern
                }

        # Nenhuma heurística aplicou - retornar None (usar ML)
        return None

    def predict(self, text: str) -> dict:
        """
        Recebe um texto (pergunta do aluno) e retorna a intenção prevista com a maior confiança.

        ESTRATÉGIA HÍBRIDA (Hotfix):
        1. Primeiro tenta heurísticas (keywords) - rápido e preciso para casos óbvios
        2. Se heurística não match, usa modelo ML (DistilBERT) - mais sofisticado
        3. Se ML tem confiança baixa (<55%), fallback para "Conceitual"

        :param text: A pergunta do aluno.
        :return: Um dicionário com a 'label' prevista, 'confidence' e 'source'.
        """
        if not text or not isinstance(text, str):
            return {"label": "N/A", "confidence": 0.0, "error": "Input inválido.", "source": "Error"}

        # ============================================================
        # CAMADA 1: HEURÍSTICAS (Hotfix para corrigir viés Conceitual)
        # ============================================================
        heuristic_result = self._apply_heuristics(text)
        if heuristic_result:
            # Log heuristic classification
            log = {
                "timestamp": datetime.datetime.now().isoformat(),
                "input": text,
                "label": heuristic_result["label"],
                "confidence": heuristic_result["confidence"],
                "source": "Heuristic",
                "pattern_matched": heuristic_result.get("pattern", "N/A")
            }
            try:
                with open("logs/nlu_inference.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(log, ensure_ascii=False) + "\n")
            except Exception as log_error:
                print(f"Warning: Could not write to log file: {log_error}")

            return {
                "label": heuristic_result["label"],
                "confidence": heuristic_result["confidence"],
                "source": "Heuristic"
            }

        # ============================================================
        # CAMADA 2: MODELO ML (Fallback quando heurística não match)
        # ============================================================
        try:
            # Run inference using the pipeline
            # With top_k=None, returns list of all scores: [{"label": "X", "score": 0.9}, ...]
            all_scores = self.pipeline(text)[0]  # [0] to get first (and only) input result

            # Sort by score descending and get top prediction
            sorted_scores = sorted(all_scores, key=lambda x: x["score"], reverse=True)
            top_prediction = sorted_scores[0]

            label = top_prediction["label"]
            score = top_prediction["score"]

            # Fallback para Conceitual se confiança muito baixa
            if score < 0.55:
                label = "Conceitual"  # fallback

            log = {
                "timestamp": datetime.datetime.now().isoformat(),
                "input": text,
                "label": label,
                "confidence": float(score),
                "source": "ML_Model",
                "all_scores": sorted_scores  # Log all scores for debugging
            }

            try:
                with open("logs/nlu_inference.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(log, ensure_ascii=False) + "\n")
            except Exception as log_error:
                print(f"Warning: Could not write to log file: {log_error}")

            return {"label": label, "confidence": score, "source": "ML_Model"}

        except Exception as e:
            print(f"Erro durante a predição: {e}")
            return {"label": "N/A", "confidence": 0.0, "error": str(e), "source": "Error"}

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