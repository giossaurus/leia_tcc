"""
Carregador de Modelo NLU (Natural Language Understanding)
Wrapper padronizado para o classificador de intenções (DistilBERT fine-tuned).

Este módulo será constante em todos os experimentos da Fase 4.
"""

import sys
from pathlib import Path
import logging

# Configurar caminho para importar o classificador
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

from src.classifier.classifier import NLUClassifier

logger = logging.getLogger(__name__)


class NLULoader:
    """
    Classe responsável por carregar o modelo NLU de forma padronizada.
    """

    DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "leia_classifier_1k_final"

    @staticmethod
    def load_nlu_model(model_path: str = None) -> NLUClassifier:
        """
        Carrega o classificador NLU (DistilBERT fine-tuned).

        Args:
            model_path: Caminho para o modelo treinado. Se None, usa o caminho padrão.

        Returns:
            Instância de NLUClassifier

        Raises:
            ValueError: Se o modelo não puder ser carregado
        """
        if model_path is None:
            model_path = str(NLULoader.DEFAULT_MODEL_PATH)
            logger.info(f"Usando modelo NLU padrão: {model_path}")
        else:
            logger.info(f"Carregando modelo NLU de: {model_path}")

        # Verificar se o caminho existe
        if not Path(model_path).exists():
            error_msg = f"Modelo NLU não encontrado em: {model_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            # Carregar o classificador
            nlu_classifier = NLUClassifier(model_path=model_path)
            logger.info("Modelo NLU carregado com sucesso")
            return nlu_classifier

        except Exception as e:
            logger.error(f"Erro ao carregar modelo NLU: {str(e)}")
            raise ValueError(f"Falha ao carregar modelo NLU de {model_path}: {str(e)}")

    @staticmethod
    def test_nlu_model(nlu_classifier: NLUClassifier) -> None:
        """
        Testa o classificador NLU com perguntas de exemplo.

        Args:
            nlu_classifier: Instância de NLUClassifier a ser testada
        """
        test_questions = [
            ("O que é fotossíntese?", "Conceitual"),
            ("Como faço para calcular a área de um círculo?", "Procedimental"),
            ("Qual a diferença entre mitose e meiose?", "Comparativo"),
            ("O que significa esse gráfico de temperatura?", "Análise de Exemplo"),
        ]

        logger.info("\n=== Testando Classificador NLU ===")
        print("\n=== Testando Classificador NLU ===")

        all_correct = True
        for question, expected_label in test_questions:
            result = nlu_classifier.predict(question)
            predicted_label = result['label']
            confidence = result['confidence']

            is_correct = predicted_label == expected_label
            all_correct = all_correct and is_correct

            status = "✓" if is_correct else "✗"
            print(f"\n{status} Pergunta: {question}")
            print(f"  Esperado: {expected_label}")
            print(f"  Previsto: {predicted_label} (confiança: {confidence:.2%})")

        if all_correct:
            logger.info("✓ Todos os testes passaram!")
            print("\n✓ Todos os testes passaram!")
        else:
            logger.warning("✗ Alguns testes falharam")
            print("\n✗ Alguns testes falharam")


# --- Função de conveniência para uso rápido ---
def load_nlu_model(model_path: str = None) -> NLUClassifier:
    """
    Função de conveniência para carregar o modelo NLU.
    Wrapper para NLULoader.load_nlu_model().

    Args:
        model_path: Caminho para o modelo treinado. Se None, usa o caminho padrão.

    Returns:
        Instância de NLUClassifier
    """
    return NLULoader.load_nlu_model(model_path)


# --- Bloco de Teste ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("\n=== Teste do NLULoader ===\n")

    try:
        # Carregar modelo NLU
        print("Carregando modelo NLU...")
        nlu = load_nlu_model()

        # Testar com perguntas de exemplo
        NLULoader.test_nlu_model(nlu)

        print("\n✓ Teste concluído com sucesso!")

    except Exception as e:
        print(f"\n✗ Erro durante o teste: {str(e)}")
        import traceback
        traceback.print_exc()
