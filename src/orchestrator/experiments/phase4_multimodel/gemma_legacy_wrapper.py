"""
Gemma Legacy Wrapper - Integração dos Experimentos Legados com Nova Infraestrutura

Este wrapper permite executar os experimentos originais do /gemma
(agent_langchain_final, agent_with_memory, agent_with_rag, agent_with_tools)
com a nova infraestrutura de logging, métricas e monitoramento da Fase 4.

Mantém os experimentos originais intactos (para documentação do TCC)
mas adiciona instrumentação robusta para comparação com novos modelos.
"""

import sys
from pathlib import Path
import time
import datetime
from typing import Dict, Any, Optional
import logging

# Configurar paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Imports dos experimentos legados (gemma)
from src.orchestrator.gemma.agent_langchain_final import LeIAAgentDefinitive
from src.orchestrator.gemma.agent_with_memory import LeIAAgentWithMemory
from src.orchestrator.gemma.agent_with_rag import LeIAAgentRAG
from src.orchestrator.gemma.agent_with_tools import LeIAAgentWithTools

# Imports da nova infraestrutura
from src.orchestrator.core.utils.memory_monitor import MemoryMonitor, MemoryTracker
from src.orchestrator.core.utils.logging_utils import ExperimentLogger, MetricsAggregator
from src.orchestrator.experiments.phase4_multimodel.test_scenarios import get_scenario

logger = logging.getLogger(__name__)


class GemmaLegacyWrapper:
    """
    Wrapper que adiciona instrumentação aos experimentos legados do Gemma.
    """

    # Mapeia nome do experimento para a classe correspondente
    AGENT_CLASSES = {
        "agent_langchain_final": LeIAAgentDefinitive,
        "agent_with_memory": LeIAAgentWithMemory,
        "agent_with_rag": LeIAAgentRAG,
        "agent_with_tools": LeIAAgentWithTools,
    }

    def __init__(
        self,
        agent_type: str = "agent_langchain_final",
        experiment_name: str = "gemma_legacy",
        nlu_model_path: Optional[str] = None,
        nlg_model_name: str = "google/gemma-3-1b-it"
    ):
        """
        Inicializa o wrapper.

        Args:
            agent_type: Tipo de agente legacy (agent_langchain_final, agent_with_memory, etc.)
            experiment_name: Nome do experimento para logging
            nlu_model_path: Caminho para o modelo NLU (None = usar padrão)
            nlg_model_name: Nome do modelo NLG Gemma
        """
        if agent_type not in self.AGENT_CLASSES:
            raise ValueError(f"agent_type deve ser um de: {list(self.AGENT_CLASSES.keys())}")

        self.agent_type = agent_type
        self.experiment_name = experiment_name
        self.nlg_model_name = nlg_model_name

        # Definir caminho padrão do NLU
        if nlu_model_path is None:
            nlu_model_path = str(ROOT_DIR / "models" / "leia_classifier_1k_final")

        logger.info(f"Inicializando GemmaLegacyWrapper: {agent_type}")

        # Criar logger do experimento
        self.exp_logger = ExperimentLogger(
            experiment_name=experiment_name,
            model_name=f"gemma_3_1b_{agent_type}"
        )

        # Tracker de memória
        self.mem_tracker = MemoryTracker(name=f"Gemma_{agent_type}")
        self.mem_tracker.set_baseline()

        # Carregar agente legado
        logger.info(f"Carregando agente legado: {agent_type}")
        agent_class = self.AGENT_CLASSES[agent_type]
        self.agent = agent_class(
            nlu_model_path=nlu_model_path,
            nlg_model_name=nlg_model_name
        )

        # Medir memória pós-carregamento
        self.mem_tracker.measure_current()
        self.memory_delta = self.mem_tracker.get_delta()

        # Salvar metadados
        self._save_metadata()

        logger.info("GemmaLegacyWrapper inicializado com sucesso")

    def _save_metadata(self) -> None:
        """Salva metadados do experimento."""
        self.exp_logger.log_experiment_metadata(
            model_config={
                "model_id": self.nlg_model_name,
                "agent_type": self.agent_type,
                "is_legacy_experiment": True,
            },
            test_scenario={
                "note": "Experimento legado com instrumentação da Fase 4",
            },
            system_info={
                "gpu_info": MemoryMonitor.get_gpu_info(),
                "baseline_memory": self.mem_tracker.baseline,
                "memory_delta": self.memory_delta,
            }
        )

    def run_interaction(
        self,
        user_input: str,
        session_id: str = "default",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Executa uma interação com o agente legado, instrumentada com métricas.

        Args:
            user_input: Entrada do usuário
            session_id: ID da sessão
            **kwargs: Argumentos adicionais específicos do agente (ex: discipline)

        Returns:
            Dicionário com resultado e metadados
        """
        start_time = time.time()

        # Capturar memória antes
        mem_before = MemoryMonitor.get_full_memory_snapshot()

        # Executar interação no agente legado
        try:
            # Cada tipo de agente tem interface levemente diferente
            if self.agent_type == "agent_langchain_final":
                response = self.agent.run_query(user_input, session_id=session_id)

            elif self.agent_type == "agent_with_memory":
                discipline = kwargs.get("discipline", "Assunto Geral")
                response = self.agent.run_query(user_input, discipline=discipline, session_id=session_id)

            elif self.agent_type == "agent_with_rag":
                discipline = kwargs.get("discipline", "Assunto Geral")
                response = self.agent.run_query(user_input, discipline=discipline)

            elif self.agent_type == "agent_with_tools":
                discipline = kwargs.get("discipline", "Assunto Geral")
                response = self.agent.run_query(user_input, discipline=discipline, session_id=session_id)

            else:
                response = "Erro: tipo de agente não suportado"

        except Exception as e:
            logger.error(f"Erro na interação: {str(e)}")
            response = f"[ERRO]: {str(e)}"

        # Capturar memória depois
        mem_after = MemoryMonitor.get_full_memory_snapshot()

        # Calcular latência
        latency_ms = round((time.time() - start_time) * 1000, 2)

        # Montar resultado com metadados
        result = {
            "timestamp": datetime.datetime.now().isoformat(),
            "session_id": session_id,
            "model_name": f"gemma_3_1b_{self.agent_type}",
            "user_input": user_input,
            "agent_response": response,
            "latency_ms": latency_ms,
            "vram_used_mb": mem_after["vram"]["allocated_mb"],
            "ram_used_mb": mem_after["ram"]["used_mb"],
            "agent_type": self.agent_type,
            "is_legacy": True,
        }

        # Logar interação
        self.exp_logger.log_interaction(
            user_input=user_input,
            agent_response=response,
            metadata=result
        )

        return result

    def run_test_scenario(self, scenario_name: str = "standard") -> Dict[str, Any]:
        """
        Executa um cenário de teste completo.

        Args:
            scenario_name: Nome do cenário (standard, scaffolding, etc.)

        Returns:
            Dicionário com métricas agregadas
        """
        logger.info(f"\nExecutando cenário: {scenario_name}")

        # Criar novo logger para este cenário específico
        self.exp_logger = ExperimentLogger(
            experiment_name=self.experiment_name,
            model_name=f"gemma_3_1b_{self.agent_type}",
            scenario_name=scenario_name  # Adiciona nome do cenário
        )

        # Carregar cenário
        scenario = get_scenario(scenario_name)

        # Agregador de métricas
        aggregator = MetricsAggregator()

        # Executar cada turno
        session_id = f"{self.experiment_name}_{scenario_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        for turn_num, test_turn in enumerate(scenario.turns, 1):
            logger.info(f"\n--- Turno {turn_num}/{len(scenario.turns)} ---")
            logger.info(f"Foco: {test_turn.test_focus}")

            result = self.run_interaction(
                user_input=test_turn.user_input,
                session_id=session_id,
                discipline="Assunto Geral"  # Para agentes que precisam
            )

            # Adicionar info do test_turn
            result["test_turn"] = {
                "expected_intent": test_turn.expected_intent,
                "test_focus": test_turn.test_focus,
                "description": test_turn.description,
            }

            # Agregar métricas
            aggregator.add_interaction(result)

            logger.info(f"✓ Turno {turn_num} concluído (latência: {result['latency_ms']}ms)")

        # Calcular métricas finais
        final_metrics = aggregator.compute_metrics()
        final_metrics["memory_delta"] = self.memory_delta
        final_metrics["agent_type"] = self.agent_type
        final_metrics["scenario"] = scenario_name

        # Salvar métricas
        self.exp_logger.log_final_metrics(final_metrics)

        logger.info(f"\n--- Métricas Finais ({self.agent_type}) ---")
        logger.info(f"Latência média: {final_metrics.get('latency_avg_ms', 'N/A')} ms")
        logger.info(f"VRAM máxima: {final_metrics.get('vram_max_mb', 'N/A')} MB")

        return final_metrics


# --- Script de Execução ---
if __name__ == '__main__':
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Executar experimentos legados do Gemma com instrumentação")
    parser.add_argument(
        "--agent-type",
        type=str,
        default="agent_langchain_final",
        choices=list(GemmaLegacyWrapper.AGENT_CLASSES.keys()),
        help="Tipo de agente legado a testar"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="standard",
        help="Nome do cenário de teste"
    )

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"GEMMA LEGACY EXPERIMENT: {args.agent_type}")
    print(f"{'='*80}\n")

    # Criar wrapper
    wrapper = GemmaLegacyWrapper(
        agent_type=args.agent_type,
        experiment_name="gemma_legacy_phase4"
    )

    # Executar cenário
    try:
        metrics = wrapper.run_test_scenario(scenario_name=args.scenario)

        print(f"\n✓ Experimento concluído!")
        print(f"Logs salvos em: {wrapper.exp_logger.get_log_dir()}")

    except Exception as e:
        print(f"\n✗ Erro: {str(e)}")
        import traceback
        traceback.print_exc()
