# src/orchestrator/core/utils/logging_utils.py

"""
LoggingUtils - Sistema de Logging para Experimentos

Salva resultados de interações em formato JSONL para análise posterior.
Cada modelo tem seu próprio arquivo de log para facilitar comparações.
"""

import json
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class ExperimentLogger:
    """
    Logger especializado para experimentos da Fase 4.
    Salva interações em formato JSONL (JSON Lines).
    """

    def __init__(
        self,
        experiment_name: str,
        model_name: str,
        base_log_dir: str = "logs/experiments",
        scenario_name: Optional[str] = None
    ):
        """
        Inicializa o logger do experimento.

        Args:
            experiment_name: Nome do experimento (ex: "phase4_multimodel")
            model_name: Nome do modelo sendo testado (ex: "llama-3-8b")
            base_log_dir: Diretório base para logs
            scenario_name: Nome do cenário de teste (ex: "standard", "scaffolding")
        """
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.base_log_dir = Path(base_log_dir)
        self.scenario_name = scenario_name

        # Criar diretório do experimento
        # Estrutura: logs/experiments/{experiment}/{model}/{scenario}/
        model_dir = self.base_log_dir / experiment_name / self._sanitize_model_name(model_name)

        if scenario_name:
            # Se tem cenário, cria subdiretório para o cenário
            self.experiment_dir = model_dir / scenario_name
        else:
            # Se não tem cenário, usa diretamente o diretório do modelo
            self.experiment_dir = model_dir

        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Arquivos de log
        self.interactions_file = self.experiment_dir / "interactions.jsonl"
        self.metrics_file = self.experiment_dir / "metrics.json"
        self.metadata_file = self.experiment_dir / "metadata.json"

        # Estado do experimento
        self.turn_counter = 0
        self.start_time = datetime.datetime.now()

        logger.info(f"ExperimentLogger inicializado: {self.experiment_dir}")

    @staticmethod
    def _sanitize_model_name(model_name: str) -> str:
        """
        Sanitiza o nome do modelo para usar como nome de diretório.

        Args:
            model_name: Nome original do modelo

        Returns:
            Nome sanitizado (sem caracteres especiais)
        """
        # Substituir caracteres problemáticos
        sanitized = model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
        return sanitized

    def log_interaction(
        self,
        user_input: str,
        agent_response: str,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Registra uma interação (turno da conversa).

        Args:
            user_input: Mensagem do usuário
            agent_response: Resposta do agente
            metadata: Metadados adicionais (NLU, trace, latência, etc.)
        """
        self.turn_counter += 1

        # Construir registro completo
        log_entry = {
            "turn_number": self.turn_counter,
            "timestamp": datetime.datetime.now().isoformat(),
            "experiment_name": self.experiment_name,
            "model_name": self.model_name,
            "user_input": user_input,
            "agent_response": agent_response,
            **metadata  # Inclui todos os metadados (NLU, trace, latência, VRAM, etc.)
        }

        # Salvar em JSONL (append)
        try:
            with open(self.interactions_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

            logger.debug(f"Interação {self.turn_counter} registrada")

        except Exception as e:
            logger.error(f"Erro ao registrar interação: {str(e)}")

    def log_experiment_metadata(
        self,
        model_config: Dict[str, Any],
        test_scenario: Dict[str, Any],
        system_info: Dict[str, Any]
    ) -> None:
        """
        Registra metadados do experimento (configuração, cenário de teste, sistema).

        Args:
            model_config: Configuração do modelo (quantização, temperatura, etc.)
            test_scenario: Cenário de teste aplicado
            system_info: Informações do sistema (GPU, RAM, etc.)
        """
        metadata = {
            "experiment_name": self.experiment_name,
            "model_name": self.model_name,
            "start_time": self.start_time.isoformat(),
            "model_config": model_config,
            "test_scenario": test_scenario,
            "system_info": system_info,
        }

        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            logger.info("Metadados do experimento salvos")

        except Exception as e:
            logger.error(f"Erro ao salvar metadados: {str(e)}")

    def log_final_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Registra métricas finais do experimento (resumo).

        Args:
            metrics: Dicionário com métricas finais (latência média, VRAM máxima, etc.)
        """
        final_metrics = {
            "experiment_name": self.experiment_name,
            "model_name": self.model_name,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.datetime.now().isoformat(),
            "total_turns": self.turn_counter,
            "duration_seconds": (datetime.datetime.now() - self.start_time).total_seconds(),
            **metrics
        }

        try:
            with open(self.metrics_file, "w", encoding="utf-8") as f:
                json.dump(final_metrics, f, ensure_ascii=False, indent=2)

            logger.info("Métricas finais salvas")

        except Exception as e:
            logger.error(f"Erro ao salvar métricas finais: {str(e)}")

    def get_log_dir(self) -> Path:
        """
        Retorna o diretório de logs do experimento.

        Returns:
            Path para o diretório
        """
        return self.experiment_dir


class MetricsAggregator:
    """
    Agregador de métricas para análise de múltiplas interações.
    """

    def __init__(self):
        """Inicializa o agregador."""
        self.latencies: List[float] = []
        self.vram_snapshots: List[float] = []
        self.ram_snapshots: List[float] = []
        self.traces: List[str] = []
        self.nlu_confidences: List[float] = []

    def add_interaction(self, interaction_data: Dict[str, Any]) -> None:
        """
        Adiciona dados de uma interação ao agregador.

        Args:
            interaction_data: Dicionário com dados da interação
        """
        # Latência
        if "latency_ms" in interaction_data:
            self.latencies.append(interaction_data["latency_ms"])

        # VRAM
        if "vram_used_mb" in interaction_data:
            self.vram_snapshots.append(interaction_data["vram_used_mb"])

        # RAM
        if "ram_used_mb" in interaction_data:
            self.ram_snapshots.append(interaction_data["ram_used_mb"])

        # Trace
        if "agent_trace" in interaction_data:
            self.traces.append(interaction_data["agent_trace"])

        # Confiança NLU
        if "nlu_confidence" in interaction_data:
            self.nlu_confidences.append(interaction_data["nlu_confidence"])

    def compute_metrics(self) -> Dict[str, Any]:
        """
        Calcula métricas agregadas.

        Returns:
            Dicionário com métricas calculadas
        """
        metrics = {}

        # Latência
        if self.latencies:
            metrics["latency_avg_ms"] = round(sum(self.latencies) / len(self.latencies), 2)
            metrics["latency_min_ms"] = round(min(self.latencies), 2)
            metrics["latency_max_ms"] = round(max(self.latencies), 2)

        # VRAM
        if self.vram_snapshots:
            metrics["vram_avg_mb"] = round(sum(self.vram_snapshots) / len(self.vram_snapshots), 2)
            metrics["vram_max_mb"] = round(max(self.vram_snapshots), 2)

        # RAM
        if self.ram_snapshots:
            metrics["ram_avg_mb"] = round(sum(self.ram_snapshots) / len(self.ram_snapshots), 2)
            metrics["ram_max_mb"] = round(max(self.ram_snapshots), 2)

        # Distribuição de traces
        if self.traces:
            trace_counts = {}
            for trace in self.traces:
                trace_counts[trace] = trace_counts.get(trace, 0) + 1
            metrics["trace_distribution"] = trace_counts

        # Confiança NLU
        if self.nlu_confidences:
            metrics["nlu_confidence_avg"] = round(sum(self.nlu_confidences) / len(self.nlu_confidences), 4)

        return metrics


def load_experiment_logs(log_dir: Path) -> List[Dict[str, Any]]:
    """
    Carrega logs de um experimento (arquivo JSONL).

    Args:
        log_dir: Diretório do experimento

    Returns:
        Lista de dicionários com as interações
    """
    interactions_file = log_dir / "interactions.jsonl"

    if not interactions_file.exists():
        logger.warning(f"Arquivo de interações não encontrado: {interactions_file}")
        return []

    interactions = []
    try:
        with open(interactions_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    interactions.append(json.loads(line))

        logger.info(f"Carregadas {len(interactions)} interações de {interactions_file}")
        return interactions

    except Exception as e:
        logger.error(f"Erro ao carregar logs: {str(e)}")
        return []


def load_all_experiments(base_log_dir: str = "logs/experiments") -> Dict[str, List[Dict[str, Any]]]:
    """
    Carrega logs de todos os experimentos disponíveis.

    Args:
        base_log_dir: Diretório base dos logs

    Returns:
        Dicionário {model_name: [interações]}
    """
    base_path = Path(base_log_dir)

    if not base_path.exists():
        logger.warning(f"Diretório de logs não encontrado: {base_path}")
        return {}

    all_experiments = {}

    # Procurar por diretórios de experimentos
    for experiment_dir in base_path.iterdir():
        if experiment_dir.is_dir():
            # Cada subdiretório é um modelo
            for model_dir in experiment_dir.iterdir():
                if model_dir.is_dir():
                    model_name = model_dir.name
                    interactions = load_experiment_logs(model_dir)
                    if interactions:
                        all_experiments[model_name] = interactions

    logger.info(f"Carregados logs de {len(all_experiments)} modelos")
    return all_experiments


# --- Bloco de Teste ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("\n=== Teste do ExperimentLogger ===\n")

    # Criar logger
    exp_logger = ExperimentLogger(
        experiment_name="test_experiment",
        model_name="test-model-1b"
    )

    print(f"Diretório de logs: {exp_logger.get_log_dir()}")

    # Simular metadados
    exp_logger.log_experiment_metadata(
        model_config={
            "model_id": "test-model-1b",
            "quantization": "4-bit",
            "temperature": 0.7
        },
        test_scenario={
            "name": "standard_test",
            "num_turns": 5
        },
        system_info={
            "gpu": "NVIDIA RTX 3090",
            "cuda": "11.8"
        }
    )

    # Simular interações
    print("\nSimulando interações...")
    for i in range(3):
        exp_logger.log_interaction(
            user_input=f"Pergunta de teste {i+1}",
            agent_response=f"Resposta de teste {i+1}",
            metadata={
                "nlu_label": "Conceitual",
                "nlu_confidence": 0.85 + (i * 0.05),
                "agent_trace": "EXECUTED_STANDARD",
                "latency_ms": 100 + (i * 10),
                "vram_used_mb": 2000 + (i * 50)
            }
        )

    # Agregar métricas
    print("\nAgregando métricas...")
    aggregator = MetricsAggregator()

    logs = load_experiment_logs(exp_logger.get_log_dir())
    for log in logs:
        aggregator.add_interaction(log)

    metrics = aggregator.compute_metrics()
    print(f"Métricas calculadas: {json.dumps(metrics, indent=2)}")

    # Salvar métricas finais
    exp_logger.log_final_metrics(metrics)

    print(f"\n✓ Teste concluído!")
    print(f"Logs salvos em: {exp_logger.get_log_dir()}")
