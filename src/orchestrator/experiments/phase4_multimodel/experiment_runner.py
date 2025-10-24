# src/orchestrator/experiments/phase4_multimodel/experiment_runner.py

"""
Experiment Runner - Arnês de Teste da Fase 4

Orquestra a execução de experimentos comparativos entre múltiplos modelos NLG.
Aplica o mesmo roteiro de teste a cada modelo e coleta métricas padronizadas.

Workflow:
1. Define lista de modelos a testar
2. Para cada modelo:
   a. Medir baseline de memória
   b. Carregar NLU e NLG
   c. Medir memória pós-carregamento
   d. Criar FreireAgent
   e. Executar cenário de teste
   f. Salvar logs e métricas
   g. Limpar memória
3. Gerar relatório comparativo
"""

import sys
from pathlib import Path
import logging
import datetime
import traceback
from typing import List, Dict, Any, Optional

# Configurar paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Imports dos módulos da Fase 4
from src.orchestrator.core.loaders.model_loader import load_nlg_model, ModelLoader
from src.orchestrator.core.loaders.nlu_loader import load_nlu_model
from src.orchestrator.core.agents.freire_agent import FreireAgent
from src.orchestrator.core.utils.memory_monitor import MemoryMonitor, MemoryTracker
from src.orchestrator.core.utils.logging_utils import ExperimentLogger, MetricsAggregator
from src.orchestrator.experiments.phase4_multimodel.test_scenarios import get_scenario, get_all_scenario_names

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/experiment_runner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Gerenciador de experimentos para a Fase 4.
    """

    # Lista de modelos a testar (pode ser customizada)
    DEFAULT_MODELS = [
        "google/gemma-3-1b-it",                    # Baseline (já testado)
        "meta-llama/Llama-3.2-3B-Instruct",       # Llama 3 (8B é muito grande, usando 3B)
        "mistralai/Mistral-7B-Instruct-v0.2",     # Mistral 7B
        "microsoft/Phi-3-mini-4k-instruct",       # Phi-3 Mini
    ]

    def __init__(
        self,
        experiment_name: str = "phase4_multimodel",
        models_to_test: Optional[List[str]] = None,
        test_scenario: str = "standard",
        enable_tools: bool = False,
        use_quantization: bool = True,
        quantization_bits: int = 4,
    ):
        """
        Inicializa o ExperimentRunner.

        Args:
            experiment_name: Nome do experimento
            models_to_test: Lista de modelos a testar (None = usar DEFAULT_MODELS)
            test_scenario: Nome do cenário de teste a aplicar
            enable_tools: Se True, habilita ferramentas (ReAct)
            use_quantization: Se True, usa quantização nos modelos
            quantization_bits: Bits de quantização (4 ou 8)
        """
        self.experiment_name = experiment_name
        self.models_to_test = models_to_test or self.DEFAULT_MODELS
        self.test_scenario_name = test_scenario
        self.enable_tools = enable_tools
        self.use_quantization = use_quantization
        self.quantization_bits = quantization_bits

        # Carregar cenário de teste
        self.test_scenario = get_scenario(test_scenario)

        # Resultados consolidados
        self.results: Dict[str, Dict[str, Any]] = {}

        logger.info(f"ExperimentRunner inicializado: {experiment_name}")
        logger.info(f"Modelos a testar: {len(self.models_to_test)}")
        logger.info(f"Cenário: {test_scenario} ({len(self.test_scenario.turns)} turnos)")

    def run_all_experiments(self) -> Dict[str, Dict[str, Any]]:
        """
        Executa experimentos para todos os modelos configurados.

        Returns:
            Dicionário com resultados consolidados {model_name: results}
        """
        logger.info("\n" + "="*80)
        logger.info(f"INICIANDO EXPERIMENTOS: {self.experiment_name}")
        logger.info("="*80 + "\n")

        # Informações do sistema
        gpu_info = MemoryMonitor.get_gpu_info()
        logger.info(f"GPU: {gpu_info.get('device_name', 'N/A')}")
        logger.info(f"CUDA: {gpu_info.get('cuda_version', 'N/A')}")

        # Carregar NLU uma vez (constante para todos os modelos)
        logger.info("\n--- Carregando NLU (constante para todos os testes) ---")
        try:
            nlu = load_nlu_model()
        except Exception as e:
            logger.error(f"Falha ao carregar NLU: {str(e)}")
            return {}

        # Executar experimento para cada modelo
        for i, model_name in enumerate(self.models_to_test, 1):
            logger.info("\n" + "="*80)
            logger.info(f"MODELO {i}/{len(self.models_to_test)}: {model_name}")
            logger.info("="*80)

            try:
                result = self._run_single_experiment(model_name, nlu)
                self.results[model_name] = result

                logger.info(f"✓ Experimento concluído para {model_name}")

            except Exception as e:
                logger.error(f"✗ Erro no experimento com {model_name}: {str(e)}")
                logger.error(traceback.format_exc())
                self.results[model_name] = {"error": str(e), "status": "failed"}

            # Limpar memória entre modelos
            MemoryMonitor.clear_gpu_cache()
            logger.info("Cache limpo entre experimentos\n")

        # Gerar relatório final
        self._generate_final_report()

        logger.info("\n" + "="*80)
        logger.info("TODOS OS EXPERIMENTOS CONCLUÍDOS")
        logger.info("="*80 + "\n")

        return self.results

    def _run_single_experiment(
        self,
        model_name: str,
        nlu
    ) -> Dict[str, Any]:
        """
        Executa experimento para um único modelo.

        Args:
            model_name: Nome/ID do modelo
            nlu: Classificador NLU (já carregado)

        Returns:
            Dicionário com resultados do experimento
        """
        # 1. Inicializar logger do experimento
        exp_logger = ExperimentLogger(
            experiment_name=self.experiment_name,
            model_name=model_name,
            scenario_name=self.test_scenario_name  # Adiciona nome do cenário
        )

        # 2. Inicializar tracker de memória
        mem_tracker = MemoryTracker(name=model_name)
        mem_tracker.set_baseline()
        MemoryMonitor.print_memory_summary(f"Baseline (antes de carregar {model_name})")

        # 3. Carregar modelo NLG
        logger.info(f"Carregando modelo NLG: {model_name}")
        try:
            llm, tokenizer, device_info = load_nlg_model(
                model_name,
                use_quantization=self.use_quantization,
                quantization_bits=self.quantization_bits
            )
            logger.info(f"Modelo carregado: {device_info}")

        except Exception as e:
            logger.error(f"Falha ao carregar modelo: {str(e)}")
            raise

        # 4. Medir memória pós-carregamento
        mem_tracker.measure_current()
        MemoryMonitor.print_memory_summary(f"Pós-carregamento ({model_name})")
        memory_delta = mem_tracker.get_delta()

        # 5. Criar FreireAgent
        logger.info("Criando FreireAgent...")
        agent = FreireAgent(
            nlu_classifier=nlu,
            nlg_llm=llm,
            model_name=model_name,
            enable_tools=self.enable_tools,
            session_id=f"{self.experiment_name}_{model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # 6. Salvar metadados do experimento
        exp_logger.log_experiment_metadata(
            model_config={
                "model_id": model_name,
                "use_quantization": self.use_quantization,
                "quantization_bits": self.quantization_bits,
                "device": device_info,
                "enable_tools": self.enable_tools,
            },
            test_scenario={
                "name": self.test_scenario_name,
                "num_turns": len(self.test_scenario.turns),
                "description": self.test_scenario.description,
            },
            system_info={
                "gpu_info": MemoryMonitor.get_gpu_info(),
                "baseline_memory": mem_tracker.baseline,
                "memory_delta": memory_delta,
            }
        )

        # 7. Executar cenário de teste
        logger.info(f"\nExecutando cenário: {self.test_scenario_name} ({len(self.test_scenario.turns)} turnos)")
        metrics_aggregator = MetricsAggregator()

        for turn_num, test_turn in enumerate(self.test_scenario.turns, 1):
            logger.info(f"\n--- Turno {turn_num}/{len(self.test_scenario.turns)} ---")
            logger.info(f"Foco: {test_turn.test_focus}")

            try:
                # Capturar memória antes da interação
                mem_before = MemoryMonitor.get_full_memory_snapshot()

                # Executar turno
                result = agent.chat(test_turn.user_input)

                # Capturar memória depois da interação
                mem_after = MemoryMonitor.get_full_memory_snapshot()

                # Adicionar dados de memória ao resultado
                result["vram_used_mb"] = mem_after["vram"]["allocated_mb"]
                result["ram_used_mb"] = mem_after["ram"]["used_mb"]
                result["test_turn"] = {
                    "expected_intent": test_turn.expected_intent,
                    "test_focus": test_turn.test_focus,
                    "description": test_turn.description,
                }

                # Logar interação
                exp_logger.log_interaction(
                    user_input=test_turn.user_input,
                    agent_response=result["response"],
                    metadata=result
                )

                # Adicionar ao agregador de métricas
                metrics_aggregator.add_interaction(result)

                logger.info(f"✓ Turno {turn_num} concluído (latência: {result['latency_ms']}ms)")

            except Exception as e:
                logger.error(f"✗ Erro no turno {turn_num}: {str(e)}")
                logger.error(traceback.format_exc())

        # 8. Calcular e salvar métricas finais
        final_metrics = metrics_aggregator.compute_metrics()
        final_metrics["memory_delta"] = memory_delta
        final_metrics["model_size_estimate_mb"] = memory_delta["vram"].get("allocated_mb_delta", 0)

        exp_logger.log_final_metrics(final_metrics)

        logger.info(f"\n--- Métricas Finais ({model_name}) ---")
        logger.info(f"Latência média: {final_metrics.get('latency_avg_ms', 'N/A')} ms")
        logger.info(f"VRAM máxima: {final_metrics.get('vram_max_mb', 'N/A')} MB")
        logger.info(f"Distribuição de traces: {final_metrics.get('trace_distribution', {})}")

        # 9. Limpar modelo
        logger.info(f"\nDescarregando {model_name}...")
        ModelLoader.unload_model(llm)
        del agent
        del llm
        del tokenizer

        return final_metrics

    def _generate_final_report(self) -> None:
        """
        Gera relatório consolidado de todos os experimentos.
        """
        logger.info("\n" + "="*80)
        logger.info("RELATÓRIO CONSOLIDADO")
        logger.info("="*80 + "\n")

        if not self.results:
            logger.warning("Nenhum resultado disponível para relatório")
            return

        # Tabela comparativa
        logger.info(f"{'Modelo':<40} {'Latência (ms)':<15} {'VRAM Máx (MB)':<15} {'Status':<10}")
        logger.info("-"*80)

        for model_name, result in self.results.items():
            if result.get("status") == "failed":
                logger.info(f"{model_name:<40} {'N/A':<15} {'N/A':<15} {'FAILED':<10}")
            else:
                latency = result.get("latency_avg_ms", "N/A")
                vram = result.get("vram_max_mb", "N/A")
                logger.info(f"{model_name:<40} {latency:<15} {vram:<15} {'OK':<10}")

        logger.info("\n" + "="*80 + "\n")


# --- Bloco Principal ---
if __name__ == '__main__':
    print("\n" + "="*80)
    print("FASE 4 - EXPERIMENTO MULTI-MODELO")
    print("="*80 + "\n")

    # Configuração do experimento
    runner = ExperimentRunner(
        experiment_name="phase4_multimodel",
        models_to_test=[
            "google/gemma-3-1b-it",  # Começar com modelo pequeno para teste
            # Adicione mais modelos após validar que funciona
            # "meta-llama/Llama-3.2-3B-Instruct",
            # "mistralai/Mistral-7B-Instruct-v0.2",
            # "microsoft/Phi-3-mini-4k-instruct",
        ],
        test_scenario="standard",  # Opções: standard, scaffolding, react, stress, edge_cases
        enable_tools=False,  # ReAct framework (busca)
        use_quantization=True,
        quantization_bits=4,
    )

    # Executar experimentos
    try:
        results = runner.run_all_experiments()

        print("\n✓ Experimentos concluídos com sucesso!")
        print(f"Logs salvos em: logs/experiments/{runner.experiment_name}/")
        print("\nPróximo passo: Analisar os resultados no notebook Jupyter")

    except KeyboardInterrupt:
        print("\n\n✗ Experimento interrompido pelo usuário")
        logger.warning("Experimento interrompido via KeyboardInterrupt")

    except Exception as e:
        print(f"\n✗ Erro fatal: {str(e)}")
        logger.error(f"Erro fatal no experiment runner: {str(e)}")
        logger.error(traceback.format_exc())
