import importlib
import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import psutil

# --- Path configuration ----------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from src.classifier.classifier import NLUClassifier  # noqa: E402
from src.orchestrator.phase4.backends import GenerationConfig, PipelineFactory  # noqa: E402


HYPOTHESIS = (
    "Modelos de maior capacidade (4B–8B) manterão maior estabilidade dialógica "
    "e consistência freiriana, sem perder a filosofia de leveza e execução local."
)

MODELS = [
    {
        "name": "meta-llama-3-8b-instruct",
        "provider": "llama",
        "repo_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "params": 8_000_000_000,
    },
    {
        "name": "mistral-7b-instruct-v0.2",
        "provider": "mistral",
        "repo_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "params": 7_000_000_000,
    },
    {
        "name": "phi-3-mini-4k-instruct",
        "provider": "phi",
        "repo_id": "microsoft/Phi-3-mini-4k-instruct",
        "params": 3_800_000_000,
    },
]

INTERACTION_SCRIPT = [
    {"user_input": "O que foi a Revolução Francesa?", "discipline": "História"},
    {"user_input": "Foi uma guerra da França contra os reis.", "discipline": "História"},
    {"user_input": "Explique usando uma metáfora com futebol.", "discipline": "História"},
    {"user_input": "Foi um jogo de futebol, certo?", "discipline": "História"},
]


@dataclass
class LocalModelBackend:
    name: str
    provider: str
    repo_id: str
    params: int
    config: GenerationConfig = field(default_factory=GenerationConfig)
    _pipeline_factory: Optional[PipelineFactory] = field(init=False, default=None)

    def __post_init__(self):
        module_path = f"src.orchestrator.{self.provider}.phase4_backend"
        module = importlib.import_module(module_path)
        if not hasattr(module, "build_pipeline"):
            raise AttributeError(
                f"O módulo {module_path} precisa expor a função 'build_pipeline'."
            )
        builder: PipelineFactory = getattr(module, "build_pipeline")
        self._pipeline_factory = builder(self.repo_id, self.config)

    def generate(self, prompt: str) -> str:
        if self._pipeline_factory is None:
            raise RuntimeError("Pipeline do modelo não foi inicializada.")
        outputs = self._pipeline_factory(prompt)
        generated_payload = outputs[0]
        generated_text = generated_payload.get("generated_text") or generated_payload.get("text") or ""
        return str(generated_text).strip()


class FreirePromptBuilder:
    def __init__(self):
        self.templates: Dict[str, str] = {
            "Conceitual": (
                "Você é LeIA, uma tutora freiriana. Nunca entregue definições prontas. "
                "Valide a curiosidade do aluno, convide-o a relacionar com o que já sabe "
                "e encerre com uma pergunta aberta que estimule reflexão. "
                "Histórico recente:\n{history}\n\nAluno: {question}\nLeIA:"
            ),
            "Procedimental": (
                "Você é LeIA, uma tutora freiriana. Evite mostrar o passo a passo completo. "
                "Reconheça o desafio proposto, ofereça o primeiro andaime conceitual e "
                "encerre com uma pergunta que incentive experimentação. "
                "Histórico recente:\n{history}\n\nAluno: {question}\nLeIA:"
            ),
            "Análise de Exemplo": (
                "Você é LeIA, uma tutora freiriana. Não entregue a interpretação final. "
                "Direcione o olhar do aluno para um elemento específico e encerre com "
                "uma pergunta que convide à conscientização crítica. "
                "Histórico recente:\n{history}\n\nAluno: {question}\nLeIA:"
            ),
            "Comparativo": (
                "Você é LeIA, uma tutora freiriana. Não liste respostas prontas. "
                "Convide o aluno a descrever um dos elementos da comparação e finalize "
                "com uma pergunta que estimule síntese própria. "
                "Histórico recente:\n{history}\n\nAluno: {question}\nLeIA:"
            ),
        }

    def build_prompt(self, intent: str, history: List[Dict[str, str]], question: str) -> str:
        entries = history[-4:]
        history_text = "\n".join(f"{item['role'].capitalize()}: {item['content']}" for item in entries) or "Nenhum"
        template = self.templates.get(intent, self.templates["Conceitual"])
        return template.format(history=history_text, question=question)


class Phase4ExperimentRunner:
    def __init__(self, nlu_model_path: Path, models: List[Dict[str, object]] = MODELS):
        self.hypothesis = HYPOTHESIS
        self.models = models
        self.nlu = NLUClassifier(model_path=str(nlu_model_path))
        self.prompt_builder = FreirePromptBuilder()
        self.process = psutil.Process()
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)

    def _init_backend(self, descriptor: Dict[str, object]) -> LocalModelBackend:
        return LocalModelBackend(
            name=str(descriptor["name"]),
            provider=str(descriptor["provider"]),
            repo_id=str(descriptor["repo_id"]),
            params=int(descriptor["params"]),
        )

    def _log_event(self, model_name: str, payload: Dict[str, object]) -> None:
        log_path = self.logs_dir / f"fase4_{model_name}.jsonl"
        with log_path.open("a", encoding="utf-8") as handler:
            handler.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def run(self) -> Path:
        summary_rows = []
        for model_descriptor in self.models:
            backend = self._init_backend(model_descriptor)
            session_id = str(uuid.uuid4())
            history: List[Dict[str, str]] = []
            latencies: List[float] = []
            memory_readings: List[float] = []

            for step in INTERACTION_SCRIPT:
                user_input = step["user_input"]
                nlu_output = self.nlu.predict(user_input)
                intent = nlu_output.get("label", "Conceitual")
                prompt = self.prompt_builder.build_prompt(intent, history, user_input)

                start_time = time.time()
                model_output = backend.generate(prompt)
                latency_ms = (time.time() - start_time) * 1000
                memory_mb = self.process.memory_info().rss / (1024**2)

                latencies.append(latency_ms)
                memory_readings.append(memory_mb)

                event = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "session_id": session_id,
                    "model_name": backend.name,
                    "model_params": backend.params,
                    "model_repo": backend.repo_id,
                    "user_input": user_input,
                    "nlu_label": intent,
                    "nlu_confidence": nlu_output.get("confidence"),
                    "final_prompt": prompt,
                    "model_output": model_output,
                    "latency_ms": round(latency_ms, 2),
                    "memory_mb": round(memory_mb, 2),
                    "execution_local": True,
                }
                self._log_event(backend.name, event)

                history.append({"role": "aluno", "content": user_input})
                history.append({"role": "leia", "content": model_output})

            summary_rows.append(
                {
                    "model_name": backend.name,
                    "parameters": backend.params,
                    "model_repo": backend.repo_id,
                    "media_latency_ms": round(sum(latencies) / len(latencies), 2),
                    "memoria_media_mb": round(sum(memory_readings) / len(memory_readings), 2),
                    "execucao_local": True,
                }
            )

        summary_path = self.results_dir / "fase4_quantitative_results.csv"
        header = "model_name,parameters,model_repo,media_latency_ms,memoria_media_mb,execucao_local\n"
        with summary_path.open("w", encoding="utf-8") as summary_file:
            summary_file.write(header)
            for row in summary_rows:
                summary_file.write(
                    f"{row['model_name']},{row['parameters']},{row['model_repo']},"
                    f"{row['media_latency_ms']},{row['memoria_media_mb']},"
                    f"{row['execucao_local']}\n"
                )
        return summary_path


def main():
    nlu_model_dir = ROOT_DIR / "models" / "leia_classifier_1k_final"
    runner = Phase4ExperimentRunner(nlu_model_path=nlu_model_dir)
    results_csv = runner.run()
    print(f"Resumo quantitativo salvo em: {results_csv}")


if __name__ == "__main__":
    main()
