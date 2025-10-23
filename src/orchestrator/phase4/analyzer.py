import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


class Phase4LogAnalyzer:
    def __init__(self, logs_dir: Path = Path("logs"), results_dir: Path = Path("results")):
        self.logs_dir = logs_dir
        self.results_dir = results_dir
        self.results_dir.mkdir(exist_ok=True)
        self.persona_patterns = [
            re.compile(r"\b(o que você|como você|que tal|o que te chama)\b", re.I),
            re.compile(r"\bte convido\b", re.I),
            re.compile(r"\bperceba\b", re.I),
        ]
        self.scaffolding_patterns = [
            re.compile(r"\bquando você diz\b", re.I),
            re.compile(r"\bse imaginarmos\b", re.I),
            re.compile(r"\bque parte\b", re.I),
            re.compile(r"\bo que acontece se\b", re.I),
        ]
        self.hallucination_patterns = [
            re.compile(r"\bfoi (mesmo )?um jogo\b", re.I),
            re.compile(r"\bsim,.*jogo\b", re.I),
            re.compile(r"\bpartida de futebol\b", re.I),
        ]

    def _iter_logs(self) -> Iterable[Tuple[str, Dict[str, object]]]:
        for log_file in self.logs_dir.glob("fase4_*.jsonl"):
            model_name = log_file.stem.replace("fase4_", "")
            with log_file.open("r", encoding="utf-8") as handler:
                for line in handler:
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    yield model_name, payload

    def _classify_response(self, text: str) -> Dict[str, bool]:
        persona = any(pattern.search(text) for pattern in self.persona_patterns)
        reflexive = text.strip().endswith("?") or text.count("?") > 0
        scaffolding = any(pattern.search(text) for pattern in self.scaffolding_patterns)
        hallucination = any(pattern.search(text) for pattern in self.hallucination_patterns)
        return {
            "persona": persona,
            "reflexive": reflexive,
            "scaffolding": scaffolding,
            "hallucination": hallucination,
        }

    def analyze(self) -> Dict[str, Dict[str, float]]:
        qualitative_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        totals: Dict[str, int] = defaultdict(int)
        latency_records: Dict[str, List[float]] = defaultdict(list)
        memory_records: Dict[str, List[float]] = defaultdict(list)
        params_lookup: Dict[str, int] = {}
        repo_lookup: Dict[str, str] = {}

        for model_name, record in self._iter_logs():
            output_text = str(record.get("model_output", ""))
            classification = self._classify_response(output_text)
            for key, value in classification.items():
                if value:
                    qualitative_counts[model_name][key] += 1
            totals[model_name] += 1
            latency_records[model_name].append(float(record.get("latency_ms", 0.0)))
            memory_records[model_name].append(float(record.get("memory_mb", 0.0)))
            if "model_params" in record:
                params_lookup[model_name] = int(record["model_params"])
            if "model_repo" in record:
                repo_lookup[model_name] = str(record["model_repo"])

        summary: Dict[str, Dict[str, float]] = {}
        for model_name, total in totals.items():
            if total == 0:
                continue
            summary[model_name] = {
                "persona": round(100 * qualitative_counts[model_name]["persona"] / total, 2),
                "reflexive": round(100 * qualitative_counts[model_name]["reflexive"] / total, 2),
                "scaffolding": round(100 * qualitative_counts[model_name]["scaffolding"] / total, 2),
                "hallucination": round(100 * qualitative_counts[model_name]["hallucination"] / total, 2),
                "media_latency_ms": round(
                    sum(latency_records[model_name]) / len(latency_records[model_name]), 2
                ),
                "memoria_media_mb": round(
                    sum(memory_records[model_name]) / len(memory_records[model_name]), 2
                ),
                "parameters": params_lookup.get(model_name),
                "model_repo": repo_lookup.get(model_name),
            }

        self._export_summary(summary)
        return summary

    def _export_summary(self, summary: Dict[str, Dict[str, float]]) -> Path:
        comparative_path = self.results_dir / "fase4_comparative_results.csv"
        fieldnames = [
            "model_name",
            "parameters",
            "model_repo",
            "media_latency_ms",
            "memoria_media_mb",
            "execucao_local",
            "persona_pct",
            "reflexive_pct",
            "scaffolding_pct",
            "hallucination_pct",
        ]
        with comparative_path.open("w", encoding="utf-8", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for model_name, metrics in summary.items():
                writer.writerow(
                    {
                        "model_name": model_name,
                        "parameters": metrics.get("parameters"),
                        "model_repo": metrics.get("model_repo"),
                        "media_latency_ms": metrics.get("media_latency_ms"),
                        "memoria_media_mb": metrics.get("memoria_media_mb"),
                        "execucao_local": True,
                        "persona_pct": metrics.get("persona"),
                        "reflexive_pct": metrics.get("reflexive"),
                        "scaffolding_pct": metrics.get("scaffolding"),
                        "hallucination_pct": metrics.get("hallucination"),
                    }
                )
        return comparative_path


def main():
    analyzer = Phase4LogAnalyzer()
    summary = analyzer.analyze()
    print("Resumo qualitativo por modelo:")
    for model_name, metrics in summary.items():
        print(f"- {model_name}: {metrics}")


if __name__ == "__main__":
    main()
