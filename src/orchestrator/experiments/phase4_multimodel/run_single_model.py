#!/usr/bin/env python3
"""
Script auxiliar para executar testes com um único modelo por vez.
Otimizado para Mac M1 Pro (32GB RAM).

Uso:
    python run_single_model.py --model gemma
    python run_single_model.py --model llama --scenario scaffolding
    python run_single_model.py --model phi3 --scenario standard
    python run_single_model.py --model mistral  # Requer mais RAM
"""

import argparse
import sys
from pathlib import Path

# Adicionar root ao path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

from experiment_runner import ExperimentRunner

# Mapeamento de aliases para IDs de modelos
MODELS = {
    # Família Gemma 3 (4 modelos para teste comparativo)
    "gemma3-270m": "google/gemma-3-270m-it",  # Gemma 3 270M - Ultra leve
    "gemma3-1b": "google/gemma-3-1b-it",      # Gemma 3 1B - Leve
    "gemma": "google/gemma-2-2b-it",          # Gemma 2 2B - Baseline atual
    "gemma3-4b": "google/gemma-3-4b-it",      # Gemma 3 4B - Modelo maior

    # Outros modelos
    "phi3": "microsoft/Phi-3-mini-4k-instruct",  # Phi-3 Mini (texto-only)
    "qwen": "Qwen/Qwen2.5-7B-Instruct",  # Alternativa open-source 7B
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Muito leve

    # Modelos que podem requerer autenticação
    "llama": "meta-llama/Llama-3.2-3B-Instruct",  # Llama 3.2 3B
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",  # Público, mas grande
}

# Informações sobre cada modelo
MODEL_INFO = {
    # Família Gemma 3
    "gemma3-270m": {
        "size_gb": 0.5,
        "ram_needed_gb": 2,
        "time_min": "5-10",
        "recommendation": "✓✓ Ultra rápido - Ideal para testes rápidos"
    },
    "gemma3-1b": {
        "size_gb": 1,
        "ram_needed_gb": 3,
        "time_min": "10-15",
        "recommendation": "✓✓ Muito rápido - Ótimo para Mac M1"
    },
    "gemma": {
        "size_gb": 2,
        "ram_needed_gb": 6,
        "time_min": "15-20",
        "recommendation": "✓ Ideal para Mac M1 (Baseline atual)"
    },
    "gemma3-4b": {
        "size_gb": 4,
        "ram_needed_gb": 10,
        "time_min": "20-30",
        "recommendation": "✓ Bom para Mac M1 - Modelo maior da família"
    },

    # Outros modelos
    "qwen": {
        "size_gb": 7,
        "ram_needed_gb": 14,
        "time_min": "25-35",
        "recommendation": "✓ Bom para Mac M1 - Qwen 2.5 7B"
    },
    "llama": {
        "size_gb": 8,
        "ram_needed_gb": 16,
        "time_min": "25-35",
        "recommendation": "✓ Bom para Mac M1 - Llama 3.1 8B"
    },
    "phi3": {
        "size_gb": 4,
        "ram_needed_gb": 8,
        "time_min": "20-25",
        "recommendation": "✓ Bom para Mac M1 - Phi-3 Mini"
    },
    "mistral": {
        "size_gb": 14,
        "ram_needed_gb": 20,
        "time_min": "30-40",
        "recommendation": "⚠️ Modelo grande - feche outros apps"
    }
}


def print_model_info(model_alias: str):
    """Imprime informações sobre o modelo."""
    info = MODEL_INFO[model_alias]
    model_id = MODELS[model_alias]

    # Nome amigável do modelo
    model_friendly = {
        "gemma3-270m": "Gemma 3 270M",
        "gemma3-1b": "Gemma 3 1B",
        "gemma": "Gemma 2 2B [BASELINE]",
        "gemma3-4b": "Gemma 3 4B",
        "qwen": "Qwen 2.5 7B Instruct",
        "llama": "Llama 3.1 8B Instruct",
        "phi3": "Phi-3 Mini 4K",
        "mistral": "Mistral 7B Instruct v0.2"
    }.get(model_alias, model_alias.upper())

    print(f"\n{'='*70}")
    print(f"MODELO SELECIONADO: {model_friendly}")
    print(f"{'='*70}")
    print(f"Alias: {model_alias}")
    print(f"ID HuggingFace: {model_id}")
    print(f"Tamanho: ~{info['size_gb']}GB")
    print(f"RAM necessária: ~{info['ram_needed_gb']}GB")
    print(f"Tempo estimado: {info['time_min']} minutos")
    print(f"Recomendação: {info['recommendation']}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Executar teste com um único modelo (Mac M1 otimizado)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODELS.keys()),
        help="Modelo a testar: gemma, llama, phi3, ou mistral"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="standard",
        choices=[
            "standard", "scaffolding", "react", "stress", "edge_cases",
            "standard_enem", "guardrail_enem", "scaffolding_complex_enem",
            "stress_enem", "edge_cases_enem"
        ],
        help="Cenário de teste (padrão: standard). Cenários ENEM usam questões reais do ENEM 2013-2016."
    )
    parser.add_argument(
        "--enable-tools",
        action="store_true",
        help="Habilitar ferramentas ReAct (busca externa)"
    )

    args = parser.parse_args()

    # Obter ID do modelo
    model_id = MODELS[args.model]

    # Mostrar informações
    print_model_info(args.model)

    # Aviso para modelos grandes
    if args.model == "mistral":
        print("⚠️  AVISO: Mistral 7B é um modelo grande!")
        print("   Feche TODOS os outros aplicativos antes de continuar.")
        response = input("   Continuar? (s/N): ")
        if response.lower() != "s":
            print("Cancelado pelo usuário.")
            return

    # Criar runner
    print(f"\nConfigurando experimento...")
    print(f"  Modelo: {model_id}")
    print(f"  Cenário: {args.scenario}")
    print(f"  Ferramentas: {'Habilitadas' if args.enable_tools else 'Desabilitadas'}")
    print()

    runner = ExperimentRunner(
        experiment_name="phase4_multimodel",
        models_to_test=[model_id],
        test_scenario=args.scenario,
        enable_tools=args.enable_tools,
        use_quantization=False,  # Sempre False para Mac M1
        quantization_bits=4,
    )

    # Executar
    try:
        print(f"\n{'='*70}")
        print(f"INICIANDO TESTE: {args.model.upper()}")
        print(f"{'='*70}\n")

        results = runner.run_all_experiments()

        print(f"\n{'='*70}")
        print(f"✓ TESTE CONCLUÍDO COM SUCESSO!")
        print(f"{'='*70}")
        print(f"\nLogs salvos em:")
        print(f"  logs/experiments/phase4_multimodel/{model_id.replace('/', '_')}/")
        print(f"\nArquivos gerados:")
        print(f"  - interactions.jsonl  (todas as interações)")
        print(f"  - metrics.json        (métricas agregadas)")
        print(f"  - metadata.json       (configuração)")
        print(f"\nPróximos passos:")
        print(f"  1. Executar outro modelo: python run_single_model.py --model [gemma|llama|phi3|mistral]")
        print(f"  2. Analisar resultados: jupyter notebook analysis.ipynb")
        print()

    except KeyboardInterrupt:
        print("\n\n✗ Teste interrompido pelo usuário")

    except Exception as e:
        print(f"\n\n✗ Erro durante o teste:")
        print(f"   {str(e)}")
        print(f"\nVerifique os logs em: logs/experiment_runner.log")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
