#!/bin/bash

###############################################################################
# Script para executar TODOS os 4 modelos Gemma sequencialmente
#
# Modelos testados:
# 1. Gemma 3 270M (ultra leve)
# 2. Gemma 3 1B (leve)
# 3. Gemma 2 2B (baseline atual)
# 4. Gemma 3 4B (modelo maior)
#
# Uso:
#   ./run_all_gemma_models.sh [scenario]
#
# Exemplos:
#   ./run_all_gemma_models.sh standard_enem
#   ./run_all_gemma_models.sh guardrail_enem
#   ./run_all_gemma_models.sh standard
###############################################################################

# Configuração
SCENARIO=${1:-"standard_enem"}  # Default: standard_enem
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/../../../../logs/batch_runs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BATCH_LOG="$LOG_DIR/gemma_batch_${SCENARIO}_${TIMESTAMP}.log"

# Criar diretório de logs se não existir
mkdir -p "$LOG_DIR"

# Lista de modelos Gemma para testar
GEMMA_MODELS=(
    "gemma3-270m"
    "gemma3-1b"
    "gemma"
    "gemma3-4b"
)

echo "================================================================================"
echo "TESTE BATCH: FAMÍLIA GEMMA (4 modelos)"
echo "================================================================================"
echo "Cenário: $SCENARIO"
echo "Timestamp: $TIMESTAMP"
echo "Log do batch: $BATCH_LOG"
echo "================================================================================"
echo ""

# Inicializar log
{
    echo "============================================"
    echo "BATCH RUN: Família Gemma - $SCENARIO"
    echo "Timestamp: $TIMESTAMP"
    echo "============================================"
    echo ""
} > "$BATCH_LOG"

# Contador de sucessos/falhas
SUCCESS_COUNT=0
FAILURE_COUNT=0
FAILED_MODELS=()

# Executar cada modelo
for i in "${!GEMMA_MODELS[@]}"; do
    MODEL="${GEMMA_MODELS[$i]}"
    MODEL_NUM=$((i + 1))

    # Obter nome completo do modelo para logging claro
    case "$MODEL" in
        "gemma3-270m") MODEL_FULL="Gemma 3 270M (google/gemma-3-270m-it)" ;;
        "gemma3-1b")   MODEL_FULL="Gemma 3 1B (google/gemma-3-1b-it)" ;;
        "gemma")       MODEL_FULL="Gemma 2 2B (google/gemma-2-2b-it) [BASELINE]" ;;
        "gemma3-4b")   MODEL_FULL="Gemma 3 4B (google/gemma-3-4b-it)" ;;
        *)             MODEL_FULL="$MODEL" ;;
    esac

    echo ""
    echo "================================================================================"
    echo "MODELO [$MODEL_NUM/4]: $MODEL_FULL"
    echo "================================================================================"
    echo "Alias: $MODEL"
    echo "Início: $(date)"
    echo ""

    # Logar início com nome completo do modelo
    {
        echo ""
        echo "==================================================================="
        echo "MODELO $MODEL_NUM/4: $MODEL_FULL"
        echo "==================================================================="
        echo "Alias: $MODEL"
        echo "Início: $(date)"
    } >> "$BATCH_LOG"

    # Executar modelo (capturar exit code do python, não do tee)
    python3 run_single_model.py --model "$MODEL" --scenario "$SCENARIO" 2>&1 | tee -a "$BATCH_LOG"
    EXIT_CODE=${PIPESTATUS[0]}  # Exit code do python, não do tee

    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ $MODEL_FULL concluído com sucesso!"
        echo "Término: $(date)"

        {
            echo "Status: SUCESSO ✓"
            echo "Término: $(date)"
            echo "==================================================================="
        } >> "$BATCH_LOG"

        ((SUCCESS_COUNT++))
    else
        echo ""
        echo "✗ $MODEL_FULL FALHOU (exit code: $EXIT_CODE)!"
        echo "Término: $(date)"

        {
            echo "Status: FALHA ✗ (exit code: $EXIT_CODE)"
            echo "Término: $(date)"
            echo "==================================================================="
        } >> "$BATCH_LOG"

        ((FAILURE_COUNT++))
        FAILED_MODELS+=("$MODEL_FULL")
    fi

    # Aguardar 10 segundos entre modelos (para limpeza de memória)
    if [ $MODEL_NUM -lt 4 ]; then
        echo ""
        echo "Aguardando 10s para limpeza de memória..."
        sleep 10
    fi
done

# Relatório final
echo ""
echo "================================================================================"
echo "RELATÓRIO FINAL - BATCH GEMMA"
echo "================================================================================"
echo "Cenário testado: $SCENARIO"
echo "Total de modelos: 4"
echo "Sucessos: $SUCCESS_COUNT"
echo "Falhas: $FAILURE_COUNT"

if [ $FAILURE_COUNT -gt 0 ]; then
    echo ""
    echo "Modelos que falharam:"
    for model in "${FAILED_MODELS[@]}"; do
        echo "  - $model"
    done
fi

echo ""
echo "Log completo salvo em:"
echo "  $BATCH_LOG"
echo ""
echo "Logs individuais em:"
echo "  logs/experiments/phase4_multimodel/google_gemma-*/"
echo "================================================================================"
echo "Término: $(date)"
echo "================================================================================"

# Logar relatório final
{
    echo ""
    echo "============================================"
    echo "RELATÓRIO FINAL"
    echo "============================================"
    echo "Total: 4 modelos"
    echo "Sucessos: $SUCCESS_COUNT"
    echo "Falhas: $FAILURE_COUNT"

    if [ $FAILURE_COUNT -gt 0 ]; then
        echo ""
        echo "Modelos que falharam:"
        for model in "${FAILED_MODELS[@]}"; do
            echo "  - $model"
        done
    fi

    echo ""
    echo "Término: $(date)"
    echo "============================================"
} >> "$BATCH_LOG"

# Exit code baseado no resultado
if [ $FAILURE_COUNT -eq 0 ]; then
    exit 0
else
    exit 1
fi
