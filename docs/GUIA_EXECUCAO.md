# Guia Passo a Passo - Executar Testes por Modelo (Mac M1 Pro 32GB)

Este guia detalha como executar os experimentos da Fase 4 **um modelo por vez**, otimizado para Mac M1 Pro com 32GB RAM.

## 📋 Pré-requisitos

### 1. Verificar Ambiente Python

```bash
# Verificar versão do Python (deve ser 3.9+)
python --version

# Verificar PyTorch com suporte MPS
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS disponível: {torch.backends.mps.is_available()}')"
```

**Saída esperada:**
```
PyTorch: 2.x.x
MPS disponível: True
```

### 2. Verificar Modelo NLU

```bash
# Verificar se o classificador DistilBERT está treinado
ls models/leia_classifier_1k_final/
```

**Se não existir**, treine primeiro:
```bash
python src/classifier/train_classifier.py
```

### 3. Criar Diretório de Logs

```bash
mkdir -p logs/experiments
```

## 🚀 Passo a Passo: Testar Cada Modelo Individualmente

### IMPORTANTE: Executar UM modelo por vez

Para evitar problemas de memória no Mac M1 Pro (32GB), execute os testes **sequencialmente**, um modelo por vez.

---

## 📦 Modelo 1: Gemma 3-1B (Baseline)

**Tamanho:** ~2GB | **RAM necessária:** ~5-6GB | **Tempo estimado:** 15-20 min

### 1.1. Editar experiment_runner.py

Abra o arquivo:
```bash
nano src/orchestrator/experiments/phase4_multimodel/experiment_runner.py
```

Ou use seu editor preferido (VSCode, PyCharm, etc.)

### 1.2. Configurar para Gemma APENAS

No **final do arquivo** (linha ~300), edite:

```python
if __name__ == '__main__':
    print("\n" + "="*80)
    print("FASE 4 - EXPERIMENTO MULTI-MODELO")
    print("="*80 + "\n")

    # Configuração do experimento
    runner = ExperimentRunner(
        experiment_name="phase4_multimodel",
        models_to_test=[
            "google/gemma-3-1b-it",  # ✓ APENAS GEMMA
            # Comentar outros modelos:
            # "meta-llama/Llama-3.2-3B-Instruct",
            # "mistralai/Mistral-7B-Instruct-v0.2",
            # "microsoft/Phi-3-mini-4k-instruct",
        ],
        test_scenario="standard",  # 5 turnos
        enable_tools=False,
        use_quantization=False,  # ⚠️ IMPORTANTE: False para M1
        quantization_bits=4,
    )
```

### 1.3. Executar Teste do Gemma

```bash
cd /Users/giossaurus/Developer/leia_tcc

# Executar
python src/orchestrator/experiments/phase4_multimodel/experiment_runner.py
```

### 1.4. Monitorar Execução

Você verá:
```
================================================================================
INICIANDO EXPERIMENTOS: phase4_multimodel
================================================================================

GPU: Apple M1 Pro
CUDA: None

--- Carregando NLU (constante para todos os testes) ---
Modelo NLU carregado com sucesso de '...' (Device: CPU)

================================================================================
MODELO 1/1: google/gemma-3-1b-it
================================================================================

==================================================
Baseline (antes de carregar google/gemma-3-1b-it)
==================================================
RAM:  8234.56 MB / 32768.00 MB (25.1%)
GPU:  MPS (Apple Silicon) - Memória Unificada
      Total: 32768.00 MB, Livre: 24533.44 MB
      Uso: 25.1% (compartilhada com RAM)
==================================================

Carregando modelo NLG: google/gemma-3-1b-it
Dispositivo detectado: mps
...

Executando cenário: standard (5 turnos)

--- Turno 1/5 ---
Foco: Pergunta conceitual direta
[ALUNO]: O que é fotossíntese?
...
✓ Turno 1 concluído (latência: 1234.56ms)
...
```

### 1.5. Verificar Logs

```bash
# Logs salvos em:
ls logs/experiments/phase4_multimodel/google_gemma-3-1b-it/

# Arquivos gerados:
# - interactions.jsonl  (todas as interações)
# - metrics.json        (métricas agregadas)
# - metadata.json       (config do experimento)
```

### 1.6. Limpar Memória Antes do Próximo Modelo

```bash
# Fechar processos Python (força limpeza)
pkill -9 python

# Ou reiniciar o terminal
```

---

## 📦 Modelo 2: Llama 3.2-3B

**Tamanho:** ~6GB | **RAM necessária:** ~10-12GB | **Tempo estimado:** 20-25 min

### 2.1. Editar experiment_runner.py

```python
    runner = ExperimentRunner(
        experiment_name="phase4_multimodel",
        models_to_test=[
            # "google/gemma-3-1b-it",  # ✗ Comentar Gemma
            "meta-llama/Llama-3.2-3B-Instruct",  # ✓ APENAS LLAMA
            # "mistralai/Mistral-7B-Instruct-v0.2",
            # "microsoft/Phi-3-mini-4k-instruct",
        ],
        test_scenario="standard",
        enable_tools=False,
        use_quantization=False,  # ⚠️ False para M1
    )
```

### 2.2. Executar

```bash
python src/orchestrator/experiments/phase4_multimodel/experiment_runner.py
```

**Primeira execução:** O modelo será baixado do HuggingFace (~6GB). Isso pode demorar 10-20 min dependendo da internet.

**Execuções subsequentes:** Modelo será carregado do cache local (~2-3 min).

### 2.3. Verificar Logs

```bash
ls logs/experiments/phase4_multimodel/meta-llama_Llama-3.2-3B-Instruct/
```

### 2.4. Limpar Memória

```bash
pkill -9 python
```

---

## 📦 Modelo 3: Phi-3-Mini-4k

**Tamanho:** ~7GB | **RAM necessária:** ~10-12GB | **Tempo estimado:** 20-25 min

### 3.1. Editar experiment_runner.py

```python
    runner = ExperimentRunner(
        experiment_name="phase4_multimodel",
        models_to_test=[
            "microsoft/Phi-3-mini-4k-instruct",  # ✓ APENAS PHI-3
        ],
        test_scenario="standard",
        enable_tools=False,
        use_quantization=False,
    )
```

### 3.2. Executar

```bash
python src/orchestrator/experiments/phase4_multimodel/experiment_runner.py
```

### 3.3. Verificar Logs

```bash
ls logs/experiments/phase4_multimodel/microsoft_Phi-3-mini-4k-instruct/
```

### 3.4. Limpar Memória

```bash
pkill -9 python
```

---

## 📦 Modelo 4: Mistral 7B (OPCIONAL - Modelo Maior)

**Tamanho:** ~14GB | **RAM necessária:** ~18-20GB | **Tempo estimado:** 30-40 min

⚠️ **AVISO:** Modelo grande! Feche **TODOS** outros aplicativos (Chrome, Slack, etc.) antes de executar.

### 4.1. Fechar Aplicativos

```bash
# Verificar memória disponível
top -l 1 | grep PhysMem

# Deve mostrar > 20GB livres
```

### 4.2. Editar experiment_runner.py

```python
    runner = ExperimentRunner(
        experiment_name="phase4_multimodel",
        models_to_test=[
            "mistralai/Mistral-7B-Instruct-v0.2",  # ✓ APENAS MISTRAL
        ],
        test_scenario="standard",
        enable_tools=False,
        use_quantization=False,
    )
```

### 4.3. Executar

```bash
python src/orchestrator/experiments/phase4_multimodel/experiment_runner.py
```

**Se ocorrer erro "Out of Memory":**
- Feche mais aplicativos
- Ou pule este modelo (opcional para o TCC)

### 4.4. Verificar Logs

```bash
ls logs/experiments/phase4_multimodel/mistralai_Mistral-7B-Instruct-v0.2/
```

---

## 📊 Analisar Todos os Resultados (Jupyter Notebook)

Após executar **todos os modelos desejados**, analise os resultados:

### 1. Abrir Notebook

```bash
cd /Users/giossaurus/Developer/leia_tcc

# Iniciar Jupyter
jupyter notebook src/orchestrator/experiments/phase4_multimodel/analysis.ipynb
```

### 2. Executar Células

No notebook:
1. **Carregar dados**: Executa célula de importação
2. **Tabela técnica**: Latência, VRAM, RAM
3. **Gráficos**: Comparação visual
4. **Análise qualitativa**: Inspeção de diálogos

### 3. Exportar Resultados

O notebook gera:
```
src/orchestrator/experiments/phase4_multimodel/results/
├── technical_metrics.csv
├── trace_analysis.csv
└── [gráficos em PNG]
```

---

## 🔍 Testar Diferentes Cenários

Após testar todos os modelos com `test_scenario="standard"`, você pode repetir com outros cenários:

### Cenário: Scaffolding (5 turnos)

Testa suporte quando aluno diz "não sei":

```python
test_scenario="scaffolding"
```

### Cenário: ReAct (3 turnos)

Testa uso de ferramentas de busca:

```python
test_scenario="react",
enable_tools=True  # ⚠️ Habilitar ferramentas
```

### Cenário: Stress Test (8 turnos)

Testa consistência em conversa longa:

```python
test_scenario="stress"
```

### Cenário: Edge Cases (4 turnos)

Testa casos extremos:

```python
test_scenario="edge_cases"
```

---

## 🛠️ Troubleshooting

### Erro: "Out of Memory"

**Solução 1:** Fechar aplicativos
```bash
# Ver processos
top

# Fechar Chrome, Slack, etc.
```

**Solução 2:** Usar modelo menor
```python
# Trocar:
"mistralai/Mistral-7B-Instruct-v0.2"  # 14GB

# Por:
"microsoft/Phi-3-mini-4k-instruct"    # 7GB
```

**Solução 3:** Reduzir `max_new_tokens`

Editar `src/orchestrator/core/loaders/model_loader.py` (linha ~40):
```python
MODEL_CONFIGS = {
    "gemma": {
        "max_new_tokens": 150,  # Reduza de 250 para 150
        ...
    }
}
```

### Erro: "Model not found"

**Sintoma:**
```
OSError: meta-llama/Llama-3.2-3B-Instruct does not appear to be a model identifier...
```

**Causa:** Modelo não existe no HuggingFace ou nome incorreto.

**Solução:** Verificar nome correto:
```bash
# Verificar no HuggingFace Hub:
# https://huggingface.co/models
```

### Erro: "MPS not available"

**Sintoma:**
```
RuntimeError: MPS not available
```

**Solução:**
```bash
# Reinstalar PyTorch com suporte MPS
pip install --upgrade torch torchvision torchaudio
```

### Modelo Demora Muito para Carregar

**Primeira execução:** Download do HuggingFace (normal)

**Verificar cache:**
```bash
ls ~/.cache/huggingface/hub/
```

**Limpar cache (se necessário):**
```bash
rm -rf ~/.cache/huggingface/hub/
```

---

## 📝 Checklist de Execução

Para facilitar, use este checklist:

- [ ] Verificar PyTorch com MPS (`torch.backends.mps.is_available() == True`)
- [ ] Verificar modelo NLU treinado (`models/leia_classifier_1k_final/`)
- [ ] Criar diretório de logs (`logs/experiments/`)
- [ ] **Modelo 1:** Gemma 3-1B
  - [ ] Editar `experiment_runner.py`
  - [ ] Executar teste
  - [ ] Verificar logs
  - [ ] Limpar memória
- [ ] **Modelo 2:** Llama 3.2-3B
  - [ ] Editar `experiment_runner.py`
  - [ ] Executar teste
  - [ ] Verificar logs
  - [ ] Limpar memória
- [ ] **Modelo 3:** Phi-3-Mini-4k
  - [ ] Editar `experiment_runner.py`
  - [ ] Executar teste
  - [ ] Verificar logs
  - [ ] Limpar memória
- [ ] **Modelo 4 (opcional):** Mistral 7B
  - [ ] Fechar todos aplicativos
  - [ ] Editar `experiment_runner.py`
  - [ ] Executar teste
  - [ ] Verificar logs
- [ ] **Análise:**
  - [ ] Abrir Jupyter Notebook
  - [ ] Executar células de análise
  - [ ] Gerar tabelas e gráficos
  - [ ] Exportar resultados

---

## ⏱️ Tempo Total Estimado

| Tarefa | Tempo |
|--------|-------|
| Setup inicial | 10 min |
| Gemma 3-1B | 15-20 min |
| Llama 3.2-3B | 20-25 min |
| Phi-3-Mini | 20-25 min |
| Mistral 7B (opcional) | 30-40 min |
| Análise (Jupyter) | 15-20 min |
| **TOTAL** | **~2-3 horas** |

**Nota:** Primeiro download de cada modelo adiciona 10-20 min.

---

## 📧 Precisa de Ajuda?

Se encontrar problemas:
1. Verificar logs em `logs/experiment_runner.log`
2. Consultar [README_MAC_M1.md](README_MAC_M1.md)
3. Verificar issues no GitHub do projeto

---

**Última atualização:** 2025-10-23
**Testado em:** macOS Tahoe 26.01, M1 Pro 32GB RAM
