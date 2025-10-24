"""
Carregador de Modelos NLG (Natural Language Generation)
Suporta múltiplos modelos com quantização para execução em hardware de consumidor.

Modelos suportados:
- Gemma (google/gemma-2-2b-it)
- Mistral (mistralai/Mistral-7B-Instruct-v0.2)
- Phi-3 (microsoft/Phi-3-Mini-4k-Instruct)
"""

import torch
import yaml
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Classe responsável por carregar modelos NLG de forma padronizada.
    Configurações podem ser lidas de config.yaml ou passadas como parâmetros.
    """

    # Configurações padrão (fallback se config.yaml não existir)
    # NOTA: Prioridade de configuração:
    #   1. Parâmetros passados explicitamente (**kwargs)
    #   2. Valores em config.yaml
    #   3. Valores padrão abaixo (fallback)
    MODEL_CONFIGS = {
        "gemma": {
            "max_new_tokens": 768,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
        },
        "llama": {
            "max_new_tokens": 768,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
        },
        "mistral": {
            "max_new_tokens": 768,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
        },
        "phi": {
            "max_new_tokens": 768,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
        },
        "qwen": {
            "max_new_tokens": 768,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
        }
    }

    @staticmethod
    def _load_config_from_yaml() -> Dict[str, Any]:
        """
        Carrega configurações do arquivo config.yaml.

        Returns:
            Dicionário com configurações de NLG ou dict vazio se arquivo não existir
        """
        # Caminho para config.yaml (na raiz do projeto)
        config_path = Path(__file__).resolve().parent.parent.parent.parent.parent / "config.yaml"

        if not config_path.exists():
            logger.debug(f"config.yaml não encontrado em {config_path}. Usando configurações padrão.")
            return {}

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                nlg_config = config.get('nlg', {})
                logger.info(f"Configurações carregadas de {config_path}")
                return nlg_config
        except Exception as e:
            logger.warning(f"Erro ao carregar config.yaml: {str(e)}. Usando configurações padrão.")
            return {}

    @staticmethod
    def _detect_model_family(model_name: str) -> str:
        """
        Detecta a família do modelo baseado no nome/ID.

        Args:
            model_name: Nome ou ID do modelo (ex: "meta-llama/Llama-3-8B-Instruct")

        Returns:
            Nome da família do modelo (gemma, llama, mistral, phi, qwen)
        """
        model_name_lower = model_name.lower()

        if "gemma" in model_name_lower:
            return "gemma"
        elif "llama" in model_name_lower:
            return "llama"
        elif "mistral" in model_name_lower:
            return "mistral"
        elif "phi" in model_name_lower:
            return "phi"
        elif "qwen" in model_name_lower:
            return "qwen"
        else:
            logger.warning(f"Família do modelo não reconhecida para '{model_name}'. Usando configuração padrão.")
            return "llama"  # fallback

    @staticmethod
    def load_nlg_model(
        model_name: str,
        use_quantization: bool = True,
        quantization_bits: int = 4,
        device: Optional[str] = None,
        **kwargs
    ) -> Tuple[HuggingFacePipeline, AutoTokenizer, str]:
        """
        Carrega um modelo NLG com suporte a quantização.

        Args:
            model_name: Nome/ID do modelo no HuggingFace Hub
            use_quantization: Se True, aplica quantização (4-bit ou 8-bit)
            quantization_bits: Número de bits para quantização (4 ou 8)
            device: Dispositivo para carregar o modelo ("cuda", "cpu", ou None para auto-detect)
            **kwargs: Parâmetros adicionais para sobrescrever configurações padrão

        Returns:
            Tupla contendo (HuggingFacePipeline, Tokenizer, device_info)

        Raises:
            ValueError: Se o modelo não puder ser carregado
        """
        logger.info(f"Iniciando carregamento do modelo: {model_name}")

        # Detectar dispositivo (suporte para CUDA, MPS/Apple Silicon, e CPU)
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon (M1/M2/M3)
            else:
                device = "cpu"

        logger.info(f"Dispositivo detectado: {device}")

        # Detectar família do modelo
        model_family = ModelLoader._detect_model_family(model_name)

        # Carregar configurações com prioridade: defaults < YAML < kwargs
        # 1. Começar com valores padrão (hardcoded)
        model_config = ModelLoader.MODEL_CONFIGS[model_family].copy()

        # 2. Sobrescrever com valores do config.yaml (se existir)
        yaml_configs = ModelLoader._load_config_from_yaml()
        if model_family in yaml_configs:
            model_config.update(yaml_configs[model_family])
            logger.debug(f"Configurações do config.yaml aplicadas para família '{model_family}'")

        # 3. Sobrescrever com kwargs explícitos (maior prioridade)
        model_config.update(kwargs)

        try:
            # Configurar quantização (se habilitada e GPU disponível)
            quantization_config = None
            if use_quantization and device == "cuda":
                if quantization_bits == 4:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                    logger.info("Aplicando quantização 4-bit (NF4)")
                elif quantization_bits == 8:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                    logger.info("Aplicando quantização 8-bit")
                else:
                    raise ValueError(f"quantization_bits deve ser 4 ou 8, recebido: {quantization_bits}")
            elif use_quantization and device in ["cpu", "mps"]:
                logger.warning(f"Quantização desabilitada: não suportada em {device.upper()}. Carregando modelo em precisão completa.")
                if device == "mps":
                    logger.info("Apple Silicon detectado: usando FP16 para melhor performance.")

            # Carregar tokenizer
            logger.info(f"Carregando tokenizer para {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True  # Necessário para modelos com código customizado (ex: Phi-4)
            )

            # Configurar pad_token se não existir
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("pad_token não encontrado, usando eos_token como pad_token")

            # Carregar modelo
            logger.info(f"Carregando modelo {model_name} em {device}...")

            # Determinar dtype baseado no dispositivo
            if device == "cuda" and not use_quantization:
                torch_dtype = torch.float16
            elif device == "mps":
                torch_dtype = torch.float16  # MPS funciona melhor com FP16
            else:
                torch_dtype = torch.float32

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True  # Necessário para modelos com código customizado (ex: Phi-4)
            )

            # Mover para dispositivo se necessário (CPU ou MPS)
            if device in ["cpu", "mps"]:
                model = model.to(device)
                logger.info(f"Modelo movido para {device}")

            # Criar pipeline do HuggingFace
            logger.info("Criando pipeline de geração de texto...")

            # Determinar device parameter para o pipeline
            if device == "cuda":
                pipeline_device = 0
            elif device == "mps":
                pipeline_device = -1  # Pipeline usa modelo já no device MPS
            else:
                pipeline_device = -1  # CPU

            nlg_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=model_config["max_new_tokens"],
                do_sample=True,
                temperature=model_config["temperature"],
                top_k=model_config["top_k"],
                top_p=model_config["top_p"],
                return_full_text=False,
                device=pipeline_device
            )

            # Envolver no LangChain HuggingFacePipeline
            llm = HuggingFacePipeline(pipeline=nlg_pipeline)

            # Informações sobre o dispositivo
            device_info = f"{device.upper()}"
            if use_quantization and device == "cuda":
                device_info += f" ({quantization_bits}-bit quantized)"

            logger.info(f"Modelo {model_name} carregado com sucesso em {device_info}")

            return llm, tokenizer, device_info

        except Exception as e:
            logger.error(f"Erro ao carregar modelo {model_name}: {str(e)}")
            raise ValueError(f"Falha ao carregar modelo {model_name}: {str(e)}")

    @staticmethod
    def unload_model(llm: HuggingFacePipeline) -> None:
        """
        Descarrega um modelo da memória e limpa o cache da GPU.

        Args:
            llm: Instância do HuggingFacePipeline a ser descarregada
        """
        try:
            # Deletar o pipeline e modelo
            del llm

            # Limpar cache da GPU se disponível
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("Cache da GPU limpo")

            logger.info("Modelo descarregado com sucesso")

        except Exception as e:
            logger.warning(f"Erro ao descarregar modelo: {str(e)}")


# --- Função de conveniência para uso rápido ---
def load_nlg_model(model_name: str, **kwargs) -> Tuple[HuggingFacePipeline, AutoTokenizer, str]:
    """
    Função de conveniência para carregar um modelo NLG.
    Wrapper para ModelLoader.load_nlg_model().

    Args:
        model_name: Nome/ID do modelo no HuggingFace Hub
        **kwargs: Parâmetros adicionais (use_quantization, quantization_bits, etc.)

    Returns:
        Tupla contendo (HuggingFacePipeline, Tokenizer, device_info)
    """
    return ModelLoader.load_nlg_model(model_name, **kwargs)


# --- Bloco de Teste ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("\n=== Teste do ModelLoader ===\n")

    # Teste com Gemma (modelo pequeno para teste rápido)
    test_model = "google/gemma-3-1b-it"

    try:
        print(f"Testando carregamento de: {test_model}")
        llm, tokenizer, device_info = load_nlg_model(
            test_model,
            use_quantization=True,
            quantization_bits=4
        )

        print(f"\n✓ Modelo carregado com sucesso!")
        print(f"  - Dispositivo: {device_info}")
        print(f"  - Tokenizer vocab size: {len(tokenizer)}")

        # Teste de geração
        print("\nTestando geração de texto...")
        test_prompt = "O que é educação freiriana?"
        response = llm.invoke(test_prompt)
        print(f"Prompt: {test_prompt}")
        print(f"Resposta: {response[:200]}...")

        # Descarregar modelo
        print("\nDescarregando modelo...")
        ModelLoader.unload_model(llm)
        print("✓ Teste concluído com sucesso!")

    except Exception as e:
        print(f"\n✗ Erro durante o teste: {str(e)}")
        import traceback
        traceback.print_exc()
