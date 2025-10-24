
"""
Monitor de Uso de Memória (RAM e VRAM)

Monitora o consumo de recursos durante os experimentos.
Essencial para a análise de performance técnica na Fase 4.
"""

import psutil
import torch
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """
    Monitora uso de RAM e VRAM (GPU) do sistema.
    """

    @staticmethod
    def get_ram_usage() -> Dict[str, float]:
        """
        Obtém informações sobre o uso de RAM do sistema.

        Returns:
            Dicionário com métricas de RAM:
            - used_mb: RAM utilizada em MB
            - available_mb: RAM disponível em MB
            - percent: Percentual de uso
            - total_mb: RAM total em MB
        """
        try:
            mem = psutil.virtual_memory()
            return {
                "used_mb": round(mem.used / (1024 ** 2), 2),
                "available_mb": round(mem.available / (1024 ** 2), 2),
                "percent": round(mem.percent, 2),
                "total_mb": round(mem.total / (1024 ** 2), 2),
            }
        except Exception as e:
            logger.error(f"Erro ao obter uso de RAM: {str(e)}")
            return {
                "used_mb": 0.0,
                "available_mb": 0.0,
                "percent": 0.0,
                "total_mb": 0.0,
                "error": str(e)
            }

    @staticmethod
    def get_vram_usage(device: int = 0) -> Dict[str, float]:
        """
        Obtém informações sobre o uso de VRAM da GPU.

        Args:
            device: Índice da GPU (padrão: 0)

        Returns:
            Dicionário com métricas de VRAM:
            - allocated_mb: VRAM alocada em MB (ou 0 se memória unificada)
            - reserved_mb: VRAM reservada em MB
            - free_mb: VRAM livre em MB (estimativa)
            - total_mb: VRAM total em MB
            - percent: Percentual de uso (baseado em allocated)
            - unified_memory: True se Apple Silicon (memória unificada)
        """
        # Verificar MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Apple Silicon: memória unificada (RAM = VRAM)
            # Usar psutil para medir toda a memória
            mem = psutil.virtual_memory()
            return {
                "allocated_mb": 0.0,  # Não há separação VRAM/RAM
                "reserved_mb": 0.0,
                "free_mb": round(mem.available / (1024 ** 2), 2),
                "total_mb": round(mem.total / (1024 ** 2), 2),
                "percent": round(mem.percent, 2),
                "available": True,
                "unified_memory": True,  # Flag indicando memória unificada
                "device_type": "mps"
            }

        # Verificar CUDA (NVIDIA)
        if not torch.cuda.is_available():
            return {
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "free_mb": 0.0,
                "total_mb": 0.0,
                "percent": 0.0,
                "available": False,
                "unified_memory": False
            }

        try:
            # Obter estatísticas de memória da GPU NVIDIA
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            total = torch.cuda.get_device_properties(device).total_memory

            allocated_mb = round(allocated / (1024 ** 2), 2)
            reserved_mb = round(reserved / (1024 ** 2), 2)
            total_mb = round(total / (1024 ** 2), 2)
            free_mb = round((total - reserved) / (1024 ** 2), 2)
            percent = round((allocated / total) * 100, 2)

            return {
                "allocated_mb": allocated_mb,
                "reserved_mb": reserved_mb,
                "free_mb": free_mb,
                "total_mb": total_mb,
                "percent": percent,
                "available": True,
                "unified_memory": False,
                "device_type": "cuda"
            }

        except Exception as e:
            logger.error(f"Erro ao obter uso de VRAM: {str(e)}")
            return {
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "free_mb": 0.0,
                "total_mb": 0.0,
                "percent": 0.0,
                "available": False,
                "unified_memory": False,
                "error": str(e)
            }

    @staticmethod
    def get_full_memory_snapshot() -> Dict[str, Dict[str, float]]:
        """
        Obtém um snapshot completo do uso de memória (RAM + VRAM).

        Returns:
            Dicionário com métricas de RAM e VRAM
        """
        return {
            "ram": MemoryMonitor.get_ram_usage(),
            "vram": MemoryMonitor.get_vram_usage(),
        }

    @staticmethod
    def print_memory_summary(prefix: str = "") -> None:
        """
        Imprime um resumo formatado do uso de memória.

        Args:
            prefix: Texto para exibir antes do resumo (ex: "Antes do carregamento")
        """
        snapshot = MemoryMonitor.get_full_memory_snapshot()
        ram = snapshot["ram"]
        vram = snapshot["vram"]

        print(f"\n{'='*50}")
        if prefix:
            print(f"{prefix}")
        print(f"{'='*50}")
        print(f"RAM:  {ram['used_mb']:.2f} MB / {ram['total_mb']:.2f} MB ({ram['percent']:.1f}%)")

        if vram["available"]:
            if vram.get("unified_memory", False):
                # Apple Silicon: memória unificada
                print(f"GPU:  MPS (Apple Silicon) - Memória Unificada")
                print(f"      Total: {vram['total_mb']:.2f} MB, Livre: {vram['free_mb']:.2f} MB")
                print(f"      Uso: {vram['percent']:.1f}% (compartilhada com RAM)")
            else:
                # NVIDIA CUDA: memória dedicada
                print(f"VRAM: {vram['allocated_mb']:.2f} MB / {vram['total_mb']:.2f} MB ({vram['percent']:.1f}%)")
                print(f"      (Reservada: {vram['reserved_mb']:.2f} MB, Livre: {vram['free_mb']:.2f} MB)")
        else:
            print("GPU:  Não disponível (usando CPU)")

        print(f"{'='*50}\n")

    @staticmethod
    def clear_gpu_cache() -> None:
        """
        Limpa o cache da GPU (se disponível).
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("Cache da GPU limpo")
        else:
            logger.info("GPU não disponível, nada a limpar")

    @staticmethod
    def get_gpu_info() -> Dict[str, any]:
        """
        Obtém informações sobre a GPU disponível.

        Returns:
            Dicionário com informações da GPU
        """
        if not torch.cuda.is_available():
            return {
                "available": False,
                "device_count": 0,
            }

        try:
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            device_capability = torch.cuda.get_device_capability(current_device)

            return {
                "available": True,
                "device_count": device_count,
                "current_device": current_device,
                "device_name": device_name,
                "compute_capability": f"{device_capability[0]}.{device_capability[1]}",
                "cuda_version": torch.version.cuda,
            }

        except Exception as e:
            logger.error(f"Erro ao obter informações da GPU: {str(e)}")
            return {
                "available": False,
                "error": str(e)
            }


class MemoryTracker:
    """
    Rastreia o uso de memória ao longo de um experimento.
    Permite medir o delta de memória antes/depois de operações.
    """

    def __init__(self, name: str = "Experimento"):
        """
        Args:
            name: Nome do experimento/operação sendo rastreada
        """
        self.name = name
        self.baseline: Optional[Dict] = None
        self.current: Optional[Dict] = None

    def set_baseline(self) -> None:
        """Captura o estado inicial da memória (baseline)."""
        self.baseline = MemoryMonitor.get_full_memory_snapshot()
        logger.info(f"[{self.name}] Baseline de memória capturada")

    def measure_current(self) -> None:
        """Captura o estado atual da memória."""
        self.current = MemoryMonitor.get_full_memory_snapshot()
        logger.info(f"[{self.name}] Medição atual de memória capturada")

    def get_delta(self) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Calcula a diferença entre baseline e estado atual.

        Returns:
            Dicionário com deltas de RAM e VRAM, ou None se baseline não foi definido
        """
        if self.baseline is None:
            logger.warning("Baseline não foi definida. Chame set_baseline() primeiro.")
            return None

        if self.current is None:
            self.measure_current()

        delta = {
            "ram": {},
            "vram": {}
        }

        # Delta de RAM
        for key in ["used_mb", "percent"]:
            baseline_val = self.baseline["ram"].get(key, 0)
            current_val = self.current["ram"].get(key, 0)
            delta["ram"][f"{key}_delta"] = round(current_val - baseline_val, 2)

        # Delta de VRAM (se disponível)
        if self.baseline["vram"]["available"] and self.current["vram"]["available"]:
            for key in ["allocated_mb", "reserved_mb", "percent"]:
                baseline_val = self.baseline["vram"].get(key, 0)
                current_val = self.current["vram"].get(key, 0)
                delta["vram"][f"{key}_delta"] = round(current_val - baseline_val, 2)

        return delta

    def print_report(self) -> None:
        """Imprime um relatório formatado do uso de memória."""
        if self.baseline is None:
            print(f"[{self.name}] Nenhuma baseline definida.")
            return

        delta = self.get_delta()

        print(f"\n{'='*60}")
        print(f"Relatório de Memória: {self.name}")
        print(f"{'='*60}")

        print(f"\nRAM:")
        print(f"  Baseline: {self.baseline['ram']['used_mb']:.2f} MB ({self.baseline['ram']['percent']:.1f}%)")
        print(f"  Atual:    {self.current['ram']['used_mb']:.2f} MB ({self.current['ram']['percent']:.1f}%)")
        print(f"  Delta:    +{delta['ram']['used_mb_delta']:.2f} MB (+{delta['ram']['percent_delta']:.1f}%)")

        if self.baseline["vram"]["available"]:
            print(f"\nVRAM:")
            print(f"  Baseline: {self.baseline['vram']['allocated_mb']:.2f} MB ({self.baseline['vram']['percent']:.1f}%)")
            print(f"  Atual:    {self.current['vram']['allocated_mb']:.2f} MB ({self.current['vram']['percent']:.1f}%)")
            print(f"  Delta:    +{delta['vram']['allocated_mb_delta']:.2f} MB (+{delta['vram']['percent_delta']:.1f}%)")

        print(f"{'='*60}\n")


# --- Bloco de Teste ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("\n=== Teste do MemoryMonitor ===\n")

    # 1. Informações da GPU
    print("1. Informações da GPU:")
    gpu_info = MemoryMonitor.get_gpu_info()
    if gpu_info["available"]:
        print(f"   GPU: {gpu_info['device_name']}")
        print(f"   CUDA: {gpu_info['cuda_version']}")
        print(f"   Compute Capability: {gpu_info['compute_capability']}")
    else:
        print("   GPU não disponível")

    # 2. Snapshot de memória
    print("\n2. Snapshot de Memória:")
    MemoryMonitor.print_memory_summary("Estado Inicial")

    # 3. Teste do MemoryTracker
    print("3. Teste do MemoryTracker:")
    tracker = MemoryTracker(name="Teste de Alocação")
    tracker.set_baseline()

    print("   Alocando tensor de teste (500 MB)...")
    if torch.cuda.is_available():
        # Alocar ~500 MB na GPU
        test_tensor = torch.randn(500, 1024, 256, device="cuda")
    else:
        # Alocar ~500 MB na RAM
        test_tensor = torch.randn(500, 1024, 256)

    tracker.measure_current()
    tracker.print_report()

    # Limpar
    del test_tensor
    MemoryMonitor.clear_gpu_cache()

    print("\n✓ Teste concluído!")
