"""
run_benchmark.py — Hardware Profiling for Mini-DeBERTa
"""

import torch
import time
import numpy as np

# Import your master Config and Model Class
from config import Config
from Model.model import PhishingTransformer

def run_hardware_profile():
    print(f"{'='*55}\n  STARTING HARDWARE PROFILING (CUDA)\n{'='*55}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[!] Error: CUDA not detected. Hardware profiling requires a GPU.")
        return

    # 1. Initialize Config & Model Instance
    print("Initializing model architecture...")
    config_obj = Config()
    model = PhishingTransformer(config_obj).to(device)
    model.eval()

    # 2. Setup Dummy Tensors (Simulating 1 Email + 26 Features)
    dummy_input_ids = torch.randint(0, 30522, (1, 512), dtype=torch.long).to(device)
    dummy_attention_mask = torch.ones((1, 512), dtype=torch.long).to(device)

    # ADDED: token_type_ids (all zeros) for DeBERTa
    dummy_token_type_ids = torch.zeros((1, 512), dtype=torch.long).to(device)

    dummy_features = torch.rand((1, 26), dtype=torch.float32).to(device)

    # Helper function to run forward pass with all 4 required arguments
    def forward_pass():
        return model(dummy_input_ids, dummy_attention_mask, dummy_token_type_ids, dummy_features)

    # 3. GPU Warm-up (Required for accurate CUDA timing)
    print("Warming up GPU...")
    with torch.no_grad():
        for _ in range(50):
            forward_pass()

    # 4. Measure Inference Latency
    print("Measuring inference latency (1000 iterations)...")
    latencies = []
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for _ in range(1000):
            starter.record()
            forward_pass()
            ender.record()

            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            latencies.append(curr_time)

    # 5. Measure VRAM Allocation
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    # Calculate metrics
    mean_latency_ms = np.mean(latencies)
    std_latency_ms = np.std(latencies)
    throughput_eps = 1000 / mean_latency_ms

    print(f"\n{'='*55}\n  PROFILING RESULTS\n{'='*55}")
    print(f"Hardware            : {torch.cuda.get_device_name(0)}")
    print(f"Peak VRAM Allocated : {peak_vram_mb:.2f} MB")
    print(f"Inference Latency   : {mean_latency_ms:.2f} ms ± {std_latency_ms:.2f} ms")
    print(f"Throughput          : {throughput_eps:.0f} emails / second")

if __name__ == "__main__":
    run_hardware_profile()