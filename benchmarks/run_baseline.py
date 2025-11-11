#!/usr/bin/env python3
"""
Benchmark script for CoBaLI baseline (Phase 1)
"""

import sys
import time
import subprocess
import argparse
from pathlib import Path

def run_benchmark(model_path, num_requests=10, prompt_length=128, output_length=128):
    """Run baseline benchmark"""
    
    print("=" * 60)
    print("CoBaLI Baseline Benchmark")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Requests: {num_requests}")
    print(f"Prompt length: {prompt_length}")
    print(f"Output length: {output_length}")
    print()
    
    # Find executable
    project_root = Path(__file__).parent.parent
    exe_path = project_root / "build" / "cobali_main"
    
    if not exe_path.exists():
        print(f"Error: Executable not found at {exe_path}")
        print("Please run: ./scripts/build_cpp.sh")
        sys.exit(1)
    
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("Please run: ./scripts/download_model.sh")
        sys.exit(1)
    
    # Run benchmark
    print("Running benchmark...")
    start_time = time.time()
    
    for i in range(num_requests):
        print(f"Request {i+1}/{num_requests}...", end=" ", flush=True)
        
        cmd = [
            str(exe_path),
            "baseline",
            model_path,
            "--max-tokens", str(output_length),
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("FAILED")
            print(result.stderr)
            sys.exit(1)
        
        print("OK")
    
    elapsed = time.time() - start_time
    
    # Results
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Throughput: {num_requests / elapsed:.2f} requests/sec")
    print(f"Average latency: {elapsed / num_requests * 1000:.2f} ms/request")
    print(f"Token throughput: {num_requests * output_length / elapsed:.2f} tokens/sec")
    print()

def main():
    parser = argparse.ArgumentParser(description="CoBaLI Baseline Benchmark")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--num-requests", type=int, default=10, help="Number of requests")
    parser.add_argument("--prompt-length", type=int, default=128, help="Prompt length")
    parser.add_argument("--output-length", type=int, default=128, help="Output length")
    
    args = parser.parse_args()
    
    run_benchmark(
        args.model,
        args.num_requests,
        args.prompt_length,
        args.output_length
    )

if __name__ == "__main__":
    main()

