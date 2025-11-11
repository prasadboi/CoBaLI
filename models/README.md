# Models Directory

This directory contains GGUF model files for CoBaLI.

## Downloading Models

### Qwen 0.5B (Recommended for development)

```bash
# Using the download script
../scripts/download_model.sh

# Or manually from Hugging Face
huggingface-cli download \
    Qwen/Qwen2-0.5B-Instruct-GGUF \
    qwen2-0_5b-instruct-q4_0.gguf \
    --local-dir . \
    --local-dir-use-symlinks False
```

**Model Details:**
- Name: Qwen2 0.5B Instruct
- Quantization: Q4_0
- Size: ~300 MB
- Context: 32K tokens (we use 2048 for this project)

### Qwen 3B (For testing scalability)

```bash
huggingface-cli download \
    Qwen/Qwen2-3B-Instruct-GGUF \
    qwen2-3b-instruct-q4_0.gguf \
    --local-dir . \
    --local-dir-use-symlinks False
```

**Model Details:**
- Size: ~1.8 GB
- Better quality, longer inference time
- Use for Phase 4 benchmarks

## Supported Formats

CoBaLI uses llama.cpp, which supports:
- GGUF (recommended)
- GGML (legacy)

## Quantization Levels

| Quantization | Size | Quality | Speed |
|--------------|------|---------|-------|
| Q4_0 | Smallest | Good | Fastest |
| Q5_0 | Medium | Better | Fast |
| Q8_0 | Large | Best | Slower |

We recommend Q4_0 for development and benchmarking.

## Adding Your Own Models

1. Convert to GGUF format (if needed)
2. Place in this directory
3. Update model path in config files

```yaml
# configs/cobali_config.yaml
model:
  path: "models/your-model.gguf"
```

## Note

Model files are gitignored (too large for git). Download them locally on each machine.

