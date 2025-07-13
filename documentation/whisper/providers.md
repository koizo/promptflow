## Whisper Providers

The speech transcription flow supports three Whisper providers, each with different advantages:

### 1. Local Whisper (Default)
```bash
curl -X POST http://localhost:8000/api/v1/speech-transcription/execute \
  -F "file=@audio.wav" \
  -F "whisper_provider=local" \
  -F "whisper_model=base"
```
- **Models**: tiny, base, small, medium, large
- **Pros**: No API costs, works offline, good baseline performance
- **Cons**: Limited to OpenAI's original models
- **Best for**: Development, cost-sensitive deployments

### 2. OpenAI Whisper API
```bash
curl -X POST http://localhost:8000/api/v1/speech-transcription/execute \
  -F "file=@audio.wav" \
  -F "whisper_provider=openai"
```
- **Requirements**: OPENAI_API_KEY environment variable
- **Pros**: Highest quality, latest improvements, no local compute
- **Cons**: Per-request costs, requires internet
- **Best for**: Production with quality requirements

### 3. HuggingFace Models (NEW!)
```bash
curl -X POST http://localhost:8000/api/v1/speech-transcription/execute \
  -F "file=@audio.wav" \
  -F "whisper_provider=huggingface" \
  -F "hf_model_name=openai/whisper-large-v3" \
  -F "device=auto"
```

#### Popular HuggingFace Models:
- **openai/whisper-tiny** (39M params, fastest)
- **openai/whisper-base** (74M params, good balance)
- **openai/whisper-large-v3** (1550M params, best quality)
- **distil-whisper/distil-large-v2** (756M params, **6x faster**)
- **Custom fine-tuned models** (username/model-name)

#### Device Options:
- **auto**: GPU if available, fallback to CPU
- **cuda**: Force GPU usage (requires NVIDIA GPU)
- **cpu**: Force CPU usage (slower but universal)

#### Advantages:
- **Latest Models**: Access to newest Whisper variants
- **Faster Options**: Distil-Whisper models with 6x speedup
- **Custom Models**: Use domain-specific fine-tuned models
- **No API Costs**: Run locally without per-request charges
- **Flexibility**: Easy model switching via configuration