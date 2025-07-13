# Speech Transcription

Advanced speech-to-text capabilities with multiple provider support, real-time processing, and high accuracy across different audio formats and languages.

## Overview

The Speech Transcription feature provides robust audio-to-text conversion using state-of-the-art speech recognition models. It supports multiple providers, languages, and audio formats with advanced features like speaker diarization and timestamp extraction.

## Features

- **Multi-Provider Support**: Local Whisper, OpenAI Whisper API, HuggingFace models
- **High Accuracy**: State-of-the-art speech recognition models
- **Multi-Language**: Support for 100+ languages
- **Speaker Diarization**: Identify different speakers in audio
- **Timestamp Extraction**: Precise timing information for transcribed text
- **Multiple Formats**: MP3, WAV, M4A, FLAC, OGG support
- **Real-time Processing**: Stream processing capabilities

## Provider Options

### Local Whisper (Recommended for Privacy)
- **Best for**: Sensitive content, offline processing, cost control
- **Models**: tiny, base, small, medium, large, large-v2, large-v3
- **Performance**: Fast local processing, no API costs
- **Privacy**: Complete data privacy, no external API calls
- **Resource Usage**: Requires local GPU/CPU resources

### OpenAI Whisper API (Recommended for Accuracy)
- **Best for**: Highest accuracy, production workloads, scalability
- **Models**: whisper-1 (latest production model)
- **Performance**: Excellent accuracy, cloud-based processing
- **Features**: Advanced noise handling, punctuation, formatting
- **Cost**: Pay-per-use API pricing

### HuggingFace Models (Recommended for Customization)
- **Best for**: Custom models, specific domains, research
- **Models**: Various Whisper implementations and fine-tuned models
- **Flexibility**: Custom model loading and configuration
- **Community**: Access to community-trained models

## Supported Models

### Local Whisper Models
```yaml
# Ultra-fast, lowest accuracy
model_size: "tiny"      # 39 MB, ~32x realtime

# Balanced speed and accuracy
model_size: "base"      # 74 MB, ~16x realtime
model_size: "small"     # 244 MB, ~6x realtime

# High accuracy
model_size: "medium"    # 769 MB, ~2x realtime
model_size: "large"     # 1550 MB, ~1x realtime

# Highest accuracy (recommended)
model_size: "large-v2"  # 1550 MB, latest improvements
model_size: "large-v3"  # 1550 MB, newest version
```

### HuggingFace Models
```yaml
# OpenAI Whisper implementations
hf_model_name: "openai/whisper-tiny"
hf_model_name: "openai/whisper-base"
hf_model_name: "openai/whisper-small"
hf_model_name: "openai/whisper-medium"
hf_model_name: "openai/whisper-large-v2"

# Fine-tuned models
hf_model_name: "openai/whisper-large-v2-en"  # English-optimized
hf_model_name: "openai/whisper-medium-es"    # Spanish-optimized
```

## Configuration Options

### Basic Configuration
```yaml
inputs:
  - name: "file"              # Audio file (required)
  - name: "provider"          # "local", "openai", "huggingface"
  - name: "language"          # Target language (auto-detect if not specified)
  - name: "model_size"        # Model size for local/HF providers
```

### Advanced Configuration
```yaml
inputs:
  - name: "temperature"       # Sampling temperature (0.0-1.0)
  - name: "initial_prompt"    # Context prompt for better accuracy
  - name: "word_timestamps"   # Include word-level timestamps
  - name: "speaker_diarization" # Enable speaker identification
  - name: "noise_reduction"   # Audio preprocessing options
  - name: "chunk_length"      # Audio chunk size for processing
```

## Usage Examples

### Basic Transcription (Local Whisper)
```bash
curl -X POST "http://localhost:8000/api/v1/speech-transcription/execute" \
  -F "file=@audio.mp3" \
  -F "provider=local" \
  -F "model_size=base" \
  -F "language=en"
```

**Response:**
```json
{
  "text": "This is the transcribed text from the audio file.",
  "language": "en",
  "confidence": 0.94,
  "duration": 45.2,
  "processing_time": 3.8,
  "provider": "local",
  "model_used": "base"
}
```

### High-Accuracy Transcription (OpenAI)
```bash
curl -X POST "http://localhost:8000/api/v1/speech-transcription/execute" \
  -F "file=@interview.wav" \
  -F "provider=openai" \
  -F "language=en" \
  -F "word_timestamps=true"
```

**Response:**
```json
{
  "text": "Welcome to our podcast. Today we're discussing artificial intelligence.",
  "segments": [
    {
      "text": "Welcome to our podcast.",
      "start": 0.0,
      "end": 2.1,
      "confidence": 0.98
    },
    {
      "text": "Today we're discussing artificial intelligence.",
      "start": 2.5,
      "end": 5.8,
      "confidence": 0.96
    }
  ],
  "language": "en",
  "duration": 120.5,
  "provider": "openai"
}
```

### Multi-Language Auto-Detection
```bash
curl -X POST "http://localhost:8000/api/v1/speech-transcription/execute" \
  -F "file=@multilingual_audio.mp3" \
  -F "provider=local" \
  -F "model_size=medium" \
  -F "language=auto"
```

### Custom Model (HuggingFace)
```bash
curl -X POST "http://localhost:8000/api/v1/speech-transcription/execute" \
  -F "file=@audio.wav" \
  -F "provider=huggingface" \
  -F "hf_model_name=openai/whisper-large-v2" \
  -F "device=cuda"
```

## Response Format

### Standard Response
```json
{
  "success": true,
  "outputs": {
    "text": "Complete transcribed text from the audio file",
    "language": "en",
    "language_confidence": 0.99,
    "duration": 120.5,
    "word_count": 245,
    "confidence": 0.94,
    "processing_time": 8.2,
    "provider": "local",
    "model_used": "base",
    "audio_file": "audio.mp3"
  },
  "metadata": {
    "transcription_provider": "local",
    "model_size": "base",
    "audio_duration": 120.5,
    "detected_language": "en",
    "processing_speed": "15.0x realtime",
    "audio_format": "mp3",
    "sample_rate": 44100,
    "channels": 2
  }
}
```

### With Timestamps
```json
{
  "segments": [
    {
      "id": 0,
      "text": "Hello and welcome to our show.",
      "start": 0.0,
      "end": 2.5,
      "confidence": 0.98,
      "words": [
        {"word": "Hello", "start": 0.0, "end": 0.5, "confidence": 0.99},
        {"word": "and", "start": 0.6, "end": 0.8, "confidence": 0.97},
        {"word": "welcome", "start": 0.9, "end": 1.4, "confidence": 0.98}
      ]
    }
  ]
}
```

### With Speaker Diarization
```json
{
  "speakers": [
    {
      "speaker_id": "SPEAKER_00",
      "segments": [
        {
          "text": "Good morning everyone.",
          "start": 0.0,
          "end": 2.1,
          "confidence": 0.95
        }
      ]
    },
    {
      "speaker_id": "SPEAKER_01",
      "segments": [
        {
          "text": "Thank you for having me.",
          "start": 2.5,
          "end": 4.2,
          "confidence": 0.93
        }
      ]
    }
  ]
}
```

## Performance Comparison

| Provider | Speed | Accuracy | Cost | Privacy | Use Case |
|----------|-------|----------|------|---------|----------|
| **Local Whisper** | ⚡ Fast | 📊 High | 💰 Free | 🔒 Complete | Privacy-sensitive, offline |
| **OpenAI API** | 🚀 Very Fast | 🎯 Highest | 💳 Pay-per-use | ⚠️ Cloud | Production, highest quality |
| **HuggingFace** | ⚡ Fast | 📊 High | 💰 Free | 🔒 Complete | Custom models, research |

## Language Support

### Supported Languages (100+)
- **English** (en) - Highest accuracy
- **Spanish** (es) - High accuracy
- **French** (fr) - High accuracy
- **German** (de) - High accuracy
- **Italian** (it) - High accuracy
- **Portuguese** (pt) - High accuracy
- **Russian** (ru) - High accuracy
- **Japanese** (ja) - High accuracy
- **Korean** (ko) - High accuracy
- **Chinese** (zh) - High accuracy
- **Arabic** (ar) - High accuracy
- **Hindi** (hi) - High accuracy
- And 88+ more languages...

### Language Detection
```yaml
# Automatic language detection
language: "auto"

# Specify expected language for better accuracy
language: "en"

# Multi-language support (some providers)
languages: ["en", "es", "fr"]
```

## Audio Format Support

### Supported Formats
- **MP3** - Most common, good compression
- **WAV** - Uncompressed, highest quality
- **M4A** - Apple format, good quality
- **FLAC** - Lossless compression
- **OGG** - Open source format
- **WEBM** - Web-optimized format

### Audio Quality Recommendations
- **Sample Rate**: 16kHz minimum, 44.1kHz recommended
- **Bit Depth**: 16-bit minimum, 24-bit for best quality
- **Channels**: Mono or stereo supported
- **Duration**: Up to 25MB file size (OpenAI limit)

## Best Practices

### Audio Quality
1. **Clear Recording**: Use good microphones, minimize background noise
2. **Proper Levels**: Avoid clipping, maintain consistent volume
3. **Format Selection**: Use WAV or FLAC for best quality
4. **Preprocessing**: Apply noise reduction if needed

### Model Selection
1. **Local Processing**: Use "base" or "small" for speed, "large-v2" for accuracy
2. **OpenAI API**: Best for production workloads requiring highest accuracy
3. **HuggingFace**: Good for custom models and research applications

### Performance Optimization
1. **Chunk Processing**: Split long audio files for better performance
2. **GPU Acceleration**: Use CUDA for faster local processing
3. **Batch Processing**: Process multiple files in parallel
4. **Model Caching**: Keep models loaded for repeated use

### Language Handling
1. **Specify Language**: Always specify if known for better accuracy
2. **Auto-Detection**: Use for unknown languages, but expect lower accuracy
3. **Context Prompts**: Use initial prompts for domain-specific terminology

## Common Use Cases

### Content Creation
- **Podcast Transcription**: Convert audio content to text
- **Video Subtitles**: Generate subtitles for video content
- **Meeting Notes**: Transcribe meetings and calls
- **Interview Processing**: Convert interviews to text

### Accessibility
- **Hearing Impaired**: Provide text alternatives for audio
- **Language Learning**: Transcribe for language practice
- **Content Analysis**: Analyze spoken content at scale

### Business Applications
- **Customer Service**: Transcribe support calls
- **Legal Documentation**: Convert depositions and hearings
- **Medical Records**: Transcribe patient consultations
- **Research**: Process interview data and focus groups

## Troubleshooting

### Low Accuracy Issues
- **Check Audio Quality**: Ensure clear, noise-free recording
- **Specify Language**: Don't rely on auto-detection for critical content
- **Use Larger Models**: Switch to "medium" or "large" for better accuracy
- **Add Context**: Use initial prompts for domain-specific content

### Performance Issues
- **Model Size**: Use smaller models for faster processing
- **GPU Acceleration**: Enable CUDA for local processing
- **Chunk Audio**: Split long files into smaller segments
- **Check Resources**: Monitor CPU/GPU usage

### Common Errors
- **File Format**: Ensure audio format is supported
- **File Size**: Check file size limits (25MB for OpenAI)
- **Model Loading**: Verify model is properly downloaded/cached
- **API Limits**: Check API rate limits and quotas

## Integration Examples

### Meeting Transcription Flow
```yaml
name: "meeting_transcription"
description: "Transcribe meeting audio with speaker identification"

inputs:
  - name: "file"
    type: "file"
    required: true

steps:
  - name: "transcribe_audio"
    executor: "whisper_processor"
    config:
      file_path: "{{ inputs.file }}"
      provider: "local"
      model_size: "medium"
      language: "en"
      word_timestamps: true
      speaker_diarization: true
      
  - name: "format_transcript"
    executor: "response_formatter"
    config:
      template: "meeting_transcript"
      data: "{{ steps.transcribe_audio }}"
```

### Multi-Language Processing
```yaml
name: "multilingual_transcription"
description: "Process audio in multiple languages"

inputs:
  - name: "file"
    type: "file"
    required: true
  - name: "expected_language"
    type: "string"
    default: "auto"

steps:
  - name: "transcribe_multilingual"
    executor: "whisper_processor"
    config:
      file_path: "{{ inputs.file }}"
      provider: "openai"
      language: "{{ inputs.expected_language }}"
      temperature: 0.0
      initial_prompt: "This is a professional conversation."
```

## API Reference

### Endpoints
- **POST** `/api/v1/speech-transcription/execute` - Process audio transcription
- **GET** `/api/v1/speech-transcription/status/{flow_id}` - Check processing status
- **GET** `/api/v1/speech-transcription/info` - Get flow information

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | required | Audio file to transcribe |
| `provider` | string | "local" | Transcription provider |
| `language` | string | "auto" | Target language |
| `model_size` | string | "base" | Model size (local/HF) |
| `temperature` | float | 0.0 | Sampling temperature |
| `word_timestamps` | boolean | false | Include word timestamps |

### Response Codes
- **200**: Success - Transcription completed
- **400**: Bad Request - Invalid parameters or file
- **422**: Unprocessable Entity - Transcription failed
- **500**: Internal Server Error - System error

## Advanced Features

### Custom Prompts
Improve accuracy with context-specific prompts:

```yaml
initial_prompt: "This is a medical consultation discussing patient symptoms and treatment options."
```

### Noise Reduction
Configure audio preprocessing:

```yaml
preprocessing:
  - noise_reduction: { strength: 0.5 }
  - normalize_audio: { target_db: -20 }
  - high_pass_filter: { cutoff: 80 }
```

### Batch Processing
Process multiple files efficiently:

```yaml
batch_config:
  - chunk_size: 30  # seconds
  - overlap: 2      # seconds
  - parallel_jobs: 4
```
