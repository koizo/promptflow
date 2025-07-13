# PromptFlow Documentation

Comprehensive documentation for the PromptFlow AI Inference Platform - build powerful AI workflows with YAML configuration.

## 📚 Documentation Structure

### Platform Architecture
Understanding the system design and components:

- **[Architecture Overview](architecture.md)** - Complete system architecture, components, and data flow

### AI Capabilities
Detailed guides for each AI feature available in the platform:

- **[OCR Processing](ai-capabilities/ocr-processing.md)** - Extract text from images and documents with multi-language support
- **[Speech Transcription](ai-capabilities/speech-transcription.md)** - Convert audio to text with multiple provider support
- **[Sentiment Analysis](ai-capabilities/sentiment-analysis.md)** - Analyze text sentiment with emotion detection and aspect analysis
- **[Image Classification](ai-capabilities/image-classification.md)** - Classify images using multiple model architectures
- **[LLM Analysis](ai-capabilities/llm-analysis.md)** - Advanced text analysis using Large Language Models with custom prompts

### Flow Development
Learn how to create and configure AI workflows:

- **[YAML Configuration](flows/yaml-configuration.md)** - Complete guide to creating flows with YAML
- **[Flow Examples](flows/examples/)** - Real-world flow examples and templates

### Executor Development
Extend the platform with custom AI capabilities:

- **[Creating Executors](executors/creating-executors.md)** - Build custom AI executors and integrations
- **[Executor Patterns](executors/patterns/)** - Common patterns and best practices

## 🚀 Quick Start

### 1. Understanding the Platform
PromptFlow combines AI executors through YAML-defined flows to create powerful processing pipelines:

```
YAML Flow → Flow Engine → Executors → Auto-Generated API
```

### 2. Basic Flow Example
```yaml
name: "document_analysis"
description: "Extract and analyze text from documents"

inputs:
  - name: "file"
    type: "file"
    required: true

steps:
  - name: "extract_text"
    executor: "ocr_processor"
    config:
      image_path: "{{ inputs.file }}"
      
  - name: "analyze_sentiment"
    executor: "sentiment_analyzer"
    config:
      text: "{{ steps.extract_text.text }}"
```

### 3. API Usage
```bash
curl -X POST "http://localhost:8000/api/v1/document-analysis/execute" \
  -F "file=@document.pdf"
```

## 🎯 AI Capabilities Overview

### OCR Processing
- **Multi-language support** (100+ languages)
- **High accuracy** text extraction
- **Bounding box detection** for precise text location
- **Confidence scoring** and filtering

### Speech Transcription
- **Multiple providers** (Local Whisper, OpenAI, HuggingFace)
- **Real-time processing** capabilities
- **Speaker diarization** and timestamp extraction
- **100+ language support**

### Sentiment Analysis
- **Dual provider architecture** (HuggingFace + LLM)
- **Emotion detection** and aspect-based analysis
- **Multiple analysis types** (basic, detailed, comprehensive)
- **95-99% accuracy** on clear sentiment cases

### Image Classification
- **Multiple model architectures** (ViT, ResNet, EfficientNet, ConvNeXT)
- **Dual provider support** (HuggingFace + OpenAI Vision)
- **Fast processing** (0.1-0.8 seconds)
- **High accuracy** (95-99% on ImageNet classes)

### LLM Analysis
- **Multiple LLM providers** (Ollama local, extensible for OpenAI/Anthropic)
- **Custom prompt engineering** for specific analysis needs
- **Chunked text processing** for large documents
- **Model selection** (mistral, llama3.2, codellama, etc.)

## 🏗️ Architecture Overview

### Core Components
- **Flow Engine** - Orchestrates AI processing workflows
- **Executor Registry** - Manages available AI capabilities
- **Template Engine** - Handles dynamic configuration
- **State Management** - Tracks execution progress via Redis

### Execution Modes
- **Asynchronous** (Production) - Non-blocking, scalable processing
- **Synchronous** (Development) - Immediate results for testing

### Container Architecture
```yaml
services:
  app:              # FastAPI application
  redis:            # State storage
  celery-worker-*:  # Async processing workers
  celery-flower:    # Worker monitoring
```

## 📖 Detailed Guides

### For Users
1. **[YAML Configuration Guide](flows/yaml-configuration.md)** - Learn to create flows
2. **[OCR Processing](ai-capabilities/ocr-processing.md)** - Extract text from documents
3. **[Speech Transcription](ai-capabilities/speech-transcription.md)** - Convert audio to text
4. **[Sentiment Analysis](ai-capabilities/sentiment-analysis.md)** - Analyze text sentiment
5. **[Image Classification](ai-capabilities/image-classification.md)** - Classify images

### For Developers
1. **[Creating Executors](executors/creating-executors.md)** - Build custom AI capabilities
2. **[Development Patterns](executors/creating-executors.md#development-patterns)** - Common implementation patterns
3. **[Testing Guide](executors/creating-executors.md#testing-your-executor)** - Test your executors
4. **[Integration Guide](executors/creating-executors.md#integration-and-registration)** - Deploy your executors

## 🔧 Configuration Examples

### Multi-Modal Analysis
```yaml
name: "multimodal_analysis"
description: "Analyze both text and image content"

inputs:
  - name: "text_content"
    type: "string"
    required: false
  - name: "image_file"
    type: "file"
    required: false

steps:
  - name: "analyze_text"
    executor: "sentiment_analyzer"
    config:
      text: "{{ inputs.text_content }}"
      analysis_type: "comprehensive"
    condition: "{{ inputs.text_content | length > 0 }}"

  - name: "classify_image"
    executor: "vision_classifier"
    config:
      image: "{{ inputs.image_file }}"
      provider: "huggingface"
      top_k: 5
    condition: "{{ inputs.image_file is defined }}"
```

### Batch Processing
```yaml
name: "batch_sentiment_analysis"
description: "Process multiple texts for sentiment"

inputs:
  - name: "texts"
    type: "array"
    required: true
    validation:
      min_items: 1
      max_items: 100

steps:
  - name: "batch_analyze"
    executor: "sentiment_analyzer"
    config:
      texts: "{{ inputs.texts }}"
      provider: "huggingface"
      batch_processing: true
```

## 🚀 Performance Guidelines

### Speed Optimization
- **HuggingFace providers** for fast, local processing
- **Model caching** for repeated operations
- **GPU acceleration** when available
- **Batch processing** for multiple items

### Accuracy Optimization
- **OpenAI/LLM providers** for highest accuracy
- **Larger models** for better results
- **Proper input preprocessing**
- **Confidence threshold tuning**

## 🔍 Troubleshooting

### Common Issues
1. **Template Errors** - Check variable references and defaults
2. **Model Loading** - Verify model availability and caching
3. **File Processing** - Ensure proper file formats and sizes
4. **Performance** - Monitor resource usage and optimize accordingly

### Debug Tools
```bash
# Validate flow syntax
curl -X POST "http://localhost:8000/api/v1/flows/validate" \
  -F "flow_file=@my_flow.yaml"

# Check system health
curl http://localhost:8000/health

# Monitor workers
curl http://localhost:8000/flower
```

## 📊 API Reference

### Core Endpoints
- **Flow Execution**: `POST /api/v1/{flow-name}/execute`
- **Status Check**: `GET /api/v1/{flow-name}/status/{flow_id}`
- **Flow Info**: `GET /api/v1/{flow-name}/info`
- **Health Check**: `GET /health`
- **Documentation**: `GET /docs`

### Response Format
```json
{
  "success": true,
  "outputs": {
    "result": "processed_data",
    "confidence": 0.95
  },
  "metadata": {
    "processing_time": 1.23,
    "executor": "sentiment_analyzer"
  }
}
```

## 🤝 Contributing

### Adding New AI Capabilities
1. **Create Executor** - Implement your AI logic
2. **Write Tests** - Comprehensive unit and integration tests
3. **Add Documentation** - Update this documentation
4. **Create Examples** - Provide usage examples

### Documentation Updates
1. **Follow Structure** - Use existing documentation patterns
2. **Include Examples** - Provide practical examples
3. **Test Instructions** - Verify all examples work
4. **Update Index** - Add new content to this index

## 📝 License

MIT License - see [LICENSE](../LICENSE) file for details.

## 🆘 Support

- **Documentation Issues** - Create GitHub issues for documentation problems
- **Feature Requests** - Suggest new AI capabilities or improvements
- **Bug Reports** - Report issues with existing functionality
- **Community** - Join discussions about AI workflow automation

---

*This documentation is continuously updated as new features are added to the platform.*
