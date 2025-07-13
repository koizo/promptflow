# 🤖 AI Flow Generator (BETA)

**Generate YAML workflows from natural language descriptions using LLM**

The AI Flow Generator is a powerful BETA feature that transforms plain English descriptions into production-ready AI workflows automatically. Using advanced language models, it understands your requirements and creates complete YAML flow definitions with proper executors, configurations, and response formatting.

## 🚀 Quick Start

```bash
# Generate a flow from natural language
cli/generate_flow.sh "Extract text from PDF and analyze sentiment"

# Interactive mode with clarifying questions
cli/generate_flow.sh --interactive "Process customer feedback documents"

# Preview without saving
cli/generate_flow.sh --preview-only "OCR an image and summarize the text"

# Custom output location
cli/generate_flow.sh "Analyze invoice data" --output custom_flows/invoice_analyzer.yaml
```

## ✨ Key Capabilities

### 🧠 Natural Language Processing
- **Plain English Input** - Describe workflows in natural language
- **Intent Recognition** - Understands complex AI workflow requirements
- **Context Awareness** - Considers multi-modal processing needs

### 🎯 Smart Executor Selection
- **Automatic Mapping** - Chooses appropriate AI executors based on description
- **Multi-Modal Support** - Combines image, audio, text, and document processing
- **Provider Optimization** - Selects best providers (HuggingFace, OpenAI, Local)

### 🏗️ Complete Flow Generation
- **Full YAML Structure** - Generates complete flow definitions
- **Production Ready** - Includes proper async configuration
- **Response Formatting** - Automatically adds response_formatter steps
- **Snake Case Naming** - Follows platform naming conventions

## 🎯 Supported AI Workflows

### 🖼️ Image Classification
Transform images into structured predictions using computer vision models.

**Example Input:**
```bash
cli/generate_flow.sh "Classify product images for e-commerce catalog"
```

**Generated Flow Features:**
- Vision models (ViT, ResNet, EfficientNet)
- HuggingFace or OpenAI Vision providers
- Confidence scoring and top-k predictions
- Structured JSON responses

### 🎤 Speech Transcription
Convert audio files to text with high accuracy and language detection.

**Example Input:**
```bash
cli/generate_flow.sh "Transcribe customer service calls and extract key points"
```

**Generated Flow Features:**
- Whisper models (local or OpenAI API)
- Multiple language support
- Timestamp and confidence data
- Segment-based transcription

### 💭 Sentiment Analysis
Analyze text emotions, sentiment, and aspects with detailed insights.

**Example Input:**
```bash
cli/generate_flow.sh "Analyze customer feedback sentiment and emotions"
```

**Generated Flow Features:**
- HuggingFace specialized models
- Emotion detection and aspect analysis
- Confidence scoring and detailed breakdowns
- Multi-language sentiment support

### 📄 Document Processing
Extract and analyze content from various document formats.

**Example Input:**
```bash
cli/generate_flow.sh "Process invoices with OCR and extract key information"
```

**Generated Flow Features:**
- Multi-format support (PDF, Word, Excel, PowerPoint)
- OCR text extraction with confidence scores
- LLM-powered content analysis
- Structured data extraction

### 🔗 Multi-Modal Pipelines
Combine multiple AI capabilities in sophisticated workflows.

**Example Input:**
```bash
cli/generate_flow.sh "Analyze customer feedback documents with OCR, sentiment analysis, and image classification"
```

**Generated Flow Features:**
- Chained executor dependencies
- Cross-modal data flow
- Comprehensive analysis results
- Unified response formatting

## 📋 Generated Flow Structure

### Complete YAML Template
```yaml
name: "workflow_name_in_snake_case"
version: "1.0.0"
description: "AI-generated workflow description"

inputs:
  - name: "input_parameter"
    type: "file|string|integer|boolean|float"
    required: true|false
    default: "default_value"
    description: "Parameter description"

steps:
  - name: "process_input"
    executor: "appropriate_executor"
    config:
      parameter: "{{ inputs.input_parameter }}"
      provider: "optimal_provider"
      model: "recommended_model"
    depends_on: []

  - name: "format_response"
    executor: "response_formatter"
    config:
      data: "{{ steps.process_input.output }}"
      template: "structured"
      format: "json"
    depends_on: ["process_input"]

outputs:
  - name: "result"
    value: "{{ steps.format_response.formatted_response }}"
    description: "Formatted workflow result"

config:
  execution:
    mode: "async"
    timeout: 300
    max_retries: 3
    queue: "workflow_name_queue"
    worker: "celery-worker-workflow_name"
  
  validation:
    required_inputs: ["input_parameter"]
    
  metadata:
    category: "workflow_category"
    tags: ["relevant", "tags", "here"]
    author: "AI Inference Platform"
```

### Key Components

#### 🏷️ Naming Convention
- **Snake Case Names** - All flow names converted to snake_case automatically
- **Descriptive IDs** - Clear, meaningful step and parameter names
- **Queue Matching** - Queue names match flow names for consistency

#### ⚙️ Configuration Management
- **Async Execution** - Production-ready async processing
- **Timeout Handling** - Appropriate timeouts for different AI tasks
- **Retry Logic** - Built-in retry mechanisms for reliability
- **Worker Assignment** - Dedicated workers per flow type

#### 📤 Response Formatting
- **Mandatory Formatter** - All flows include response_formatter as final step
- **Structured Output** - Consistent JSON response format
- **Callback Ready** - Properly formatted for webhook callbacks
- **Metadata Inclusion** - Rich metadata for debugging and monitoring

## 🛠️ Command Line Interface

### Basic Usage
```bash
cli/generate_flow.sh "<natural language description>"
```

### Advanced Options

#### Interactive Mode
```bash
cli/generate_flow.sh --interactive "Process customer documents"
```
- Asks clarifying questions about inputs, outputs, and requirements
- Refines the description based on your responses
- Generates more accurate and detailed flows

#### Preview Mode
```bash
cli/generate_flow.sh --preview-only "Analyze audio sentiment"
```
- Shows generated flow structure without saving
- Displays inputs, steps, outputs, and configuration
- Perfect for testing and validation

#### Custom Output
```bash
cli/generate_flow.sh "Image classification" --output flows/custom/my_classifier.yaml
```
- Save to specific location
- Organize flows in custom directories
- Maintain project structure

### Example Commands

#### Simple AI Tasks
```bash
# Image processing
cli/generate_flow.sh "Classify product images"

# Audio processing  
cli/generate_flow.sh "Transcribe meeting recordings"

# Text analysis
cli/generate_flow.sh "Analyze customer reviews sentiment"

# Document processing
cli/generate_flow.sh "Extract text from scanned documents"
```

#### Complex Multi-Modal Workflows
```bash
# Document analysis pipeline
cli/generate_flow.sh "Process customer feedback forms with OCR and sentiment analysis"

# Media content analysis
cli/generate_flow.sh "Analyze video thumbnails and transcribe audio content"

# Business intelligence
cli/generate_flow.sh "Extract data from invoices and classify by urgency"
```

## 🎯 Best Practices

### 📝 Writing Effective Descriptions

#### ✅ Good Descriptions
```bash
# Specific and clear
"Classify product images for e-commerce with confidence scores"

# Mentions input/output types
"Transcribe audio files and extract key discussion points"

# Specifies analysis type
"Analyze customer feedback sentiment with emotion detection"
```

#### ❌ Avoid Vague Descriptions
```bash
# Too generic
"Process files"

# Unclear intent
"Do AI stuff with images"

# Missing context
"Analyze text"
```

### 🔧 Flow Optimization

#### Input Parameters
- **Rich Inputs** - Generated flows may have basic inputs; consider adding more parameters manually
- **Default Values** - Add sensible defaults for optional parameters
- **Validation** - Include input validation and constraints

#### Configuration Tuning
- **Timeouts** - Adjust based on expected processing time
- **Concurrency** - Set appropriate worker concurrency limits
- **Retry Logic** - Configure retries based on failure patterns

#### Output Enhancement
- **Multiple Outputs** - Consider adding granular outputs for debugging
- **Metadata** - Include processing metadata and confidence scores
- **Error Handling** - Add error outputs for failure scenarios

## 🧪 Examples Gallery

### Image Classification Flow
**Input:** `"Classify images using computer vision models"`

**Generated Flow:**
```yaml
name: "image_classification"
version: "1.0.0"
description: "Classify images using computer vision models"

inputs:
  - name: "input_image"
    type: "file"
    required: true
    description: "Image file to classify"

steps:
  - name: "handle_file"
    executor: "file_handler"
    config:
      file_content: "{{ inputs.input_image }}"
    depends_on: []

  - name: "classify_image"
    executor: "vision_classifier"
    config:
      file: "{{ steps.handle_file.temp_path }}"
      provider: "huggingface"
      model: "google/vit-base-patch16-224"
      top_k: 5
    depends_on: ["handle_file"]

  - name: "format_response"
    executor: "response_formatter"
    config:
      data: "{{ steps.classify_image.predictions }}"
      template: "structured"
      format: "json"
    depends_on: ["classify_image"]

outputs:
  - name: "classification_result"
    value: "{{ steps.format_response.formatted_response }}"
    description: "Image classification results"

config:
  execution:
    mode: "async"
    timeout: 300
    max_retries: 3
    queue: "image_classification"
    worker: "celery-worker-image_classification"
  
  validation:
    required_inputs: ["input_image"]
    
  metadata:
    category: "computer_vision"
    tags: ["image", "classification", "vision"]
    author: "AI Inference Platform"
```

### Audio Sentiment Analysis Flow
**Input:** `"Transcribe audio and analyze sentiment"`

**Generated Flow:**
```yaml
name: "audio_sentiment_analysis"
version: "1.0.0"
description: "Transcribe audio files and analyze sentiment"

inputs:
  - name: "audio_file"
    type: "file"
    required: true
    description: "Audio file to transcribe and analyze"

steps:
  - name: "transcribe_audio"
    executor: "whisper_processor"
    config:
      file: "{{ inputs.audio_file }}"
      provider: "local"
      model_size: "base"
      language: "en"
    depends_on: []

  - name: "analyze_sentiment"
    executor: "sentiment_analyzer"
    config:
      text: "{{ steps.transcribe_audio.transcription }}"
      provider: "huggingface"
      analysis_type: "comprehensive"
    depends_on: ["transcribe_audio"]

  - name: "format_response"
    executor: "response_formatter"
    config:
      data: "{{ steps.analyze_sentiment.output }}"
      template: "structured"
      format: "json"
    depends_on: ["analyze_sentiment"]

outputs:
  - name: "analysis_result"
    value: "{{ steps.format_response.formatted_response }}"
    description: "Audio transcription and sentiment analysis"

config:
  execution:
    mode: "async"
    timeout: 300
    max_retries: 3
    queue: "audio_sentiment_analysis"
    worker: "celery-worker-audio_sentiment_analysis"
  
  validation:
    required_inputs: ["audio_file"]
    
  metadata:
    category: "audio_processing"
    tags: ["audio", "transcription", "sentiment", "whisper"]
    author: "AI Inference Platform"
```

### Document Processing Pipeline
**Input:** `"Process customer feedback documents with OCR and sentiment analysis"`

**Generated Flow:**
```yaml
name: "customer_feedback_analysis"
version: "1.0.0"
description: "Process customer feedback documents with OCR and sentiment analysis"

inputs:
  - name: "feedback_document"
    type: "file"
    required: true
    description: "Customer feedback document to process"

steps:
  - name: "handle_document"
    executor: "file_handler"
    config:
      file_content: "{{ inputs.feedback_document }}"
    depends_on: []

  - name: "extract_content"
    executor: "document_extractor"
    config:
      file_path: "{{ steps.handle_document.temp_path }}"
      extract_images: true
    depends_on: ["handle_document"]

  - name: "ocr_processing"
    executor: "ocr_processor"
    config:
      image_path: "{{ steps.extract_content.images }}"
      provider: "tesseract"
      languages: ["en"]
    depends_on: ["extract_content"]

  - name: "analyze_sentiment"
    executor: "sentiment_analyzer"
    config:
      text: "{{ steps.ocr_processing.text }}"
      provider: "huggingface"
      analysis_type: "comprehensive"
    depends_on: ["ocr_processing"]

  - name: "format_response"
    executor: "response_formatter"
    config:
      data: "{{ steps.analyze_sentiment.output }}"
      template: "detailed"
      format: "json"
    depends_on: ["analyze_sentiment"]

outputs:
  - name: "feedback_analysis"
    value: "{{ steps.format_response.formatted_response }}"
    description: "Complete customer feedback analysis"

config:
  execution:
    mode: "async"
    timeout: 600
    max_retries: 3
    queue: "customer_feedback_analysis"
    worker: "celery-worker-customer_feedback_analysis"
  
  validation:
    required_inputs: ["feedback_document"]
    
  metadata:
    category: "document_processing"
    tags: ["document", "ocr", "sentiment", "feedback"]
    author: "AI Inference Platform"
```

## 🔧 Customization & Enhancement

### Manual Flow Refinement

After generating a flow, you may want to enhance it manually:

#### Adding Rich Input Parameters
```yaml
inputs:
  - name: "file"
    type: "file"
    required: true
    description: "Input file to process"
  
  # Add provider selection
  - name: "provider"
    type: "string"
    required: false
    default: "huggingface"
    enum: ["huggingface", "openai"]
    description: "AI provider to use"
  
  # Add model selection
  - name: "model"
    type: "string"
    required: false
    default: "google/vit-base-patch16-224"
    description: "Model to use for processing"
  
  # Add confidence threshold
  - name: "confidence_threshold"
    type: "float"
    required: false
    default: 0.5
    description: "Minimum confidence threshold"
```

#### Enhanced Configuration
```yaml
config:
  execution:
    mode: "async"
    timeout: 300
    max_retries: 3
    queue: "flow_name"
    worker: "celery-worker-flow_name"
    max_concurrent: 2  # Add concurrency control
  
  # Add callback configuration
  callbacks:
    enabled: true
    max_retries: 3
    retry_delay: 60
  
  # Add cleanup settings
  cleanup_temp_files: true
  
  validation:
    required_inputs: ["file"]
    # Add input validation rules
    file_size_limit: 10485760  # 10MB
    allowed_formats: [".jpg", ".png", ".pdf"]
  
  metadata:
    category: "computer_vision"
    tags: ["image", "classification"]
    author: "AI Inference Platform"
    version: "1.0.0"
    created_by: "flow_generator"
```

#### Multiple Outputs
```yaml
outputs:
  - name: "primary_result"
    value: "{{ steps.format_response.formatted_response }}"
    description: "Main processing result"
  
  # Add granular outputs
  - name: "confidence_score"
    value: "{{ steps.process_data.confidence }}"
    description: "Processing confidence score"
  
  - name: "processing_time"
    value: "{{ steps.process_data.processing_time }}"
    description: "Time taken for processing"
  
  - name: "model_used"
    value: "{{ inputs.model }}"
    description: "Model used for processing"
```

## 🚨 Limitations & Known Issues

### Current Limitations

#### ⚠️ BETA Status
- **Active Development** - Features and behavior may change
- **Limited Testing** - Not all edge cases have been thoroughly tested
- **Manual Refinement** - Complex workflows may require manual adjustments

#### 🔧 Technical Constraints
- **Basic Input Generation** - Generated flows have simple input parameters
- **Generic Configuration** - May need manual tuning for specific use cases
- **Limited Error Handling** - Basic error handling in generated flows

#### 🎯 Accuracy Considerations
- **LLM Dependency** - Quality depends on underlying language model
- **Context Understanding** - May misinterpret complex or ambiguous descriptions
- **Executor Knowledge** - Limited to available executors in the platform

### Troubleshooting

#### Common Issues

**Flow Generation Fails**
```bash
# Check LLM connectivity
curl http://localhost:11434/api/tags

# Verify flow generator script permissions
chmod +x cli/generate_flow.sh

# Check Python environment
source venv/bin/activate
python cli/flow_generator.py --help
```

**Generated Flow Validation Errors**
- **Snake Case Names** - Flow names automatically converted to snake_case
- **Missing Executors** - Only uses available executors in the platform
- **Config Structure** - Validates required config sections

**Flow Execution Issues**
- **Parameter Mapping** - Check if generated parameter names match executor expectations
- **Dependencies** - Verify step dependencies are correctly chained
- **Output References** - Ensure outputs reference correct step outputs

## 🔮 Future Enhancements

### Planned Features

#### 🎯 Enhanced Generation
- **Rich Input Parameters** - Generate flows with comprehensive input schemas
- **Advanced Configuration** - More sophisticated config generation
- **Error Handling** - Built-in error handling and recovery steps

#### 🧠 Improved Intelligence
- **Context Learning** - Learn from existing flows to improve generation
- **Template Library** - Pre-built templates for common patterns
- **Validation Enhancement** - More comprehensive flow validation

#### 🔧 Developer Experience
- **Visual Flow Builder** - GUI for flow generation and editing
- **Flow Testing** - Built-in testing and validation tools
- **Performance Optimization** - Automatic performance tuning suggestions

### Contributing

We welcome contributions to improve the Flow Generator:

#### 🐛 Bug Reports
- Report issues with flow generation accuracy
- Document edge cases and limitations
- Suggest improvements to generated flows

#### 💡 Feature Requests
- New executor support
- Enhanced configuration options
- Improved natural language understanding

#### 🔧 Code Contributions
- Improve LLM prompts and templates
- Add new flow patterns and examples
- Enhance validation and error handling

## 📚 Related Documentation

- **[YAML Configuration Guide](yaml-configuration.md)** - Manual flow creation
- **[AI Capabilities Overview](../ai-capabilities/)** - Available AI executors
- **[Architecture Guide](../architecture.md)** - Platform architecture
- **[API Documentation](../../main.py)** - REST API reference

---

> **⚠️ BETA Notice:** The AI Flow Generator is in active development. Generated flows may require manual refinement for complex use cases. We're continuously improving the LLM prompts, validation logic, and generation accuracy. Please report any issues or suggestions to help us improve this feature.
