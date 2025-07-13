# Image Classification

Advanced image classification capabilities with dual provider support, multiple model architectures, and high-accuracy object recognition for comprehensive visual understanding.

## Overview

The Image Classification feature provides sophisticated image recognition using state-of-the-art computer vision models. It supports multiple providers, model architectures, and delivers detailed predictions with confidence scores for accurate visual content analysis.

## Features

- **Dual Provider Architecture**: HuggingFace models and OpenAI Vision API
- **Multiple Model Architectures**: Vision Transformer, ResNet, EfficientNet, ConvNeXT
- **High Accuracy**: 95-99% accuracy on ImageNet classes
- **Fast Processing**: 0.1-0.8 seconds per classification
- **Flexible Input**: File uploads, base64 data, file paths
- **Confidence Scoring**: Detailed confidence metrics for all predictions
- **Model Caching**: Improved performance for repeated classifications

## Provider Options

### HuggingFace Provider (Recommended for Speed)
- **Best for**: High-volume processing, real-time classification, offline deployment
- **Models**: Multiple pre-trained architectures (ViT, ResNet, EfficientNet, ConvNeXT)
- **Performance**: 0.1-0.8 seconds per classification
- **Accuracy**: 95-99% on standard ImageNet classes
- **Device Support**: Auto-detection (CPU/CUDA)
- **Cost**: Free, local processing

### OpenAI Provider (Recommended for Detailed Analysis)
- **Best for**: Complex scenes, custom descriptions, natural language output
- **Models**: GPT-4 Vision, GPT-4 Omni models
- **Performance**: 2-5 seconds per classification
- **Accuracy**: Context-aware, handles complex scenes
- **Features**: Natural language descriptions, custom prompts
- **Flexibility**: Custom classification prompts and detailed analysis

## Supported Models

### HuggingFace Models

#### Vision Transformer (Recommended)
```yaml
# Best balance of accuracy and speed
hf_model_name: "google/vit-base-patch16-224"     # 86M params, fast
hf_model_name: "google/vit-large-patch16-224"    # 307M params, high accuracy
```

#### ResNet Models
```yaml
# Reliable and well-tested
hf_model_name: "microsoft/resnet-50"            # 25M params, reliable
hf_model_name: "microsoft/resnet-101"           # 44M params, higher accuracy
```

#### EfficientNet Models
```yaml
# Efficient and lightweight
hf_model_name: "google/efficientnet-b0"         # 5M params, very fast
hf_model_name: "google/efficientnet-b4"         # 19M params, balanced
hf_model_name: "google/efficientnet-b7"         # 66M params, high accuracy
```

#### ConvNeXT Models
```yaml
# Modern architecture
hf_model_name: "facebook/convnext-tiny-224"     # 28M params, modern
hf_model_name: "facebook/convnext-base-224"     # 89M params, high performance
```

### OpenAI Vision Models
```yaml
# Latest models
openai_model: "gpt-4-vision-preview"    # High accuracy, detailed analysis
openai_model: "gpt-4o"                  # Latest model, best performance
openai_model: "gpt-4o-mini"             # Fast and cost-effective
```

## Configuration Options

### Basic Configuration
```yaml
inputs:
  - name: "file"              # Image file (required)
  - name: "provider"          # "huggingface" or "openai"
  - name: "top_k"             # Number of predictions
  - name: "confidence_threshold" # Minimum confidence
```

### Advanced Configuration
```yaml
inputs:
  - name: "hf_model_name"     # HuggingFace model selection
  - name: "openai_model"      # OpenAI model selection
  - name: "device"            # Device for HF models
  - name: "classification_prompt" # Custom prompt (OpenAI)
  - name: "image_preprocessing" # Image preprocessing options
```

## Usage Examples

### Quick Image Classification (HuggingFace)
```bash
curl -X POST "http://localhost:8000/api/v1/image-classification/execute" \
  -F "file=@cat.jpg" \
  -F "provider=huggingface" \
  -F "top_k=3" \
  -F "hf_model_name=google/vit-base-patch16-224"
```

**Response:**
```json
{
  "top_prediction": {
    "label": "tiger cat",
    "score": 0.892
  },
  "predictions": [
    {"label": "tiger cat", "score": 0.892},
    {"label": "tabby cat", "score": 0.087},
    {"label": "Egyptian cat", "score": 0.021}
  ],
  "processing_time_seconds": 0.097,
  "provider": "huggingface",
  "model_used": "google/vit-base-patch16-224"
}
```

### Detailed Analysis (OpenAI Vision)
```bash
curl -X POST "http://localhost:8000/api/v1/image-classification/execute" \
  -F "file=@scene.jpg" \
  -F "provider=openai" \
  -F "openai_model=gpt-4-vision-preview" \
  -F "classification_prompt=Describe this image in detail and identify all objects"
```

**Response:**
```json
{
  "description": "A orange tabby cat sitting on a wooden table near a window. The cat appears relaxed and is looking directly at the camera. The lighting is natural, coming from the window, and the background shows a cozy indoor setting.",
  "classifications": [
    "domestic cat",
    "furniture",
    "indoor scene",
    "pet",
    "tabby cat"
  ],
  "confidence": 0.95,
  "detailed_analysis": {
    "objects": ["cat", "table", "window"],
    "scene_type": "indoor",
    "lighting": "natural",
    "mood": "peaceful"
  },
  "processing_time_seconds": 3.2,
  "provider": "openai",
  "model_used": "gpt-4-vision-preview"
}
```

### High-Accuracy Classification
```bash
curl -X POST "http://localhost:8000/api/v1/image-classification/execute" \
  -F "file=@wildlife.jpg" \
  -F "provider=huggingface" \
  -F "hf_model_name=google/vit-large-patch16-224" \
  -F "top_k=5" \
  -F "confidence_threshold=0.1"
```

### Custom Classification Prompt (OpenAI)
```bash
curl -X POST "http://localhost:8000/api/v1/image-classification/execute" \
  -F "file=@product.jpg" \
  -F "provider=openai" \
  -F "classification_prompt=Analyze this product image for e-commerce: identify the product type, color, condition, and any notable features"
```

## Response Format

### HuggingFace Response
```json
{
  "success": true,
  "outputs": {
    "top_prediction": {
      "label": "golden retriever",
      "score": 0.943
    },
    "predictions": [
      {"label": "golden retriever", "score": 0.943},
      {"label": "Nova Scotia duck tolling retriever", "score": 0.032},
      {"label": "kuvasz", "score": 0.015},
      {"label": "Great Pyrenees", "score": 0.008},
      {"label": "Labrador retriever", "score": 0.002}
    ],
    "total_predictions": 1000,
    "filtered_predictions": 5,
    "confidence_threshold": 0.1,
    "processing_time_seconds": 0.156,
    "provider": "huggingface",
    "model_used": "google/vit-base-patch16-224",
    "device": "cpu",
    "image_info": {
      "format": "JPEG",
      "size": [640, 480],
      "mode": "RGB"
    }
  }
}
```

### OpenAI Vision Response
```json
{
  "success": true,
  "outputs": {
    "description": "This image shows a golden retriever dog sitting in a grassy field during what appears to be golden hour. The dog has a happy expression with its tongue out and appears to be well-groomed with a shiny, golden coat.",
    "classifications": [
      "golden retriever",
      "dog",
      "pet",
      "outdoor scene",
      "grass field",
      "natural lighting"
    ],
    "confidence": 0.98,
    "detailed_analysis": {
      "breed": "Golden Retriever",
      "age_estimate": "adult",
      "condition": "healthy and well-groomed",
      "setting": "outdoor grass field",
      "lighting": "golden hour/sunset",
      "mood": "happy and relaxed"
    },
    "objects_detected": [
      {"object": "dog", "confidence": 0.99},
      {"object": "grass", "confidence": 0.95},
      {"object": "field", "confidence": 0.90}
    ],
    "processing_time_seconds": 2.8,
    "provider": "openai",
    "model_used": "gpt-4-vision-preview"
  }
}
```

## Performance Comparison

| Provider | Speed | Accuracy | Features | Cost | Use Case |
|----------|-------|----------|----------|------|----------|
| **HuggingFace** | ⚡ 0.1-0.8s | 📊 95-99% | Standard classification | 💰 Free | High-volume, real-time |
| **OpenAI** | 🐌 2-5s | 🧠 Context-aware | Detailed analysis, custom prompts | 💳 Pay-per-use | Complex scenes, descriptions |

## Model Performance Comparison

| Model | Speed | Accuracy | Parameters | Use Case |
|-------|-------|----------|------------|----------|
| **EfficientNet-B0** | ⚡⚡⚡ | 📊📊📊 | 5M | Ultra-fast processing |
| **ResNet-50** | ⚡⚡ | 📊📊📊📊 | 25M | Reliable, proven |
| **ViT-Base** | ⚡⚡ | 📊📊📊📊📊 | 86M | Best balance (recommended) |
| **ViT-Large** | ⚡ | 📊📊📊📊📊 | 307M | Highest accuracy |

## Image Format Support

### Supported Formats
- **JPEG/JPG** - Most common, good compression
- **PNG** - Lossless, supports transparency
- **WebP** - Modern format, excellent compression
- **TIFF** - High quality, uncompressed
- **BMP** - Basic bitmap format

### Image Quality Recommendations
- **Resolution**: 224x224 minimum, higher for better accuracy
- **Quality**: High quality images for best results
- **Lighting**: Good lighting and contrast
- **Focus**: Sharp, well-focused images
- **Size**: Up to 20MB file size supported

## Best Practices

### Image Quality
1. **High Resolution**: Use images with good resolution (224x224+)
2. **Good Lighting**: Ensure proper lighting and contrast
3. **Sharp Focus**: Avoid blurry or out-of-focus images
4. **Proper Framing**: Center the main subject in the image

### Model Selection
1. **Speed Priority**: Use EfficientNet-B0 or ResNet-50
2. **Accuracy Priority**: Use ViT-Large or ConvNeXT-Base
3. **Balanced**: Use ViT-Base (recommended for most cases)
4. **Custom Needs**: Use OpenAI for detailed analysis

### Performance Optimization
1. **Model Caching**: Keep models loaded for repeated use
2. **GPU Acceleration**: Use CUDA for faster processing
3. **Batch Processing**: Process multiple images efficiently
4. **Image Preprocessing**: Optimize images before classification

### Provider Selection
1. **HuggingFace**: High-volume, cost-sensitive applications
2. **OpenAI**: Complex analysis, custom descriptions needed
3. **Hybrid Approach**: Use both for validation and comparison

## Common Use Cases

### E-commerce
- **Product Classification**: Automatically categorize product images
- **Quality Control**: Assess product condition and quality
- **Inventory Management**: Organize products by visual characteristics
- **Search Enhancement**: Enable visual search capabilities

### Content Moderation
- **Image Filtering**: Identify inappropriate or harmful content
- **Brand Safety**: Ensure images align with brand guidelines
- **Automated Tagging**: Generate tags for content organization
- **Compliance**: Meet regulatory requirements for content

### Healthcare & Medical
- **Medical Imaging**: Assist in medical image analysis (with proper validation)
- **Diagnostic Support**: Support healthcare professionals with image insights
- **Research**: Analyze medical research images and data
- **Documentation**: Classify and organize medical imagery

### Security & Surveillance
- **Object Detection**: Identify objects in security footage
- **Access Control**: Verify identity through image analysis
- **Threat Detection**: Identify potential security threats
- **Monitoring**: Automated surveillance and alerting

### Research & Education
- **Scientific Classification**: Classify specimens and research images
- **Educational Tools**: Create interactive learning experiences
- **Data Analysis**: Process large datasets of images
- **Documentation**: Organize and categorize research materials

## Troubleshooting

### Low Accuracy Issues
- **Check Image Quality**: Ensure high resolution and good lighting
- **Verify Model**: Try different models for your specific use case
- **Adjust Confidence**: Lower threshold for difficult images
- **Preprocessing**: Clean up and optimize images

### Performance Issues
- **Model Selection**: Use smaller models for speed
- **GPU Acceleration**: Enable CUDA for faster processing
- **Image Size**: Optimize image dimensions
- **Batch Processing**: Process multiple images efficiently

### Common Errors
- **File Format**: Ensure image format is supported
- **File Size**: Check file size limits (20MB max)
- **Model Loading**: Verify model is properly cached
- **Memory Issues**: Monitor resource usage for large images

## Integration Examples

### Product Classification Flow
```yaml
name: "product_classification"
description: "Classify product images for e-commerce"

inputs:
  - name: "file"
    type: "file"
    required: true

steps:
  - name: "classify_product"
    executor: "vision_classifier"
    config:
      image: "{{ inputs.file }}"
      provider: "huggingface"
      hf_model_name: "google/vit-base-patch16-224"
      top_k: 5
      confidence_threshold: 0.1
      
  - name: "format_results"
    executor: "response_formatter"
    config:
      template: "product_classification"
      data: "{{ steps.classify_product }}"
```

### Content Moderation Flow
```yaml
name: "image_content_moderation"
description: "Analyze images for content moderation"

inputs:
  - name: "file"
    type: "file"
    required: true

steps:
  - name: "analyze_content"
    executor: "vision_classifier"
    config:
      image: "{{ inputs.file }}"
      provider: "openai"
      openai_model: "gpt-4-vision-preview"
      classification_prompt: "Analyze this image for content moderation. Identify any inappropriate content, violence, or policy violations."
```

### Multi-Model Validation
```yaml
name: "multi_model_classification"
description: "Use multiple models for validation"

inputs:
  - name: "file"
    type: "file"
    required: true

steps:
  - name: "hf_classification"
    executor: "vision_classifier"
    config:
      image: "{{ inputs.file }}"
      provider: "huggingface"
      hf_model_name: "google/vit-base-patch16-224"
      
  - name: "openai_classification"
    executor: "vision_classifier"
    config:
      image: "{{ inputs.file }}"
      provider: "openai"
      openai_model: "gpt-4-vision-preview"
      
  - name: "compare_results"
    executor: "data_combiner"
    config:
      hf_results: "{{ steps.hf_classification }}"
      openai_results: "{{ steps.openai_classification }}"
```

## API Reference

### Endpoints
- **POST** `/api/v1/image-classification/execute` - Classify image
- **GET** `/api/v1/image-classification/status/{flow_id}` - Check processing status
- **GET** `/api/v1/image-classification/info` - Get flow information

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | required | Image file to classify |
| `provider` | string | "huggingface" | Classification provider |
| `top_k` | integer | 5 | Number of predictions |
| `confidence_threshold` | float | 0.1 | Minimum confidence |
| `hf_model_name` | string | "google/vit-base-patch16-224" | HuggingFace model |
| `openai_model` | string | "gpt-4-vision-preview" | OpenAI model |
| `device` | string | "auto" | Device for HF models |

### Response Codes
- **200**: Success - Classification completed
- **400**: Bad Request - Invalid parameters or file
- **422**: Unprocessable Entity - Classification failed
- **500**: Internal Server Error - System error

## Advanced Features

### Custom Classification Prompts (OpenAI)
Tailor analysis for specific domains:

```yaml
classification_prompt: "Analyze this medical X-ray image and identify any abnormalities, focusing on bone structure and potential fractures."
```

### Image Preprocessing
Configure image preprocessing for optimal results:

```yaml
preprocessing:
  - resize: { width: 224, height: 224 }
  - normalize: { mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] }
  - enhance_contrast: { factor: 1.1 }
```

### Batch Processing
Process multiple images efficiently:

```yaml
batch_config:
  - images: ["image1.jpg", "image2.jpg", "image3.jpg"]
  - parallel_processing: true
  - aggregate_results: true
```

### Confidence Filtering
Advanced confidence-based filtering:

```yaml
confidence_config:
  - threshold: 0.8
  - fallback_threshold: 0.5
  - max_predictions: 10
  - filter_similar: true
```
