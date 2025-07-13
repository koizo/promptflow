# Sentiment Analysis

Advanced sentiment analysis capabilities with dual provider support, emotion detection, and aspect-based analysis for comprehensive text understanding.

## Overview

The Sentiment Analysis feature provides sophisticated text sentiment classification using state-of-the-art models. It supports multiple providers, analysis types, and delivers detailed insights including emotions, confidence scores, and aspect-based sentiment analysis.

## Features

- **Dual Provider Architecture**: HuggingFace models and LLM-based analysis
- **Multiple Analysis Types**: Basic, detailed, comprehensive, and emotion-focused
- **High Accuracy**: 95-99% confidence on clear sentiment cases
- **Emotion Detection**: Identify specific emotions beyond basic sentiment
- **Aspect-Based Analysis**: Analyze sentiment for different aspects of text
- **Real-Time Processing**: Fast inference with model caching
- **Confidence Scoring**: Detailed confidence metrics for all predictions

## Provider Options

### HuggingFace Provider (Recommended for Speed)
- **Best for**: High-volume processing, real-time analysis, cost efficiency
- **Models**: Specialized sentiment models (RoBERTa, BERT-based)
- **Performance**: 0.02-0.05 seconds per analysis
- **Accuracy**: 95-99% confidence on clear sentiment cases
- **Device Support**: Auto-detection (CPU/CUDA)
- **Cost**: Free, local processing

### LLM Provider (Recommended for Nuanced Analysis)
- **Best for**: Complex text, detailed analysis, custom prompts
- **Models**: Reuses existing LLM infrastructure (Ollama, OpenAI)
- **Performance**: 2-5 seconds per analysis
- **Accuracy**: Context-aware, handles nuanced sentiment
- **Features**: Aspect-based analysis, emotion detection, reasoning
- **Flexibility**: Custom prompts and analysis types

## Analysis Types

### Basic Analysis
Simple sentiment classification with confidence scores.

```yaml
analysis_type: "basic"
```

**Output:**
- Sentiment: positive/negative/neutral
- Confidence score (0.0-1.0)
- Processing time

### Detailed Analysis
Enhanced analysis with emotions and key phrases.

```yaml
analysis_type: "detailed"
```

**Output:**
- Basic sentiment + confidence
- Primary emotions detected
- Key phrases with sentiment
- Reasoning for classification

### Comprehensive Analysis
Complete analysis with aspect-based sentiment and insights.

```yaml
analysis_type: "comprehensive"
```

**Output:**
- All detailed analysis features
- Aspect-based sentiment breakdown
- Emotion intensity scores
- Actionable insights and recommendations

### Emotion-Focused Analysis
Specialized emotion detection and classification.

```yaml
analysis_type: "emotions"
```

**Output:**
- Detailed emotion classification
- Emotion intensity scores
- Emotional context analysis
- Mood indicators

## Supported Models

### HuggingFace Models
```yaml
# Default high-accuracy model (recommended)
hf_model_name: "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Alternative models
hf_model_name: "nlptown/bert-base-multilingual-uncased-sentiment"
hf_model_name: "cardiffnlp/twitter-roberta-base-emotion"
hf_model_name: "j-hartmann/emotion-english-distilroberta-base"
```

### LLM Models
```yaml
# Local models (via Ollama)
llm_model: "mistral"
llm_model: "llama2"
llm_model: "codellama"

# OpenAI models
llm_model: "gpt-3.5-turbo"
llm_model: "gpt-4"
```

## Configuration Options

### Basic Configuration
```yaml
inputs:
  - name: "text"              # Text to analyze (required)
  - name: "provider"          # "huggingface" or "llm"
  - name: "analysis_type"     # Analysis depth level
```

### Advanced Configuration
```yaml
inputs:
  - name: "hf_model_name"     # HuggingFace model selection
  - name: "llm_model"         # LLM model selection
  - name: "device"            # Device for HF models
  - name: "custom_prompt"     # Custom analysis prompt (LLM)
  - name: "aspects"           # Specific aspects to analyze
  - name: "language"          # Text language (for multilingual models)
```

## Usage Examples

### Basic Sentiment Analysis (HuggingFace)
```bash
curl -X POST "http://localhost:8000/api/v1/sentiment-analysis/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I love this product! It works perfectly.",
    "provider": "huggingface",
    "analysis_type": "basic"
  }'
```

**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 0.988,
  "all_scores": {
    "negative": 0.005,
    "neutral": 0.007,
    "positive": 0.988
  },
  "processing_time_seconds": 0.03,
  "provider": "huggingface",
  "model_used": "cardiffnlp/twitter-roberta-base-sentiment-latest"
}
```

### Comprehensive Analysis (LLM)
```bash
curl -X POST "http://localhost:8000/api/v1/sentiment-analysis/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The product quality is excellent but shipping was slow and expensive.",
    "provider": "llm",
    "analysis_type": "comprehensive",
    "llm_model": "mistral"
  }'
```

**Response:**
```json
{
  "sentiment": "mixed",
  "confidence": 0.78,
  "emotions": ["satisfaction", "frustration"],
  "aspects": [
    {
      "aspect": "product quality",
      "sentiment": "positive",
      "confidence": 0.9,
      "phrases": ["excellent"]
    },
    {
      "aspect": "shipping",
      "sentiment": "negative",
      "confidence": 0.8,
      "phrases": ["slow", "expensive"]
    }
  ],
  "key_phrases": [
    {"phrase": "excellent quality", "sentiment": "positive"},
    {"phrase": "slow and expensive", "sentiment": "negative"}
  ],
  "insights": "Customer appreciates product quality but frustrated with shipping experience",
  "recommendations": ["Improve shipping speed", "Consider shipping cost optimization"],
  "processing_time_seconds": 3.2,
  "provider": "llm",
  "model_used": "mistral"
}
```

### Emotion Detection
```bash
curl -X POST "http://localhost:8000/api/v1/sentiment-analysis/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I am so excited about the upcoming vacation!",
    "provider": "huggingface",
    "analysis_type": "emotions",
    "hf_model_name": "j-hartmann/emotion-english-distilroberta-base"
  }'
```

**Response:**
```json
{
  "primary_emotion": "joy",
  "emotion_scores": {
    "joy": 0.92,
    "excitement": 0.85,
    "anticipation": 0.78,
    "surprise": 0.12,
    "fear": 0.03,
    "anger": 0.02,
    "sadness": 0.01
  },
  "sentiment": "positive",
  "confidence": 0.94,
  "emotional_intensity": "high",
  "processing_time_seconds": 0.04
}
```

### Multi-Aspect Analysis
```bash
curl -X POST "http://localhost:8000/api/v1/sentiment-analysis/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The hotel room was clean and comfortable, but the service was terrible and the food was mediocre.",
    "provider": "llm",
    "analysis_type": "comprehensive",
    "aspects": ["room", "service", "food"]
  }'
```

## Response Format

### Basic Response (HuggingFace)
```json
{
  "success": true,
  "outputs": {
    "sentiment": "positive",
    "confidence": 0.95,
    "all_scores": {
      "negative": 0.02,
      "neutral": 0.03,
      "positive": 0.95
    },
    "processing_time_seconds": 0.03,
    "provider": "huggingface",
    "model_used": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "text_length": 45,
    "analysis_type": "basic"
  }
}
```

### Comprehensive Response (LLM)
```json
{
  "success": true,
  "outputs": {
    "sentiment": "mixed",
    "confidence": 0.82,
    "primary_emotion": "satisfaction",
    "emotions": ["satisfaction", "disappointment"],
    "emotion_scores": {
      "satisfaction": 0.7,
      "disappointment": 0.6,
      "neutral": 0.3
    },
    "aspects": [
      {
        "aspect": "product",
        "sentiment": "positive",
        "confidence": 0.9,
        "key_phrases": ["great quality", "works well"]
      },
      {
        "aspect": "delivery",
        "sentiment": "negative",
        "confidence": 0.8,
        "key_phrases": ["late delivery", "poor packaging"]
      }
    ],
    "key_phrases": [
      {"phrase": "great quality", "sentiment": "positive", "confidence": 0.9},
      {"phrase": "late delivery", "sentiment": "negative", "confidence": 0.8}
    ],
    "insights": "Customer satisfied with product but disappointed with delivery service",
    "recommendations": [
      "Maintain product quality standards",
      "Improve delivery reliability and packaging"
    ],
    "reasoning": "Mixed sentiment due to positive product experience offset by negative delivery experience",
    "processing_time_seconds": 4.1,
    "provider": "llm",
    "model_used": "mistral",
    "analysis_type": "comprehensive"
  }
}
```

## Performance Comparison

| Provider | Speed | Accuracy | Features | Cost | Use Case |
|----------|-------|----------|----------|------|----------|
| **HuggingFace** | ⚡ 0.02-0.05s | 📊 95-99% | Basic-Detailed | 💰 Free | High-volume, real-time |
| **LLM** | 🐌 2-5s | 🧠 Context-aware | All features | 💳 Variable | Complex analysis, insights |

## Language Support

### HuggingFace Models
- **English**: Highest accuracy with specialized models
- **Multilingual**: Support for 100+ languages with multilingual models
- **Domain-Specific**: Twitter, reviews, news-optimized models

### LLM Models
- **Multi-Language**: Native support for major languages
- **Context-Aware**: Better handling of cultural nuances
- **Custom Prompts**: Adaptable to specific languages and domains

## Best Practices

### Text Preprocessing
1. **Clean Text**: Remove unnecessary formatting and noise
2. **Length Optimization**: Ideal text length 10-500 words
3. **Context Preservation**: Maintain important context clues
4. **Language Consistency**: Use consistent language throughout

### Provider Selection
1. **HuggingFace for Speed**: High-volume, real-time applications
2. **LLM for Depth**: Complex analysis, nuanced understanding
3. **Hybrid Approach**: Use both for validation and comparison

### Model Selection
1. **Domain Matching**: Choose models trained on similar data
2. **Language Specific**: Use language-specific models when available
3. **Performance Testing**: Validate accuracy on your specific use case

### Analysis Type Selection
1. **Basic**: Simple classification needs
2. **Detailed**: When emotions and key phrases are important
3. **Comprehensive**: Full business intelligence and insights
4. **Emotions**: Specialized emotion detection requirements

## Common Use Cases

### Customer Feedback Analysis
- **Product Reviews**: Analyze customer satisfaction and issues
- **Support Tickets**: Prioritize urgent or negative feedback
- **Survey Responses**: Extract insights from open-ended responses
- **Social Media**: Monitor brand sentiment across platforms

### Content Moderation
- **Comment Filtering**: Identify negative or toxic content
- **Content Scoring**: Rate content based on sentiment
- **Community Management**: Monitor discussion sentiment
- **Brand Safety**: Ensure content aligns with brand values

### Business Intelligence
- **Market Research**: Analyze consumer sentiment trends
- **Competitive Analysis**: Monitor competitor sentiment
- **Campaign Effectiveness**: Measure marketing campaign reception
- **Risk Assessment**: Identify potential PR issues early

### Healthcare & Wellness
- **Patient Feedback**: Analyze patient satisfaction and concerns
- **Mental Health**: Monitor emotional well-being indicators
- **Treatment Effectiveness**: Assess patient response to treatments
- **Support Services**: Prioritize urgent mental health cases

## Troubleshooting

### Low Accuracy Issues
- **Check Text Quality**: Ensure clear, well-formed text
- **Verify Language**: Confirm model supports text language
- **Adjust Analysis Type**: Try different analysis depths
- **Model Selection**: Test different models for your use case

### Performance Issues
- **Provider Choice**: Use HuggingFace for speed, LLM for accuracy
- **Batch Processing**: Process multiple texts efficiently
- **Model Caching**: Keep models loaded for repeated use
- **Device Optimization**: Use GPU acceleration when available

### Common Errors
- **Text Length**: Ensure text is within supported length limits
- **Model Loading**: Verify models are properly downloaded/cached
- **API Limits**: Check rate limits for LLM providers
- **Memory Issues**: Monitor resource usage for large batches

## Integration Examples

### Customer Feedback Analysis Flow
```yaml
name: "customer_feedback_analysis"
description: "Analyze customer feedback with detailed insights"

inputs:
  - name: "feedback_text"
    type: "string"
    required: true

steps:
  - name: "analyze_sentiment"
    executor: "sentiment_analyzer"
    config:
      text: "{{ inputs.feedback_text }}"
      provider: "llm"
      analysis_type: "comprehensive"
      aspects: ["product", "service", "delivery", "value"]
      
  - name: "format_insights"
    executor: "response_formatter"
    config:
      template: "feedback_report"
      data: "{{ steps.analyze_sentiment }}"
```

### Social Media Monitoring
```yaml
name: "social_media_sentiment"
description: "Monitor social media sentiment in real-time"

inputs:
  - name: "posts"
    type: "array"
    required: true

steps:
  - name: "batch_sentiment_analysis"
    executor: "sentiment_analyzer"
    config:
      texts: "{{ inputs.posts }}"
      provider: "huggingface"
      analysis_type: "detailed"
      batch_processing: true
```

## API Reference

### Endpoints
- **POST** `/api/v1/sentiment-analysis/execute` - Analyze text sentiment
- **GET** `/api/v1/sentiment-analysis/status/{flow_id}` - Check processing status
- **GET** `/api/v1/sentiment-analysis/info` - Get flow information

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | Text to analyze |
| `provider` | string | "huggingface" | Analysis provider |
| `analysis_type` | string | "basic" | Analysis depth |
| `hf_model_name` | string | auto | HuggingFace model |
| `llm_model` | string | "mistral" | LLM model |
| `aspects` | array | null | Specific aspects to analyze |

### Response Codes
- **200**: Success - Analysis completed
- **400**: Bad Request - Invalid parameters
- **422**: Unprocessable Entity - Analysis failed
- **500**: Internal Server Error - System error

## Advanced Features

### Custom Prompts (LLM)
Tailor analysis for specific domains:

```yaml
custom_prompt: "Analyze this medical patient feedback focusing on treatment satisfaction, side effects, and overall care quality."
```

### Batch Processing
Process multiple texts efficiently:

```yaml
batch_config:
  - texts: ["text1", "text2", "text3"]
  - parallel_processing: true
  - aggregate_results: true
```

### Confidence Thresholds
Filter results by confidence:

```yaml
confidence_threshold: 0.8
fallback_analysis: true  # Use alternative method for low confidence
```
