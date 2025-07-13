# LLM Analysis

Advanced text analysis capabilities using Large Language Models with support for multiple providers, custom prompts, and intelligent text processing workflows.

## Overview

The LLM Analysis feature provides sophisticated text analysis using state-of-the-art Large Language Models. It supports custom prompts, multiple model providers, chunked text processing, and intelligent analysis combination for comprehensive text understanding.

## Features

- **Multiple LLM Providers**: Ollama (local), with extensible architecture for OpenAI, Anthropic, etc.
- **Custom Prompts**: Flexible prompt engineering for specific analysis needs
- **Chunked Processing**: Handle large texts by processing in chunks with intelligent combination
- **Model Selection**: Choose from various models optimized for different tasks
- **Template Integration**: Seamless integration with flow template variables
- **Error Handling**: Robust error handling with graceful degradation

## Supported Providers

### Ollama Provider (Default)
- **Best for**: Local processing, privacy-sensitive content, cost control
- **Models**: mistral, llama3.2, llama3.1, codellama
- **Performance**: Fast local inference, no API costs
- **Privacy**: Complete data privacy, no external API calls
- **Flexibility**: Easy model switching and custom model support

### Future Providers (Extensible Architecture)
- **OpenAI**: GPT-3.5, GPT-4 models (planned)
- **Anthropic**: Claude models (planned)
- **HuggingFace**: Transformers integration (planned)
- **Custom**: Easy integration of custom providers

## Supported Models

### Ollama Models
```yaml
# General purpose models
model: "mistral"        # 7B params, fast and capable
model: "llama3.2"       # Latest Llama model, high quality
model: "llama3.1"       # Previous Llama version, reliable

# Specialized models
model: "codellama"      # Optimized for code analysis and generation
```

### Model Characteristics
| Model | Size | Strengths | Best For |
|-------|------|-----------|----------|
| **mistral** | 7B | Fast, balanced | General analysis, summaries |
| **llama3.2** | 8B | Latest, high quality | Complex reasoning, detailed analysis |
| **llama3.1** | 8B | Reliable, proven | Production workloads |
| **codellama** | 7B | Code-focused | Code analysis, technical docs |

## Configuration Options

### Basic Configuration
```yaml
inputs:
  - name: "text"              # Text to analyze (required if no chunks)
  - name: "prompt"            # Analysis prompt (required)
  - name: "model"             # LLM model (default: "mistral")
  - name: "provider"          # LLM provider (default: "ollama")
```

### Advanced Configuration
```yaml
inputs:
  - name: "chunks"            # List of text chunks for batch processing
  - name: "combine_chunks"    # Combine chunk analyses (default: true)
  - name: "system_prompt"     # System-level prompt for context
  - name: "temperature"       # Creativity level (0.0-1.0, default: 0.7)
  - name: "max_tokens"        # Maximum response length
```

## Usage Examples

### Basic Text Analysis
```bash
curl -X POST "http://localhost:8000/api/v1/document-analysis/execute" \
  -F "file=@document.pdf" \
  -F "analysis_prompt=Summarize the key points and main conclusions"
```

**Flow Configuration:**
```yaml
steps:
  - name: "analyze_content"
    executor: "llm_analyzer"
    config:
      text: "{{ steps.extract_text.text }}"
      prompt: "{{ inputs.analysis_prompt }}"
      model: "mistral"
      provider: "ollama"
```

### Custom Analysis with Specific Model
```yaml
steps:
  - name: "code_analysis"
    executor: "llm_analyzer"
    config:
      text: "{{ inputs.code_content }}"
      prompt: "Analyze this code for potential bugs, security issues, and optimization opportunities"
      model: "codellama"
      provider: "ollama"
```

### Chunked Text Processing
```yaml
steps:
  - name: "analyze_large_document"
    executor: "llm_analyzer"
    config:
      chunks: "{{ steps.extract_text.chunks }}"
      prompt: "Analyze this section of the document and identify key themes and important information"
      model: "llama3.2"
      combine_chunks: true
```

### Multi-Step Analysis Pipeline
```yaml
steps:
  - name: "initial_summary"
    executor: "llm_analyzer"
    config:
      text: "{{ steps.extract_text.text }}"
      prompt: "Provide a brief summary of this document"
      model: "mistral"
      
  - name: "detailed_analysis"
    executor: "llm_analyzer"
    config:
      text: "{{ steps.extract_text.text }}"
      prompt: "Based on this summary: {{ steps.initial_summary.analysis }}, provide detailed analysis focusing on key insights and recommendations"
      model: "llama3.2"
```

## Response Format

### Single Text Analysis
```json
{
  "success": true,
  "outputs": {
    "analysis": "Comprehensive analysis of the provided text...",
    "prompt": "Analyze this document and provide key insights",
    "model": "mistral",
    "provider": "ollama",
    "text_length": 1250,
    "analysis_type": "single_text"
  },
  "metadata": {
    "input_characters": 1250,
    "output_characters": 450,
    "model_used": "mistral",
    "provider_used": "ollama"
  }
}
```

### Chunked Analysis
```json
{
  "success": true,
  "outputs": {
    "chunk_analyses": [
      {
        "chunk_index": 0,
        "analysis": "Analysis of first chunk...",
        "chunk_text_length": 500,
        "analysis_length": 150
      },
      {
        "chunk_index": 1,
        "analysis": "Analysis of second chunk...",
        "chunk_text_length": 480,
        "analysis_length": 140
      }
    ],
    "combined_analysis": "Unified analysis combining insights from all chunks...",
    "prompt": "Analyze each section for key themes",
    "model": "llama3.2",
    "provider": "ollama",
    "total_chunks": 2,
    "successful_analyses": 2,
    "analysis_type": "chunked_text"
  },
  "metadata": {
    "total_input_characters": 980,
    "total_output_characters": 420,
    "chunks_processed": 2,
    "model_used": "llama3.2",
    "provider_used": "ollama",
    "combined_analysis_generated": true
  }
}
```

## Prompt Engineering

### Effective Prompt Patterns

#### Analysis Prompts
```yaml
# Structured analysis
prompt: |
  Analyze the following text and provide:
  1. Main themes and topics
  2. Key insights and findings
  3. Important entities (people, organizations, dates)
  4. Conclusions and recommendations
  
  Text: {{ text }}

# Comparative analysis
prompt: |
  Compare and contrast the following documents, focusing on:
  - Similarities and differences
  - Conflicting information
  - Complementary insights
  
  Document: {{ text }}

# Domain-specific analysis
prompt: |
  As a legal expert, analyze this contract for:
  - Key terms and conditions
  - Potential risks or issues
  - Important dates and obligations
  
  Contract: {{ text }}
```

#### Question-Answering Prompts
```yaml
# Specific questions
prompt: |
  Based on the following document, answer these questions:
  1. What is the main purpose?
  2. Who are the key stakeholders?
  3. What are the next steps?
  
  Document: {{ text }}

# Information extraction
prompt: |
  Extract the following information from this text:
  - Names and contact information
  - Dates and deadlines
  - Financial amounts
  - Action items
  
  Text: {{ text }}
```

### Best Practices for Prompts

1. **Be Specific**: Clear, detailed instructions yield better results
2. **Use Structure**: Numbered lists and bullet points help organize output
3. **Provide Context**: Include relevant background information
4. **Set Expectations**: Specify desired output format and length
5. **Use Examples**: Include examples of desired output when possible

## Performance Optimization

### Model Selection Guidelines

#### For Speed (< 2 seconds)
- **mistral**: Best balance of speed and quality
- **Use for**: Real-time analysis, quick summaries, simple tasks

#### For Quality (2-10 seconds)
- **llama3.2**: Latest model with improved capabilities
- **Use for**: Complex analysis, detailed reasoning, important documents

#### For Specialized Tasks
- **codellama**: Code analysis and technical documentation
- **Use for**: Software documentation, code reviews, technical analysis

### Chunking Strategies

#### Optimal Chunk Sizes
```yaml
# For detailed analysis
chunk_size: 1000        # ~200 words, good for detailed analysis
chunk_overlap: 200      # Maintain context between chunks

# For quick processing
chunk_size: 2000        # ~400 words, faster processing
chunk_overlap: 100      # Minimal overlap for speed

# For complex documents
chunk_size: 500         # ~100 words, maximum detail
chunk_overlap: 100      # High overlap for context preservation
```

#### Chunking Best Practices
1. **Preserve Context**: Use appropriate overlap between chunks
2. **Logical Boundaries**: Chunk at paragraph or section breaks when possible
3. **Consistent Size**: Maintain similar chunk sizes for uniform analysis
4. **Combine Results**: Use `combine_chunks: true` for unified insights

## Common Use Cases

### Document Analysis
- **Legal Documents**: Contract analysis, compliance checking
- **Research Papers**: Key findings extraction, methodology analysis
- **Business Reports**: Executive summaries, trend identification
- **Technical Documentation**: Feature analysis, requirement extraction

### Content Processing
- **News Articles**: Fact extraction, sentiment analysis, bias detection
- **Social Media**: Trend analysis, opinion mining, content categorization
- **Customer Feedback**: Issue identification, satisfaction analysis
- **Academic Papers**: Literature review, citation analysis

### Code Analysis
- **Code Reviews**: Bug detection, security analysis, optimization suggestions
- **Documentation**: API documentation analysis, usage pattern identification
- **Architecture**: System design analysis, dependency mapping
- **Quality Assessment**: Code quality metrics, maintainability analysis

## Integration Examples

### Document Processing Pipeline
```yaml
name: "comprehensive_document_analysis"
description: "Multi-stage document analysis with LLM"

steps:
  - name: "extract_text"
    executor: "document_extractor"
    config:
      file_path: "{{ inputs.file }}"
      
  - name: "initial_summary"
    executor: "llm_analyzer"
    config:
      text: "{{ steps.extract_text.text }}"
      prompt: "Provide a concise summary of this document's main points"
      model: "mistral"
      
  - name: "detailed_analysis"
    executor: "llm_analyzer"
    config:
      text: "{{ steps.extract_text.text }}"
      prompt: |
        Based on this summary: {{ steps.initial_summary.analysis }}
        
        Provide detailed analysis including:
        1. Key insights and findings
        2. Important entities and relationships
        3. Actionable recommendations
        4. Potential concerns or risks
      model: "llama3.2"
      
  - name: "extract_entities"
    executor: "llm_analyzer"
    config:
      text: "{{ steps.extract_text.text }}"
      prompt: |
        Extract structured information in JSON format:
        {
          "people": ["names of people mentioned"],
          "organizations": ["companies, institutions"],
          "dates": ["important dates and deadlines"],
          "amounts": ["financial figures"],
          "locations": ["addresses, cities, countries"]
        }
      model: "mistral"
```

### Multi-Language Analysis
```yaml
name: "multilingual_analysis"
description: "Analyze documents in multiple languages"

steps:
  - name: "detect_language"
    executor: "llm_analyzer"
    config:
      text: "{{ steps.extract_text.text }}"
      prompt: "Identify the primary language of this text and provide a confidence score"
      model: "mistral"
      
  - name: "analyze_content"
    executor: "llm_analyzer"
    config:
      text: "{{ steps.extract_text.text }}"
      prompt: |
        Analyze this text in its original language ({{ steps.detect_language.analysis }}).
        Provide analysis in English including:
        - Summary of main content
        - Cultural context and nuances
        - Key terminology and concepts
      model: "llama3.2"
```

### Comparative Analysis
```yaml
name: "document_comparison"
description: "Compare multiple documents using LLM"

steps:
  - name: "analyze_doc1"
    executor: "llm_analyzer"
    config:
      text: "{{ steps.extract_text1.text }}"
      prompt: "Analyze this document and identify key themes, arguments, and conclusions"
      model: "llama3.2"
      
  - name: "analyze_doc2"
    executor: "llm_analyzer"
    config:
      text: "{{ steps.extract_text2.text }}"
      prompt: "Analyze this document and identify key themes, arguments, and conclusions"
      model: "llama3.2"
      
  - name: "compare_documents"
    executor: "llm_analyzer"
    config:
      text: |
        Document 1 Analysis: {{ steps.analyze_doc1.analysis }}
        
        Document 2 Analysis: {{ steps.analyze_doc2.analysis }}
      prompt: |
        Compare these two document analyses and provide:
        1. Key similarities and differences
        2. Conflicting information or viewpoints
        3. Complementary insights
        4. Overall relationship between documents
      model: "llama3.2"
```

## Error Handling and Troubleshooting

### Common Issues

#### Model Not Available
```json
{
  "success": false,
  "error": "Model 'gpt-4' not available in provider 'ollama'"
}
```
**Solution**: Check available models with `ollama list` or use supported models

#### Prompt Too Long
```json
{
  "success": false,
  "error": "Input text exceeds model context limit"
}
```
**Solution**: Use chunked processing or reduce input text size

#### Provider Connection Error
```json
{
  "success": false,
  "error": "Failed to connect to Ollama server"
}
```
**Solution**: Ensure Ollama is running and accessible

### Best Practices for Error Handling

1. **Validate Inputs**: Check text length and prompt format before processing
2. **Use Fallbacks**: Implement fallback models or simplified prompts
3. **Monitor Resources**: Track model memory usage and response times
4. **Graceful Degradation**: Provide partial results when possible

## API Reference

### Endpoints
LLM Analysis is integrated into flows and accessed through flow-specific endpoints:
- **Document Analysis**: `POST /api/v1/document-analysis/execute`
- **OCR Analysis**: `POST /api/v1/ocr-analysis/execute`
- **Custom Flows**: `POST /api/v1/{flow-name}/execute`

### Configuration Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required* | Text to analyze |
| `chunks` | array | required* | Text chunks for batch processing |
| `prompt` | string | required | Analysis prompt |
| `model` | string | "mistral" | LLM model to use |
| `provider` | string | "ollama" | LLM provider |
| `combine_chunks` | boolean | true | Combine chunk analyses |
| `temperature` | float | 0.7 | Response creativity (0.0-1.0) |
| `max_tokens` | integer | null | Maximum response length |

*Either `text` or `chunks` must be provided

### Response Codes
- **200**: Success - Analysis completed
- **400**: Bad Request - Invalid parameters
- **422**: Unprocessable Entity - Analysis failed
- **500**: Internal Server Error - System error

## Advanced Features

### Custom System Prompts
Set context and behavior for the LLM:

```yaml
config:
  system_prompt: "You are an expert financial analyst. Provide precise, data-driven analysis."
  prompt: "Analyze this financial report"
  text: "{{ document_text }}"
```

### Temperature Control
Adjust creativity vs. consistency:

```yaml
config:
  temperature: 0.1    # More consistent, factual responses
  temperature: 0.7    # Balanced creativity and consistency
  temperature: 1.0    # More creative, varied responses
```

### Token Limits
Control response length:

```yaml
config:
  max_tokens: 500     # Short, concise responses
  max_tokens: 2000    # Detailed analysis
  max_tokens: null    # No limit (model default)
```

### Batch Processing
Process multiple texts efficiently:

```yaml
config:
  chunks: [
    {"text": "First document content..."},
    {"text": "Second document content..."},
    {"text": "Third document content..."}
  ]
  prompt: "Analyze each document for key themes"
  combine_chunks: true
```

This comprehensive LLM Analysis capability provides the foundation for sophisticated text understanding and processing workflows, enabling powerful AI-driven document analysis and content processing.
