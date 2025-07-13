# YAML Flow Configuration

Complete guide to creating and configuring AI workflows using YAML definitions in the PromptFlow platform.

## Overview

YAML flows are the core of the PromptFlow platform, allowing you to define complex AI workflows through simple, declarative configuration files. Each flow combines multiple AI executors in a sequence to create powerful, automated processing pipelines.

## Basic Flow Structure

### Minimal Flow Example
```yaml
name: "simple_flow"
version: "1.0.0"
description: "A basic flow example"

inputs:
  - name: "text"
    type: "string"
    required: true

steps:
  - name: "process_text"
    executor: "llm_analyzer"
    config:
      text: "{{ inputs.text }}"

config:
  execution:
    mode: "async"
    timeout: 300
```

### Complete Flow Structure
```yaml
# Flow Metadata
name: "flow_name"                    # Unique flow identifier
version: "1.0.0"                     # Semantic version
description: "Flow description"       # Human-readable description
author: "Your Name"                  # Optional: Flow author
tags: ["ai", "nlp", "processing"]   # Optional: Categorization tags

# Input Definitions
inputs:
  - name: "input_name"
    type: "string|file|integer|float|boolean|array|object"
    required: true|false
    default: "default_value"
    description: "Input description"
    validation:
      min_length: 1
      max_length: 1000
      pattern: "regex_pattern"

# Processing Steps
steps:
  - name: "step_name"
    executor: "executor_name"
    config:
      parameter: "{{ inputs.input_name }}"
      static_value: "fixed_value"
    condition: "{{ inputs.condition == 'value' }}"
    retry:
      max_attempts: 3
      delay: 1.0

# Flow Configuration
config:
  execution:
    mode: "async|sync"
    timeout: 300
    max_retries: 3
  
  output:
    format: "json|yaml|text"
    include_metadata: true
    
  error_handling:
    on_failure: "stop|continue|retry"
    fallback_step: "fallback_step_name"
```

## Input Definitions

### Input Types

#### String Input
```yaml
inputs:
  - name: "text_input"
    type: "string"
    required: true
    description: "Text to process"
    validation:
      min_length: 1
      max_length: 5000
      pattern: "^[a-zA-Z0-9\\s]+$"  # Alphanumeric and spaces only
```

#### File Input
```yaml
inputs:
  - name: "file_input"
    type: "file"
    required: true
    description: "File to process"
    validation:
      allowed_types: ["image/jpeg", "image/png", "application/pdf"]
      max_size: 10485760  # 10MB in bytes
      min_size: 1024      # 1KB minimum
```

#### Numeric Inputs
```yaml
inputs:
  - name: "confidence_threshold"
    type: "float"
    required: false
    default: 0.8
    description: "Confidence threshold (0.0-1.0)"
    validation:
      minimum: 0.0
      maximum: 1.0
      
  - name: "max_results"
    type: "integer"
    required: false
    default: 10
    description: "Maximum number of results"
    validation:
      minimum: 1
      maximum: 100
```

#### Boolean Input
```yaml
inputs:
  - name: "include_metadata"
    type: "boolean"
    required: false
    default: true
    description: "Include metadata in response"
```

#### Array Input
```yaml
inputs:
  - name: "languages"
    type: "array"
    required: false
    default: ["en"]
    description: "Languages to process"
    validation:
      min_items: 1
      max_items: 5
      allowed_values: ["en", "es", "fr", "de", "it"]
```

#### Object Input
```yaml
inputs:
  - name: "processing_options"
    type: "object"
    required: false
    default:
      quality: "high"
      format: "json"
    description: "Processing configuration options"
    validation:
      required_properties: ["quality"]
      allowed_properties: ["quality", "format", "timeout"]
```

### Input Validation

#### String Validation
```yaml
validation:
  min_length: 10           # Minimum string length
  max_length: 1000         # Maximum string length
  pattern: "^[A-Z][a-z]+$" # Regex pattern
  allowed_values: ["option1", "option2", "option3"]
  forbidden_values: ["spam", "test"]
```

#### Numeric Validation
```yaml
validation:
  minimum: 0.0             # Minimum value (inclusive)
  maximum: 100.0           # Maximum value (inclusive)
  exclusive_minimum: 0.0   # Minimum value (exclusive)
  exclusive_maximum: 100.0 # Maximum value (exclusive)
  multiple_of: 0.1         # Must be multiple of this value
```

#### File Validation
```yaml
validation:
  allowed_types: ["image/jpeg", "image/png", "text/plain"]
  max_size: 10485760       # Maximum file size in bytes
  min_size: 1024           # Minimum file size in bytes
  allowed_extensions: [".jpg", ".png", ".txt"]
```

#### Array Validation
```yaml
validation:
  min_items: 1             # Minimum array length
  max_items: 10            # Maximum array length
  unique_items: true       # All items must be unique
  allowed_values: ["en", "es", "fr"]
  item_type: "string"      # Type of array items
```

## Step Definitions

### Basic Step
```yaml
steps:
  - name: "analyze_text"
    executor: "sentiment_analyzer"
    config:
      text: "{{ inputs.text }}"
      provider: "huggingface"
      analysis_type: "detailed"
```

### Step with Conditions
```yaml
steps:
  - name: "conditional_step"
    executor: "llm_analyzer"
    config:
      text: "{{ inputs.text }}"
    condition: "{{ inputs.text | length > 100 }}"
```

### Step with Retry Logic
```yaml
steps:
  - name: "api_call_step"
    executor: "external_api"
    config:
      endpoint: "{{ inputs.api_endpoint }}"
    retry:
      max_attempts: 3
      delay: 2.0
      backoff_multiplier: 2.0
      retry_on_errors: ["timeout", "rate_limit"]
```

### Step with Error Handling
```yaml
steps:
  - name: "risky_step"
    executor: "external_service"
    config:
      data: "{{ inputs.data }}"
    error_handling:
      on_failure: "continue"
      fallback_value: "default_result"
      log_errors: true
```

### Parallel Steps
```yaml
steps:
  - name: "parallel_processing"
    type: "parallel"
    steps:
      - name: "sentiment_analysis"
        executor: "sentiment_analyzer"
        config:
          text: "{{ inputs.text }}"
          
      - name: "entity_extraction"
        executor: "entity_extractor"
        config:
          text: "{{ inputs.text }}"
```

## Template Variables

### Input References
```yaml
config:
  text: "{{ inputs.text_input }}"           # Direct input reference
  file_path: "{{ inputs.file_input }}"      # File input reference
  threshold: "{{ inputs.confidence | default(0.8) }}"  # With default value
```

### Step Output References
```yaml
config:
  # Reference previous step output
  processed_text: "{{ steps.preprocess.cleaned_text }}"
  
  # Reference nested output
  sentiment: "{{ steps.analyze.outputs.sentiment }}"
  
  # Reference with fallback
  result: "{{ steps.process.result | default('no_result') }}"
```

### Template Filters

#### String Filters
```yaml
config:
  # String manipulation
  uppercase: "{{ inputs.text | upper }}"
  lowercase: "{{ inputs.text | lower }}"
  title_case: "{{ inputs.text | title }}"
  
  # String operations
  length: "{{ inputs.text | length }}"
  truncate: "{{ inputs.text | truncate(100) }}"
  replace: "{{ inputs.text | replace('old', 'new') }}"
```

#### Numeric Filters
```yaml
config:
  # Math operations
  rounded: "{{ inputs.score | round(2) }}"
  absolute: "{{ inputs.value | abs }}"
  minimum: "{{ inputs.values | min }}"
  maximum: "{{ inputs.values | max }}"
```

#### Array Filters
```yaml
config:
  # Array operations
  first_item: "{{ inputs.array | first }}"
  last_item: "{{ inputs.array | last }}"
  array_length: "{{ inputs.array | length }}"
  joined: "{{ inputs.array | join(', ') }}"
  unique: "{{ inputs.array | unique }}"
```

#### Conditional Filters
```yaml
config:
  # Conditional logic
  default_value: "{{ inputs.optional | default('fallback') }}"
  conditional: "{{ inputs.score > 0.8 | ternary('high', 'low') }}"
```

## Configuration Options

### Execution Configuration
```yaml
config:
  execution:
    mode: "async"              # async|sync
    timeout: 300               # Timeout in seconds
    max_retries: 3             # Maximum retry attempts
    retry_delay: 1.0           # Delay between retries (seconds)
    max_concurrent: 5          # Max concurrent step execution
    priority: "normal"         # low|normal|high
```

### Output Configuration
```yaml
config:
  output:
    format: "json"             # json|yaml|text|xml
    include_metadata: true     # Include execution metadata
    include_timing: true       # Include timing information
    include_errors: false      # Include error details
    pretty_print: true         # Format output for readability
```

### Error Handling Configuration
```yaml
config:
  error_handling:
    on_failure: "stop"         # stop|continue|retry
    max_error_count: 5         # Max errors before stopping
    log_errors: true           # Log errors to system
    include_stack_trace: false # Include stack traces in errors
    fallback_flow: "error_handler_flow"  # Fallback flow name
```

### Caching Configuration
```yaml
config:
  caching:
    enabled: true              # Enable result caching
    ttl: 3600                  # Cache TTL in seconds
    key_template: "{{ inputs.text | hash }}"  # Cache key template
    cache_provider: "redis"    # redis|memory|file
```

## Flow Examples

### Document Analysis Flow
```yaml
name: "document_analysis"
version: "1.0.0"
description: "Extract and analyze text from documents"

inputs:
  - name: "file"
    type: "file"
    required: true
    description: "Document file to analyze"
    validation:
      allowed_types: ["application/pdf", "image/jpeg", "image/png"]
      max_size: 20971520  # 20MB

  - name: "languages"
    type: "array"
    required: false
    default: ["en"]
    description: "Expected languages in document"

  - name: "analysis_depth"
    type: "string"
    required: false
    default: "standard"
    description: "Analysis depth level"
    validation:
      allowed_values: ["basic", "standard", "comprehensive"]

steps:
  - name: "extract_text"
    executor: "ocr_processor"
    config:
      image_path: "{{ inputs.file }}"
      languages: "{{ inputs.languages }}"
      confidence_threshold: 0.8
      return_bboxes: true

  - name: "analyze_sentiment"
    executor: "sentiment_analyzer"
    config:
      text: "{{ steps.extract_text.text }}"
      provider: "llm"
      analysis_type: "{{ inputs.analysis_depth }}"
    condition: "{{ steps.extract_text.text | length > 10 }}"

  - name: "extract_entities"
    executor: "entity_extractor"
    config:
      text: "{{ steps.extract_text.text }}"
      entity_types: ["PERSON", "ORG", "DATE", "MONEY"]
    condition: "{{ inputs.analysis_depth != 'basic' }}"

  - name: "format_results"
    executor: "response_formatter"
    config:
      template: "document_analysis_report"
      data:
        ocr_results: "{{ steps.extract_text }}"
        sentiment: "{{ steps.analyze_sentiment }}"
        entities: "{{ steps.extract_entities }}"

config:
  execution:
    mode: "async"
    timeout: 600
    max_retries: 2
  
  output:
    format: "json"
    include_metadata: true
    
  error_handling:
    on_failure: "stop"
    log_errors: true
```

### Multi-Modal Content Analysis
```yaml
name: "multimodal_content_analysis"
version: "1.0.0"
description: "Analyze both text and image content"

inputs:
  - name: "text_content"
    type: "string"
    required: false
    description: "Text content to analyze"

  - name: "image_file"
    type: "file"
    required: false
    description: "Image file to analyze"
    validation:
      allowed_types: ["image/jpeg", "image/png", "image/webp"]

  - name: "analysis_type"
    type: "string"
    required: false
    default: "comprehensive"
    validation:
      allowed_values: ["basic", "detailed", "comprehensive"]

steps:
  - name: "analyze_text_sentiment"
    executor: "sentiment_analyzer"
    config:
      text: "{{ inputs.text_content }}"
      provider: "huggingface"
      analysis_type: "{{ inputs.analysis_type }}"
    condition: "{{ inputs.text_content | length > 0 }}"

  - name: "classify_image"
    executor: "vision_classifier"
    config:
      image: "{{ inputs.image_file }}"
      provider: "huggingface"
      top_k: 5
      confidence_threshold: 0.1
    condition: "{{ inputs.image_file is defined }}"

  - name: "analyze_image_with_ai"
    executor: "vision_classifier"
    config:
      image: "{{ inputs.image_file }}"
      provider: "openai"
      classification_prompt: "Analyze this image for emotional content and mood"
    condition: "{{ inputs.image_file is defined and inputs.analysis_type == 'comprehensive' }}"

  - name: "combine_results"
    executor: "data_combiner"
    config:
      text_analysis: "{{ steps.analyze_text_sentiment }}"
      image_classification: "{{ steps.classify_image }}"
      image_analysis: "{{ steps.analyze_image_with_ai }}"
      combination_strategy: "merge"

config:
  execution:
    mode: "async"
    timeout: 300
    max_concurrent: 3
```

### Batch Processing Flow
```yaml
name: "batch_sentiment_analysis"
version: "1.0.0"
description: "Process multiple texts for sentiment analysis"

inputs:
  - name: "texts"
    type: "array"
    required: true
    description: "Array of texts to analyze"
    validation:
      min_items: 1
      max_items: 100
      item_type: "string"

  - name: "batch_size"
    type: "integer"
    required: false
    default: 10
    description: "Number of texts to process in each batch"
    validation:
      minimum: 1
      maximum: 50

steps:
  - name: "batch_process"
    type: "batch"
    executor: "sentiment_analyzer"
    config:
      texts: "{{ inputs.texts }}"
      batch_size: "{{ inputs.batch_size }}"
      provider: "huggingface"
      analysis_type: "basic"
      parallel_processing: true

  - name: "aggregate_results"
    executor: "data_combiner"
    config:
      results: "{{ steps.batch_process.results }}"
      aggregation_type: "sentiment_summary"
      include_statistics: true

config:
  execution:
    mode: "async"
    timeout: 1800  # 30 minutes for large batches
    max_concurrent: 5
```

## Advanced Features

### Dynamic Step Generation
```yaml
steps:
  - name: "dynamic_processing"
    type: "dynamic"
    generator: "{{ inputs.processing_config }}"
    template:
      name: "process_{{ item.name }}"
      executor: "{{ item.executor }}"
      config: "{{ item.config }}"
```

### Conditional Flows
```yaml
steps:
  - name: "route_processing"
    type: "conditional"
    conditions:
      - condition: "{{ inputs.file_type == 'image' }}"
        steps:
          - name: "process_image"
            executor: "vision_classifier"
            
      - condition: "{{ inputs.file_type == 'audio' }}"
        steps:
          - name: "process_audio"
            executor: "whisper_processor"
            
      - default: true
        steps:
          - name: "process_text"
            executor: "llm_analyzer"
```

### Loop Processing
```yaml
steps:
  - name: "iterative_processing"
    type: "loop"
    iterator: "{{ inputs.items }}"
    max_iterations: 100
    steps:
      - name: "process_item"
        executor: "item_processor"
        config:
          item: "{{ loop.item }}"
          index: "{{ loop.index }}"
```

## Best Practices

### Flow Design
1. **Single Responsibility**: Each flow should have a clear, single purpose
2. **Modular Steps**: Break complex processing into smaller, reusable steps
3. **Error Handling**: Always include appropriate error handling
4. **Documentation**: Provide clear descriptions for flows and inputs

### Input Validation
1. **Validate Early**: Use input validation to catch errors early
2. **Reasonable Limits**: Set appropriate size and length limits
3. **Type Safety**: Always specify input types explicitly
4. **Default Values**: Provide sensible defaults for optional inputs

### Performance Optimization
1. **Async Processing**: Use async mode for production workloads
2. **Parallel Steps**: Use parallel processing where possible
3. **Caching**: Enable caching for expensive operations
4. **Timeouts**: Set appropriate timeouts for all operations

### Security Considerations
1. **Input Sanitization**: Validate and sanitize all inputs
2. **File Type Restrictions**: Restrict allowed file types
3. **Size Limits**: Enforce reasonable file size limits
4. **Access Control**: Implement proper access controls

## Troubleshooting

### Common Issues

#### Template Errors
```yaml
# ❌ Incorrect
config:
  text: "{{ inputs.nonexistent }}"

# ✅ Correct
config:
  text: "{{ inputs.text | default('') }}"
```

#### Type Mismatches
```yaml
# ❌ Incorrect
inputs:
  - name: "count"
    type: "string"  # Should be integer
    
# ✅ Correct
inputs:
  - name: "count"
    type: "integer"
    validation:
      minimum: 1
```

#### Missing Dependencies
```yaml
# ❌ Incorrect - step2 depends on step1 but step1 might fail
steps:
  - name: "step1"
    executor: "risky_executor"
  - name: "step2"
    executor: "dependent_executor"
    config:
      data: "{{ steps.step1.result }}"

# ✅ Correct - add condition or error handling
steps:
  - name: "step1"
    executor: "risky_executor"
    error_handling:
      on_failure: "continue"
      fallback_value: "default"
  - name: "step2"
    executor: "dependent_executor"
    config:
      data: "{{ steps.step1.result | default('default') }}"
    condition: "{{ steps.step1.success }}"
```

### Validation Tools
Use the built-in validation tools to check your YAML flows:

```bash
# Validate flow syntax
curl -X POST "http://localhost:8000/api/v1/flows/validate" \
  -F "flow_file=@my_flow.yaml"

# Test flow with sample data
curl -X POST "http://localhost:8000/api/v1/flows/test" \
  -F "flow_file=@my_flow.yaml" \
  -F "test_data=@test_inputs.json"
```
