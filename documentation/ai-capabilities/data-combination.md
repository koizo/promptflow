# Data Combination

The Data Combiner is a powerful utility executor for combining and merging results from multiple flow steps. It supports various combination strategies and data transformation operations, making it essential for creating comprehensive AI workflows.

## 🎯 Overview

The Data Combiner allows you to:
- **Merge** dictionary results with sophisticated conflict resolution
- **Concatenate** lists and arrays from multiple sources
- **Join** text content with custom separators and formatting
- **Structure** data with custom templates and field extraction
- **Aggregate** numeric data with statistical operations
- **Transform** outputs with formatting, filtering, and renaming

## 🔧 Configuration

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `sources` | `List[str \| Any]` | List of step names to combine or direct values |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `strategy` | `string` | `"merge"` | Combination strategy (`merge`, `concat`, `join`, `structured`, `aggregate`) |
| `output_key` | `string` | `"combined_result"` | Key name for the combined result |
| `join_separator` | `string` | `" "` | Separator for text joining |
| `merge_strategy` | `string` | `"overwrite"` | Conflict resolution (`overwrite`, `keep_first`, `combine`) |
| `include_metadata` | `boolean` | `false` | Include combination metadata in output |
| `structure_template` | `object` | `{}` | Template for structured output |
| `aggregations` | `List[string]` | `["count", "sum"]` | Aggregation functions to apply |
| `transform` | `object` | `{}` | Output transformation rules |

## 🔄 Combination Strategies

### 1. Merge Strategy
Combines dictionary objects with configurable conflict resolution.

```yaml
- name: "merge_results"
  executor: "data_combiner"
  config:
    sources: ["step1", "step2", "step3"]
    strategy: "merge"
    merge_strategy: "overwrite"  # or "keep_first", "combine"
    output_key: "merged_data"
```

**Merge Strategies:**
- `overwrite`: Last value wins (default)
- `keep_first`: First value is kept
- `combine`: Values are combined (strings joined, numbers in lists)

### 2. Concatenation Strategy
Combines lists or converts values to lists and concatenates them.

```yaml
- name: "concat_lists"
  executor: "data_combiner"
  config:
    sources: ["list_step1", "list_step2"]
    strategy: "concat"
    output_key: "combined_list"
```

### 3. Join Strategy
Joins data as text with configurable separators.

```yaml
- name: "join_text"
  executor: "data_combiner"
  config:
    sources: ["text_step1", "text_step2"]
    strategy: "join"
    join_separator: " | "
    output_key: "joined_text"
```

### 4. Structured Strategy
Creates structured output with custom templates.

```yaml
- name: "structure_data"
  executor: "data_combiner"
  config:
    sources: ["ocr_result", "sentiment_result", "llm_result"]
    strategy: "structured"
    structure_template:
      extracted_text: "0.text"           # First source, text field
      confidence: "0.confidence"         # First source, confidence field
      sentiment: "1.sentiment"           # Second source, sentiment field
      summary: "2.summary"               # Third source, summary field
    output_key: "structured_analysis"
```

### 5. Aggregate Strategy
Performs statistical aggregation on numeric data.

```yaml
- name: "aggregate_scores"
  executor: "data_combiner"
  config:
    sources: ["score1", "score2", "score3"]
    strategy: "aggregate"
    aggregations: ["count", "sum", "avg", "min", "max"]
    output_key: "score_statistics"
```

## 🔧 Data Transformations

Apply transformations to the final output:

```yaml
- name: "transform_output"
  executor: "data_combiner"
  config:
    sources: ["step1", "step2"]
    strategy: "merge"
    output_key: "data"
    transform:
      rename:
        data: "processed_data"
      include_only: ["processed_data", "metadata"]
      format:
        processed_data: "json"
```

## 📊 Use Cases

### Document Processing Pipeline
Combine OCR, sentiment analysis, and LLM results:

```yaml
name: "document_analysis"
steps:
  - name: "extract_text"
    executor: "ocr_processor"
    config:
      image_path: "{{ inputs.file }}"
  
  - name: "analyze_sentiment"
    executor: "sentiment_analyzer"
    config:
      text: "{{ steps.extract_text.text }}"
  
  - name: "llm_analysis"
    executor: "llm_analyzer"
    config:
      text: "{{ steps.extract_text.text }}"
      prompt: "Summarize key points"
  
  - name: "combine_analysis"
    executor: "data_combiner"
    config:
      sources: ["extract_text", "analyze_sentiment", "llm_analysis"]
      strategy: "structured"
      structure_template:
        text: "0.text"
        confidence: "0.confidence"
        sentiment: "1.sentiment"
        summary: "2.summary"
      include_metadata: true
```

### Multi-Document Aggregation
Aggregate results from multiple document analyses:

```yaml
- name: "aggregate_documents"
  executor: "data_combiner"
  config:
    sources: ["doc1_analysis", "doc2_analysis", "doc3_analysis"]
    strategy: "aggregate"
    aggregations: ["count", "avg", "concat_text"]
    output_key: "document_statistics"
```

### Error Recovery Combination
Combine results with fallback handling:

```yaml
- name: "combine_with_fallback"
  executor: "data_combiner"
  config:
    sources: ["primary_result", "backup_result", "default_values"]
    strategy: "merge"
    merge_strategy: "keep_first"  # Use first successful result
    include_metadata: true
```

## 🎯 Advanced Features

### Template-Based Field Extraction
Extract specific fields from complex nested data:

```yaml
structure_template:
  user_info: "0.user.profile"      # Extract nested user profile
  scores: "1.analysis.scores"      # Extract analysis scores
  summary: "2.summary"             # Extract summary text
  metadata: 0                      # Include entire first source
```

### Statistical Aggregations
Perform comprehensive statistical analysis:

```yaml
aggregations: 
  - "count"        # Count of numeric values
  - "sum"          # Sum of all values
  - "avg"          # Average value
  - "min"          # Minimum value
  - "max"          # Maximum value
  - "concat_text"  # Concatenate text values
```

### Output Transformations
Clean and format final outputs:

```yaml
transform:
  rename:
    old_key: "new_key"
  include_only: ["important_field1", "important_field2"]
  exclude: ["temporary_field"]
  format:
    json_field: "json"     # Convert to JSON string
    text_field: "upper"    # Convert to uppercase
```

## 📈 Performance & Scalability

- **Efficient Processing**: Handles hundreds of sources efficiently
- **Memory Optimized**: Streaming processing for large datasets
- **Async Compatible**: Non-blocking execution for high throughput
- **Error Resilient**: Continues processing with partial failures

## 🔗 Integration Examples

### With OCR Processing
```yaml
sources: ["ocr_processor"]
strategy: "structured"
structure_template:
  text: "0.text"
  confidence: "0.confidence"
```

### With Sentiment Analysis
```yaml
sources: ["sentiment_analyzer"]
strategy: "merge"
merge_strategy: "combine"
```

### With LLM Analysis
```yaml
sources: ["llm_analyzer"]
strategy: "join"
join_separator: "\n\n"
```

## 🧪 Testing & Validation

The Data Combiner includes comprehensive test coverage:
- **100% Test Coverage** with 13 comprehensive tests
- **All Combination Strategies** thoroughly validated
- **Error Scenarios** and edge cases covered
- **Integration Testing** with real workflow scenarios
- **Performance Testing** with large datasets

## 🎉 Best Practices

1. **Use Structured Strategy** for complex multi-step workflows
2. **Include Metadata** for debugging and monitoring
3. **Apply Transformations** to clean up final outputs
4. **Use Aggregation** for statistical analysis of multiple results
5. **Handle Errors Gracefully** with appropriate merge strategies
6. **Test Templates** thoroughly with sample data
7. **Document Sources** clearly in flow descriptions

The Data Combiner is essential for building sophisticated AI workflows that require data aggregation, transformation, and structured output generation.
