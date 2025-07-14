# DataCombiner Executor

The DataCombiner executor is a powerful utility for combining and merging results from multiple flow steps. It supports various combination strategies and data transformation operations, making it essential for creating comprehensive AI workflows.

## Overview

The DataCombiner allows you to:
- **Merge** dictionary results with conflict resolution
- **Concatenate** lists and arrays
- **Join** text content with custom separators
- **Structure** data with custom templates
- **Aggregate** numeric data with statistics
- **Transform** outputs with formatting and filtering

## Configuration

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

## Combination Strategies

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

**Text Extraction Priority:**
1. `text` field from dictionaries
2. `content` field from dictionaries
3. JSON representation for complex objects
4. String conversion for other types

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
      full_ocr: 0                        # Entire first source
    output_key: "structured_analysis"
```

**Template Reference Format:**
- `"N.field"`: Extract field from Nth source (0-indexed)
- `N`: Include entire Nth source
- Invalid references return `null`

### 5. Aggregate Strategy

Performs statistical aggregation on numeric data.

```yaml
- name: "aggregate_scores"
  executor: "data_combiner"
  config:
    sources: ["score1", "score2", "score3"]
    strategy: "aggregate"
    aggregations: ["count", "sum", "avg", "min", "max", "concat_text"]
    output_key: "score_statistics"
```

**Available Aggregations:**
- `count`: Count of numeric values
- `sum`: Sum of numeric values
- `avg`: Average of numeric values
- `min`: Minimum value
- `max`: Maximum value
- `concat_text`: Concatenated text values

## Data Transformations

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
      exclude: ["temporary_field"]
      format:
        processed_data: "json"  # or "upper", "lower"
```

**Transformation Options:**
- `rename`: Rename output keys
- `include_only`: Keep only specified keys
- `exclude`: Remove specified keys
- `format`: Apply formatting (`json`, `upper`, `lower`)

## Metadata

When `include_metadata: true`, the combiner adds processing information:

```json
{
  "combination_metadata": {
    "sources_count": 3,
    "strategy": "merge",
    "timestamp": "2024-01-15T10:30:00Z",
    "source_info": {
      "step1": {
        "type": "step_result",
        "success": true,
        "timestamp": "2024-01-15T10:29:45Z"
      }
    }
  }
}
```

## Usage Examples

### Document Processing Pipeline

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

```yaml
- name: "combine_with_fallback"
  executor: "data_combiner"
  config:
    sources: ["primary_result", "backup_result", "default_values"]
    strategy: "merge"
    merge_strategy: "keep_first"  # Use first successful result
    include_metadata: true
```

## Error Handling

The DataCombiner handles various error scenarios:

- **Missing Sources**: Returns error if no valid sources found
- **Invalid Step References**: Skips missing steps, continues with available data
- **Unknown Strategy**: Returns descriptive error message
- **Template Errors**: Returns `null` for invalid template references
- **Type Mismatches**: Gracefully handles mixed data types

## Performance Considerations

- **Memory Usage**: Large datasets are processed efficiently with streaming where possible
- **Async Processing**: All operations are async-compatible
- **Caching**: Results can be cached when used in flows with caching enabled
- **Scalability**: Handles hundreds of sources efficiently

## Integration with Other Executors

The DataCombiner works seamlessly with all other executors:

```yaml
# Combine OCR + Sentiment + LLM
sources: ["ocr_processor", "sentiment_analyzer", "llm_analyzer"]

# Combine multiple image classifications
sources: ["vision_classifier_1", "vision_classifier_2", "vision_classifier_3"]

# Combine audio transcriptions
sources: ["whisper_local", "whisper_openai", "whisper_huggingface"]
```

## Best Practices

1. **Use Structured Strategy** for complex multi-step workflows
2. **Include Metadata** for debugging and monitoring
3. **Apply Transformations** to clean up final outputs
4. **Use Aggregation** for statistical analysis of multiple results
5. **Handle Errors Gracefully** with appropriate merge strategies
6. **Test Templates** thoroughly with sample data
7. **Document Sources** clearly in flow descriptions

## Testing

The DataCombiner includes comprehensive tests covering:
- All combination strategies
- Error scenarios and edge cases
- Template processing and validation
- Transformation operations
- Integration scenarios
- Performance with large datasets

Run tests with:
```bash
python -m pytest tests/test_data_combiner.py -v
```

## API Reference

### DataCombiner Class

```python
class DataCombiner(BaseExecutor):
    async def execute(context: FlowContext, config: Dict[str, Any]) -> ExecutionResult
    def get_required_config() -> List[str]
    def get_optional_config() -> Dict[str, Any]
```

### Configuration Schema

```json
{
  "sources": ["step1", "step2"],
  "strategy": "merge|concat|join|structured|aggregate",
  "output_key": "string",
  "merge_strategy": "overwrite|keep_first|combine",
  "join_separator": "string",
  "include_metadata": "boolean",
  "structure_template": {
    "key": "source_reference"
  },
  "aggregations": ["count", "sum", "avg", "min", "max"],
  "transform": {
    "rename": {"old": "new"},
    "include_only": ["key1", "key2"],
    "exclude": ["key3"],
    "format": {"key": "json|upper|lower"}
  }
}
```

The DataCombiner executor is essential for building sophisticated AI workflows that require data aggregation, transformation, and structured output generation.
