# OCR Processing

Optical Character Recognition (OCR) capabilities for extracting text from images and documents with high accuracy and multiple language support.

## Overview

The OCR Processing feature provides robust text extraction from images using advanced OCR engines. It supports multiple languages, confidence scoring, and bounding box detection for precise text location.

## Features

- **Multi-Language Support**: Extract text in 100+ languages
- **High Accuracy**: Advanced OCR engines with confidence scoring
- **Bounding Box Detection**: Precise text location coordinates
- **Confidence Filtering**: Filter results by confidence threshold
- **Multiple Formats**: Support for PNG, JPEG, TIFF, PDF images

## Supported Languages

### Primary Languages
- **English** (en) - Default, highest accuracy
- **Spanish** (spa) - High accuracy for Latin American and European Spanish
- **French** (fra) - Optimized for French text recognition
- **German** (deu) - German language support
- **Portuguese** (por) - Brazilian and European Portuguese

### Extended Language Support
The OCR processor supports 100+ languages through Tesseract. Common languages include:
- Chinese (Simplified/Traditional)
- Japanese, Korean
- Arabic, Hebrew
- Russian, Ukrainian
- Italian, Dutch
- And many more...

## Configuration Options

### Basic Configuration
```yaml
inputs:
  - name: "image_path"        # Path to image file (required)
  - name: "provider"          # OCR provider (default: "tesseract")
  - name: "languages"         # Languages to detect (default: ["en"])
  - name: "confidence_threshold" # Minimum confidence (default: 0.0)
```

### Advanced Configuration
```yaml
inputs:
  - name: "return_bboxes"     # Include bounding boxes (default: true)
  - name: "return_confidence" # Include confidence scores (default: true)
  - name: "preprocessing"     # Image preprocessing options
  - name: "dpi"              # Image DPI for processing (default: 300)
```

## Usage Examples

### Basic Text Extraction
```bash
curl -X POST "http://localhost:8000/api/v1/ocr-analysis/execute" \
  -F "file=@document.png" \
  -F "languages=en"
```

**Response:**
```json
{
  "text": "Extracted text content from the image...",
  "confidence": 0.95,
  "word_count": 150,
  "languages": ["en"],
  "processing_time": 2.3
}
```

### Multi-Language Processing
```bash
curl -X POST "http://localhost:8000/api/v1/ocr-analysis/execute" \
  -F "file=@multilingual_document.png" \
  -F "languages=en,spa,fra" \
  -F "confidence_threshold=0.8"
```

### High-Confidence Text Only
```bash
curl -X POST "http://localhost:8000/api/v1/ocr-analysis/execute" \
  -F "file=@document.png" \
  -F "confidence_threshold=0.9" \
  -F "return_bboxes=true"
```

## Response Format

### Standard Response
```json
{
  "success": true,
  "outputs": {
    "text": "Clean, processed text extracted from image",
    "original_text": "Raw OCR output before cleaning",
    "raw_ocr_text": "Unfiltered OCR text",
    "confidence": 0.92,
    "word_count": 145,
    "total_words": 160,
    "provider": "tesseract",
    "languages": ["en"],
    "image_file": "document.png",
    "confidence_threshold": 0.8,
    "text_blocks_count": 12
  },
  "metadata": {
    "ocr_provider": "tesseract",
    "languages_used": ["en"],
    "image_processed": "/tmp/document.png",
    "extraction_confidence": 0.92,
    "character_count": 1250,
    "words_extracted": 145,
    "total_text_blocks": 12,
    "processing_time": 2.34
  }
}
```

### With Bounding Boxes
```json
{
  "text_blocks": [
    {
      "text": "Header Text",
      "confidence": 0.98,
      "bbox": {
        "x": 100,
        "y": 50,
        "width": 200,
        "height": 30
      }
    },
    {
      "text": "Body paragraph content...",
      "confidence": 0.94,
      "bbox": {
        "x": 100,
        "y": 100,
        "width": 400,
        "height": 120
      }
    }
  ]
}
```

## Performance Optimization

### Image Preprocessing
- **Resolution**: 300 DPI recommended for optimal accuracy
- **Contrast**: High contrast images yield better results
- **Noise Reduction**: Clean images without artifacts
- **Orientation**: Properly oriented text (not rotated)

### Language Selection
- **Specific Languages**: Use only required languages for faster processing
- **Primary Language First**: List most likely language first
- **Avoid Over-specification**: Too many languages can reduce accuracy

### Confidence Thresholds
- **High Accuracy**: Use 0.8+ threshold for critical applications
- **Balanced**: 0.6-0.8 for general use cases
- **Maximum Coverage**: 0.0-0.5 for difficult images

## Best Practices

### Image Quality
1. **High Resolution**: Use images with at least 300 DPI
2. **Good Lighting**: Ensure even lighting without shadows
3. **Sharp Focus**: Avoid blurry or out-of-focus images
4. **Proper Orientation**: Ensure text is right-side up

### Language Configuration
1. **Specify Languages**: Always specify expected languages
2. **Order Matters**: List most likely language first
3. **Limit Languages**: Use only necessary languages for better performance

### Error Handling
1. **Check Confidence**: Always validate confidence scores
2. **Fallback Strategy**: Have backup processing for low-confidence results
3. **Validation**: Verify critical extracted text manually

### Performance Tips
1. **Batch Processing**: Process multiple images in parallel
2. **Caching**: Cache results for repeated processing
3. **Preprocessing**: Optimize images before OCR processing
4. **Resource Management**: Monitor memory usage for large images

## Common Use Cases

### Document Digitization
- **Invoices**: Extract invoice numbers, amounts, dates
- **Receipts**: Process expense receipts for accounting
- **Contracts**: Digitize legal documents and agreements
- **Forms**: Extract data from filled forms

### Content Management
- **Archive Digitization**: Convert physical documents to digital
- **Search Indexing**: Make scanned documents searchable
- **Data Entry**: Automate manual data entry processes
- **Compliance**: Extract information for regulatory compliance

### Accessibility
- **Screen Readers**: Convert images to accessible text
- **Translation**: Extract text for translation services
- **Content Analysis**: Analyze text content from images

## Troubleshooting

### Low Accuracy Issues
- **Check Image Quality**: Ensure high resolution and good contrast
- **Verify Language**: Confirm correct language specification
- **Adjust Threshold**: Lower confidence threshold for difficult images
- **Preprocess Image**: Clean up noise and improve contrast

### Performance Issues
- **Reduce Languages**: Limit to necessary languages only
- **Optimize Images**: Resize large images appropriately
- **Check Resources**: Monitor CPU and memory usage
- **Batch Processing**: Process multiple images efficiently

### Common Errors
- **File Not Found**: Verify image file path and permissions
- **Unsupported Format**: Check image format compatibility
- **Language Not Available**: Verify language code is supported
- **Memory Issues**: Reduce image size or batch size

## Integration Examples

### Document Analysis Flow
```yaml
name: "document_ocr_analysis"
description: "Extract and analyze text from document images"

inputs:
  - name: "file"
    type: "file"
    required: true

steps:
  - name: "extract_text"
    executor: "ocr_processor"
    config:
      image_path: "{{ inputs.file }}"
      languages: ["en", "spa"]
      confidence_threshold: 0.8
      
  - name: "analyze_content"
    executor: "llm_analyzer"
    config:
      text: "{{ steps.extract_text.text }}"
      analysis_type: "document_summary"
```

### Multi-Language Processing
```yaml
name: "multilingual_ocr"
description: "Process documents in multiple languages"

inputs:
  - name: "file"
    type: "file"
    required: true
  - name: "target_languages"
    type: "array"
    default: ["en", "spa", "fra"]

steps:
  - name: "extract_multilingual_text"
    executor: "ocr_processor"
    config:
      image_path: "{{ inputs.file }}"
      languages: "{{ inputs.target_languages }}"
      return_bboxes: true
      confidence_threshold: 0.7
```

## API Reference

### Endpoints
- **POST** `/api/v1/ocr-analysis/execute` - Process OCR extraction
- **GET** `/api/v1/ocr-analysis/status/{flow_id}` - Check processing status
- **GET** `/api/v1/ocr-analysis/info` - Get flow information

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | required | Image file to process |
| `languages` | array | ["en"] | Languages to detect |
| `confidence_threshold` | float | 0.0 | Minimum confidence score |
| `return_bboxes` | boolean | true | Include bounding boxes |
| `return_confidence` | boolean | true | Include confidence scores |

### Response Codes
- **200**: Success - OCR processing completed
- **400**: Bad Request - Invalid parameters or file
- **422**: Unprocessable Entity - OCR processing failed
- **500**: Internal Server Error - System error

## Advanced Features

### Custom Preprocessing
Configure image preprocessing for optimal OCR results:

```yaml
preprocessing:
  - resize: { width: 1200, height: 1600 }
  - enhance_contrast: { factor: 1.2 }
  - denoise: { strength: 0.5 }
  - deskew: { auto: true }
```

### Region of Interest (ROI)
Extract text from specific image regions:

```yaml
roi:
  - name: "header"
    bbox: { x: 0, y: 0, width: 800, height: 100 }
  - name: "body"
    bbox: { x: 0, y: 100, width: 800, height: 600 }
```

### Output Formatting
Customize output format and structure:

```yaml
output_format:
  - preserve_layout: true
  - include_formatting: true
  - export_format: "structured_json"
```
