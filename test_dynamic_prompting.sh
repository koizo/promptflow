#!/bin/bash

# Test script for dynamic prompting in OCR analysis flow
echo "🧪 Testing Dynamic Prompting in OCR Analysis Flow"
echo "=================================================="

BASE_URL="http://localhost:8000/api/v1/ocr-analysis/execute"
IMAGE_FILE="examples/ocr.jpg"

echo ""
echo "1️⃣  Testing SUMMARY analysis..."
curl -s -X POST $BASE_URL \
  -F "file=@$IMAGE_FILE" \
  -F "analysis_type=summary" \
  -F "ocr_provider=tesseract" \
  -F "llm_model=mistral" | jq -r '.analysis' | head -3
echo ""

echo "2️⃣  Testing KEY_POINTS analysis..."
curl -s -X POST $BASE_URL \
  -F "file=@$IMAGE_FILE" \
  -F "analysis_type=key_points" \
  -F "ocr_provider=tesseract" \
  -F "llm_model=mistral" | jq -r '.analysis' | head -3
echo ""

echo "3️⃣  Testing ENTITIES analysis..."
curl -s -X POST $BASE_URL \
  -F "file=@$IMAGE_FILE" \
  -F "analysis_type=entities" \
  -F "ocr_provider=tesseract" \
  -F "llm_model=mistral" | jq -r '.analysis' | head -3
echo ""

echo "4️⃣  Testing COMPREHENSIVE analysis..."
curl -s -X POST $BASE_URL \
  -F "file=@$IMAGE_FILE" \
  -F "analysis_type=comprehensive" \
  -F "ocr_provider=tesseract" \
  -F "llm_model=mistral" | jq -r '.analysis' | head -5
echo ""

echo "5️⃣  Testing CUSTOM analysis with custom prompt..."
curl -s -X POST $BASE_URL \
  -F "file=@$IMAGE_FILE" \
  -F "analysis_type=custom" \
  -F "custom_prompt=Analyze this text as if you were a linguistics professor. What can you tell me about the language structure and word choice?" \
  -F "ocr_provider=tesseract" \
  -F "llm_model=mistral" | jq -r '.analysis' | head -5
echo ""

echo "6️⃣  Testing CUSTOM analysis without custom prompt (fallback)..."
curl -s -X POST $BASE_URL \
  -F "file=@$IMAGE_FILE" \
  -F "analysis_type=custom" \
  -F "ocr_provider=tesseract" \
  -F "llm_model=mistral" | jq -r '.analysis' | head -3
echo ""

echo "✅ Dynamic prompting tests completed!"
echo ""
echo "📋 Available analysis types:"
echo "   - summary: Concise summary of the text"
echo "   - key_points: Key points extraction"
echo "   - entities: Named entity recognition"
echo "   - comprehensive: Full analysis with multiple aspects"
echo "   - custom: Use your own custom prompt"
echo ""
echo "🎯 Custom prompting allows complete flexibility in how the LLM analyzes the extracted text!"
