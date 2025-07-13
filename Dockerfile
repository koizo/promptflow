# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including Tesseract with proper language data
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-por \
    tesseract-ocr-fra \
    tesseract-ocr-deu \
    tesseract-ocr-spa \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libreoffice \
    pandoc \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set Tesseract environment variables properly
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata/
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Verify Tesseract installation and language data
RUN tesseract --version && \
    tesseract --list-langs && \
    ls -la /usr/share/tesseract-ocr/5/tessdata/

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p flows logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
