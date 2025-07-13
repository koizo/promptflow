#!/bin/bash
# Auto-generated worker startup script
# Start separate Celery workers for each async flow

echo '🚀 Starting Celery workers for async flows...'

echo '📋 Starting worker for ocr_analysis flow...'
celery -A celery_app worker --loglevel=info -Q ocr_analysis --concurrency=2 --detach

echo '🌸 Starting Celery Flower monitoring...'
celery -A celery_app flower --detach

echo '✅ All workers started!'
echo '📊 Monitor at: http://localhost:5555'
echo '🛑 Stop with: pkill -f celery'