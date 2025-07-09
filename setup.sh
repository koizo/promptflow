#!/bin/bash

# AI Inference Platform Setup Script

echo "🚀 Setting up AI Inference Platform..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "✅ Please edit .env file with your configuration"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads logs flows

# Set permissions
chmod +x setup.sh
chmod 755 uploads logs flows

# Build and start services
echo "🔨 Building Docker images..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check service health
echo "🔍 Checking service health..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ App service is running"
else
    echo "⚠️  App service might not be ready yet"
fi

if docker-compose exec redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis service is running"
else
    echo "⚠️  Redis service might not be ready yet"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Available services:"
echo "  - API Server: http://localhost:8000"
echo "  - Redis: localhost:6379"
echo "  - Redis UI: http://localhost:8081 (run 'make tools' to enable)"
echo ""
echo "🛠️  Useful commands:"
echo "  - make help    : Show all available commands"
echo "  - make logs    : View service logs"
echo "  - make shell   : Open shell in app container"
echo "  - make down    : Stop all services"
echo ""
