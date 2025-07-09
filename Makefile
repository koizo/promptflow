.PHONY: help build up down logs shell redis-cli clean restart dev prod

# Default target
help:
	@echo "Available commands:"
	@echo "  build     - Build the Docker images"
	@echo "  up        - Start all services"
	@echo "  down      - Stop all services"
	@echo "  logs      - Show logs from all services"
	@echo "  shell     - Open shell in the app container"
	@echo "  redis-cli - Open Redis CLI"
	@echo "  clean     - Remove all containers, images, and volumes"
	@echo "  restart   - Restart all services"
	@echo "  dev       - Start development environment"
	@echo "  prod      - Start production environment"
	@echo "  tools     - Start with Redis Commander UI"

# Build Docker images
build:
	docker-compose build

# Start all services
up:
	docker-compose up -d

# Stop all services
down:
	docker-compose down

# Show logs
logs:
	docker-compose logs -f

# Open shell in app container
shell:
	docker-compose exec app /bin/bash

# Open Redis CLI
redis-cli:
	docker-compose exec redis redis-cli

# Clean up everything
clean:
	docker-compose down -v --rmi all --remove-orphans
	docker system prune -f

# Restart services
restart: down up

# Development environment (with file watching)
dev:
	docker-compose up --build

# Production environment
prod:
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Start with tools (Redis Commander)
tools:
	docker-compose --profile tools up -d

# Check service health
health:
	docker-compose ps
	@echo "\nHealth checks:"
	@curl -s http://localhost:8000/health || echo "App: Not ready"
	@docker-compose exec redis redis-cli ping || echo "Redis: Not ready"
