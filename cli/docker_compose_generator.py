"""
Docker Compose Generator

Generates docker-compose.yml dynamically based on async flows.
Creates separate worker services per async flow for isolation.

Usage:
    # From project root:
    python cli/docker_compose_generator.py
    
    # From CLI directory:
    cd cli && python docker_compose_generator.py
    
    # With virtual environment:
    source venv/bin/activate && python cli/docker_compose_generator.py

Requirements:
    - Must be run with virtual environment activated
    - Generates docker-compose.yml and start_workers.sh in project root
"""

import yaml
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.flow_engine.flow_runner import FlowRunner
from core.flow_engine.celery_config import get_async_flows

logger = logging.getLogger(__name__)


def generate_worker_commands(flow_runner: FlowRunner) -> List[Dict[str, Any]]:
    """Generate worker commands - separate worker per async flow."""
    
    commands = []
    async_flows = get_async_flows(flow_runner)
    
    for flow_config in async_flows:
        flow_name = flow_config['name']
        max_concurrent = flow_config['max_concurrent']
        
        # Separate worker per flow
        commands.append({
            'name': f'celery-worker-{flow_name}',
            'command': f"celery -A celery_app worker --loglevel=info -Q {flow_name} --concurrency={max_concurrent}",
            'flow_name': flow_name,
            'max_concurrent': max_concurrent,
            'queue': flow_config['queue']
        })
    
    return commands


def generate_docker_compose_services(flow_runner: FlowRunner) -> Dict[str, Any]:
    """Generate docker-compose services with separate workers per flow."""
    
    base_services = {
        'app': {
            'build': '.',
            'ports': ['8000:8000'],
            'depends_on': ['redis'],
            'environment': [
                'REDIS_URL=redis://redis:6379',
                'OLLAMA_BASE_URL=http://host.docker.internal:11434'
            ],
            'volumes': [
                './logs:/app/logs',
                'temp_files:/app/temp'
            ],
            'healthcheck': {
                'test': ['CMD', 'curl', '-f', 'http://localhost:8000/health'],
                'interval': '30s',
                'timeout': '10s',
                'retries': 3
            }
        },
        'redis': {
            'image': 'redis:7-alpine',
            'ports': ['6379:6379'],
            'volumes': [
                'redis_data:/data'
            ],
            'command': 'redis-server --appendonly yes',
            'healthcheck': {
                'test': ['CMD', 'redis-cli', 'ping'],
                'interval': '30s',
                'timeout': '10s',
                'retries': 3
            }
        }
    }
    
    # Add separate worker service per async flow
    worker_commands = generate_worker_commands(flow_runner)
    
    for worker in worker_commands:
        base_services[worker['name']] = {
            'build': '.',
            'command': worker['command'],
            'depends_on': ['redis'],
            'environment': [
                'REDIS_URL=redis://redis:6379',
                f'FLOW_NAME={worker["flow_name"]}',
                f'MAX_CONCURRENT={worker["max_concurrent"]}',
                f'QUEUE_NAME={worker["queue"]}',
                'OLLAMA_BASE_URL=http://host.docker.internal:11434'
            ],
            'volumes': [
                'temp_files:/app/temp'
            ],
            'healthcheck': {
                'test': ['CMD', 'celery', '-A', 'celery_app', 'inspect', 'ping'],
                'interval': '30s',
                'timeout': '10s',
                'retries': 3
            }
        }
    
    # Add monitoring service
    base_services['celery-flower'] = {
        'build': '.',
        'command': 'celery -A celery_app flower --port=5555',
        'ports': ['5555:5555'],
        'depends_on': ['redis'],
        'environment': [
            'REDIS_URL=redis://redis:6379'
        ],
        'healthcheck': {
            'test': ['CMD', 'curl', '-f', 'http://localhost:5555'],
            'interval': '30s',
            'timeout': '10s',
            'retries': 3
        }
    }
    
    return base_services


def generate_docker_compose_config(flow_runner: FlowRunner) -> Dict[str, Any]:
    """Generate complete docker-compose configuration."""
    
    services = generate_docker_compose_services(flow_runner)
    
    config = {
        'version': '3.8',
        'services': services,
        'volumes': {
            'redis_data': {},
            'temp_files': {}
        },
        'networks': {
            'default': {
                'name': 'ai-inference-platform'
            }
        }
    }
    
    return config


def update_docker_compose(flow_runner: FlowRunner, output_path: Path = None) -> bool:
    """
    Update docker-compose.yml based on current flows.
    
    Args:
        flow_runner: FlowRunner instance to discover flows from
        output_path: Path to write docker-compose.yml (default: ./docker-compose.yml)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if output_path is None:
            output_path = Path('docker-compose.yml')
        
        # Generate configuration
        compose_config = generate_docker_compose_config(flow_runner)
        
        # Write to file
        with open(output_path, 'w') as f:
            yaml.dump(compose_config, f, default_flow_style=False, sort_keys=False)
        
        # Log summary
        services = compose_config['services']
        worker_services = [name for name in services.keys() if name.startswith('celery-worker-')]
        
        logger.info(f"📝 Updated {output_path} with {len(services)} services:")
        logger.info(f"  - app: Main application")
        logger.info(f"  - redis: Redis broker/backend/state store")
        for worker in worker_services:
            flow_name = worker.replace('celery-worker-', '')
            logger.info(f"  - {worker}: Dedicated worker for {flow_name} flow")
        logger.info(f"  - celery-flower: Monitoring dashboard")
        
        logger.info(f"✅ Docker Compose configuration updated successfully")
        logger.info(f"💡 Run 'docker-compose up -d' to start all services")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to update docker-compose.yml: {e}")
        return False


def generate_worker_startup_script(flow_runner: FlowRunner) -> str:
    """Generate a startup script for running workers locally."""
    
    async_flows = get_async_flows(flow_runner)
    
    script_lines = [
        "#!/bin/bash",
        "# Auto-generated worker startup script",
        "# Start separate Celery workers for each async flow",
        "",
        "echo '🚀 Starting Celery workers for async flows...'",
        ""
    ]
    
    for flow_config in async_flows:
        flow_name = flow_config['name']
        max_concurrent = flow_config['max_concurrent']
        queue = flow_config['queue']
        
        script_lines.extend([
            f"echo '📋 Starting worker for {flow_name} flow...'",
            f"celery -A celery_app worker --loglevel=info -Q {queue} --concurrency={max_concurrent} --detach",
            ""
        ])
    
    script_lines.extend([
        "echo '🌸 Starting Celery Flower monitoring...'",
        "celery -A celery_app flower --detach",
        "",
        "echo '✅ All workers started!'",
        "echo '📊 Monitor at: http://localhost:5555'",
        "echo '🛑 Stop with: pkill -f celery'"
    ])
    
    return "\n".join(script_lines)


def create_worker_startup_script(flow_runner: FlowRunner, output_path: Path = None) -> bool:
    """Create a startup script for local development."""
    
    try:
        if output_path is None:
            output_path = Path('start_workers.sh')
        
        script_content = generate_worker_startup_script(flow_runner)
        
        with open(output_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        output_path.chmod(0o755)
        
        logger.info(f"📝 Created worker startup script: {output_path}")
        logger.info(f"💡 Run './{output_path}' to start workers locally")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create startup script: {e}")
        return False


async def main():
    """Test the docker-compose generator."""
    
    print("🧪 Testing Docker Compose Generator...")
    
    # Find project root directory (where flows directory is located)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    flows_dir = project_root / "flows"
    
    # Initialize flow runner with correct flows directory
    flow_runner = FlowRunner(flows_dir)
    await flow_runner.initialize()
    
    try:
        # Generate docker-compose.yml in project root
        compose_path = project_root / "docker-compose.yml"
        success = update_docker_compose(flow_runner, compose_path)
        
        if success:
            print("✅ Docker Compose generation successful")
        else:
            print("❌ Docker Compose generation failed")
        
        # Generate startup script in project root
        script_path = project_root / "start_workers.sh"
        script_success = create_worker_startup_script(flow_runner, script_path)
        
        if script_success:
            print("✅ Worker startup script generation successful")
        else:
            print("❌ Worker startup script generation failed")
            
    finally:
        await flow_runner.stop()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
