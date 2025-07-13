# PromptFlow Architecture

Comprehensive guide to the PromptFlow platform architecture, system design, and component interactions.

## Overview

PromptFlow is built as a distributed, microservices-based AI inference platform that orchestrates AI workflows through YAML configuration. The architecture emphasizes scalability, modularity, and extensibility while maintaining simplicity for end users.

## High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   YAML Flows    │───▶│  Flow Engine    │───▶│   Executors     │
│                 │    │                 │    │                 │
│ • Flow Configs  │    │ • Orchestration │    │ • AI Processing │
│ • Input Schemas │    │ • State Mgmt    │    │ • Multi-Provider│
│ • Step Defs     │    │ • Template Eng  │    │ • Model Caching │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Auto-Generated │    │   Async Queue   │    │  State Storage  │
│   REST APIs     │    │                 │    │                 │
│                 │    │ • Celery Tasks  │    │ • Redis Cache   │
│ • FastAPI       │    │ • Worker Pools  │    │ • Execution     │
│ • OpenAPI Docs  │    │ • Load Balance  │    │ • Progress      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Core Components

### 1. Flow Engine
The central orchestration component that manages workflow execution.

**Location**: `core/flow_engine/`

**Key Components**:
- **FlowRunner** - Main execution coordinator
- **TemplateEngine** - Jinja2-based variable resolution
- **StateManager** - Execution state tracking
- **StepExecutor** - Individual step processing

**Responsibilities**:
- Parse and validate YAML flow definitions
- Resolve template variables and dependencies
- Coordinate step execution order
- Handle error recovery and retries
- Track execution progress and state

```python
# Flow execution lifecycle
class FlowRunner:
    async def execute_flow(self, flow_config, inputs):
        # 1. Validate flow configuration
        flow = await self._validate_flow(flow_config)
        
        # 2. Create execution context
        context = self._create_context(flow, inputs)
        
        # 3. Execute steps in order
        for step in flow.steps:
            result = await self._execute_step(step, context)
            context.update_step_result(step.name, result)
        
        # 4. Return final results
        return self._format_results(context)
```

### 2. Executor Registry
Manages available AI processing components and their lifecycle.

**Location**: `core/executors/`

**Key Components**:
- **BaseExecutor** - Abstract base class for all executors
- **ExecutorRegistry** - Registry and factory for executors
- **ExecutionResult** - Standardized result format

**Executor Types**:
- **AI Processing**: OCR, Speech, Sentiment, Vision
- **File Handling**: Upload, validation, conversion
- **Data Processing**: Transformation, combination, formatting
- **External APIs**: Third-party service integrations

```python
# Executor registration and instantiation
class ExecutorRegistry:
    def __init__(self):
        self.executors = {}
        self._register_built_in_executors()
    
    def register(self, name: str, executor_class: Type[BaseExecutor]):
        self.executors[name] = executor_class
    
    def get_executor(self, name: str) -> BaseExecutor:
        if name not in self.executors:
            raise ValueError(f"Unknown executor: {name}")
        return self.executors[name]()
```

### 3. API Layer
FastAPI-based REST API that auto-generates endpoints from flow definitions.

**Location**: `main.py`

**Key Features**:
- **Auto-Generated Endpoints** - One endpoint per flow
- **OpenAPI Documentation** - Automatic API docs
- **Request Validation** - Pydantic-based validation
- **File Upload Support** - Multipart form handling
- **Async Processing** - Non-blocking request handling

**Endpoint Structure**:
```
POST /api/v1/{flow-name}/execute     # Execute flow
GET  /api/v1/{flow-name}/status/{id} # Check execution status
GET  /api/v1/{flow-name}/info        # Flow information
GET  /api/v1/{flow-name}/health      # Health check
```

### 4. Async Processing Layer
Celery-based distributed task processing for scalable execution.

**Location**: `celery_app.py`

**Components**:
- **Celery Workers** - Distributed task processors
- **Redis Broker** - Message queue and result backend
- **Task Routing** - Queue-based load balancing
- **Worker Monitoring** - Flower dashboard

**Worker Architecture**:
```yaml
# Specialized workers for different AI capabilities
celery-worker-ocr:           # OCR processing
celery-worker-speech:        # Speech transcription
celery-worker-sentiment:     # Sentiment analysis
celery-worker-vision:        # Image classification
celery-worker-general:       # General processing
```

### 5. State Management
Redis-based state storage for execution tracking and caching.

**Location**: `core/state_store.py`

**Storage Types**:
- **Execution State** - Flow progress and results
- **Model Cache** - AI model instances and weights
- **Session Data** - Temporary execution context
- **Configuration Cache** - Parsed flow definitions

## Data Flow Architecture

### 1. Request Processing Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Client    │───▶│  FastAPI    │───▶│   Celery    │───▶│   Worker    │
│   Request   │    │   Server    │    │   Broker    │    │   Process   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │                   │
       │                   ▼                   ▼                   ▼
       │            ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
       │            │   Request   │    │    Task     │    │    Flow     │
       │            │ Validation  │    │   Queue     │    │  Execution  │
       │            └─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Response  │◀───│   Task ID   │◀───│   Redis     │◀───│  Executors  │
│   (Async)   │    │  (Immediate)│    │   State     │    │ Processing  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### 2. Execution State Lifecycle

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Queued    │───▶│  Running    │───▶│  Completed  │    │   Failed    │
│             │    │             │    │             │    │             │
│ • Task ID   │    │ • Progress  │    │ • Results   │    │ • Error     │
│ • Timestamp │    │ • Current   │    │ • Metadata  │    │ • Stack     │
│ • Config    │    │   Step      │    │ • Timing    │    │ • Retry     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Redis State Store                           │
│                                                                     │
│ execution:{id} = {                                                  │
│   status: "running",                                                │
│   progress: { current_step: "analyze", percentage: 60 },           │
│   results: { step1: {...}, step2: {...} },                        │
│   metadata: { started_at: "...", worker: "..." }                   │
│ }                                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Interactions

### 1. Flow Definition to API Generation

```python
# 1. YAML Flow Definition
flow_config = {
    "name": "sentiment_analysis",
    "inputs": [
        {"name": "text", "type": "string", "required": True}
    ],
    "steps": [
        {
            "name": "analyze",
            "executor": "sentiment_analyzer",
            "config": {"text": "{{ inputs.text }}"}
        }
    ]
}

# 2. API Endpoint Auto-Generation
@app.post("/api/v1/sentiment-analysis/execute")
async def execute_sentiment_analysis(
    text: str = Form(...),
    execution_mode: str = Form("async")
):
    # Validate inputs against flow schema
    inputs = {"text": text}
    
    # Queue execution task
    task = execute_flow_async.delay("sentiment_analysis", inputs)
    
    # Return task ID for status tracking
    return {"task_id": task.id, "status": "queued"}

# 3. Celery Task Execution
@celery_app.task
def execute_flow_async(flow_name: str, inputs: dict):
    # Load flow configuration
    flow = flow_registry.get_flow(flow_name)
    
    # Execute with flow engine
    runner = FlowRunner()
    result = await runner.execute_flow(flow, inputs)
    
    # Store results in Redis
    state_store.set_execution_result(task.id, result)
    
    return result
```

### 2. Template Resolution and Variable Binding

```python
# Template resolution process
class TemplateEngine:
    def resolve_config(self, config: dict, context: dict) -> dict:
        """
        Resolve Jinja2 templates in configuration.
        
        Context includes:
        - inputs: User-provided inputs
        - steps: Previous step results
        - system: System variables
        """
        template_str = json.dumps(config)
        template = self.jinja_env.from_string(template_str)
        resolved_str = template.render(**context)
        return json.loads(resolved_str)

# Example resolution:
config = {
    "text": "{{ inputs.text }}",
    "provider": "{{ inputs.provider | default('huggingface') }}",
    "confidence": "{{ steps.preprocess.confidence | default(0.8) }}"
}

context = {
    "inputs": {"text": "I love this!", "provider": "llm"},
    "steps": {"preprocess": {"confidence": 0.9}}
}

# Resolved config:
{
    "text": "I love this!",
    "provider": "llm", 
    "confidence": 0.9
}
```

### 3. Executor Lifecycle and Caching

```python
# Executor instantiation and caching
class ExecutorManager:
    def __init__(self):
        self.executor_cache = {}
        self.model_cache = {}
    
    async def execute_step(self, step_config: dict, context: dict):
        executor_name = step_config["executor"]
        
        # Get or create executor instance
        if executor_name not in self.executor_cache:
            executor_class = self.registry.get_executor(executor_name)
            self.executor_cache[executor_name] = executor_class()
        
        executor = self.executor_cache[executor_name]
        
        # Execute with resolved configuration
        resolved_config = self.template_engine.resolve_config(
            step_config["config"], context
        )
        
        return await executor.execute(context, resolved_config)
```

## Scalability Architecture

### 1. Horizontal Scaling

**Worker Scaling**:
```yaml
# Scale workers based on queue depth
services:
  celery-worker-sentiment:
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
```

**Load Balancing**:
```python
# Queue-based load balancing
CELERY_ROUTES = {
    'execute_ocr_flow': {'queue': 'ocr_processing'},
    'execute_speech_flow': {'queue': 'speech_processing'},
    'execute_sentiment_flow': {'queue': 'sentiment_processing'},
    'execute_vision_flow': {'queue': 'vision_processing'},
}
```

### 2. Vertical Scaling

**Resource Optimization**:
- **CPU-Intensive**: OCR, Speech processing
- **Memory-Intensive**: Large language models
- **GPU-Accelerated**: Vision models, ML inference

**Model Caching Strategy**:
```python
# Multi-level caching
class ModelCache:
    def __init__(self):
        self.memory_cache = {}      # In-process cache
        self.redis_cache = Redis()  # Shared cache
        self.disk_cache = Path()    # Persistent cache
    
    async def get_model(self, model_key: str):
        # 1. Check memory cache (fastest)
        if model_key in self.memory_cache:
            return self.memory_cache[model_key]
        
        # 2. Check Redis cache (fast)
        cached_model = await self.redis_cache.get(model_key)
        if cached_model:
            model = self.deserialize_model(cached_model)
            self.memory_cache[model_key] = model
            return model
        
        # 3. Load from disk/download (slow)
        model = await self.load_model(model_key)
        await self.cache_model(model_key, model)
        return model
```

## Security Architecture

### 1. Input Validation and Sanitization

```python
# Multi-layer validation
class SecurityManager:
    def validate_request(self, flow_name: str, inputs: dict):
        # 1. Schema validation
        flow_schema = self.get_flow_schema(flow_name)
        self.validate_against_schema(inputs, flow_schema)
        
        # 2. Content validation
        self.validate_content_safety(inputs)
        
        # 3. Rate limiting
        self.check_rate_limits(request.client.host)
        
        # 4. File validation
        for key, value in inputs.items():
            if isinstance(value, UploadFile):
                self.validate_file_upload(value)
```

### 2. Execution Isolation

**Process Isolation**:
- Each worker runs in separate process
- Containerized execution environment
- Resource limits and quotas

**Data Isolation**:
- Temporary file cleanup
- Memory management
- Secure credential handling

### 3. API Security

```python
# Security middleware
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    # CORS handling
    if request.method == "OPTIONS":
        return handle_cors_preflight(request)
    
    # Rate limiting
    await rate_limiter.check_rate_limit(request.client.host)
    
    # Input validation
    if request.method == "POST":
        await validate_request_size(request)
    
    response = await call_next(request)
    return add_security_headers(response)
```

## Monitoring and Observability

### 1. Metrics Collection

**System Metrics**:
- Request throughput and latency
- Worker queue depths and processing times
- Resource utilization (CPU, memory, GPU)
- Error rates and failure patterns

**Business Metrics**:
- Flow execution success rates
- AI model accuracy and confidence
- Processing volume by capability
- User adoption and usage patterns

### 2. Logging Architecture

```python
# Structured logging
import structlog

logger = structlog.get_logger()

class ExecutionLogger:
    def log_execution_start(self, flow_id: str, flow_name: str):
        logger.info(
            "Flow execution started",
            flow_id=flow_id,
            flow_name=flow_name,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def log_step_completion(self, flow_id: str, step_name: str, duration: float):
        logger.info(
            "Step completed",
            flow_id=flow_id,
            step_name=step_name,
            duration_seconds=duration,
            timestamp=datetime.utcnow().isoformat()
        )
```

### 3. Health Monitoring

```python
# Health check endpoints
@app.get("/health")
async def health_check():
    checks = {
        "api": await check_api_health(),
        "redis": await check_redis_health(),
        "celery": await check_celery_health(),
        "workers": await check_worker_health()
    }
    
    overall_status = "healthy" if all(checks.values()) else "unhealthy"
    
    return {
        "status": overall_status,
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat()
    }
```

## Deployment Architecture

### 1. Container Orchestration

```yaml
# docker-compose.yml structure
version: '3.8'
services:
  # API Gateway
  app:
    build: .
    ports: ["8000:8000"]
    depends_on: [redis]
    
  # Message Broker & Cache
  redis:
    image: redis:alpine
    ports: ["6379:6379"]
    
  # Specialized Workers
  celery-worker-ocr:
    build: .
    command: celery worker --queues=ocr_processing
    depends_on: [redis]
    
  celery-worker-speech:
    build: .
    command: celery worker --queues=speech_processing
    depends_on: [redis]
    
  # Monitoring
  celery-flower:
    build: .
    command: celery flower
    ports: ["5555:5555"]
    depends_on: [redis]
```

### 2. Production Considerations

**High Availability**:
- Redis clustering for state persistence
- Load balancer for API endpoints
- Worker auto-scaling based on queue depth

**Data Persistence**:
- Model cache persistence across restarts
- Execution history and audit logs
- Configuration backup and versioning

**Disaster Recovery**:
- Automated backups of Redis state
- Model cache reconstruction procedures
- Graceful degradation strategies

## Extension Points

### 1. Adding New AI Capabilities

```python
# 1. Create new executor
class CustomAIExecutor(BaseExecutor):
    async def execute(self, context, config):
        # Implement AI processing logic
        return ExecutionResult(success=True, outputs=result)

# 2. Register executor
executor_registry.register("custom_ai", CustomAIExecutor)

# 3. Create flow definition
flow_config = {
    "name": "custom_ai_flow",
    "steps": [{"executor": "custom_ai", "config": {...}}]
}

# 4. API automatically available at:
# POST /api/v1/custom-ai-flow/execute
```

### 2. Custom Provider Integration

```python
# Provider abstraction pattern
class BaseProvider(ABC):
    @abstractmethod
    async def process(self, config: dict) -> dict:
        pass

class CustomProvider(BaseProvider):
    async def process(self, config: dict) -> dict:
        # Integrate with external service
        return await self.call_external_api(config)

# Multi-provider executor
class MultiProviderExecutor(BaseExecutor):
    def __init__(self):
        self.providers = {
            'custom': CustomProvider(),
            'existing': ExistingProvider()
        }
```

### 3. Custom Flow Patterns

```yaml
# Advanced flow patterns
name: "conditional_processing"
steps:
  - name: "route_decision"
    executor: "decision_maker"
    config:
      input: "{{ inputs.data }}"
      
  - name: "path_a"
    executor: "processor_a"
    condition: "{{ steps.route_decision.path == 'a' }}"
    
  - name: "path_b"
    executor: "processor_b"
    condition: "{{ steps.route_decision.path == 'b' }}"
```

## Performance Optimization

### 1. Caching Strategies

**Multi-Level Caching**:
- L1: In-memory executor cache
- L2: Redis shared cache
- L3: Persistent disk cache

**Cache Invalidation**:
- TTL-based expiration
- Version-based invalidation
- Manual cache clearing

### 2. Resource Management

**Memory Optimization**:
- Model sharing across workers
- Lazy loading of large models
- Garbage collection tuning

**CPU Optimization**:
- Async I/O for network calls
- Parallel processing where possible
- Efficient serialization formats

### 3. Network Optimization

**Request Optimization**:
- Connection pooling
- Request batching
- Compression for large payloads

**Response Optimization**:
- Streaming for large results
- Partial response support
- Efficient serialization

This architecture provides a solid foundation for scalable, maintainable AI workflow processing while remaining flexible enough to accommodate new AI capabilities and integration patterns.
