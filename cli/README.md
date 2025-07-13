# CLI Tools

Command-line tools for managing and operating the AI Inference Platform.

## Available Tools

| Tool | Description | Usage |
|------|-------------|-------|
| **Flow Generator** | Generate YAML workflows from natural language | `cli/generate_flow.sh "Extract text and analyze sentiment"` |
| **Flow Validator** | Validate flow definitions and check syntax | `python cli/validate_flows.py` |
| **Flow Runner** | Execute flows locally for testing | `python cli/run_flow.py flow_name '{"input": "data"}'` |
| **Docker Generator** | Generate container infrastructure dynamically | `python cli/docker_compose_generator.py` |

