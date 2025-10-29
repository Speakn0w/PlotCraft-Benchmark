# PlotCraft

A benchmark for evaluating LLM's ability to generate data visualization code. Supports both single-turn and multi-turn evaluation modes.

## Requirements

- Python 3.13
- OS: Linux

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Set your API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or edit the shell scripts (`run_single_turn.sh` / `run_multi_turn.sh`) to set API credentials directly.

## Usage

### Single-Turn Evaluation

```bash
bash run_single_turn.sh <MODEL_NAME> <API_URL>

# Example
bash run_single_turn.sh gpt-4o https://api.openai.com/v1
```

### Multi-Turn Evaluation

```bash
bash run_multi_turn.sh <MODEL_NAME> <API_URL>

# Example
bash run_multi_turn.sh gpt-4o https://api.openai.com/v1
```

### Python Command

```bash
# Single-turn
python evaluate_single_turn.py \
  --generation_model_name "gpt-4o" \
  --evaluation_model_name "gemini-2.5-pro" \
  --data_dir "data" \
  --results_dir "results_single_turn"

# Multi-turn
python evaluate_multi_turn.py \
  --generation_model_name "gpt-4o" \
  --evaluation_model_name "gemini-2.5-pro" \
  --data_dir "data" \
  --results_dir "results_multi_turn"
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--generation_model_name` | Model name for code generation | `gpt-4o` |
| `--evaluation_model_name` | Model name for evaluation | `gpt-4o` |
| `--data_dir` | Data directory path | `data` |
| `--results_dir` | Results output directory | `results` |
| `--difficulty` | Difficulty level (simple/middle/hard) | All |
| `--timeout` | Code execution timeout (seconds) | `120` |
| `--max_retries` | Maximum retry count | `5` |
| `--max_qps` | Maximum requests per second | `20` |

## Output Files

Results are saved in the specified results directory:

- `generated_code_*.py` - Generated Python code
- `generated_plot_*.png` - Generated visualization
- `score_*.json` - Evaluation scores
- `statistics.json` - Overall statistics
- `*_cache.jsonl` - API request cache

## Evaluation Metrics

### Task Compliance
- Layout Compliance
- Chart Type Compliance
- Visualization Requirement Fulfillment
- Complete Task Fulfillment

### Chart Quality
- Clarity (No Overlap)
- Layout Quality
- Color Quality
- Text Clarity
- Formatting and Professional Standards

