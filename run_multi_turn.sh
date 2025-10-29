#!/bin/bash


GENERATION_MODEL=$1
URL=$2
GEN_URL=$URL
EVALUATION_MODEL="gemini-2.5-pro"
DATA_DIR="data"
RESULTS_DIR="results_multi_round"
GENERATION_PROMPT="prompts/benchmark_generate_prompt.txt"
EVALUATION_PROMPT="prompts/eval.txt"


GENERATION_BASE_URL=""
EVALUATION_BASE_URL=""

API_KEY=""
API_KEY_GEN=""



python -u evaluate_multi_turn.py \
  --mode both \
  --generation_model_name "$GENERATION_MODEL" \
  --evaluation_model_name "$EVALUATION_MODEL" \
  --generation_base_url "$GEN_URL" \
  --evaluation_base_url "$EVALUATION_BASE_URL" \
  --generation_api_key "$API_KEY_GEN" \
  --evaluation_api_key "$API_KEY" \
  --data_dir "$DATA_DIR" \
  --results_dir "$RESULTS_DIR" \
  --generation_prompt_path "$GENERATION_PROMPT" \
  --evaluation_prompt_path "$EVALUATION_PROMPT" \
  --max_data_rows 5 \
  --timeout 120 \
  --max_retries 20 \
  --max_qps 20
