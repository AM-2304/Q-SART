#!/bin/bash

echo "============================================================"
echo "🚀 RUNNING ACADEMIC Q-SART EVALUATION PIPELINE"
echo "============================================================"

# List of 14 highly diverse models for academic research
# Encompasses Groq production models, OpenAI-family models (via Groq), 
# Mistral family (via Hugging Face), and local Gemma4 NPC models.
models=(
    # Local Gemma4 NPC Models (Update these paths to where you downloaded the GGUFs)
    # "models/Gemma4NPC-it-DPO-GGUF/merged_float16.Q4_K_M.gguf"
    # "models/Gemma4NPC-E4B-Q4_K_M.gguf"
    # "models/Gemma4NPC-E2B-Q4_K_M.gguf"

    # GROQ Production Models
    "groq/llama-3.1-8b-instant"
    "groq/llama-3.3-70b-versatile"
    "groq/meta-llama/llama-4-scout-17b-16e-instruct"
    "groq/allam-2-7b"
    "groq/qwen/qwen3-32b"

    # OpenAI-Family Models (Hosted on Groq)
    "groq/openai/gpt-oss-120b"
    "groq/openai/gpt-oss-20b"
    "groq/openai/gpt-oss-safeguard-20b"

    # Mistral-Family Models (via Hugging Face Serverless API)
    "huggingface/mistralai/Mistral-7B-Instruct-v0.3"
    "huggingface/mistralai/Mixtral-8x7B-Instruct-v0.1"

    # Hugging Face Qwen Model
    "huggingface/Qwen/Qwen2.5-7B-Instruct"
)

mkdir -p experiments/results/

for model in "${models[@]}"; do
    echo "------------------------------------------------------------"
    echo "Evaluating Model: $model"
    echo "------------------------------------------------------------"
    
    # Generate clean name for output file
    safe_name=$(echo "$model" | awk -F/ '{print $NF}' | tr ':' '_')
    output_file="experiments/results/${safe_name}_scores.json"
    
    python3 src/evaluate.py --model "$model" --tasks "data/tasks.json" --output "$output_file"
done

echo "============================================================"
echo "🏁 All Q-SART evaluations completed! Results saved in experiments/results/"
echo "============================================================"
