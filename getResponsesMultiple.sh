#!/bin/bash

# Define models and which query types to run for each (1=enable, 0=disable)
declare -A MODEL_CONFIGS=(
    # Format: "enabled|yesno|radio|checkbox"
    ["meta-llama/Meta-Llama-3.1-8B-Instruct"]="1|0|0|1"
    ["mistralai/Mistral-7B-Instruct-v0.1"]="1|0|0|1"
    ["Qwen/Qwen2.5-7B-Instruct"]="1|0|0|1"
)

# Base paths
QUERY_YESNO_PATH="queries_yesno.csv"
QUERY_RADIO_PATH="queries_radio.csv"
QUERY_CHECKBOX_PATH="queries_checkbox.csv"
# OUTPUT_BASE="benchmark"
OUTPUT_BASE="test_responses"

for model in "${!MODEL_CONFIGS[@]}"; do
    # Split the configuration string
    IFS='|' read -r -a config <<< "${MODEL_CONFIGS[$model]}"
    enabled="${config[0]}"
    yesno="${config[1]}"
    radio="${config[2]}"
    checkbox="${config[3]}"
    
    if [[ "$enabled" -eq 1 ]]; then
        # Extract simple model name for file paths
        model_simple=$(echo "$model" | awk -F'/' '{print $NF}' | tr '[:upper:]' '[:lower:]')
        
        echo "Running model: $model"
        echo "  Config: yesno=$yesno, radio=$radio, checkbox=$checkbox"
        
        # Check if this is Qwen model to add quantize argument
        quantize_arg=""
        if [[ "$model" == "Qwen/Qwen2.5-7B-Instruct" ]]; then
            quantize_arg="-quantize 8"
        fi
        
        # Yes/No queries
        if [[ "$yesno" -eq 1 ]]; then
            echo "  Running Yes/No queries..."
            CUDA_LAUNCH_BLOCKING=1 python3 getResponses.py \
                -model "$model" \
                -query_result_path "${OUTPUT_BASE}/yesno_responses_${model_simple}.csv" \
                -query_dataset_path "$QUERY_YESNO_PATH" \
                -qtype "yes/no" \
                $quantize_arg
        fi
        
        # Radio queries
        if [[ "$radio" -eq 1 ]]; then
            echo "  Running Radio queries..."
            CUDA_LAUNCH_BLOCKING=1 python3 getResponses.py \
                -model "$model" \
                -query_result_path "${OUTPUT_BASE}/radio_responses_${model_simple}.csv" \
                -query_dataset_path "$QUERY_RADIO_PATH" \
                -qtype "radio" \
                $quantize_arg
        fi
        
        # Checkbox queries
        if [[ "$checkbox" -eq 1 ]]; then
            echo "  Running Checkbox queries..."
            CUDA_LAUNCH_BLOCKING=1 python3 getResponses.py \
                -model "$model" \
                -query_result_path "${OUTPUT_BASE}/checkbox_responses_${model_simple}.csv" \
                -query_dataset_path "$QUERY_CHECKBOX_PATH" \
                -qtype "checkbox" \
                $quantize_arg
        fi
    fi
done