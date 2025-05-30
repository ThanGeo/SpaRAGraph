#!/bin/bash

# Get the absolute path to the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# RESPONSES_DIR="$SCRIPT_DIR/benchmark_external_reasoning/1st-run"
# RESPONSES_DIR="$SCRIPT_DIR/benchmark_external_reasoning/2nd-run"
# RESPONSES_DIR="$SCRIPT_DIR/benchmark_external_reasoning/3rd-run"
RESPONSES_DIR="$SCRIPT_DIR/benchmark_external_reasoning"


# RESPONSES_DIR="$SCRIPT_DIR/benchmark_plain_llm/1st-run"
# RESPONSES_DIR="$SCRIPT_DIR/benchmark_plain_llm/2nd-run"
# RESPONSES_DIR="$SCRIPT_DIR/benchmark_plain_llm/3rd-run"


# RESPONSES_DIR="$SCRIPT_DIR/benchmark_internal_reasoning"


SRC_DIR="$SCRIPT_DIR/../../benchmark/spatial-inference-benchmark/src"

echo $RESPONSES_DIR

# Process all yesno files (BINARY)
for file in "$RESPONSES_DIR"/yesno_*.csv; do
    if [ -f "$file" ]; then
        model=$(echo "$file" | sed -E 's/.*yesno_responses_(.*)\.csv/\1/')
        echo "Evaluating model: $model (BINARY)"
        python3 "$SRC_DIR/evaluate.py" -response_path "$file" -label_type BINARY
        echo ""
    fi
done

# Process all radio files (MULTICLASS)
for file in "$RESPONSES_DIR"/radio_*.csv; do
    if [ -f "$file" ]; then
        model=$(echo "$file" | sed -E 's/.*radio_responses_(.*)\.csv/\1/')
        echo "Evaluating model: $model (MULTICLASS)"
        python3 "$SRC_DIR/evaluate.py" -response_path "$file" -label_type MULTICLASS
        echo ""
    fi
done

# Process all checkbox files (MULTILABEL)
for file in "$RESPONSES_DIR"/checkbox_*.csv; do
    if [ -f "$file" ]; then
        model=$(echo "$file" | sed -E 's/.*checkbox_responses_(.*)\.csv/\1/')
        echo "Evaluating model: $model (MULTILABEL)"
        python3 "$SRC_DIR/evaluate.py" -response_path "$file" -label_type MULTILABEL
        echo ""
    fi
done

echo "All evaluations completed!"
