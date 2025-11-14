#!/bin/bash

# Maximum number of super epochs
N=100   # <-- change this to whatever value you want

MODEL_BASE="models/Llama-3.2-1B-Instruct-r128-ccf-squad"

for ((i=1; i<=N; i++)); do
    MODEL_PATH="${MODEL_BASE}/super_epoch_${i}"
    echo "Running super epoch ${i}..."
    
    python3 src/generation_text.py \
        --model "$MODEL_PATH" \
        --is_adapter
    
done

echo "Done!"
