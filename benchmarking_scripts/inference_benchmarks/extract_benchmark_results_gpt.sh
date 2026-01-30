#!/bin/bash

# Extract benchmark results from llama-3.3 tp8 JSON files
# Output format: input,output,reqs,tokens

DIR="${1:-~/dkennetz}"

for file in "${DIR}"/gpt-oss*tp8*.json; do
    if [ ! -f "$file" ]; then
        continue
    fi
    
    basename=$(basename "$file" .json)
    input=$(echo "$basename" | grep -oE '[0-9]+-[0-9]+-tp8' | cut -d'-' -f1)
    output=$(echo "$basename" | grep -oE '[0-9]+-[0-9]+-tp8' | cut -d'-' -f2)
    reqs=$(jq -r '.requests_per_second' "$file")
    tokens=$(jq -r '.tokens_per_second' "$file")
    
    echo "${input},${output},${reqs},${tokens}"
done | sort -t, -k1,1n -k2,2n
