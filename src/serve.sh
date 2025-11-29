#!/bin/bash

vllm serve "output/models/unsloth/cot/gemma-3-1b-it-final_251120" \
    --gpu-memory-utilization 0.8 \
    --max-num-seqs 128 \
    --enable-chunked-prefill \
    --port 8010