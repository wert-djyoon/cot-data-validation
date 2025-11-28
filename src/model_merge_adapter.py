from pathlib import Path


# Data
ROOT_DIR = Path("/home/work/djyoon/project/cot-data-validation")
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
VERSION = "final_251120"
INPUT_DATA_PATH = DATA_DIR / "가공데이터"
OUTPUT_DATA_PATH = DATA_DIR / VERSION

# Training
BASE_MODEL = "unsloth/gemma-3-1b-it"
TOKENIZER_MODEL = BASE_MODEL
BASE_MODEL_NAME = BASE_MODEL.split("/")[1]
EXP_NAME = f"cot/{BASE_MODEL_NAME}-{VERSION}-v4"
OUTPUT_MODEL_PATH = OUTPUT_DIR / "models" / "unsloth" / EXP_NAME
SEED = 42

from glob import glob
from os.path import exists

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ---------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------
print(f"1. 베이스 모델 로드 중: {BASE_MODEL}")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, dtype=torch.bfloat16, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

adapter_paths = [p for p in sorted(glob(f"{OUTPUT_MODEL_PATH}/checkpoint-*"))]
for adapter_path in adapter_paths:
    save_directory = f"{OUTPUT_MODEL_PATH}-{adapter_path.split('/')[-1]}"
    if exists(save_directory):
        print("✅ 이미 병합된 모델이 존재합니다. vLLM에서 이 경로를 사용하세요.")
    else:
        print(f"2. LoRA 어댑터 로드 중: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)

        print("3. 모델 병합 중 (Merge and Unload)...")
        model = model.merge_and_unload()

        print(f"4. 병합된 모델 저장 중: {save_directory}")
        model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)

        print("✅ 저장 완료! vLLM에서 이 경로를 사용하세요.")