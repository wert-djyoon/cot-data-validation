"""
Version: 4
Model: unsloth/gemma-3-1b-it
Training: 16bit LoRA (r=8, alpha=8)
"""

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


import json
import random
from glob import glob
from pathlib import Path
from os.path import exists

from datasets import Dataset, DatasetDict, load_from_disk


SPECIAL_ANS = "answer"
TEST_SIZE = 0.1  # train:val:test = 8:1:1


def generate_dataset(
    input_path: str | Path, output_path: str | Path, test_size: float = TEST_SIZE
) -> DatasetDict:
    """Generate dataset from source data path"""
    if exists(output_path):
        return load_from_disk(output_path)
    else:
        random.seed(SEED)
        random.shuffle(data)

        data = load_json(input_path)
        n = len(data)
        ds = DatasetDict(
            {
                "train": Dataset.from_list(data[int(n * 2 * test_size) :]),
                "val": Dataset.from_list(
                    data[int(n * test_size) : int(n * 2 * test_size)]
                ),
                "test": Dataset.from_list(data[: int(n * test_size)]),
            }
        )

        # Generate qa set
        for id in ds:
            ds[id] = ds[id].map(
                generate_pair, remove_columns=["instruction", "context", "target_steps"]
            )

        ds.save_to_disk(output_path)
        print(ds)
        print(f"Dataset is saved to {output_path}.")

    return ds


def load_json(base_path: Path | str) -> list[dict]:
    """base_path 하위 모든 폴더를 재귀 탐색하여 JSON 파일 로드"""
    json_files = glob(f"{base_path}/**/*.json", recursive=True)
    data = []
    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                data.extend([build_record(o) for o in obj])
            else:
                data.append(build_record(obj))
        except Exception as e:
            print(f"⚠️ {f} 로드 실패: {e}")
    print(f"✅ 총 {len(data)}개 JSON 파일 로드 완료 (경로: {base_path})")
    return data


def build_record(obj):
    """단일 JSON 객체 → CoT 학습용 포맷 (가치 산출 중심, auto context generation)"""
    p = obj.get("patent_info", {})
    c = obj.get("Company_info", {})
    ins = obj.get("instruction_id", {})
    val = obj.get("valuation_id", {})

    # ----- Instruction -----
    instruction = (
        ins.get("input") or f"{ins.get('title_ko','')} 특허의 가치를 단계별로 계산하라."
    )

    # ----- Dynamic Context Builder -----
    ctx_parts = []

    # 1. 특허 기본 정보
    for key, label in [
        ("invention_title", "특허명"),
        ("application_number", "출원번호"),
        ("open_number", "공개번호"),
        ("register_number", "등록번호"),
        ("ipc_all", "IPC"),
        ("applicant_name", "출원인"),
    ]:
        val_ = p.get(key)
        if val_:
            ctx_parts.append(f"{label}: {val_}")

    # 2. 회사/산업 정보
    for key, label in [
        ("company_name", "회사명"),
        ("industry", "산업분류"),
        ("ksic", "KSIC 코드"),
        ("sales", "매출액"),
        ("net_income", "당기순이익"),
        ("asset", "총자산"),
        ("liabilities", "부채"),
        ("equity", "자본"),
    ]:
        val_ = c.get(key)
        if val_ not in [None, "", 0]:
            ctx_parts.append(f"{label}: {val_}")

    # 3. 가치평가 관련 파라미터
    for key, label in [
        ("royalty_rate", "로열티율(%)"),
        ("useful_life_years", "경제적 수명(년)"),
        ("wacc", "WACC(%)"),
        ("business_risk", "사업위험도"),
    ]:
        val_ = val.get(key, ins.get(key))
        if val_ not in [None, "", 0]:
            ctx_parts.append(f"{label}: {val_}")

    # 4. 키워드 및 요약
    if ins.get("keywords"):
        ctx_parts.append(f"핵심 키워드: {', '.join(ins['keywords'])}")
    if ins.get("abstract_ko"):
        ctx_parts.append(f"요약: {ins['abstract_ko'][:300]}...")

    # 5. 목표
    ctx_parts.append(
        "이 데이터를 기반으로 로열티 공제법(Royalty Relief Method)을 사용하여 특허의 경제적 가치를 계산하라."
    )

    # ----- Target Steps & Answer -----
    steps = "\n".join(ins.get("output", []))
    ans = ins.get("answer")

    return {
        "instruction": instruction.strip(),
        "context": "\n".join(ctx_parts).strip(),
        "target_steps": steps.strip(),
        "target_answer": ans,
    }


def generate_pair(d: dict) -> dict:
    question = (
        f"당신은 특허 가치평가 전문가입니다.\n"
        f"주어진 데이터를 바탕으로 로열티 공제법(Royalty Relief Method)을 사용하여 특허의 경제적 가치를 추정하세요.\n"
        f"모든 계산은 단계별로 명확히 제시하고, 각 단계마다 수식과 계산 근거를 설명하세요.\n"
        f"최종 단계에서는 할인 후 현재가치를 계산하여 결과를 제시합니다.\n\n"
        f"[INSTRUCTION]\n{d['instruction']}\n\n"
        f"[CONTEXT]\n{d.get('context','없음')}\n\n"
        f"[OUTPUT FORMAT]\n"
        f"1. 단계별 계산 과정 (예: 매출 추정 → 로열티율 적용 → 세후 조정 → 할인 계산)\n"
        f'2. 마지막 줄에는 반드시 "<{SPECIAL_ANS}>{{정수}}원</{SPECIAL_ANS}>" 형식으로 최종 특허 가치를 표시하세요.\n\n'
        f"[EXAMPLE]\n"
        "① ...\n"
        "② ...\n"
        "...\n\n"
        f"<{SPECIAL_ANS}>152,000,000원</{SPECIAL_ANS}>"
    )
    steps = d.get("target_steps", "")
    ans = d.get("target_answer", None)
    if ans is None:
        answer = steps
    else:
        answer = f"{steps}\n\n<{SPECIAL_ANS}>{int(ans):,}원</{SPECIAL_ANS}>"
    return {"question": question, "answer": answer}


def combine_datasets(
    ds1: DatasetDict, ds2: DatasetDict, ds_dir: str, version: str
) -> DatasetDict:
    """두 DatasetDict를 합쳐서 새로운 DatasetDict 생성"""
    combined = {}
    for split in ["train", "val", "test"]:
        combined_data = ds1[split].to_list() + ds2[split].to_list()
        combined[split] = Dataset.from_list(combined_data)
    ds = DatasetDict(combined)

    # Save to disk
    ds.save_to_disk(ds_dir / version)
    print(
        f"Data length: train({len(ds['train'])}), val({len(ds['val'])}), test({len(ds['test'])})"
    )
    return ds


dataset = generate_dataset(INPUT_DATA_PATH, OUTPUT_DATA_PATH)


#####################################################################################
# 1. Load model and tokenizer
#####################################################################################
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template


model, tokenizer = FastModel.from_pretrained(
    model_name=BASE_MODEL,
    load_in_4bit=False,
    load_in_8bit=False,
    load_in_16bit=True,
    full_finetuning=False,
)
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,  # Turn off for just text!
    finetune_language_layers=True,  # Should leave on!
    finetune_attention_modules=True,  # Attention good for GRPO
    finetune_mlp_modules=True,  # SHould leave on always!
    r=8,  # Larger = higher accuracy, but might overfit
    lora_alpha=8,  # Recommended alpha == r at least
    lora_dropout=0,
    bias="none",
    random_state=SEED,
)
tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


#####################################################################################
# 2. Load dataset
#####################################################################################
from datasets import load_from_disk


dataset = load_from_disk(OUTPUT_DATA_PATH)
train_dataset_raw = dataset["train"]
eval_dataset_raw = dataset["val"]
test_dataset_raw = dataset["test"]

del dataset


#####################################################################################
# 3. Format prompts
#####################################################################################
def formatting_prompts_func(examples):
    convos = [
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        for question, answer in zip(examples["question"], examples["answer"])
    ]
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False
        ).removeprefix(
            tokenizer.bos_token
        )  # NOTE: avoid duplicate bos_token as `trainer` adds it again
        for convo in convos
    ]
    return {"text": texts}


train_dataset = train_dataset_raw.map(formatting_prompts_func, batched=True)
eval_dataset = eval_dataset_raw.map(formatting_prompts_func, batched=True)
test_dataset = test_dataset_raw.map(formatting_prompts_func, batched=True)


#####################################################################################
# 4. Prepare trainer
#####################################################################################
import re
from time import time
from random import sample

import torch
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback
from unsloth.chat_templates import train_on_responses_only


trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    args=SFTConfig(
        output_dir=OUTPUT_MODEL_PATH,
        max_length=2250,
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=1,  # 3000 steps
        warmup_steps=30,
        learning_rate=2e-4,  # Reduce to 2e-5 for long training runs
        logging_steps=150,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=SEED,
        report_to="none",  # Use TrackIO/WandB etc
        save_strategy="steps",
        save_steps=600,
    ),
    # # NOTE: EVALUATION
    # callbacks=[
    #     GenerationEvalCallback(
    #         tokenizer,
    #         eval_dataset,
    #         batch_size=64,
    #         eval_steps=600,
    #     ),
    # ],
)
trainer = train_on_responses_only(
    trainer,
    instruction_part="<start_of_turn>user\n",
    response_part="<start_of_turn>model\n",
)

#####################################################################################
# 5. Run training
#####################################################################################
trainer.train()


#####################################################################################
# 6. Save model, tokenizer
#####################################################################################
model.save_pretrained_merged(OUTPUT_MODEL_PATH, tokenizer, save_method="merged_16bit")
