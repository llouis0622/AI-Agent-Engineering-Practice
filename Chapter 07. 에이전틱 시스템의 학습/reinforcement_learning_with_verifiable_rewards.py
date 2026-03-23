import json
import logging
import os
import re
from typing import Any, Dict, List
import torch
from datasets import load_dataset
from huggingface_hub import constants as hf_constants
from trl import GRPOConfig, GRPOTrainer

RLVR_DATA = "ch07/training_data/rlvr_it_help_desk_training_data.jsonl"


def _is_model_cached(repo_id: str) -> bool:
    """Hugging Face 모델이 로컬 캐시에 있는지 확인"""
    if os.path.exists(repo_id) and os.path.isdir(repo_id):
        return True
    cache_folder = "models--" + repo_id.replace("/", "--")
    cache_path = os.path.join(hf_constants.HF_HUB_CACHE, cache_folder)
    return os.path.exists(cache_path)


GRPO_MODEL = "Qwen/Qwen2-0.5B-Instruct"
logger = logging.getLogger(__name__)
if not _is_model_cached(GRPO_MODEL):
    logger.warning("로컬 경로를 찾을 수 없습니다. Hub에서 '%s'를 다운로드합니다.", GRPO_MODEL)

dataset = load_dataset("json", data_files=RLVR_DATA, split="train")


def reward_tool_call_quality(completions: List[str], **kwargs) -> List[float]:
    """
    함수 호출 품질에 대한 세밀한 보상 함수.

    보상 체계:
    - 올바른 도구명 + 유효한 JSON + 필수 매개변수: +1.0
    - 올바른 도구명 + 유효한 JSON + 누락된 매개변수: +0.5
    - 올바른 도구명 + 잘못된 JSON: +0.2
    - 잘못된 도구명 + 유효한 JSON: -0.3
    - 도구 호출 없음 또는 완전히 잘못됨: -1.0
    """
    labels = kwargs.get('label', [])
    expected_params = kwargs.get('required_params', [])  # 선택사항: 필수 매개변수 이름 목록

    rewards = []
    num_generations = kwargs.get('num_generations', getattr(trainer.args, 'num_generations', 1))

    for i, completion in enumerate(completions):
        label_idx = i // num_generations
        if label_idx >= len(labels):
            rewards.append(-1.0)
            continue

        label = labels[label_idx]
        expected_tool = label.lower().strip()

        tool_match = re.search(
            r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
            completion,
            re.DOTALL
        )

        if not tool_match:
            rewards.append(-1.0)
            continue

        tool_json_str = tool_match.group(1)

        try:
            tool_call = json.loads(tool_json_str)
        except json.JSONDecodeError:
            if expected_tool in tool_json_str.lower():
                rewards.append(0.2)
            else:
                rewards.append(-0.5)
            continue

        tool_name = tool_call.get('name', '').lower().strip()

        tool_name_correct = (
                expected_tool in tool_name or
                tool_name in expected_tool or
                tool_name == expected_tool
        )

        if not tool_name_correct:
            rewards.append(-0.3)
            continue

        if expected_params and label_idx < len(expected_params):
            required = expected_params[label_idx]
            provided_params = tool_call.get('parameters', tool_call.get('arguments', {}))

            if isinstance(provided_params, dict):
                has_all_required = all(
                    param in provided_params and provided_params[param] not in [None, '', []]
                    for param in required
                )

                if has_all_required:
                    rewards.append(1.0)
                else:
                    rewards.append(0.5)
            else:
                rewards.append(0.5)
        else:
            rewards.append(1.0)

    return rewards


def reward_format_compliance(completions: List[str], **kwargs) -> List[float]:
    """
    형식 준수를 위한 보상 함수.
    올바른 XML 태그, JSON 구조 등을 확인합니다.
    """
    rewards = []

    for completion in completions:
        reward = 0.0

        if '<tool_call>' in completion and '</tool_call>' in completion:
            reward += 0.3

        if completion.count('{') == completion.count('}'):
            reward += 0.2

        tool_match = re.search(r'<tool_call>(.*?)</tool_call>', completion, re.DOTALL)
        if tool_match:
            tool_content = tool_match.group(1).strip()
            if '"name"' in tool_content or "'name'" in tool_content:
                reward += 0.3

            try:
                json.loads(tool_content)
                reward += 0.2
            except:
                pass

        rewards.append(reward)

    return rewards


def combined_reward(completions: List[str], **kwargs) -> List[float]:
    """여러 보상 신호의 가중치 결합."""
    quality_rewards = reward_tool_call_quality(completions, **kwargs)
    format_rewards = reward_format_compliance(completions, **kwargs)

    combined = [
        0.8 * q + 0.2 * f
        for q, f in zip(quality_rewards, format_rewards)
    ]

    return combined


grpo_config = GRPOConfig(
    output_dir="ch07/fine_tuned_model/qwen-helpdesk-rlvr",
    num_generations=4,
    learning_rate=5e-6,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=1,
    save_strategy="epoch",
    report_to=None,
)

trainer = GRPOTrainer(
    model=GRPO_MODEL,
    reward_funcs=combined_reward,
    train_dataset=dataset,
    args=grpo_config,
)

print(f"✅ 모델 로드 완료")
print(f"훈련 장치: {trainer.model.device}")

print("훈련을 시작합니다...")
trainer.train()

print(f"✅ 훈련 완료! 모델 저장: {grpo_config.output_dir}")
