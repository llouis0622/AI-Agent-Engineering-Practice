import argparse
import json
import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH = "ch07/fine_tuned_model/gemma-2-2B-function-call-ft"

DEFAULT_PROMPTS = [
    "오늘 날씨가 어때?",
    "서울에서 부산까지 거리가 얼마나 돼?",
    "내 이메일을 확인해줘.",
    "Python으로 Hello World를 출력하는 코드를 작성해줘.",
    "지금 몇 시야?",
]


def load_model(adapter_path: str = ADAPTER_PATH):
    """베이스 모델 + LoRA 어댑터 로드"""
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"어댑터 경로를 찾을 수 없습니다: {adapter_path}")

    print("모델 로딩 중...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    device_map = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "auto")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    base.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    print("✅ 모델 로드 완료\n")
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """프롬프트에 대해 응답 생성 (Hermes chat template 사용)"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=False)


def extract_tool_call(text: str) -> str | None:
    """생성된 텍스트에서 <tool_call> 블록 추출"""
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    return match.group(1) if match else None


def main():
    parser = argparse.ArgumentParser(description="SFT 함수 호출 모델 테스트")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="테스트할 프롬프트 (미지정 시 기본 예시들 실행)",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=ADAPTER_PATH,
        help=f"LoRA 어댑터 경로 (기본: {ADAPTER_PATH})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="생성할 최대 토큰 수",
    )
    args = parser.parse_args()

    model, tokenizer = load_model(args.adapter)
    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS

    for i, prompt in enumerate(prompts, 1):
        print(f"{'=' * 60}")
        print(f"[{i}] 사용자: {prompt}")
        print("-" * 60)
        response = generate(model, tokenizer, prompt, args.max_tokens)
        print(f"모델: {response.strip()}")
        tool_json = extract_tool_call(response)
        if tool_json:
            try:
                parsed = json.loads(tool_json)
                print(f"\n[추출된 도구 호출] {json.dumps(parsed, ensure_ascii=False, indent=2)}")
            except json.JSONDecodeError:
                pass
        print()


if __name__ == "__main__":
    main()
