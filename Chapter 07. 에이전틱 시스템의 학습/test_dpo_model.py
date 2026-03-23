import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH = "ch07/fine_tuned_model/phi3-mini-helpdesk-dpo"

DEFAULT_PROMPTS = [
    "비밀번호를 잊어버려서 이메일이 잠겼습니다.",
    "VPN이 매시간 연결이 끊깁니다.",
    "3층 로비의 프린터가 용지 걸림입니다.",
    "Google Drive의 재무 공유 폴더에 접근 권한이 필요합니다.",
]


def load_model(adapter_path: str = ADAPTER_PATH):
    """베이스 모델 + LoRA 어댑터 로드"""
    if not os.path.exists(adapter_path) and adapter_path == ADAPTER_PATH:
        fallback = "phi3-mini-helpdesk-dpo"
        if os.path.exists(fallback):
            adapter_path = fallback
            print(f"참고: {ADAPTER_PATH} 없음. {fallback} 사용 (이전 학습 결과)")
    print("모델 로딩 중...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    print("✅ 모델 로드 완료\n")
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """프롬프트에 대해 응답 생성"""
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


def main():
    parser = argparse.ArgumentParser(description="DPO 파인튜닝 모델 테스트")
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
        default=256,
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
        print()


if __name__ == "__main__":
    main()
