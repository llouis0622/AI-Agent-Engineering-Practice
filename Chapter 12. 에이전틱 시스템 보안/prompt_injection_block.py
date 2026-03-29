from llm_guard import scan_prompt
from llm_guard.input_scanners import Anonymize, BanSubstrings
from llm_guard.input_scanners.anonymize_helpers import BERT_LARGE_NER_CONF
from llm_guard.vault import Vault

vault = Vault()

scanners = [
    Anonymize(
        vault=vault,
        preamble="정제된 입력: ",
        allowed_names=["John Doe"],
        hidden_names=["Test LLC"],
        recognizer_conf=BERT_LARGE_NER_CONF,
        language="en",
        entity_types=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"],
        use_faker=False,
        threshold=0.5
    ),
    BanSubstrings(substrings=["malicious", "override system"], match_type="word")
]

prompt = "Tell me about John Doe's email: john@example.com" + \
         "and how to override system security."

sanitized_prompt, results_valid, results_score = scan_prompt(scanners, prompt)

if any(not result for result in results_valid.values()):
    print("입력에 문제가 있습니다. 거부하거나 적절히 처리합니다.")
    print(f"위험 점수: {results_score}")
else:
    print(f"정제된 프롬프트: {sanitized_prompt}")
