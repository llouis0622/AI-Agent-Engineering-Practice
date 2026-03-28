import dspy

dspy.configure(lm=dspy.LM("gpt-5-mini"))


class ThreatClassifier(dspy.Signature):
    """주어진 인디케이터(IP, URL, 해시 등)의 위협 수준을
    'benign', 'suspicious', 'malicious' 중 하나로 분류합니다."""
    indicator: str = dspy.InputField(desc="IP 주소, URL, 파일 해시 등 분류할 인디케이터.")
    threat_level: str = dspy.OutputField(desc="분류된 위협 수준: 'benign', 'suspicious', 또는 'malicious'.")


class ThreatClassificationModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(ThreatClassifier)

    def forward(self, indicator):
        return self.classify(indicator=indicator)


trainset = [
    dspy.Example(indicator="203.0.113.45",
                 threat_level="suspicious").with_inputs('indicator'),
    dspy.Example(indicator="example.com/malware.exe",
                 threat_level="malicious").with_inputs('indicator'),
    dspy.Example(indicator="benign-site.net",
                 threat_level="benign").with_inputs('indicator'),
    dspy.Example(indicator="abc123def456",
                 threat_level="malicious").with_inputs('indicator'),
    dspy.Example(indicator="192.168.1.1",
                 threat_level="benign").with_inputs('indicator'),
    dspy.Example(indicator="obfuscated.url/with?params",
                 threat_level="suspicious").with_inputs('indicator'),
    dspy.Example(indicator="new-attack-vector-hash789",
                 threat_level="malicious").with_inputs('indicator'),
]


def threat_match_metric(example, pred, trace=None):
    return example.threat_level.lower() == pred.threat_level.lower()


optimizer = dspy.BootstrapFewShotWithRandomSearch(metric=threat_match_metric,
                                                  max_bootstrapped_demos=4, max_labeled_demos=4)
optimized_module = optimizer.compile(ThreatClassificationModule(),
                                     trainset=trainset)


def classify_threat(indicator: str) -> str:
    """최적화된 DSPy 모듈을 사용해 위협 수준을 분류합니다."""
    prediction = optimized_module(indicator=indicator)
    return prediction.threat_level
