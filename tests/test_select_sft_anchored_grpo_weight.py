from scripts.select_sft_anchored_grpo_weight import parse_candidate


def test_parse_candidate() -> None:
    label, weight, path = parse_candidate("sftw010=0.10=result.jsonl")
    assert label == "sftw010"
    assert weight == 0.10
    assert path.name == "result.jsonl"
