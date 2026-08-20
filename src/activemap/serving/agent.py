"""Structured local-HuggingFace policy for the ActiveMap agent."""

from __future__ import annotations

import json
from typing import Any, cast

from activemap.agent.records import AgentAction, AgentObservation
from activemap.agent.tools import GreedyAgentPolicy


def _first_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("model output does not contain a JSON object")


class HuggingFaceAgentPolicy:
    def __init__(
        self,
        model_path: str,
        *,
        device: str,
        max_new_tokens: int = 160,
        fallback_to_greedy: bool = True,
        enable_thinking: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map=device,
        ).eval()
        self.max_new_tokens = max_new_tokens
        self.fallback_to_greedy = fallback_to_greedy
        self.enable_thinking = enable_thinking
        self.greedy = GreedyAgentPolicy()
        self.last_source = "uninitialized"
        self.last_raw_output = ""

    def _prompt(self, observation: AgentObservation) -> str:
        schema = (
            '{"action":"ACQUIRE","evidence_id":"id"} | '
            '{"action":"USE_TOOL","tool_call":{"call_id":"id","tool":"TOOL",'
            '"inputs":{},"parameters":{}}} | '
            '{"action":"COMMIT","edit":"ADD|DELETE|RESHAPE"} | '
            '{"action":"REJECT"}'
        )
        return (
            "You control an editable-map maintenance system. Select exactly one valid action. "
            "Respect remaining budget and only use listed evidence IDs/tools. Return JSON only.\n"
            f"Allowed schemas: {schema}\n"
            f"Observation: {observation.model_dump_json()}"
        )

    def act(self, observation: AgentObservation) -> AgentAction:
        import torch

        messages = [{"role": "user", "content": self._prompt(observation)}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        encoded = self.tokenizer(prompt, return_tensors="pt")
        target_device = self.model.device
        encoded = {key: value.to(target_device) for key, value in encoded.items()}
        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, encoded["input_ids"].shape[1] :]
        self.last_raw_output = cast(
            str, self.tokenizer.decode(generated, skip_special_tokens=True)
        )
        try:
            action = AgentAction.model_validate(_first_json_object(self.last_raw_output))
            self.last_source = "model"
            return action
        except (ValueError, TypeError):
            if not self.fallback_to_greedy:
                raise
            self.last_source = "greedy_fallback"
            return self.greedy.act(observation)
