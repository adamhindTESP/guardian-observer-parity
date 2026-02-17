# planner/qwen_wrapper.py
from typing import Optional
from planner.base import PlannerInterface
import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM


class QwenPlanner(PlannerInterface):
    """Deterministic Qwen2.5-7B-Instruct wrapper"""

    def __init__(
        self,
        base_model: str,
        lora_path: Optional[str] = None,
        device: Optional[str] = "cpu",
        **kwargs
    ):
        self.device = device or "cpu"

        print(f">>> Loading Qwen: {base_model} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        if self.device != "cuda":
            self.model.to(self.device)

        self.model.eval()
        print(">>> Qwen ready")

    def propose(self, prompt: str) -> str:
        """
        Deterministic JSON-only generation.
        """

        prompt_data = json.loads(prompt)

        system_prompt = (
            "You are a safety-constrained robot planner.\n"
            "Output ONLY a valid JSON object with EXACT keys:\n"
            "force_n (float), velocity_mps (float), distance_m (float).\n"
            "No markdown. No explanation. No backticks.\n"
            "Example format:\n"
            '{"force_n": 0.5, "velocity_mps": 0.02, "distance_m": 0.3}'
        )

        user_prompt = f"Input:\n{json.dumps(prompt_data, indent=2)}\n\nOutput JSON:"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        completion = self.tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]) :],
            skip_special_tokens=True,
        ).strip()

        # Remove possible fenced blocks
        completion = completion.replace("```json", "").replace("```", "").strip()

        return completion

    def close(self):
        if hasattr(self, "model"):
            del self.model
        if self.device == "cuda":
            torch.cuda.empty_cache()
