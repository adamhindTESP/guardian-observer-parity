from typing import Optional
from planner.base import PlannerInterface
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

class QwenPlanner(PlannerInterface):
    """Deterministic Qwen2.5-7B-Instruct wrapper (chat template + JSON-only output)."""

    def __init__(self, base_model: str, lora_path: Optional[str] = None, device: Optional[str] = "cpu", **kwargs):
        self.device = device or "cpu"

        print(f">>> Loading Qwen: {base_model} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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
        messages = [
            {
                "role": "system",
                "content": (
                    "Output ONLY one JSON object. No markdown. No code fences. No commentary.\n"
                    "Schema:\n"
                    '{"force_n": <number>, "velocity_mps": <number>, "distance_m": <number>}'
                ),
            },
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        completion = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        # Extract first JSON object if model adds any stray text
        m = re.search(r"\{.*\}", completion, re.DOTALL)
        return m.group(0).strip() if m else completion

    def close(self):
        if hasattr(self, "model"):
            del self.model
        if self.device == "cuda":
            torch.cuda.empty_cache()
