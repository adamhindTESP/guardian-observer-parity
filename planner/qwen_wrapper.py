# planner/qwen_wrapper.py
from typing import Optional
from planner.base import PlannerInterface
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM


class QwenPlanner(PlannerInterface):
    """Deterministic Qwen2.5-7B-Instruct wrapper - RAW PROMPT MODE"""

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
        """
        RAW PROMPT MODE - No chat template. Pure causal JSON generation.
        """
        prompt_data = json.loads(prompt)

        # ULTRA-STRICT PROMPT - No conversation, no thinking, JSON ONLY
        strict_prompt = f"""Return ONLY valid JSON object. No explanation. No markdown. No text.

{{"force_n": ?, "velocity_mps": ?, "distance_m": ?}}

Input: {json.dumps(prompt_data)}
JSON:"""

        inputs = self.tokenizer(
            strict_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,  # <- Tighter limit
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,  # <- Prevent loops
            )

        completion = self.tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True
        ).strip()

        # Clean any trailing junk but preserve JSON
        if completion.startswith("```"):
            completion = completion.split("```")[1].strip()
        if "```" in completion:
            completion = completion.split("```")[0].strip()

        print(f"RAW PLANNER OUTPUT: '{completion}'")  # <- DEBUG: Remove after testing
        
        return completion

    def close(self):
        if hasattr(self, "model"):
            del self.model
        if self.device == "cuda":
            torch.cuda.empty_cache()
