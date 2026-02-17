# planner/qwen_wrapper.py
from typing import Optional
from planner.base import PlannerInterface
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM


class QwenPlanner(PlannerInterface):
    """Deterministic Qwen2.5-7B-Instruct wrapper - CHAT TEMPLATE + CLEAN JSON"""

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
        CHAT TEMPLATE MODE - Matches Qwen training. Clean JSON only.
        """
        prompt_data = json.loads(prompt)

        # ULTRA-CLEAN SYSTEM PROMPT - No example (reduces formatting)
        system_prompt = (
            "You are a robot motion planner. "
            "Output ONLY valid JSON: {\"force_n\": float, \"velocity_mps\": float, \"distance_m\": float}. "
            "No text. No markdown. No explanation."
        )

        # DIRECT USER PROMPT - No "Input:" framing
        user_prompt = json.dumps(prompt_data)

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
                max_new_tokens=64,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.eos_token_id,
                # NO repetition_penalty - conflicts with deterministic mode
            )

        completion = self.tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True,
        ).strip()

        # Clean any lingering markdown (chat models sometimes add)
        completion = completion.replace("```json", "").replace("```", "").strip()
        completion = completion.replace("```", "").strip()

        print(f"🔍 RAW PLANNER OUTPUT: '{completion}'")  # DEBUG - REMOVE AFTER SUCCESS
        
        return completion

    def close(self):
        if hasattr(self, "model"):
            del self.model
        if self.device == "cuda":
            torch.cuda.empty_cache()
