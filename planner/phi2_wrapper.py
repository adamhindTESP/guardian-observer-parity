# planner/phi2_wrapper.py

from typing import Dict, Any, Optional
from planner.base import PlannerInterface

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class Phi2Planner(PlannerInterface):
    """
    Phi-2 wrapper for local inference.

    Loads base model + optional LoRA locally.
    Does NOT ship weights in repo.
    """

    def __init__(
        self,
        base_model: Optional[str] = None,
        lora_path: Optional[str] = None,
        device: Optional[str] = "cpu",
        **kwargs
    ):
        if base_model is None:
            raise ValueError("Phi2 requires --base-model")

        self.device = device or "cpu"

        print(">>> Loading Phi-2 base model...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
        ).to(self.device)

        if lora_path:
            print(">>> Applying LoRA:", lora_path)
            self.model = PeftModel.from_pretrained(self.model, lora_path)

        self.model.eval()
        print(">>> Phi-2 ready.")

    def propose(self, instruction: str, context: Dict[str, Any]) -> str:
        """
        Generate structured physics proposal.

        Returns raw text response (parsed by caller).
        """

        prompt = f"""Return JSON only.
Schema: {{"output": {{"force_n": number, "velocity_mps": number, "distance_m": number}}}}

INSTRUCTION: {instruction}
CONTEXT_JSON: {json.dumps(context, sort_keys=True)}"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return text