
# planner/tinyllama_wrapper.py
from typing import Optional
from planner.base import PlannerInterface
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class TinyLlamaPlanner(PlannerInterface):
    
    def __init__(self, base_model: str, device: Optional[str] = None, **kwargs):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f">>> Loading TinyLlama ({base_model}) on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()
        print(">>> TinyLlama ready.")

    def propose(self, prompt: str) -> str:  # ✅ FIXED: single str input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Strip prompt tokens
        gen_start = inputs["input_ids"].shape[1]
        generated = self.tokenizer.decode(output[0][gen_start:], skip_special_tokens=True)
        return generated
