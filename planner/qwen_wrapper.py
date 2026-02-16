# planner/qwen_wrapper.py
from typing import Optional
from planner.base import PlannerInterface
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

class QwenPlanner(PlannerInterface):
    """Qwen2.5-7B-Instruct wrapper for Guardian planner"""
    
    def __init__(self, base_model: str, lora_path: Optional[str] = None, device: Optional[str] = "cpu", **kwargs):
        self.device = device or "cpu"
        
        print(f">>> Loading Qwen: {base_model} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        if self.device != "cuda":
            self.model.to(self.device)
        
        self.model.eval()
        print(">>> Qwen ready")
    
    def propose(self, prompt: str) -> str:
        """Generate JSON output for given prompt"""
        
        # Parse incoming JSON prompt
        prompt_data = json.loads(prompt)
        
        # Qwen chat format
        messages = [
            {"role": "system", "content": "You are a safety-constrained robot planner. Output ONLY valid JSON with keys: force_n, velocity_mps, distance_m. No explanations."},
            {"role": "user", "content": f"Input: {json.dumps(prompt_data)}\n\nOutput JSON:"}
        ]
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        completion = self.tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):], 
            skip_special_tokens=True
        ).strip()
        
        return completion
    
    def close(self):
        if hasattr(self, "model"):
            del self.model
        if self.device == "cuda":
            torch.cuda.empty_cache()
