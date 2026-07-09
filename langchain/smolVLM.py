from langchain_core.language_models import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from typing import Optional, List, Any
import torch
from transformers import AutoProcessor, AutoModel  # Change this

class SmolVLM(LLM):
    """Custom LLM wrapper for SmolVLM model."""

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_id: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    processor: Any = None
    model: Any = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(  # ✅ Use AutoModel instead
            self.model_id,
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2" if self.device == "cuda" else "eager",
        ).to(self.device)
    
    @property
    def _llm_type(self) -> str:
        return "smolvlm"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        chat_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=chat_prompt, return_tensors="pt")
        inputs = inputs.to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=500)
        output = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return output