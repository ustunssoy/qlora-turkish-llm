import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from config import ModelConfig, TrainingConfig
from dataset_loader import format_llama3_prompt


class TurkishLLMPipeline:
    def __init__(self, adapter_path: str = None):
        self.model_cfg = ModelConfig()
        self.train_cfg = TrainingConfig()
        self.adapter_path = adapter_path or os.path.join(self.train_cfg.output_dir, "final_adapter")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.model_cfg.load_in_4bit,
            bnb_4bit_quant_type=self.model_cfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.model_cfg.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=self.model_cfg.bnb_4bit_compute_dtype,
        )

        print("[*] Tokenizer yükleniyor...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_cfg.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("[*] Temel 4-bit model yükleniyor...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_cfg.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=self.model_cfg.torch_dtype,
        )

        if os.path.exists(self.adapter_path):
            print(f"[*] LoRA adaptörü ekleniyor: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        else:
            print(f"[!] Uyarı: LoRA adaptörü bulunamadı ({self.adapter_path}), temel model kullanılacak.")

        self.model.eval()

    def generate(
        self,
        instruction: str,
        user_input: str = "",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        prompt = format_llama3_prompt(instruction, user_input)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        gen_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


if __name__ == "__main__":
    pipeline = TurkishLLMPipeline()
    
    test_prompts = [
        "Türkiye'de yenilenebilir enerji kaynaklarının gelişimini 2 maddede açıkla.",
        "Aşağıdaki e-postayı profesyonel bir üslupla yanıtla:\n'Fiyat teklifinizi aldık, indirim yapabilir misiniz?'",
    ]

    print("\n--- Test Çıkarımları ---")
    for prompt in test_prompts:
        print(f"\n[Soru/Talimat]: {prompt}")
        cevap = pipeline.generate(prompt)
        print(f"[Model Yanıtı]:\n{cevap}\n{'-'*50}")
