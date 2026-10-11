import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig as PeftLoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer

from config import ModelConfig, LoraConfig, TrainingConfig, DataConfig
from dataset_loader import load_and_prepare_dataset


def build_bnb_config(model_cfg: ModelConfig) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=model_cfg.load_in_4bit,
        bnb_4bit_quant_type=model_cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=model_cfg.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=model_cfg.bnb_4bit_compute_dtype,
    )


def train():
    model_cfg = ModelConfig()
    lora_cfg = LoraConfig()
    train_cfg = TrainingConfig()
    data_cfg = DataConfig()

    print("=" * 60)
    print(" Llama-3.1-8B Türkçe QLoRA Eğitimi Başlatılıyor")
    print(f" Hedef Donanım: NVIDIA T4 (16GB VRAM)")
    print(f" Model: {model_cfg.model_name}")
    print(f" Kuantizasyon: 4-bit NF4 (Double Quantization, FP16 compute)")
    print("=" * 60)

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. 4-bit Quantized Model
    bnb_config = build_bnb_config(model_cfg)
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=model_cfg.torch_dtype,
        trust_remote_code=True,
    )

    # 3. K-Bit Hazırlık ve LoRA
    model = prepare_model_for_kbit_training(
        model, 
        use_gradient_checkpointing=train_cfg.gradient_checkpointing
    )

    peft_config = PeftLoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        task_type=lora_cfg.task_type,
        target_modules=lora_cfg.target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Veri Seti
    dataset_dict = load_and_prepare_dataset(
        dataset_path=data_cfg.dataset_name_or_path,
        test_size=data_cfg.eval_split_ratio,
        seed=train_cfg.seed,
    )

    # 5. Training Arguments (T4 için optimize)
    training_args = TrainingArguments(
        output_dir=train_cfg.output_dir,
        num_train_epochs=train_cfg.num_train_epochs,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=train_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        optim=train_cfg.optim,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        warmup_ratio=train_cfg.warmup_ratio,
        weight_decay=train_cfg.weight_decay,
        fp16=train_cfg.fp16,
        bf16=train_cfg.bf16,
        logging_steps=train_cfg.logging_steps,
        save_strategy=train_cfg.save_strategy,
        save_steps=train_cfg.save_steps,
        save_total_limit=train_cfg.save_total_limit,
        max_grad_norm=train_cfg.max_grad_norm,
        evaluation_strategy="steps",
        eval_steps=train_cfg.save_steps,
        seed=train_cfg.seed,
        report_to="none",
    )

    # 6. SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["test"],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=train_cfg.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("\n[+] Eğitim başlatılıyor...")
    trainer.train()

    # 7. Kayıt
    adapter_path = os.path.join(train_cfg.output_dir, "final_adapter")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\n[✓] Eğitim tamamlandı. LoRA ağırlıkları kaydedildi: {adapter_path}")


if __name__ == "__main__":
    train()
