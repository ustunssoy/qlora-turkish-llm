import torch
from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    torch_dtype: torch.dtype = torch.float16
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: torch.dtype = torch.float16


@dataclass
class LoraConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class TrainingConfig:
    output_dir: str = "./outputs/llama3-8b-turkish-qlora"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    optim: str = "paged_adamw_8bit"
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    fp16: bool = True
    bf16: bool = False  # T4 natively accelerates fp16
    gradient_checkpointing: bool = True
    max_grad_norm: float = 0.3
    logging_steps: int = 25
    save_strategy: str = "steps"
    save_steps: int = 200
    save_total_limit: int = 2
    max_seq_length: int = 512
    seed: int = 42


@dataclass
class DataConfig:
    dataset_name_or_path: str = "data/turkish_instructions_50k.jsonl"
    eval_split_ratio: float = 0.05
    instruction_column: str = "instruction"
    input_column: str = "input"
    output_column: str = "output"
