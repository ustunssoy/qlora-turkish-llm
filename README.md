# Llama-3.1-8B Türkçe QLoRA Fine-Tuning

Tek bir **NVIDIA T4 (16 GB)** GPU üzerinde ~50k Türkçe talimat veri seti ile QLoRA ince ayar projesi.

## Özellikler
- **Model:** Meta-Llama-3.1-8B-Instruct
- **Kuantizasyon:** 4-bit NormalFloat (NF4) + Double Quantization (`bitsandbytes`)
- **Adaptör:** LoRA (`r=16`, `lora_alpha=32`, `target_modules` all linear)
- **Hafıza Optimizasyonu:** Gradient Checkpointing + Paged AdamW 8-bit + FP16 mixed precision + Gradient Accumulation (steps=8, batch=2)
- **Metrik:** ROUGE-L Baseline: `0.32` ➔ Fine-Tuned: `0.48`

## Kurulum
```bash
pip install -r requirements.txt
```

## Kullanım

### 1. Veri Hazırlığı
```bash
python dataset_loader.py
```
*(Varsayılan olarak `data/turkish_instructions_50k.jsonl` yolunu okur, yoksa şablon oluşturur.)*

### 2. Eğitim
```bash
python train.py
```

### 3. Değerlendirme (ROUGE)
```bash
python evaluate.py
```

### 4. Çıkarım (Inference)
```bash
python inference.py
```
