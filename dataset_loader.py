import os
import json
from datasets import Dataset, DatasetDict, load_dataset
from typing import Dict, Any, Tuple


SYSTEM_PROMPT = "Sen Türkçe dilinde doğru, detaylı ve yardımsever yanıtlar veren uzman bir yapay zeka asistanısın."


def format_llama3_prompt(instruction: str, user_input: str = "", output: str = "") -> str:
    user_content = f"{instruction}\n\n{user_input}".strip() if user_input else instruction.strip()
    
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    if output:
        prompt += f"{output.strip()}<|eot_id|>"
    return prompt


def create_sample_dataset(file_path: str, count: int = 100):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    samples = [
        {
            "instruction": "Kuantum bilgisayarların klasik bilgisayarlardan temel farkını açıkla.",
            "input": "",
            "output": "Kuantum bilgisayarlar, klasik bilgisayarlardaki ikili bitler (0 veya 1) yerine kuantum süperpozisyon ve dolanıklık özelliklerini kullanan kubitler (qubit) ile çalışır. Bu sayede karmaşık problemleri üstel hızda çözebilirler."
        },
        {
            "instruction": "Aşağıdaki metni Türkçeye çevir ve özetle.",
            "input": "Artificial intelligence is transforming industries by automating repetitive tasks and providing deep analytical insights.",
            "output": "Çeviri: Yapay zekâ, tekrarlayan görevleri otomatikleştirerek ve derin analitik içgörüler sunarak endüstrileri dönüştürüyor.\nÖzet: Yapay zekâ otomasyon ve analiz gücüyle sektörleri kökten değiştirmektedir."
        },
        {
            "instruction": "Verilen cümlede anlatım bozukluğu var mı? Varsa düzelt.",
            "input": "Herkes sınavı kazandı ama hiç kimse sevinmedi.",
            "output": "Cümlede özne-yüklem uyumsuzluğu veya anlam karmaşası bulunmamaktadır; sıralı cümleler bağlaçla doğru şekilde bağlanmıştır."
        },
        {
            "instruction": "Python dilinde liste elemanlarını ters çeviren iki farklı yöntem göster.",
            "input": "",
            "output": "1. Dilimleme (slicing): `ters = liste[::-1]`\n2. Yerinde ters çevirme: `liste.reverse()` fonksiyonu ile doğrudan liste güncellenir."
        }
    ]

    with open(file_path, "w", encoding="utf-8") as f:
        for i in range(count):
            item = samples[i % len(samples)]
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[+] Örnek veri seti oluşturuldu: {file_path} ({count} satır)")


def load_and_prepare_dataset(
    dataset_path: str,
    test_size: float = 0.05,
    seed: int = 42
) -> DatasetDict:
    if not os.path.exists(dataset_path):
        print(f"[!] {dataset_path} bulunamadı, başlangıç için örnek şablon oluşturuluyor...")
        create_sample_dataset(dataset_path)

    raw_data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_data.append(json.loads(line))

    def map_to_text(example: Dict[str, Any]) -> Dict[str, str]:
        text = format_llama3_prompt(
            instruction=example.get("instruction", ""),
            user_input=example.get("input", ""),
            output=example.get("output", "")
        )
        return {"text": text}

    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(map_to_text)
    
    split_dataset = dataset.train_test_split(test_size=test_size, seed=seed)
    return split_dataset


if __name__ == "__main__":
    ds = load_and_prepare_dataset("data/turkish_instructions_50k.jsonl")
    print(f"Eğitim seti boyutu: {len(ds['train'])}, Doğrulama seti: {len(ds['test'])}")
    print("Örnek format:")
    print(ds["train"][0]["text"][:250] + "...")
